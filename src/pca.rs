//! Principal component analysis of a set of spectra, with outlier statistics.
//!
//! Spectra are passed as one flat row-major slice: row `i` holds the
//! `variables` intensities of sample `i`. The model mean-centers the data and
//! keeps the leading components; it never scales individual variables, because
//! spectral variables share one unit and autoscaling would amplify noise.
//!
//! [`Pca::project`] places a further spectrum in the model and reports
//! Hotelling's T² and the Q residual, the two standard outlier statistics.
use crate::{Error, numeric};

/// Distance of one spectrum from the centre of the model and from its plane.
///
/// Both statistics are needed: T² finds a spectrum that is extreme in known
/// directions, Q finds one that carries variation the model does not describe.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Diagnostics {
    /// Hotelling's T², the squared distance inside the component plane,
    /// `Σ t²/λ` over the components.
    ///
    /// Control limits follow the F distribution:
    /// `T²_limit = k(n²−1)/(n(n−k)) · F(k, n−k; α)`.
    pub hotelling_t2: f64,
    /// Squared distance to the component plane, also called the squared
    /// prediction error.
    ///
    /// Control limits follow Jackson and Mudholkar, from the discarded
    /// eigenvalues.
    pub q_residual: f64,
}

/// Scores and diagnostics of one projected spectrum.
#[derive(Debug, Clone, PartialEq)]
pub struct Projection {
    /// One score per component.
    pub scores: Vec<f64>,
    /// Outlier statistics of the spectrum.
    pub diagnostics: Diagnostics,
}

/// A fitted principal component model.
///
/// Fitting takes O(samples × variables × min(samples, variables)) time, holds
/// one centred copy of the data besides the factors of the decomposition, and
/// runs on a single thread. The model itself stores the mean, the loadings and
/// the training scores.
///
/// # Example
/// ```
/// use chemometrics::pca::Pca;
/// // Three variables, five samples that vary along one direction only.
/// let data = [
///     1.0, 2.0, 3.0, //
///     2.0, 4.0, 6.0, //
///     3.0, 6.0, 9.0, //
///     4.0, 8.0, 12.0, //
///     5.0, 10.0, 15.0,
/// ];
/// let model = Pca::fit(&data, 3, 1)?;
/// // One component explains everything, so nothing is left over.
/// assert!((model.explained_variance_ratio()[0] - 1.0).abs() < 1e-12);
/// let projection = model.project(&[2.0, 4.0, 6.0])?;
/// assert!(projection.diagnostics.q_residual < 1e-20);
/// # Ok::<(), chemometrics::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Pca {
    samples: usize,
    variables: usize,
    components: usize,
    mean: Vec<f64>,
    loadings: Vec<f64>,
    scores: Vec<f64>,
    eigenvalues: Vec<f64>,
    ratios: Vec<f64>,
    total_variance: f64,
    /// Power-of-two scale of the training data.
    scale: numeric::Scale,
    /// Score standard deviations in scaled units, which stay precise where the
    /// eigenvalues in data units have become subnormal.
    deviations: Vec<f64>,
}

/// Number of samples in a row-major slice, or a shape error.
fn samples_of(length: usize, variables: usize) -> Result<usize, Error> {
    if variables == 0 || length == 0 || length % variables != 0 {
        return Err(Error::InvalidDataShape { length, variables });
    }
    Ok(length / variables)
}

impl Pca {
    /// Fits a model to row-major `data` with `variables` values per sample.
    ///
    /// `components` must be between one and `min(samples − 1, variables)`.
    ///
    /// # Errors
    /// Checks the data shape, then the sample count, then the component count,
    /// then non-finite values, in that order. Returns
    /// [`Error::NumericalFailure`] if a requested component is not
    /// distinguishable from rounding in the data, so its direction would be
    /// arbitrary, or if the variances of such large or small values overflow or
    /// underflow. A singular value counts as rounding at or below
    /// `f64::EPSILON · max(samples, variables) · ‖X‖_F`, with the norm of the
    /// uncentred data, since centring cannot remove rounding smaller than the
    /// values themselves. Returns [`Error::AllocationFailure`] if the model or
    /// the working memory cannot be reserved; allocations inside the
    /// decomposition's matrix kernels may still abort on failure.
    ///
    /// # Example
    /// ```
    /// use chemometrics::pca::Pca;
    /// let data = [0.0, 1.0, 1.0, 3.0, 2.0, 5.0, 3.0, 7.0];
    /// let model = Pca::fit(&data, 2, 1)?;
    /// assert_eq!(model.samples(), 4);
    /// assert_eq!(model.mean().len(), 2);
    /// # Ok::<(), chemometrics::Error>(())
    /// ```
    pub fn fit(data: &[f64], variables: usize, components: usize) -> Result<Self, Error> {
        let samples = samples_of(data.len(), variables)?;
        if samples < 2 {
            return Err(Error::TooFewSamples {
                length: samples,
                minimum: 2,
            });
        }
        let maximum = (samples - 1).min(variables);
        if components == 0 || components > maximum {
            return Err(Error::InvalidComponentCount {
                requested: components,
                maximum,
            });
        }
        if let Some(index) = data.iter().position(|x| !x.is_finite()) {
            return Err(Error::NonFiniteInput { index });
        }
        // Scale before centring: sums of values near f64::MAX would otherwise
        // overflow while forming the column means.
        let scale = numeric::Scale::new(data);
        let mut centered = numeric::zeros(data.len())?;
        // Centring leaves rounding of the order of EPSILON times the uncentred
        // values, which a tolerance relative to the centred data cannot see.
        let uncentred_norm = numeric::sum(data.iter().map(|x| {
            let x = scale.divide(*x);
            x * x
        }))
        .sqrt();
        for (value, x) in centered.iter_mut().zip(data) {
            *value = scale.divide(*x);
        }
        let mut mean = numeric::zeros(variables)?;
        for (j, value) in mean.iter_mut().enumerate() {
            let column = centered.iter().skip(j).step_by(variables).copied();
            *value = numeric::sum(column) / samples as f64;
        }
        for row in centered.chunks_exact_mut(variables) {
            for (value, m) in row.iter_mut().zip(&mean) {
                *value -= m;
            }
        }
        let degrees = (samples - 1) as f64;
        let variance = numeric::sum(centered.iter().map(|x| x * x)) / degrees;
        let svd = numeric::thin_svd(&centered, samples, variables, components)?;
        let threshold = f64::EPSILON * samples.max(variables) as f64 * uncentred_norm;
        if svd.values.iter().any(|s| *s <= threshold) {
            return Err(Error::NumericalFailure);
        }
        // Shares and T² do not depend on the scale, so they are taken from the
        // scaled values, which keep full precision for any data magnitude.
        let mut eigenvalues = numeric::zeros(components)?;
        let mut ratios = numeric::zeros(components)?;
        let mut deviations = numeric::zeros(components)?;
        for (a, value) in svd.values.iter().enumerate() {
            let scaled = value * value / degrees;
            eigenvalues[a] = scale.restore_squared(scaled);
            ratios[a] = scaled / variance;
            deviations[a] = scaled.sqrt();
        }
        // The product is bounded by the data length, so it cannot overflow.
        let mut scores = numeric::zeros(samples * components)?;
        for (row, left) in scores
            .chunks_exact_mut(components)
            .zip(svd.left.chunks_exact(components))
        {
            for ((score, u), value) in row.iter_mut().zip(left).zip(&svd.values) {
                *score = scale.restore(u * value);
            }
        }
        for value in &mut mean {
            *value = scale.restore(*value);
        }
        let mut model = Self {
            samples,
            variables,
            components,
            mean,
            loadings: svd.right,
            scores,
            eigenvalues,
            ratios,
            total_variance: scale.restore_squared(variance),
            scale,
            deviations,
        };
        model.orient();
        model.usable()?;
        Ok(model)
    }

    /// Number of samples the model was fitted to.
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// Number of variables per sample.
    pub fn variables(&self) -> usize {
        self.variables
    }

    /// Number of retained components.
    pub fn components(&self) -> usize {
        self.components
    }

    /// Mean spectrum subtracted before the decomposition.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// All loading vectors, `components` × `variables` row-major.
    ///
    /// The loadings are orthonormal. The sign of a component is fixed so that
    /// its largest-magnitude loading is positive; the first such variable wins
    /// a tie. Components with nearly equal eigenvalues are not determined by
    /// the data and may differ between platforms or library versions.
    pub fn loadings(&self) -> &[f64] {
        &self.loadings
    }

    /// Loading vector of one component, or `None` if it does not exist.
    pub fn loading(&self, component: usize) -> Option<&[f64]> {
        self.loadings.chunks_exact(self.variables).nth(component)
    }

    /// Training scores, `samples` × `components` row-major.
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Scores of one training sample, or `None` if it does not exist.
    pub fn score(&self, sample: usize) -> Option<&[f64]> {
        self.scores.chunks_exact(self.components).nth(sample)
    }

    /// Variance of each component's training scores, using `samples − 1`.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Share of the total variance carried by each component.
    ///
    /// The total is the variance of the centred data over all variables, not
    /// only the part the retained components cover, so the shares sum to one
    /// exactly when every component is retained.
    pub fn explained_variance_ratio(&self) -> &[f64] {
        &self.ratios
    }

    /// Total variance of the centred training data.
    pub fn total_variance(&self) -> f64 {
        self.total_variance
    }

    /// Projects a spectrum into a newly allocated [`Projection`].
    ///
    /// Returns [`Error::AllocationFailure`] if the scores cannot be reserved.
    pub fn project(&self, spectrum: &[f64]) -> Result<Projection, Error> {
        let mut scores = numeric::zeros(self.components)?;
        let diagnostics = self.project_into(spectrum, &mut scores)?;
        Ok(Projection {
            scores,
            diagnostics,
        })
    }

    /// Projects a spectrum into a caller-owned buffer without allocating.
    ///
    /// The buffer holds one score per component. Invalid inputs leave it
    /// unchanged; a numerical failure can leave it partially written.
    ///
    /// # Errors
    /// Checks the spectrum length, then the buffer length, then non-finite
    /// values, in that order. Returns [`Error::NumericalFailure`] if the
    /// projection is not representable.
    pub fn project_into(&self, spectrum: &[f64], scores: &mut [f64]) -> Result<Diagnostics, Error> {
        if spectrum.len() != self.variables {
            return Err(Error::InvalidSpectrumLength {
                expected: self.variables,
                actual: spectrum.len(),
            });
        }
        if scores.len() != self.components {
            return Err(Error::OutputLengthMismatch {
                expected: self.components,
                actual: scores.len(),
            });
        }
        if let Some(index) = spectrum.iter().position(|x| !x.is_finite()) {
            return Err(Error::NonFiniteInput { index });
        }
        // Both terms are scaled separately, so their difference stays finite
        // even for spectra near f64::MAX.
        let centered = |(x, m): (&f64, &f64)| self.scale.divide(*x) - self.scale.divide(*m);
        let centered_spectrum = || spectrum.iter().zip(&self.mean).map(centered);
        let loadings = || self.loadings.chunks_exact(self.variables);
        for (score, loading) in scores.iter_mut().zip(loadings()) {
            *score = numeric::sum(centered_spectrum().zip(loading).map(|(d, p)| d * p));
        }
        let unscaled: &[f64] = scores;
        // The residual is formed directly; ‖x−x̄‖² − ‖t‖² would cancel.
        let q_residual = numeric::sum(centered_spectrum().enumerate().map(|(j, d)| {
            let fitted = numeric::sum(loadings().zip(unscaled).map(|(p, t)| t * p[j]));
            let residual = d - fitted;
            residual * residual
        }));
        for score in scores.iter_mut() {
            *score = self.scale.restore(*score);
        }
        // T² is scale invariant, so scores and deviations are compared in the
        // scaled units of the training data.
        let hotelling_t2 =
            numeric::sum(scores.iter().zip(&self.deviations).map(|(t, deviation)| {
                let ratio = self.scale.divide(*t) / deviation;
                ratio * ratio
            }));
        let diagnostics = Diagnostics {
            hotelling_t2,
            q_residual: self.scale.restore_squared(q_residual),
        };
        if !diagnostics.hotelling_t2.is_finite()
            || !diagnostics.q_residual.is_finite()
            || scores.iter().any(|t| !t.is_finite())
        {
            return Err(Error::NumericalFailure);
        }
        Ok(diagnostics)
    }

    /// Fixes each component's sign by its largest-magnitude loading.
    fn orient(&mut self) {
        let Self {
            loadings,
            scores,
            variables,
            components,
            ..
        } = self;
        for (a, loading) in loadings.chunks_exact_mut(*variables).enumerate() {
            let extreme = loading.iter().fold(0.0_f64, |best, value| {
                if value.abs() > best.abs() {
                    *value
                } else {
                    best
                }
            });
            if extreme >= 0.0 {
                continue;
            }
            for value in loading.iter_mut() {
                *value = -*value;
            }
            for score in scores.chunks_exact_mut(*components) {
                score[a] = -score[a];
            }
        }
    }

    /// Rejects models whose rescaled values left the representable range.
    ///
    /// An eigenvalue that underflowed to zero would leave a model whose every
    /// projection fails, so it is rejected here rather than later.
    fn usable(&self) -> Result<(), Error> {
        let finite = |values: &[f64]| values.iter().all(|x| x.is_finite());
        if self.total_variance.is_finite()
            && self.total_variance > 0.0
            && self.eigenvalues.iter().all(|v| *v > 0.0)
            && finite(&self.mean)
            && finite(&self.ratios)
            && finite(&self.scores)
        {
            return Ok(());
        }
        Err(Error::NumericalFailure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_of_tied_extremes_fixes_the_sign() {
        let mut model = Pca {
            samples: 2,
            variables: 2,
            components: 1,
            mean: vec![0.0; 2],
            loadings: vec![-0.5, 0.5],
            scores: vec![1.0, -1.0],
            eigenvalues: vec![1.0],
            ratios: vec![1.0],
            total_variance: 1.0,
            scale: numeric::Scale::new(&[1.0]),
            deviations: vec![1.0],
        };
        model.orient();
        assert_eq!(model.loadings, [0.5, -0.5]);
        assert_eq!(model.scores, [-1.0, 1.0]);
    }
}
