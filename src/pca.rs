//! Principal component analysis of a set of spectra, with outlier statistics.
//!
//! [`Pca::fit`] builds a model from a set of spectra, and [`Pca::project`]
//! places a further spectrum in it and reports Hotelling's T² and the Q
//! residual, the two standard outlier statistics.
//!
//! # Examples
//! ```
//! use chemometrics::pca::Pca;
//!
//! // Reference spectra at five wavelengths, one row per spectrum: a single
//! // band whose height follows the concentration.
//! let wavelengths = 5;
//! let references = [
//!     0.10, 0.31, 0.50, 0.29, 0.10,
//!     0.12, 0.36, 0.61, 0.36, 0.11,
//!     0.16, 0.45, 0.75, 0.44, 0.15,
//!     0.17, 0.52, 0.85, 0.51, 0.17,
//!     0.20, 0.60, 0.99, 0.61, 0.20,
//! ];
//! // One component describes the band.
//! let model = Pca::fit(&references, wavelengths, 1)?;
//!
//! // A higher concentration than in the references stands out in T², an
//! // unexpected band in Q.
//! let high = model.project(&[0.32, 0.96, 1.60, 0.96, 0.32])?.diagnostics;
//! let band = model.project(&[0.14, 0.42, 0.70, 0.52, 0.34])?.diagnostics;
//! assert!(high.hotelling_t2 > 10.0 && high.q_residual < 0.01);
//! assert!(band.hotelling_t2 < 1.0 && band.q_residual > 0.01);
//! # Ok::<(), chemometrics::Error>(())
//! ```
//!
//! # Data layout
//! Spectra are passed as one flat row-major slice: row `i` holds the
//! `variables` intensities of sample `i`, as in a C-ordered NumPy array or a
//! row-major `ndarray` view. Spectra preprocessed one at a time are written
//! into such a slice with `apply_into`, as [`Pca::fit`] shows. Column-major
//! matrices, such as nalgebra's `DMatrix` or Fortran-ordered NumPy arrays, must
//! be transposed first; a slice of the same length in the wrong order cannot be
//! detected and gives a meaningless model.
//!
//! # Centering and scaling
//! The model mean-centers the data and keeps the leading components.
//! [`Pca::fit`] does not scale individual variables, because spectral
//! variables share one unit and autoscaling would amplify noise.
//!
//! # Wavelengths
//! Wavelengths are not part of the input, so loadings are read against the
//! caller's own axis. All spectra, including those passed to [`Pca::project`],
//! must share one wavelength grid in the same order; the length check cannot
//! detect a different grid. Each column counts equally, so a densely sampled
//! region weighs more.
use crate::{Error, numeric};
use std::fmt;

/// Maximum allowed entrywise deviation of `P Pᵀ` from the identity when
/// loading a model with [`Pca::from_parts`].
///
/// Rounding to twelve significant digits stays far below it, while
/// unnormalized or non-orthogonal rows exceed it. Precision lost to `f32`
/// storage averages out over many variables, so it cannot be detected
/// reliably; it changes T² and Q only in about the seventh digit.
const ORTHONORMAL_TOLERANCE: f64 = 1e-9;

/// Distance of one spectrum from the center of the model and from its plane.
///
/// Both statistics are needed: T² finds a spectrum that is extreme in known
/// directions, Q finds one that carries variation the model does not describe.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Diagnostics {
    /// Hotelling's T², the squared distance inside the component plane,
    /// `Σ t²/λ` over the components, with the eigenvalues λ of
    /// [`Pca::eigenvalues`].
    ///
    /// With k components, n training spectra and significance level α, control
    /// limits for new spectra follow the F distribution:
    /// `T²_limit = k(n²−1)/(n(n−k)) · F(k, n−k; α)`. The training samples took
    /// part in the fit and follow a Beta distribution instead:
    /// `T²_limit = (n−1)²/n · Beta(k/2, (n−k−1)/2; α)`.
    pub hotelling_t2: f64,
    /// Squared distance to the component plane, also called the squared
    /// prediction error.
    ///
    /// Control limits follow Jackson and Mudholkar, from the discarded
    /// eigenvalues in [`Pca::all_eigenvalues`].
    pub q_residual: f64,
}

/// Scores and diagnostics of one projected spectrum.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Projection {
    /// One score per component.
    pub scores: Vec<f64>,
    /// Outlier statistics of the spectrum.
    pub diagnostics: Diagnostics,
}

/// A fitted principal component model.
///
/// Fitting takes O(samples × variables × min(samples, variables)) time on a
/// single thread and temporarily needs a few times the memory of the data. The
/// model keeps the mean, loadings, variances and, when fitted with
/// [`Pca::fit`], the training scores, but not the data; `Debug` prints only
/// its shape and eigenvalues.
#[derive(Clone)]
pub struct Pca {
    samples: usize,
    variables: usize,
    components: usize,
    mean: Vec<f64>,
    loadings: Vec<f64>,
    scores: Vec<f64>,
    /// All `min(samples − 1, variables)` eigenvalues, the kept ones first.
    eigenvalues: Vec<f64>,
    ratios: Vec<f64>,
    total_variance: f64,
}

impl fmt::Debug for Pca {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pca")
            .field("samples", &self.samples)
            .field("variables", &self.variables)
            .field("components", &self.components)
            .field("eigenvalues", &self.eigenvalues())
            .finish_non_exhaustive()
    }
}

impl Pca {
    /// Fits a model to row-major `data` with `variables` values per sample.
    ///
    /// `components` must be between one and `min(samples − 1, variables)`.
    ///
    /// # Errors
    /// Checks the data shape, then the number of spectra, then the component
    /// count, then non-finite values, in that order; the index in
    /// [`Error::NonFiniteInput`] refers to `data`, so the affected spectrum is
    /// `index / variables`. Returns [`Error::InsufficientRank`], with the
    /// number of components the data support, if a requested component cannot
    /// be told apart from rounding, so its direction would be arbitrary.
    /// Returns [`Error::NumericalFailure`] for data beyond about 1e±150, and
    /// [`Error::AllocationFailure`] if memory cannot be reserved; allocations
    /// inside the decomposition may still abort on failure.
    ///
    /// # Examples
    /// Spectra preprocessed one at a time are written straight into the rows
    /// of `data` with `apply_into`, which rejects a spectrum of another length:
    /// ```
    /// use chemometrics::{normalize::StandardNormalVariate, pca::Pca};
    /// // Three spectra of five wavelengths each.
    /// let raw = [
    ///     [0.50, 0.62, 0.81, 0.70, 0.55],
    ///     [0.55, 0.70, 0.86, 0.72, 0.58],
    ///     [0.48, 0.58, 0.83, 0.75, 0.52],
    /// ];
    /// let variables = 5;
    /// let mut data = vec![0.0; raw.len() * variables];
    /// for (spectrum, row) in raw.iter().zip(data.chunks_mut(variables)) {
    ///     StandardNormalVariate.apply_into(spectrum, row)?;
    /// }
    /// let model = Pca::fit(&data, variables, 1)?;
    /// assert_eq!(model.samples(), 3);
    /// # Ok::<(), chemometrics::Error>(())
    /// ```
    ///
    /// Spectra that already exist as separate vectors can be joined with
    /// `concat`, which does not check that they have the same length.
    pub fn fit(data: &[f64], variables: usize, components: usize) -> Result<Self, Error> {
        let samples = numeric::samples_of(data.len(), variables)?;
        if samples < 2 {
            return Err(Error::TooFewSpectra {
                count: samples,
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
        // Centering leaves rounding of the order of EPSILON times the
        // uncentered values, which a tolerance relative to the centered data
        // cannot see.
        let uncentered_norm = numeric::sum(data.iter().map(|x| x * x)).sqrt();
        let mut centered = numeric::zeros(data.len())?;
        centered.copy_from_slice(data);
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
        // Squares beyond the representable range would make the rank test below
        // meaningless.
        if !uncentered_norm.is_finite() || !variance.is_finite() {
            return Err(Error::NumericalFailure);
        }
        let svd = numeric::thin_svd(&centered, samples, variables, components)?;
        let threshold = numeric::rank_tolerance(samples, variables, uncentered_norm);
        let supported = svd.values[..maximum]
            .iter()
            .filter(|s| **s > threshold)
            .count();
        if supported < components {
            return Err(Error::InsufficientRank {
                requested: components,
                supported,
            });
        }
        let kept = &svd.values[..components];
        // Centering removes one direction, so with no more samples than
        // variables the last singular value is rounding, not variation.
        let mut eigenvalues = numeric::zeros(maximum)?;
        for (eigenvalue, value) in eigenvalues.iter_mut().zip(&svd.values) {
            *eigenvalue = value * value / degrees;
        }
        let mut ratios = numeric::zeros(components)?;
        for (ratio, eigenvalue) in ratios.iter_mut().zip(&eigenvalues) {
            *ratio = eigenvalue / variance;
        }
        // The product is bounded by the data length, so it cannot overflow.
        let mut scores = numeric::zeros(samples * components)?;
        for (row, left) in scores
            .chunks_exact_mut(components)
            .zip(svd.left.chunks_exact(components))
        {
            for ((score, u), value) in row.iter_mut().zip(left).zip(kept) {
                *score = u * value;
            }
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
            total_variance: variance,
        };
        model.orient();
        model.usable()?;
        Ok(model)
    }

    /// Reconstructs a fitted model from its mean, loadings, full eigenvalue
    /// spectrum and training sample count, without refitting.
    ///
    /// Use this to load a validated model across platforms or library versions
    /// without depending on a file format. The loaded model carries no
    /// training scores, so [`Self::scores`] is empty and [`Self::score`]
    /// returns `None`, while [`Self::project`], [`Self::project_into`] and all
    /// variance accessors match a model from [`Self::fit`]. Loading signs are
    /// kept as given.
    ///
    /// `loadings` is `components` × `variables` row-major, where `variables`
    /// is `mean.len()`. `all_eigenvalues` must contain all
    /// `min(samples − 1, variables)` eigenvalues in nonincreasing order, as
    /// returned by [`Self::all_eigenvalues`], so total variance, explained
    /// variance ratios and Q control limits stay consistent. Loading rows must
    /// be orthonormal (`|P Pᵀ − I| ≤ 1e-9` entrywise). Values written with at
    /// least twelve significant digits pass; the check catches unnormalized or
    /// non-orthogonal rows, but not precision lost to `f32` storage.
    ///
    /// # Errors
    /// Checks `loadings` shape against `mean.len()`, then `samples`, then the
    /// component count, then `all_eigenvalues.len()`, then non-finite values,
    /// then eigenvalue values, then loading orthonormality, then the rank, in
    /// that order. Non-finite values are sought in `mean`, then `loadings`,
    /// then `all_eigenvalues`, and the index in [`Error::NonFiniteInput`]
    /// refers to the first of these slices that contains one. Returns
    /// [`Error::InvalidDataShape`], [`Error::TooFewSpectra`],
    /// [`Error::InvalidComponentCount`], [`Error::InvalidEigenvalueCount`],
    /// [`Error::NonFiniteInput`], [`Error::InvalidEigenvalues`],
    /// [`Error::NonOrthonormalLoadings`], [`Error::InsufficientRank`] if
    /// [`Self::fit`] would reject a retained component as rounding,
    /// [`Error::NumericalFailure`] if the model describes data beyond the
    /// `f64` range or retained eigenvalues are not normal numbers, and
    /// [`Error::AllocationFailure`] if memory cannot be reserved.
    ///
    /// # Examples
    /// ```
    /// use chemometrics::pca::Pca;
    /// let references = [
    ///     0.10, 0.31, 0.50, 0.29, 0.10,
    ///     0.12, 0.36, 0.61, 0.36, 0.11,
    ///     0.16, 0.45, 0.75, 0.44, 0.15,
    ///     0.17, 0.52, 0.85, 0.51, 0.17,
    ///     0.20, 0.60, 0.99, 0.61, 0.20,
    /// ];
    /// let fitted = Pca::fit(&references, 5, 1)?;
    /// let loaded = Pca::from_parts(
    ///     fitted.mean(),
    ///     fitted.loadings(),
    ///     fitted.all_eigenvalues(),
    ///     fitted.samples(),
    /// )?;
    /// assert!(loaded.scores().is_empty());
    /// let spectrum = [0.14, 0.42, 0.70, 0.52, 0.34];
    /// assert_eq!(loaded.project(&spectrum)?, fitted.project(&spectrum)?);
    /// # Ok::<(), chemometrics::Error>(())
    /// ```
    pub fn from_parts(
        mean: &[f64],
        loadings: &[f64],
        all_eigenvalues: &[f64],
        samples: usize,
    ) -> Result<Self, Error> {
        let variables = mean.len();
        let components = numeric::samples_of(loadings.len(), variables)?;
        if samples < 2 {
            return Err(Error::TooFewSpectra {
                count: samples,
                minimum: 2,
            });
        }
        let maximum = (samples - 1).min(variables);
        if components > maximum {
            return Err(Error::InvalidComponentCount {
                requested: components,
                maximum,
            });
        }
        if all_eigenvalues.len() != maximum {
            return Err(Error::InvalidEigenvalueCount {
                expected: maximum,
                actual: all_eigenvalues.len(),
            });
        }
        for slice in [mean, loadings, all_eigenvalues] {
            if let Some(index) = slice.iter().position(|x| !x.is_finite()) {
                return Err(Error::NonFiniteInput { index });
            }
        }
        if all_eigenvalues.iter().any(|v| *v < 0.0)
            || all_eigenvalues[..components].contains(&0.0)
            || all_eigenvalues.windows(2).any(|w| w[0] < w[1])
        {
            return Err(Error::InvalidEigenvalues);
        }
        for (a, row_a) in loadings.chunks_exact(variables).enumerate() {
            for (b, row_b) in loadings.chunks_exact(variables).enumerate().skip(a) {
                let dot = numeric::sum(row_a.iter().zip(row_b).map(|(x, y)| x * y));
                let expected = if a == b { 1.0 } else { 0.0 };
                if !dot.is_finite() || (dot - expected).abs() > ORTHONORMAL_TOLERANCE {
                    return Err(Error::NonOrthonormalLoadings);
                }
            }
        }
        // The rank test of `fit` needs the norm of the uncentered data, which
        // the mean and the eigenvalues determine.
        let degrees = (samples - 1) as f64;
        let uncentered_norm = (samples as f64 * numeric::sum(mean.iter().map(|m| m * m))
            + degrees * numeric::sum(all_eigenvalues.iter().copied()))
        .sqrt();
        if !uncentered_norm.is_finite() {
            return Err(Error::NumericalFailure);
        }
        let threshold = numeric::rank_tolerance(samples, variables, uncentered_norm);
        let supported = all_eigenvalues
            .iter()
            .filter(|v| (*v * degrees).sqrt() > threshold)
            .count();
        if supported < components {
            return Err(Error::InsufficientRank {
                requested: components,
                supported,
            });
        }
        let mut owned_mean = numeric::zeros(variables)?;
        owned_mean.copy_from_slice(mean);
        let mut owned_loadings = numeric::zeros(loadings.len())?;
        owned_loadings.copy_from_slice(loadings);
        let mut eigenvalues = numeric::zeros(maximum)?;
        for (slot, value) in eigenvalues.iter_mut().zip(all_eigenvalues) {
            // `fit` never reports negative zero, so a stored one is turned
            // into positive zero to match.
            *slot = value + 0.0;
        }
        let total_variance = numeric::sum(eigenvalues.iter().copied());
        let mut ratios = numeric::zeros(components)?;
        for (ratio, eigenvalue) in ratios.iter_mut().zip(&eigenvalues) {
            *ratio = eigenvalue / total_variance;
        }
        let model = Self {
            samples,
            variables,
            components,
            mean: owned_mean,
            loadings: owned_loadings,
            scores: Vec::new(),
            eigenvalues,
            ratios,
            total_variance,
        };
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
    /// The loadings are orthonormal. [`Self::fit`] fixes each component's sign
    /// so that its largest loading is positive; among loadings equal up to
    /// rounding, the first decides. [`Self::from_parts`] keeps the signs it is
    /// given. Components with nearly equal eigenvalues are not
    /// determined by the data and may differ between platforms or library
    /// versions.
    pub fn loadings(&self) -> &[f64] {
        &self.loadings
    }

    /// Loading vector of one component, or `None` if it does not exist.
    pub fn loading(&self, component: usize) -> Option<&[f64]> {
        self.loadings.chunks_exact(self.variables).nth(component)
    }

    /// Training scores, `samples` × `components` row-major, or empty for a
    /// model loaded with [`Self::from_parts`].
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Scores of one training sample, or `None` if it does not exist or the
    /// model was loaded with [`Self::from_parts`].
    pub fn score(&self, sample: usize) -> Option<&[f64]> {
        self.scores.chunks_exact(self.components).nth(sample)
    }

    /// Variance of each component's training scores, with divisor
    /// `samples − 1` as in scikit-learn's `explained_variance_`.
    ///
    /// Tools that divide by `samples` instead report eigenvalues smaller, and
    /// T² values larger, by the factor `samples / (samples − 1)`.
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues[..self.components]
    }

    /// Eigenvalues of all `min(samples − 1, variables)` components, the
    /// retained ones first, in nonincreasing order.
    ///
    /// The discarded eigenvalues `all_eigenvalues()[components..]` give the
    /// Jackson–Mudholkar limit for Q, and divided by [`Pca::total_variance`]
    /// they show how much further components would explain. Preprocessing such
    /// as SNV or detrending removes directions from the data, so trailing
    /// eigenvalues can be pure rounding.
    pub fn all_eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Share of the total variance carried by each retained component.
    ///
    /// The total covers all variables, so the shares sum to one only when every
    /// component is retained.
    pub fn explained_variance_ratio(&self) -> &[f64] {
        &self.ratios
    }

    /// Total variance of the centered training data.
    pub fn total_variance(&self) -> f64 {
        self.total_variance
    }

    /// Projects a spectrum into a newly allocated [`Projection`].
    ///
    /// # Errors
    /// Returns [`Error::InvalidSpectrumLength`], [`Error::NonFiniteInput`] or
    /// [`Error::NumericalFailure`] as [`Self::project_into`] does, and
    /// [`Error::AllocationFailure`] if the scores cannot be reserved.
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
        let centered = || spectrum.iter().zip(&self.mean).map(|(x, m)| x - m);
        let loadings = || self.loadings.chunks_exact(self.variables);
        for (score, loading) in scores.iter_mut().zip(loadings()) {
            *score = numeric::sum(centered().zip(loading).map(|(d, p)| d * p));
        }
        let scores: &[f64] = scores;
        // The residual is formed directly; ‖x−x̄‖² − ‖t‖² would cancel.
        let residuals = || {
            centered()
                .enumerate()
                .map(|(j, d)| d - numeric::sum(loadings().zip(scores).map(|(p, t)| t * p[j])))
        };
        let q_residual = numeric::sum(residuals().map(|r| r * r));
        let hotelling_t2 = numeric::sum(
            scores
                .iter()
                .zip(self.eigenvalues())
                .map(|(t, eigenvalue)| t * t / eigenvalue),
        );
        let diagnostics = Diagnostics {
            hotelling_t2,
            q_residual,
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
            let largest = loading.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            // Loadings equal in magnitude up to rounding count as tied, and the
            // first of them decides.
            let bound = largest * (1.0 - f64::EPSILON.sqrt());
            let leading = loading.iter().find(|v| v.abs() >= bound).copied();
            if leading.is_none_or(|v| v >= 0.0) {
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

    /// Rejects models whose values left the range of normal numbers.
    ///
    /// T² divides by the retained eigenvalues, so one that underflowed to zero
    /// or to a subnormal number would make every projection fail or lose its
    /// digits; it is rejected here rather than later.
    fn usable(&self) -> Result<(), Error> {
        let finite = |values: &[f64]| values.iter().all(|x| x.is_finite());
        if self.total_variance.is_normal()
            && self.eigenvalues().iter().all(|v| v.is_normal())
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
        };
        model.orient();
        assert_eq!(model.loadings, [0.5, -0.5]);
        assert_eq!(model.scores, [-1.0, 1.0]);
    }
}
