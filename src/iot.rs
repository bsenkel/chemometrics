//! Composition of a mixture from the spectra of its pure components.
//!
//! [`Iot`] implements iterative optimization technology (IOT) after Muteki et
//! al. (Ind. Eng. Chem. Res., 2013): a mixture spectrum is described as a sum
//! of the pure component spectra, each weighted by its contribution. The
//! contributions are not negative and sum to one. No calibration samples are
//! needed, only one spectrum of every pure component.
//!
//! # Examples
//! ```
//! use chemometrics::iot::Iot;
//!
//! // Pure component spectra at five wavelengths, one row per component.
//! let variables = 5;
//! let pure = [
//!     0.10, 0.80, 0.30, 0.05, 0.02,
//!     0.40, 0.10, 0.20, 0.70, 0.30,
//!     0.05, 0.05, 0.60, 0.10, 0.50,
//! ];
//! let model = Iot::new(&pure, variables)?;
//!
//! // A mixture of 20 %, 70 % and 10 % of the components.
//! let mixture: Vec<f64> = (0..variables)
//!     .map(|i| 0.2 * pure[i] + 0.7 * pure[variables + i] + 0.1 * pure[2 * variables + i])
//!     .collect();
//! let prediction = model.predict(&mixture)?;
//! for (actual, expected) in prediction.contributions.iter().zip([0.2, 0.7, 0.1]) {
//!     assert!((actual - expected).abs() < 1e-12);
//! }
//! # Ok::<(), chemometrics::Error>(())
//! ```
//!
//! # Data layout
//! Pure component spectra are passed as one flat row-major slice: row `i`
//! holds the `variables` intensities of component `i`. All spectra, including
//! the mixtures, must share one wavelength grid in the same order.
//!
//! # Assumptions
//! The spectra must add up linearly, as absorbances do under the
//! Beer–Lambert law, and pure components and mixtures must be measured under
//! similar conditions. Contributions are fractions of the pure component
//! spectra; they equal mass fractions only as far as these assumptions hold.
//! Pure spectra that resemble each other closely make the contributions
//! sensitive to noise.
//!
//! # Preprocessing
//! Preprocessing must be the same for pure components and mixtures, and
//! linear, so that the preprocessed mixture is still the sum of the
//! preprocessed components:
//! [`SavitzkyGolay`](crate::smooth::SavitzkyGolay) smoothing and derivatives,
//! [`MovingAverage`](crate::smooth::MovingAverage) and
//! [`Detrend`](crate::baseline::Detrend) are, and so are selecting or
//! weighting wavelengths.
//! [`StandardNormalVariate`](crate::normalize::StandardNormalVariate) is not,
//! because it divides each spectrum by its own standard deviation.
use crate::{
    Error,
    numeric::{self, QrError},
};
use std::fmt;

/// Estimated composition of one mixture spectrum.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Prediction {
    /// Contribution of each pure component, in the order of the pure spectra.
    ///
    /// Each lies between zero and one. They sum to one for [`Iot::new`] and to
    /// at most one for [`Iot::new_partial`].
    pub contributions: Vec<f64>,
    /// Sum of squared differences between the mixture spectrum and the pure
    /// spectra weighted by their contributions.
    ///
    /// A value well above that of typical mixtures points to a component
    /// without a pure spectrum or to spectra that do not add up linearly.
    pub residual: f64,
}

/// Pure component spectra, prepared for estimating the composition of
/// mixtures.
///
/// Construction takes O(variables × components²) time and stores about twice
/// the pure spectra; `Debug` prints only the shape.
#[derive(Clone)]
pub struct Iot {
    variables: usize,
    components: usize,
    /// Whether the contributions may sum to less than one.
    partial: bool,
    /// The pure spectra, `components` × `variables` row-major.
    pure: Vec<f64>,
    /// Orthonormal basis of the pure spectra relative to the reference, one
    /// row per component other than the reference.
    q: Vec<f64>,
    /// The same spectra expressed in that basis, column-major upper
    /// triangular.
    r: Vec<f64>,
}

impl fmt::Debug for Iot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Iot")
            .field("components", &self.components)
            .field("variables", &self.variables)
            .field("partial", &self.partial)
            .finish_non_exhaustive()
    }
}

impl Iot {
    /// Prepares row-major `pure` spectra with `variables` values each for
    /// mixtures that consist of these components only.
    ///
    /// The contributions sum to one. Between two and `variables + 1`
    /// components are supported.
    ///
    /// # Errors
    /// Checks the data shape, then the number of spectra, then the component
    /// count, then non-finite values, in that order; the index in
    /// [`Error::NonFiniteInput`] refers to `pure`, so the affected spectrum is
    /// `index / variables`. Returns [`Error::DependentSpectra`] for the first
    /// spectrum that is a mixture of the preceding ones up to rounding, such
    /// as a duplicate. Returns [`Error::NumericalFailure`] for spectra beyond
    /// about 1e±150, and [`Error::AllocationFailure`] if memory cannot be
    /// reserved.
    pub fn new(pure: &[f64], variables: usize) -> Result<Self, Error> {
        Self::build(pure, variables, false)
    }

    /// Prepares `pure` spectra for mixtures that contain one further
    /// component without a usable spectrum of its own.
    ///
    /// The contributions of the given components sum to at most one, and the
    /// remainder belongs to the further component, whose spectrum counts as
    /// zero. Between one and `variables` components are supported.
    ///
    /// # Errors
    /// As [`Self::new`], except that a single spectrum suffices and that
    /// [`Error::DependentSpectra`] also reports a spectrum of zeros.
    pub fn new_partial(pure: &[f64], variables: usize) -> Result<Self, Error> {
        Self::build(pure, variables, true)
    }

    fn build(pure: &[f64], variables: usize, partial: bool) -> Result<Self, Error> {
        let components = numeric::samples_of(pure.len(), variables)?;
        if !partial && components < 2 {
            return Err(Error::TooFewSpectra {
                count: components,
                minimum: 2,
            });
        }
        // The reference is the first pure spectrum, or the spectrum of zeros
        // of the further component. Every other component needs a direction
        // of its own among the variables.
        let first = usize::from(!partial);
        let others = components - first;
        if others > variables {
            return Err(Error::InvalidComponentCount {
                requested: components,
                maximum: variables + first,
            });
        }
        if let Some(index) = pure.iter().position(|x| !x.is_finite()) {
            return Err(Error::NonFiniteInput { index });
        }
        // Subtracting the reference leaves rounding of the order of EPSILON
        // times the original values, which a tolerance relative to the
        // differences cannot see.
        let norm = numeric::sum(pure.iter().map(|x| x * x)).sqrt();
        if !norm.is_finite() {
            return Err(Error::NumericalFailure);
        }
        let mut stored = numeric::zeros(pure.len())?;
        stored.copy_from_slice(pure);
        // The product is bounded by the data length, so it cannot overflow.
        let mut differences = numeric::zeros(others * variables)?;
        for (column, spectrum) in differences
            .chunks_exact_mut(variables)
            .zip(pure.chunks_exact(variables).skip(first))
        {
            for (i, (difference, x)) in column.iter_mut().zip(spectrum).enumerate() {
                *difference = if partial { *x } else { x - pure[i] };
            }
        }
        let tolerance = numeric::rank_tolerance(variables, others, norm);
        let qr =
            numeric::thin_qr(differences, variables, others, tolerance).map_err(|failure| {
                match failure {
                    QrError::DependentColumn(column) => Error::DependentSpectra {
                        index: first + column,
                    },
                    QrError::Other(error) => error,
                }
            })?;
        // The solver multiplies entries of `R` with each other; a square that
        // underflowed would let it stop at a wrong composition.
        let usable = (0..others).all(|j| {
            let diagonal = qr.r[j * others + j];
            (diagonal * diagonal).is_normal()
        });
        if !usable || qr.q.iter().chain(&qr.r).any(|x| !x.is_finite()) {
            return Err(Error::NumericalFailure);
        }
        Ok(Self {
            variables,
            components,
            partial,
            pure: stored,
            q: qr.q,
            r: qr.r,
        })
    }

    /// Number of pure component spectra.
    pub fn components(&self) -> usize {
        self.components
    }

    /// Number of variables per spectrum.
    pub fn variables(&self) -> usize {
        self.variables
    }

    /// Estimates the composition of a mixture spectrum.
    ///
    /// The contributions are those that describe the mixture best in the
    /// least-squares sense among all that are not negative and sum to one, or
    /// to at most one for [`Self::new_partial`]. The search ends after
    /// finitely many steps at the best composition up to rounding, rather
    /// than stopping an approximation at a threshold. Some contributions can
    /// be exactly zero; the others are then the best fit of those components
    /// alone.
    ///
    /// # Errors
    /// Returns [`Error::InvalidSpectrumLength`] for a spectrum of another
    /// length and [`Error::NonFiniteInput`] for NaN or infinity, checked in
    /// that order. Returns [`Error::NumericalFailure`] if a result is not
    /// representable or no composition can be settled on, and
    /// [`Error::AllocationFailure`] if memory cannot be reserved.
    pub fn predict(&self, mixture: &[f64]) -> Result<Prediction, Error> {
        if mixture.len() != self.variables {
            return Err(Error::InvalidSpectrumLength {
                expected: self.variables,
                actual: mixture.len(),
            });
        }
        if let Some(index) = mixture.iter().position(|x| !x.is_finite()) {
            return Err(Error::NonFiniteInput { index });
        }
        let first = usize::from(!self.partial);
        let reference = |i: usize| if self.partial { 0.0 } else { self.pure[i] };
        let basis = || self.q.chunks_exact(self.variables);
        let mut target = numeric::zeros(basis().len())?;
        for (value, direction) in target.iter_mut().zip(basis()) {
            let products = direction.iter().zip(mixture).enumerate();
            *value = numeric::sum(products.map(|(i, (q, x))| q * (x - reference(i))));
        }
        let mut contributions = solve(&self.r, &target)?;
        // The residual is formed from the differences to the reference, so an
        // offset shared by all spectra cannot round it away.
        let others = || self.pure.chunks_exact(self.variables).skip(first);
        let residuals = mixture.iter().enumerate().map(|(i, x)| {
            let weighted = others().zip(&contributions[1..]);
            x - reference(i) - numeric::sum(weighted.map(|(s, c)| c * (s[i] - reference(i))))
        });
        let residual = numeric::sum(residuals.map(|e| e * e));
        if !residual.is_finite() {
            return Err(Error::NumericalFailure);
        }
        if self.partial {
            contributions.remove(0);
        }
        Ok(Prediction {
            contributions,
            residual,
        })
    }
}

/// Minimizes `‖M x − target‖` over all `x` that are not negative and sum to
/// one. `M` is the column-major upper triangular `r`, preceded by a column of
/// zeros for the reference, so `x` holds the reference's contribution first.
///
/// An active-set method after Lawson and Hanson. Every component is either
/// free or fixed at zero. The free ones are fitted alone; if that asks for a
/// negative contribution, the current composition moves towards the fit only
/// until the first contribution reaches zero, which is then fixed. Once a fit
/// is feasible, a fixed component is released if that lowers the misfit, and
/// the composition is final when none does.
fn solve(r: &[f64], target: &[f64]) -> Result<Vec<f64>, Error> {
    let rows = target.len();
    let count = rows + 1;
    let entry = |i: usize, k: usize| if i == 0 { 0.0 } else { r[(i - 1) * rows + k] };
    let norm = |values: &[f64]| numeric::sum(values.iter().map(|x| x * x)).sqrt();
    let scale = norm(r) * (norm(r) + norm(target));
    if !scale.is_finite() {
        return Err(Error::NumericalFailure);
    }
    // Rounding leaves the gradient uncertain by about this much, so a smaller
    // gain from releasing a component is not a reason to release it.
    let tolerance = numeric::rank_tolerance(count, count, scale);
    let mut free = numeric::reserved(count)?;
    free.resize(count, true);
    let mut x = numeric::zeros(count)?;
    x.fill(1.0 / count as f64);
    let mut fit = numeric::zeros(count)?;
    let mut gradient = numeric::zeros(count)?;
    let mut residual = numeric::zeros(rows)?;
    let mut matrix = numeric::zeros(rows * rows)?;
    let mut right_side = numeric::zeros(rows)?;

    // Fits the free components alone. The first of them takes what the others
    // leave of the sum of one, which turns the fit into an ordinary
    // least-squares problem for the differences to that component.
    let mut fit_free = |free: &[bool], fit: &mut [f64]| -> Result<(), Error> {
        fit.fill(0.0);
        let Some(pivot) = free.iter().position(|free| *free) else {
            return Err(Error::NumericalFailure);
        };
        let rest = || (pivot + 1..count).filter(|i| free[*i]);
        let columns = rest().count();
        for (column, i) in matrix.chunks_exact_mut(rows).zip(rest()) {
            for (k, value) in column.iter_mut().enumerate() {
                *value = entry(i, k) - entry(pivot, k);
            }
        }
        for (k, value) in right_side.iter_mut().enumerate() {
            *value = target[k] - entry(pivot, k);
        }
        // Construction compared each spectrum with the preceding ones only, so
        // its tolerance says nothing about this subset; subsets of independent
        // spectra are independent, and only an exact zero is rejected.
        numeric::least_squares(
            &mut matrix[..columns * rows],
            rows,
            columns,
            0.0,
            &mut right_side,
        )?;
        for (i, value) in rest().zip(&right_side) {
            // Adding zero turns a negative zero from the back substitution
            // into a positive one, so an absent component reports 0.0.
            fit[i] = value + 0.0;
        }
        fit[pivot] = 1.0 - numeric::sum(right_side[..columns].iter().copied());
        if fit.iter().any(|x| !x.is_finite()) {
            return Err(Error::NumericalFailure);
        }
        Ok(())
    };

    fit_free(&free, &mut fit)?;
    // Each round releases one component; Lawson and Hanson's bound of three
    // rounds per component is far above what real mixtures need.
    for _ in 0..3 * count {
        loop {
            // The free component whose contribution reaches zero first on
            // the way to the fit.
            let reach = |i: &usize| x[*i] / (x[*i] - fit[*i]);
            let negative = (0..count).filter(|i| free[*i] && fit[*i] < 0.0);
            let Some(leaving) = negative.min_by(|a, b| reach(a).total_cmp(&reach(b))) else {
                break;
            };
            let step = reach(&leaving);
            for i in 0..count {
                x[i] += step * (fit[i] - x[i]);
                if i == leaving || (free[i] && fit[i] < 0.0 && x[i] <= 0.0) {
                    x[i] = 0.0;
                    free[i] = false;
                }
            }
            fit_free(&free, &mut fit)?;
        }
        x.copy_from_slice(&fit);

        // At the best composition the gradient of the squared misfit is the
        // same for all free components and not smaller for the fixed ones.
        for (k, value) in residual.iter_mut().enumerate() {
            *value = numeric::sum((1..count).map(|i| entry(i, k) * x[i])) - target[k];
        }
        for (i, value) in gradient.iter_mut().enumerate() {
            *value = numeric::sum(residual.iter().enumerate().map(|(k, e)| entry(i, k) * e));
        }
        if gradient.iter().any(|g| !g.is_finite()) {
            return Err(Error::NumericalFailure);
        }
        let freed = (0..count).filter(|i| free[*i]);
        let level = numeric::sum(freed.clone().map(|i| gradient[i])) / freed.count() as f64;
        loop {
            let fixed = (0..count).filter(|i| !free[*i]);
            let candidate = fixed.min_by(|a, b| gradient[*a].total_cmp(&gradient[*b]));
            let Some(released) = candidate.filter(|i| gradient[*i] - level < -tolerance) else {
                return Ok(x);
            };
            free[released] = true;
            fit_free(&free, &mut fit)?;
            if fit[released] > 0.0 {
                break;
            }
            // Rounding made the gradient promise a gain that the fit does not
            // deliver; the component stays fixed and is not tried again.
            free[released] = false;
            gradient[released] = level;
        }
    }
    Err(Error::NumericalFailure)
}
