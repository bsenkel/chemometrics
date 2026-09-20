//! Baseline correction of `f64` spectra.
//!
//! Each spectrum is corrected on its own, without reference to other spectra.
//! Output length and sample order are preserved.
use crate::{Error, numeric};

/// Value of the monic Gram polynomial of degree `degree` at `coordinate`.
///
/// Gram polynomials are orthogonal on the uniform grid `2 i / (n - 1) - 1`,
/// which is symmetric, so the three-term recurrence needs no shift term. Its
/// coefficients depend only on the degree and `count`, so every value follows
/// from `O(degree)` arithmetic without storing a basis.
fn gram(degree: usize, coordinate: f64, count: f64) -> f64 {
    let mut previous = 0.0;
    let mut current = 1.0;
    for lower in 0..degree {
        // Degrees above zero need at least two samples, so the division is
        // finite; the first step's factor is zero.
        let k = lower as f64;
        let factor =
            k * k * (count * count - k * k) / ((4.0 * k * k - 1.0) * (count - 1.0) * (count - 1.0));
        let next = coordinate * current - factor * previous;
        previous = current;
        current = next;
    }
    current
}

/// Detrending: subtracts the least-squares polynomial fitted to a whole spectrum.
///
/// The fit uses the sample position as its coordinate, scaled to `[-1, 1]`, and
/// the result is the residual. Under uniform sampling this is the same fit as
/// over the wavelength axis, ascending or descending, so no x values are needed.
///
/// Order 0 subtracts the mean, order 1 a straight line and order 2 a parabola.
/// Order 2 is the detrending step that follows
/// [`StandardNormalVariate`](crate::normalize::StandardNormalVariate) in Barnes,
/// Dhanoa and Lister's SNV and Detrend, the usual treatment of scatter and
/// curved baselines in near-infrared spectra. Orders 0 and 1 match
/// `scipy.signal.detrend` with `type="constant"` and `type="linear"`.
///
/// Every polynomial up to the fitted order is removed exactly, so adding one to
/// a spectrum does not change the result. Strong bands pull the fit towards
/// themselves and are therefore damped along with the baseline; low orders limit
/// this. Orders above roughly 3 fit band structure rather than a baseline, and
/// orders of a few dozen degrees fail with [`Error::NumericalFailure`]; that
/// limit falls as a spectrum grows longer.
///
/// A spectrum needs at least `order + 1` samples. With exactly that many, the
/// polynomial passes through every sample and the result is zeros. Application
/// takes O(n × order²) time; one value corrects spectra of any length.
///
/// # Example
/// ```
/// use chemometrics::baseline::Detrend;
/// // A parabolic baseline under a single band.
/// let baseline = |i: usize| 0.2 + 0.01 * i as f64 - 0.0004 * (i * i) as f64;
/// let band = [0.0, 0.0, 0.0, 0.1, 0.4, 0.7, 0.4, 0.1, 0.0, 0.0, 0.0];
/// let measured: Vec<_> = band.iter().enumerate().map(|(i, b)| b + baseline(i)).collect();
/// let corrected = Detrend::new(2).apply(&measured)?;
/// let expected = Detrend::new(2).apply(&band)?;
/// for (actual, expected) in corrected.iter().zip(&expected) {
///     assert!((actual - expected).abs() < 1e-12);
/// }
/// # Ok::<(), chemometrics::Error>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Detrend {
    polynomial_order: usize,
}

impl Detrend {
    /// Creates a detrending correction of the given polynomial order.
    ///
    /// Any order is accepted; whether it fits a given spectrum depends on the
    /// spectrum's length and is reported by [`Self::apply`] and
    /// [`Self::apply_into`].
    pub const fn new(polynomial_order: usize) -> Self {
        Self { polynomial_order }
    }

    /// Corrects a spectrum into a newly allocated vector.
    ///
    /// Returns [`Error::AllocationFailure`] if the result cannot be reserved.
    /// Other errors match [`Self::apply_into`].
    pub fn apply(&self, input: &[f64]) -> Result<Vec<f64>, Error> {
        numeric::validate(input, input.len(), self.minimum_length())?;
        let mut output = numeric::zeros(input.len())?;
        self.correct(input, &mut output)?;
        Ok(output)
    }

    /// Corrects into a same-length buffer without allocating.
    ///
    /// Invalid inputs leave the buffer unchanged. Numerical failure can leave
    /// a partially written buffer.
    ///
    /// # Errors
    /// Returns [`Error::TooFewSamples`] for fewer than `order + 1` samples,
    /// [`Error::OutputLengthMismatch`] for a buffer of different length and
    /// [`Error::NonFiniteInput`] for NaN or infinity, checked in that order.
    /// Returns [`Error::NumericalFailure`] if the basis is numerically rank
    /// deficient or a result is not finite.
    pub fn apply_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), Error> {
        numeric::validate(input, output.len(), self.minimum_length())?;
        self.correct(input, output)
    }

    fn minimum_length(&self) -> usize {
        self.polynomial_order.saturating_add(1)
    }

    /// Assumes `validate` already accepted these slices.
    fn correct(&self, input: &[f64], output: &mut [f64]) -> Result<(), Error> {
        let count = input.len() as f64;
        // The fit removes constants, so subtracting the shift's reference
        // sample leaves the result unchanged.
        let shift = numeric::Shift::new(input);
        for (out, x) in output.iter_mut().zip(input) {
            *out = shift.apply(*x);
        }
        // A single sample only allows order 0, which ignores the coordinate;
        // this keeps it finite instead of dividing zero by zero.
        let coordinate = |i: usize| {
            if input.len() == 1 {
                0.0
            } else {
                2.0 * i as f64 / (count - 1.0) - 1.0
            }
        };
        for degree in 0..=self.polynomial_order {
            let basis = |i: usize| gram(degree, coordinate(i), count);
            let square = numeric::sum((0..input.len()).map(basis).map(|b| b * b));
            // Gram polynomials shrink geometrically with their degree. The
            // threshold has the same form as the one in `numeric::kernels`,
            // with `sqrt(count)` the norm of the constant basis polynomial, and
            // stops the fit once a degree has shrunk to rounding level.
            let norm = square.sqrt();
            if !norm.is_finite() || norm <= f64::EPSILON * count * count.sqrt() {
                return Err(Error::NumericalFailure);
            }
            // Projecting the running residual rather than the input keeps
            // rounding in the basis from accumulating across degrees.
            let projection =
                numeric::sum(output.iter().enumerate().map(|(i, y)| y * basis(i))) / square;
            for (i, out) in output.iter_mut().enumerate() {
                *out -= projection * basis(i);
            }
        }
        for out in output.iter_mut() {
            *out = shift.scale.restore(*out);
            if !out.is_finite() {
                return Err(Error::NumericalFailure);
            }
        }
        Ok(())
    }
}
