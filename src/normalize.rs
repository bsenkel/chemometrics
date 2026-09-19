//! Per-spectrum normalization of `f64` signals.
//!
//! Each spectrum is transformed on its own, without reference to other spectra
//! or to an x-axis. Output length and sample order are preserved.
use crate::{Error, numeric};

/// Standard normal variate (SNV): centers each spectrum on its mean and divides
/// it by its sample standard deviation.
///
/// For a spectrum `x` of length `n`, sample `i` becomes `(x[i] - mean) / s` with
/// `s = sqrt(sum((x[i] - mean)²) / (n - 1))`. The result has mean zero and
/// sample standard deviation one. This matches R's `sd` and
/// `scipy.stats.zscore(x, ddof=1)`; tools that divide by `n` instead return
/// values larger by the constant factor `sqrt(n / (n - 1))`.
///
/// SNV removes multiplicative scaling and constant offsets: for `a > 0`,
/// `a * x + b` gives the same result as `x`, and `a < 0` flips its sign. A
/// sloping baseline is not removed. A constant spectrum has no scale to divide
/// by and yields zeros. At least two samples are required.
///
/// Application takes O(n) time. The transformation has no parameters, so one
/// value processes spectra of any length.
///
/// # Example
/// ```
/// use chemometrics::normalize::StandardNormalVariate;
/// let result = StandardNormalVariate.apply(&[1.0, 2.0, 3.0])?;
/// assert_eq!(result, vec![-1.0, 0.0, 1.0]);
/// # Ok::<(), chemometrics::Error>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardNormalVariate;

impl StandardNormalVariate {
    /// Normalizes a spectrum into a newly allocated vector.
    ///
    /// Returns [`Error::AllocationFailure`] if the result cannot be reserved.
    /// Other errors match [`Self::apply_into`].
    pub fn apply(&self, input: &[f64]) -> Result<Vec<f64>, Error> {
        numeric::validate(input, input.len(), 2)?;
        let mut output = numeric::zeros(input.len())?;
        normalize(input, &mut output)?;
        Ok(output)
    }

    /// Normalizes into a same-length buffer without allocating.
    ///
    /// Invalid inputs leave the buffer unchanged. Numerical failure can leave
    /// a partially written buffer.
    ///
    /// # Errors
    /// Returns [`Error::TooFewSamples`] for fewer than two samples,
    /// [`Error::OutputLengthMismatch`] for a buffer of different length and
    /// [`Error::NonFiniteInput`] for NaN or infinity, checked in that order.
    /// Returns [`Error::NumericalFailure`] if a result is not finite.
    pub fn apply_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), Error> {
        numeric::validate(input, output.len(), 2)?;
        normalize(input, output)
    }
}

/// Assumes `validate` already accepted these slices.
fn normalize(input: &[f64], output: &mut [f64]) -> Result<(), Error> {
    let shift = numeric::Shift::new(input);
    let shifted = |x: &f64| shift.apply(*x);
    let count = input.len() as f64;
    let mean = numeric::sum(input.iter().map(shifted)) / count;
    // Two passes instead of mean(y²) - mean(y)², which cancels most digits
    // for spectra with a large offset and small variation.
    let variance = numeric::sum(input.iter().map(|x| {
        let deviation = shifted(x) - mean;
        deviation * deviation
    })) / (count - 1.0);
    if variance == 0.0 {
        // Exact offsets leave zero variance only for a constant spectrum, which
        // has no spread to divide by.
        output.fill(0.0);
        return Ok(());
    }
    let standard_deviation = variance.sqrt();
    for (out, x) in output.iter_mut().zip(input) {
        let value = (shifted(x) - mean) / standard_deviation;
        if !value.is_finite() {
            return Err(Error::NumericalFailure);
        }
        *out = value;
    }
    Ok(())
}
