//! Per-spectrum normalization of `f64` signals.
//!
//! Each spectrum is transformed on its own, without reference to other spectra
//! or to an x-axis. Output length and sample order are preserved.
use crate::{Error, polynomial};

fn validate(input: &[f64], output_len: usize) -> Result<(), Error> {
    if input.len() < 2 {
        return Err(Error::TooFewSamples {
            length: input.len(),
            minimum: 2,
        });
    }
    if output_len != input.len() {
        return Err(Error::OutputLengthMismatch {
            expected: input.len(),
            actual: output_len,
        });
    }
    if let Some(index) = input.iter().position(|x| !x.is_finite()) {
        return Err(Error::NonFiniteInput { index });
    }
    Ok(())
}

/// Largest power of two not above a positive, finite `magnitude`.
///
/// Division by a power of two is exact, so scaling cannot merge distinct
/// samples or change the result beyond what unscaled arithmetic would give.
fn power_of_two_floor(magnitude: f64) -> f64 {
    let bits = magnitude.to_bits();
    let exponent = bits & (0x7ff << 52);
    if exponent == 0 {
        // Subnormal: keep only the highest set mantissa bit.
        f64::from_bits(1 << (63 - bits.leading_zeros()))
    } else {
        f64::from_bits(exponent)
    }
}

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
        validate(input, input.len())?;
        let mut output = polynomial::zeros(input.len())?;
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
        validate(input, output.len())?;
        normalize(input, output)
    }
}

/// Assumes `validate` already accepted these slices.
fn normalize(input: &[f64], output: &mut [f64]) -> Result<(), Error> {
    let first = input[0];
    if input.iter().all(|&x| x == first) {
        // A compensated mean of identical values need not be bit-exact, and
        // dividing its residual deviations would produce arbitrary values.
        output.fill(0.0);
        return Ok(());
    }
    // Scaled samples lie in (-2, 2), so deviations and their squares stay
    // finite near f64::MAX and representable for subnormal spectra.
    let scale = power_of_two_floor(input.iter().fold(0.0_f64, |a, b| a.max(b.abs())));
    let count = input.len() as f64;
    let mean = polynomial::sum(input.iter().map(|x| x / scale)) / count;
    // Two passes instead of mean(y²) - mean(y)², which cancels most digits
    // for spectra with a large offset and small variation.
    let variance = polynomial::sum(input.iter().map(|x| {
        let deviation = x / scale - mean;
        deviation * deviation
    })) / (count - 1.0);
    let deviation = variance.sqrt();
    if !deviation.is_finite() || deviation <= 0.0 {
        return Err(Error::NumericalFailure);
    }
    for (out, x) in output.iter_mut().zip(input) {
        let value = (x / scale - mean) / deviation;
        if !value.is_finite() {
            return Err(Error::NumericalFailure);
        }
        *out = value;
    }
    Ok(())
}
