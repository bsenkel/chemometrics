//! Length-preserving smoothing and differentiation of uniformly sampled `f64` signals.
//!
//! Windows have positive odd lengths measured in samples. At either edge,
//! the nearest complete window is used without padding or invented samples.
//! An ascending or descending uniform x-axis is equally valid.
use crate::{Error, numeric};

fn window_valid(window: usize) -> Result<(), Error> {
    if window == 0 || window % 2 == 0 {
        return Err(Error::InvalidWindowLength(window));
    }
    Ok(())
}

fn validate(input: &[f64], output_len: usize, window: usize) -> Result<(), Error> {
    if input.len() < window {
        return Err(Error::SignalTooShort {
            length: input.len(),
            window_length: window,
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

fn start(i: usize, window: usize, length: usize) -> usize {
    i.saturating_sub(window / 2).min(length - window)
}

/// Writes one filtered sample per output slot, using the nearest full window.
///
/// `value` receives the position of the sample inside its window and the
/// window itself. This is the single place the edge convention is applied.
/// Assumes `validate` already accepted these lengths.
fn map_windows(
    input: &[f64],
    output: &mut [f64],
    window: usize,
    value: impl Fn(usize, &[f64]) -> f64,
) -> Result<(), Error> {
    for (i, out) in output.iter_mut().enumerate() {
        let begin = start(i, window, input.len());
        let filtered = value(i - begin, &input[begin..begin + window]);
        if !filtered.is_finite() {
            return Err(Error::NumericalFailure);
        }
        *out = filtered;
    }
    Ok(())
}

/// An arithmetic moving average with shifted full windows at the edges.
///
/// Edge values repeat the mean of the first or last complete window.
/// Application takes O(signal length × window length) time.
///
/// # Example
/// ```
/// use chemometrics::smooth::MovingAverage;
/// let signal = [0.0, 1.0, 2.0, 3.0, 4.0];
/// let smoothed = MovingAverage::new(3)?.apply(&signal)?;
/// assert_eq!(smoothed, vec![1.0, 1.0, 2.0, 3.0, 3.0]);
/// assert_eq!(signal, [0.0, 1.0, 2.0, 3.0, 4.0]);
/// # Ok::<(), chemometrics::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct MovingAverage {
    window_length: usize,
}

impl MovingAverage {
    /// Creates a filter with a positive odd window length.
    ///
    /// # Errors
    /// Returns [`Error::InvalidWindowLength`] for a zero or even window length.
    pub fn new(window_length: usize) -> Result<Self, Error> {
        window_valid(window_length)?;
        Ok(Self { window_length })
    }

    /// Filters a signal into a newly allocated vector.
    ///
    /// # Errors
    /// Returns [`Error::AllocationFailure`] if the result cannot be reserved.
    /// Other errors match [`Self::apply_into`].
    pub fn apply(&self, input: &[f64]) -> Result<Vec<f64>, Error> {
        validate(input, input.len(), self.window_length)?;
        let mut output = numeric::zeros(input.len())?;
        self.filter(input, &mut output)?;
        Ok(output)
    }

    /// Filters into a same-length buffer without allocating.
    ///
    /// Invalid inputs leave the buffer unchanged. Numerical failure can leave
    /// a partially written buffer.
    ///
    /// # Errors
    /// Returns [`Error::SignalTooShort`] for a signal shorter than the window,
    /// [`Error::OutputLengthMismatch`] for a buffer of different length and
    /// [`Error::NonFiniteInput`] for NaN or infinity, checked in that order.
    /// Returns [`Error::NumericalFailure`] if a result is not finite.
    pub fn apply_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), Error> {
        validate(input, output.len(), self.window_length)?;
        self.filter(input, output)
    }

    fn filter(&self, input: &[f64], output: &mut [f64]) -> Result<(), Error> {
        map_windows(input, output, self.window_length, |_, samples| {
            let scale = samples.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
            if scale == 0.0 {
                return 0.0;
            }
            // Normalize before summation, including for values near f64::MAX.
            // The exact normalized mean is in [-1, 1]; clamp rounding drift
            // before rescaling so a finite constant cannot overflow.
            let normalized = numeric::sum(samples.iter().map(|x| x / scale)) / samples.len() as f64;
            normalized.clamp(-1.0, 1.0) * scale
        })
    }
}

/// Savitzky–Golay smoothing and differentiation with polynomial evaluation at the edges.
///
/// Fits use coordinates scaled to [-1, 1] and Householder QR. A diagonal
/// magnitude at or below `f64::EPSILON * max(rows, columns) * ||A||_F`
/// is treated as numerical rank deficiency. High polynomial orders can fail
/// this check; low orders such as 2 or 3 are typical for smoothing.
///
/// Construction stores O(window length²) coefficients. Application takes
/// O(signal length × window length) time and uses the prepared coefficients.
///
/// # Example
/// ```
/// use chemometrics::smooth::SavitzkyGolay;
/// let filter = SavitzkyGolay::new(5, 2)?;
/// let signal = [0.0, 1.0, 4.0, 9.0, 16.0];
/// let mut output = [0.0; 5];
/// filter.apply_into(&signal, &mut output)?;
/// // A quadratic fit preserves quadratic data, including the edges.
/// for (actual, expected) in output.iter().zip(signal) {
///     assert!((actual - expected).abs() < 1e-10);
/// }
/// # Ok::<(), chemometrics::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct SavitzkyGolay {
    window_length: usize,
    kernels: Vec<f64>,
}

impl SavitzkyGolay {
    /// Prepares a filter. The polynomial order must be below the window length.
    ///
    /// # Errors
    /// Returns [`Error::InvalidWindowLength`] or
    /// [`Error::InvalidPolynomialOrder`] for invalid parameters, checked in
    /// that order, and [`Error::NumericalFailure`] if the polynomial fit is
    /// rank deficient. Returns [`Error::AllocationFailure`] if buffer sizes
    /// exceed addressable capacity or the allocator cannot reserve the
    /// requested memory.
    pub fn new(window_length: usize, polynomial_order: usize) -> Result<Self, Error> {
        Self::new_derivative(window_length, polynomial_order, 0, 1.0)
    }

    /// Prepares a local polynomial derivative filter.
    ///
    /// `derivative_order` must not exceed `polynomial_order`. Order zero is
    /// smoothing, equivalent to [`Self::new`] for any valid `sample_spacing`.
    /// `sample_spacing` is the constant difference between adjacent x values;
    /// it must be finite and nonzero, even for order zero. Negative spacing
    /// supports descending axes and reverses the sign of odd derivatives.
    /// Output units are input units divided by x units to the derivative order.
    ///
    /// Evaluates each local polynomial's derivative, including at the edges.
    /// Differentiation can amplify noise.
    ///
    /// # Errors
    /// Returns an input error for an invalid window, polynomial order, derivative
    /// order, or spacing, checked in that order. Returns [`Error::NumericalFailure`]
    /// for rank deficiency, overflow or complete underflow of derivative scaling,
    /// or non-finite coefficients. Allocation errors match [`Self::new`].
    ///
    /// # Example
    /// ```
    /// use chemometrics::smooth::SavitzkyGolay;
    /// // y = x² at x = 0, 0.5, 1, 1.5, 2; dy/dx = 2x.
    /// let filter = SavitzkyGolay::new_derivative(5, 2, 1, 0.5)?;
    /// let result = filter.apply(&[0.0, 0.25, 1.0, 2.25, 4.0])?;
    /// for (actual, expected) in result.iter().zip([0.0, 1.0, 2.0, 3.0, 4.0]) {
    ///     assert!((actual - expected).abs() < 1e-10);
    /// }
    /// # Ok::<(), chemometrics::Error>(())
    /// ```
    pub fn new_derivative(
        window_length: usize,
        polynomial_order: usize,
        derivative_order: usize,
        sample_spacing: f64,
    ) -> Result<Self, Error> {
        window_valid(window_length)?;
        if polynomial_order >= window_length {
            return Err(Error::InvalidPolynomialOrder {
                order: polynomial_order,
                window_length,
            });
        }
        if derivative_order > polynomial_order {
            return Err(Error::InvalidDerivativeOrder {
                derivative_order,
                polynomial_order,
            });
        }
        if !sample_spacing.is_finite() || sample_spacing == 0.0 {
            return Err(Error::InvalidSampleSpacing);
        }
        Ok(Self {
            window_length,
            kernels: numeric::kernels(
                window_length,
                polynomial_order,
                derivative_order,
                sample_spacing,
            )?,
        })
    }

    /// Filters a signal into a newly allocated vector.
    ///
    /// # Errors
    /// Returns [`Error::AllocationFailure`] if the result cannot be reserved.
    /// Other errors match [`Self::apply_into`].
    pub fn apply(&self, input: &[f64]) -> Result<Vec<f64>, Error> {
        validate(input, input.len(), self.window_length)?;
        let mut output = numeric::zeros(input.len())?;
        self.filter(input, &mut output)?;
        Ok(output)
    }

    /// Filters into a same-length buffer without allocating.
    ///
    /// Invalid inputs leave the buffer unchanged. Numerical failure can leave
    /// a partially written buffer.
    ///
    /// # Errors
    /// Returns [`Error::SignalTooShort`] for a signal shorter than the window,
    /// [`Error::OutputLengthMismatch`] for a buffer of different length and
    /// [`Error::NonFiniteInput`] for NaN or infinity, checked in that order.
    /// Returns [`Error::NumericalFailure`] if a result is not finite.
    pub fn apply_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), Error> {
        validate(input, output.len(), self.window_length)?;
        self.filter(input, output)
    }

    fn filter(&self, input: &[f64], output: &mut [f64]) -> Result<(), Error> {
        map_windows(input, output, self.window_length, |offset, samples| {
            let row = offset * self.window_length;
            let weights = &self.kernels[row..row + self.window_length];
            numeric::sum(weights.iter().zip(samples).map(|(a, b)| a * b))
        })
    }
}
