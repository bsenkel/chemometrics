use std::fmt;

/// Invalid input, allocation failure, or a numerical failure during filtering.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Window length must be positive and odd.
    InvalidWindowLength(usize),
    /// Polynomial order must be smaller than the window length.
    InvalidPolynomialOrder {
        /// Requested polynomial order.
        order: usize,
        /// Requested window length.
        window_length: usize,
    },
    /// Input contains fewer points than one window.
    SignalTooShort {
        /// Actual input length.
        length: usize,
        /// Required minimum length.
        window_length: usize,
    },
    /// Output length differs from input length.
    OutputLengthMismatch {
        /// Required output length.
        expected: usize,
        /// Actual output length.
        actual: usize,
    },
    /// Input contains NaN or infinity.
    NonFiniteInput {
        /// Index of the first non-finite sample.
        index: usize,
    },
    /// The fit is numerically rank deficient or arithmetic became non-finite.
    NumericalFailure,
    /// A requested allocation exceeds addressable capacity or could not be reserved.
    AllocationFailure,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWindowLength(n) => {
                write!(f, "window length must be positive and odd, got {n}")
            }
            Self::InvalidPolynomialOrder {
                order,
                window_length,
            } => write!(
                f,
                "polynomial order {order} must be smaller than window length {window_length}"
            ),
            Self::SignalTooShort {
                length,
                window_length,
            } => write!(
                f,
                "signal length {length} is smaller than window length {window_length}"
            ),
            Self::OutputLengthMismatch { expected, actual } => {
                write!(f, "output length must be {expected}, got {actual}")
            }
            Self::NonFiniteInput { index } => write!(f, "non-finite input at index {index}"),
            Self::AllocationFailure => f.write_str("requested memory capacity is unavailable"),
            Self::NumericalFailure => {
                f.write_str("numerical rank deficiency or non-finite arithmetic")
            }
        }
    }
}

impl std::error::Error for Error {}
