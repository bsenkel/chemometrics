use std::fmt;

/// Invalid input, allocation failure, or a numerical failure.
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
    /// Derivative order must not exceed the polynomial order.
    InvalidDerivativeOrder {
        /// Requested derivative order.
        derivative_order: usize,
        /// Requested polynomial order.
        polynomial_order: usize,
    },
    /// Sample spacing must be finite and nonzero.
    InvalidSampleSpacing,
    /// Data length is not a positive multiple of the number of variables.
    InvalidDataShape {
        /// Actual data length.
        length: usize,
        /// Requested number of variables per sample.
        variables: usize,
    },
    /// Spectrum length differs from the number of variables in a model.
    InvalidSpectrumLength {
        /// Required spectrum length.
        expected: usize,
        /// Actual spectrum length.
        actual: usize,
    },
    /// Component count is zero or exceeds what the data supports.
    InvalidComponentCount {
        /// Requested number of components.
        requested: usize,
        /// Largest supported number of components.
        maximum: usize,
    },
    /// Data vary in fewer directions than the requested number of components,
    /// so the remaining ones would be rounding with arbitrary directions.
    InsufficientRank {
        /// Requested number of components.
        requested: usize,
        /// Number of components that stand out from rounding.
        supported: usize,
    },
    /// Input contains fewer points than one window.
    SignalTooShort {
        /// Actual input length.
        length: usize,
        /// Required minimum length.
        window_length: usize,
    },
    /// Input contains fewer samples than the operation requires: values of a
    /// spectrum for per-spectrum transforms, spectra for principal components.
    TooFewSamples {
        /// Actual number of samples.
        length: usize,
        /// Required minimum number of samples.
        minimum: usize,
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
    /// A polynomial fit is numerically rank deficient, a scaling or a result
    /// is unrepresentable, or arithmetic became non-finite.
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
            Self::InvalidDerivativeOrder { derivative_order, polynomial_order } => write!(
                f, "derivative order {derivative_order} must not exceed polynomial order {polynomial_order}"
            ),
            Self::InvalidSampleSpacing => f.write_str("sample spacing must be finite and nonzero"),
            Self::InvalidDataShape { length, variables } => write!(
                f,
                "data length {length} must be a positive multiple of {variables} variables"
            ),
            Self::InvalidSpectrumLength { expected, actual } => {
                write!(f, "spectrum length must be {expected}, got {actual}")
            }
            Self::InvalidComponentCount { requested, maximum } => write!(
                f,
                "component count {requested} must be between 1 and {maximum}"
            ),
            Self::InsufficientRank {
                requested,
                supported,
            } => write!(
                f,
                "component count {requested} exceeds the numerical rank {supported} of the data"
            ),
            Self::SignalTooShort {
                length,
                window_length,
            } => write!(
                f,
                "signal length {length} is smaller than window length {window_length}"
            ),
            Self::TooFewSamples { length, minimum } => {
                write!(f, "at least {minimum} samples are required, got {length}")
            }
            Self::OutputLengthMismatch { expected, actual } => {
                write!(f, "output length must be {expected}, got {actual}")
            }
            Self::NonFiniteInput { index } => write!(f, "non-finite input at index {index}"),
            Self::AllocationFailure => f.write_str("requested memory capacity is unavailable"),
            Self::NumericalFailure => {
                f.write_str("numerical rank deficiency, unrepresentable scaling or result, or non-finite arithmetic")
            }
        }
    }
}

impl std::error::Error for Error {}
