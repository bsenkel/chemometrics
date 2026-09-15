//! Spectral preprocessing for uniformly sampled signals, dependency-free by
//! default.
//!
//! Start with [`smooth::MovingAverage`] or [`smooth::SavitzkyGolay`].
//! [`smooth::SavitzkyGolay::new_derivative`] also computes local derivatives.
//! [`normalize::StandardNormalVariate`] removes multiplicative scaling and
//! constant offsets from each spectrum.
//! Filters preserve signal length and shift complete windows at the edges.

mod error;
pub mod normalize;
mod polynomial;
pub mod smooth;
pub use error::Error;

#[doc = include_str!("../README.md")]
#[cfg(doctest)]
struct ReadmeDoctests;
