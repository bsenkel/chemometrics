//! Spectral preprocessing for uniformly sampled signals, dependency-free by
//! default.
//!
//! Start with [`smooth::MovingAverage`] or [`smooth::SavitzkyGolay`].
//! Filters preserve signal length and shift complete windows at the edges.

mod error;
mod polynomial;
pub mod smooth;
pub use error::Error;

#[doc = include_str!("../README.md")]
#[cfg(doctest)]
struct ReadmeDoctests;
