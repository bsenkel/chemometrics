//! Spectral preprocessing and chemometric analysis of uniformly sampled
//! spectra, dependency-free by default.
//!
//! Start with [`smooth::MovingAverage`] or [`smooth::SavitzkyGolay`].
//! [`smooth::SavitzkyGolay::new_derivative`] also computes local derivatives.
//! Filters preserve signal length and shift complete windows at the edges.
//! [`normalize::StandardNormalVariate`] removes multiplicative scaling and
//! constant offsets from each spectrum, and [`baseline::Detrend`] subtracts a
//! fitted polynomial baseline. With the optional `pca` feature,
//! [`pca::Pca`] decomposes a set of spectra and reports outlier statistics.
//!
//! The crate is built for measured spectra. Values between about 1e-150 and
//! 1e150 are processed without overflow or underflow, which covers absorbance,
//! reflectance and raw detector counts with a wide margin. Beyond that range,
//! methods return [`Error::NumericalFailure`] wherever a result would overflow
//! or lose most of its digits, rather than a wrong value. Subnormal inputs
//! below about 2.2e-308 may lose precision, as in any `f64` arithmetic.

#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod baseline;
mod error;
pub mod normalize;
mod numeric;
#[cfg(feature = "pca")]
#[cfg_attr(docsrs, doc(cfg(feature = "pca")))]
pub mod pca;
pub mod smooth;
pub use error::Error;

#[doc = include_str!("../README.md")]
#[cfg(all(doctest, feature = "pca"))]
struct ReadmeDoctests;
