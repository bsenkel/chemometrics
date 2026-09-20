//! Spectral preprocessing for uniformly sampled signals, dependency-free by
//! default.
//!
//! Start with [`smooth::MovingAverage`] or [`smooth::SavitzkyGolay`].
//! [`smooth::SavitzkyGolay::new_derivative`] also computes local derivatives.
//! Filters preserve signal length and shift complete windows at the edges.
//! [`normalize::StandardNormalVariate`] removes multiplicative scaling and
//! constant offsets from each spectrum, and [`baseline::Detrend`] subtracts a
//! fitted polynomial baseline. With the optional `pca` feature,
//! [`pca::Pca`] decomposes a set of spectra and reports outlier statistics.

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
#[cfg(doctest)]
struct ReadmeDoctests;
