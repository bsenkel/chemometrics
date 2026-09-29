# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `MovingAverage` and `SavitzkyGolay` report a signal shorter than the window
  as `Error::TooFewSamples`, like the per-spectrum transforms.
- `MovingAverage` returns `Error::NumericalFailure` when the sum of a window
  exceeds the `f64` range, as for values near `f64::MAX`, instead of rescaling
  internally.
- `Pca::fit` returns `Error::NumericalFailure` for data beyond roughly 1e±150,
  whose squares or retained variances leave the range of normal `f64` numbers,
  instead of rescaling internally. Within that range results are unchanged or,
  for T², slightly more precise.
- `StandardNormalVariate` returns `Error::NumericalFailure` for spectra whose
  deviations from the mean lie beyond roughly 1e±150, where their squares leave
  the range of normal `f64` numbers, instead of rescaling internally.
- `Detrend` no longer rescales internally. Results are unchanged except for
  subnormal inputs below about 2.2e-308, which lose a little precision.

### Removed

- `Error::SignalTooShort`, replaced by `Error::TooFewSamples`.

## [0.1.4] - 2026-09-25

### Added

- `pca::Pca` behind the optional `pca` feature: principal component analysis of
  a set of spectra with Hotelling's T² and the Q residual as outlier statistics,
  validated against analytic results and NumPy/scikit-learn references.
- `Pca::all_eigenvalues` for Q control limits and for choosing the number of
  components.
- `Error::InvalidDataShape`, `Error::TooFewSpectra`,
  `Error::InvalidSpectrumLength`, `Error::InvalidComponentCount` and
  `Error::InsufficientRank` for invalid principal component inputs; the last
  reports how many components the data supports.
- An executable example that checks incoming lots of a raw material against T²
  and Q control limits.

## [0.1.3] - 2026-09-18

### Added

- `baseline::Detrend` for subtracting a fitted polynomial baseline from a
  spectrum, validated against analytic results and NumPy/SciPy references.

## [0.1.2] - 2026-09-17

### Added

- `normalize::StandardNormalVariate` for per-spectrum SNV normalization with
  the sample standard deviation, validated against analytic results and SciPy
  references.
- `Error::TooFewSamples` for inputs shorter than an operation requires.

## [0.1.1] - 2026-09-13

### Added

- `SavitzkyGolay::new_derivative` for local polynomial derivatives with signed
  sample spacing, validated against analytic results and SciPy references.

## [0.1.0] - 2026-09-06

Initial release.

### Added

- Dependency-free moving-average and Savitzky–Golay smoothing for uniformly
  sampled `f64` signals.
- Reusable filters with allocating `apply` and allocation-free `apply_into`.
- Length-preserving, shifted full windows: repeated means at the edges for
  moving averages and local polynomial evaluation for Savitzky–Golay.
- Input validation and errors for non-finite data, numerical failures,
  unrepresentable buffer sizes and failed memory reservations.
- Mathematical regression tests and stored SciPy reference results.
- Documentation, an executable example, MIT license and CI for Linux, macOS,
  Windows and the minimum supported Rust version.

[Unreleased]: https://github.com/bsenkel/chemometrics/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/bsenkel/chemometrics/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/bsenkel/chemometrics/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/bsenkel/chemometrics/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/bsenkel/chemometrics/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bsenkel/chemometrics/releases/tag/v0.1.0
