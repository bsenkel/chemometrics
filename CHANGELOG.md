# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `pca::Pca` behind the optional `pca` feature: principal component analysis of
  a set of spectra with Hotelling T² and Q residual outlier statistics and the
  eigenvalues of all components for Q control limits and choosing the number of
  components, validated against analytic results and NumPy/scikit-learn
  references.
- `Error::InvalidDataShape`, `Error::InvalidSpectrumLength` and
  `Error::InvalidComponentCount`.

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

[Unreleased]: https://github.com/bsenkel/chemometrics/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/bsenkel/chemometrics/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/bsenkel/chemometrics/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/bsenkel/chemometrics/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bsenkel/chemometrics/releases/tag/v0.1.0
