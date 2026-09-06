# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/bsenkel/chemometrics/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bsenkel/chemometrics/releases/tag/v0.1.0
