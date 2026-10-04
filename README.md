# chemometrics

[![CI](https://github.com/bsenkel/chemometrics/actions/workflows/ci.yml/badge.svg)](https://github.com/bsenkel/chemometrics/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/chemometrics.svg)](https://crates.io/crates/chemometrics)
[![docs.rs](https://img.shields.io/docsrs/chemometrics)](https://docs.rs/chemometrics)

Spectral preprocessing and chemometric analysis for uniformly sampled spectra
in Rust, dependency-free by default:

- Moving average and Savitzky–Golay smoothing
- Savitzky–Golay derivatives
- Standard normal variate (SNV) normalization
- Polynomial detrending
- Mixture composition from pure component spectra by iterative optimization
  technology (IOT), without calibration samples
- Principal component analysis with Hotelling's T² and the Q residual, behind
  the optional `pca` feature

## Installation

```sh
cargo add chemometrics
```

For principal component analysis:

```sh
cargo add chemometrics --features pca
```

The `pca` feature adds [`faer`](https://crates.io/crates/faer), a pure-Rust
linear algebra library without BLAS or LAPACK (about 50 crates), which does not
appear in this crate's API. It uses `unsafe` SIMD code, so PCA results are not
bit-identical across CPU architectures; this crate itself forbids `unsafe`.

## Examples

```rust
use chemometrics::{normalize::StandardNormalVariate, smooth::SavitzkyGolay};

fn main() -> Result<(), chemometrics::Error> {
    // An excerpt of a near-infrared absorbance spectrum, measured every 2 nm.
    let absorbance = [0.52, 0.55, 0.61, 0.70, 0.78, 0.83, 0.84, 0.81, 0.74, 0.66, 0.60];

    // A smoothed first derivative removes constant baseline offsets.
    // Window 5, polynomial order 2, spacing 2 nm: the result is per nm.
    let derivative = SavitzkyGolay::new_derivative(5, 2, 1, 2.0)?.apply(&absorbance)?;

    // SNV then removes multiplicative scatter effects.
    let preprocessed = StandardNormalVariate.apply(&derivative)?;

    println!("{preprocessed:?}");
    Ok(())
}
```

`Detrend::new(2)` and `MovingAverage::new(5)?` are applied the same way. Filters
are reusable across spectra, and `apply_into` writes into an existing buffer
without allocating.

With the `pca` feature, a model fitted to reference spectra checks new ones:

```rust
use chemometrics::pca::Pca;

fn main() -> Result<(), chemometrics::Error> {
    // Reference spectra at five wavelengths, one row per spectrum: a single
    // band whose height follows the concentration.
    let wavelengths = 5;
    let references = [
        0.10, 0.31, 0.50, 0.29, 0.10,
        0.12, 0.36, 0.61, 0.36, 0.11,
        0.16, 0.45, 0.75, 0.44, 0.15,
        0.17, 0.52, 0.85, 0.51, 0.17,
        0.20, 0.60, 0.99, 0.61, 0.20,
    ];
    // One component describes the band.
    let model = Pca::fit(&references, wavelengths, 1)?;

    // T² flags spectra that are extreme in known directions, Q those that
    // carry variation the model does not describe.
    let new_spectra = [
        ("typical", [0.14, 0.42, 0.70, 0.42, 0.14]),
        ("high concentration", [0.32, 0.96, 1.60, 0.96, 0.32]),
        ("unexpected band", [0.14, 0.42, 0.70, 0.52, 0.34]),
    ];
    for (name, spectrum) in new_spectra {
        let diagnostics = model.project(&spectrum)?.diagnostics;
        println!(
            "{name:18} T² {:5.2}  Q {:.5}",
            diagnostics.hotelling_t2, diagnostics.q_residual
        );
    }
    Ok(())
}
```

```text
typical            T²  0.04  Q 0.00004
high concentration T² 18.98  Q 0.00112
unexpected band    T²  0.01  Q 0.04562
```

The typical spectrum stays as close to the model as the references, with T² up
to 1.7 and Q up to 0.00014. The high concentration stands out in T², as known
variation beyond the usual range, and the unexpected band in Q, as variation
the model does not describe.

`cargo run --example preprocessing` shows which disturbance each preprocessing
step removes, `cargo run --example iot` the composition of powder blends from
pure component spectra, `cargo run --features pca --example pca` a complete
inspection with control limits for T² and Q, and the
[API documentation](https://docs.rs/chemometrics) describes every method.

## Conventions

- Spectra are `f64` slices on a uniform axis, ascending or descending; results
  have the same length.
- Invalid input and numerical failures return a `chemometrics::Error`; NaN and
  infinity are rejected.
- Values between about 1e-150 and 1e150 are supported; beyond that range,
  methods return `NumericalFailure` rather than a wrong value.
- Not supported: uneven sampling, `f32`, `no_std`.

## Validation

Results agree with SciPy, NumPy and scikit-learn reference values in
`tests/fixtures/` to about 1e-10, and tests check analytic invariants such as
polynomial preservation and orthonormal loadings.

## Minimum supported Rust version

Rust 1.85. The `pca` feature is tested on 1.85 with this repository's
`Cargo.lock`; a fresh dependency resolution may need a newer toolchain.

## License

[MIT](LICENSE)
