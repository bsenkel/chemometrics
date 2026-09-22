# chemometrics

[![CI](https://github.com/bsenkel/chemometrics/actions/workflows/ci.yml/badge.svg)](https://github.com/bsenkel/chemometrics/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/chemometrics.svg)](https://crates.io/crates/chemometrics)
[![docs.rs](https://img.shields.io/docsrs/chemometrics)](https://docs.rs/chemometrics)

Spectral preprocessing in Rust, dependency-free by default. Provides moving
average smoothing and Savitzky–Golay smoothing and numerical derivatives for
uniformly sampled `f64` signals, standard normal variate (SNV) normalization
and polynomial detrending of spectra. The optional `pca` feature adds
principal component analysis with outlier statistics.

```rust
use chemometrics::smooth::{MovingAverage, SavitzkyGolay};

fn main() -> Result<(), chemometrics::Error> {
    let signal = [2.0, 2.0, 5.0, 2.0, 1.0, 0.0, 1.0, 4.0, 9.0];

    // Apply each filter independently to the original signal.
    let average = MovingAverage::new(3)?.apply(&signal)?;
    let filter = SavitzkyGolay::new(5, 2)?;
    let smoothed = filter.apply(&signal)?;

    println!("Original: {signal:?}");
    println!("Moving average: {average:?}");
    println!("Savitzky–Golay: {smoothed:?}");

    // Alternatively, write into an existing buffer without allocating.
    let mut buffer = vec![0.0; signal.len()];
    filter.apply_into(&signal, &mut buffer)?;
    assert_eq!(buffer, smoothed);
    Ok(())
}
```

`signal` remains unchanged. `average` and `smoothed` are separate result
vectors; the Savitzky–Golay filter does not process `average`. Reuse `filter`
and `buffer` when processing additional spectra of the same length.

Run `cargo run --example preprocessing` for an executable example, and
`cargo run --features pca --example pca` for the principal component one.

## Savitzky–Golay derivatives

```rust
use chemometrics::smooth::SavitzkyGolay;

// Window 5, polynomial order 2, first derivative, sample spacing 0.5.
let filter = SavitzkyGolay::new_derivative(5, 2, 1, 0.5)?;
let derivative = filter.apply(&[0.0, 0.25, 1.0, 2.25, 4.0])?;
# Ok::<(), chemometrics::Error>(())
```

Derivative order must not exceed polynomial order; order zero is equivalent to
`SavitzkyGolay::new`. Sample spacing is `x[i + 1] - x[i]` and must be finite and
nonzero, even for order zero. Negative spacing supports descending axes.
Derivatives scale with `1 / spacing^d` and have units of intensity / axis units^d.
Differentiation can amplify noise; window length controls the smoothing.

## Standard normal variate

```rust
use chemometrics::{normalize::StandardNormalVariate, smooth::SavitzkyGolay};

// Absorbance sampled every 2 nm: first derivative, then SNV.
let absorbance = [0.52, 0.55, 0.61, 0.70, 0.78, 0.83, 0.84, 0.81, 0.74, 0.66, 0.60];
let derivative = SavitzkyGolay::new_derivative(5, 2, 1, 2.0)?.apply(&absorbance)?;
let normalized = StandardNormalVariate.apply(&derivative)?;
# Ok::<(), chemometrics::Error>(())
```

SNV centers each spectrum on its mean and divides it by its sample standard
deviation, with divisor `n - 1` as in R's `sd` and
`scipy.stats.zscore(x, ddof=1)`. Tools dividing by `n` return values larger by
`sqrt(n / (n - 1))`. Multiplicative scaling and constant offsets cancel; a
sloping baseline does not, which is what detrending below is for. A constant
spectrum yields zeros, and at least two samples are required
(`Error::TooFewSamples`). SNV needs no x-axis, holds no parameters and takes
O(n) time.

## Detrend

```rust
use chemometrics::{baseline::Detrend, normalize::StandardNormalVariate};

// SNV and Detrend: scatter correction, then a quadratic baseline.
let absorbance = [0.52, 0.55, 0.61, 0.70, 0.78, 0.83, 0.84, 0.81, 0.74, 0.66, 0.60];
let normalized = StandardNormalVariate.apply(&absorbance)?;
let corrected = Detrend::new(2).apply(&normalized)?;
# Ok::<(), chemometrics::Error>(())
```

`Detrend` fits a least-squares polynomial to a whole spectrum and subtracts it.
The fit uses the sample position scaled to `[-1, 1]`, which under uniform
sampling is the same fit as over the wavelength axis, ascending or descending,
so no x values are needed. Order 0 subtracts the mean, order 1 a straight line
and order 2 a parabola. Orders 0 and 1 match `scipy.signal.detrend` with
`type="constant"` and `type="linear"`; order 2 after SNV is the detrending step
of Barnes, Dhanoa and Lister's SNV and Detrend, the usual treatment of scatter
and curved baselines in near-infrared spectra.

Every polynomial up to the fitted order is removed exactly, so adding one to a
spectrum leaves the result unchanged. Strong bands pull the fit towards
themselves and are damped along with the baseline; low orders limit this, and
orders above roughly 3 fit band structure rather than a baseline. Orders of a
few dozen degrees fail with `Error::NumericalFailure`, a limit that falls as a
spectrum grows longer. A spectrum needs at least `order + 1` samples
(`Error::TooFewSamples`); with exactly that many the fit passes through every
sample and the result is zeros. Savitzky–Golay derivatives are the alternative
that needs no fit at all: the first removes offsets and the second also removes
slopes.

Detrending holds only its order, so one value corrects spectra of any length.
`apply_into` writes into a caller-owned buffer without allocating. Both take
O(n × order²) time and no working memory beyond the output, using Gram
polynomials evaluated from their recurrence.

## Principal component analysis

Principal components are behind the optional `pca` feature, because they need a
matrix decomposition and therefore a dependency:

```toml
[dependencies]
chemometrics = { version = "0.2", features = ["pca"] }
```

```rust,ignore
use chemometrics::pca::Pca;

// Four spectra of three wavelengths each, row-major.
let data = [
    1.0, 2.0, 3.0, //
    2.0, 4.1, 6.0, //
    3.0, 5.9, 9.0, //
    4.0, 8.0, 12.0,
];
let model = Pca::fit(&data, 3, 2)?;
println!("explained: {:?}", model.explained_variance_ratio());

let projection = model.project(&[2.0, 4.0, 6.0])?;
let diagnostics = projection.diagnostics;
println!("T² {}, Q {}", diagnostics.hotelling_t2, diagnostics.q_residual);
```

This block is not compiled with the README's other examples, which run without
the feature; the module documentation carries the tested version.


A set of spectra is one flat row-major slice plus the number of variables per
sample, which is the layout of a NumPy array or an `ndarray` row-major view, so
no matrix type appears in the API. `Pca::fit` mean-centers the data and keeps
the requested number of components, at most `min(samples - 1, variables)`.
`fit` does not scale individual variables: spectral variables share one unit,
and autoscaling would amplify noise-only wavelengths. Autoscaling for other data
is planned as a separate method.

`Pca::project` places a further spectrum in the model and returns its scores
together with Hotelling's T², the squared distance inside the component plane,
and the Q residual, the squared distance to it. Both statistics are needed: T²
finds a spectrum that is extreme in directions the model knows, Q finds one that
carries variation the model does not describe, such as an unexpected band.
`project_into` writes the scores into a caller-owned buffer without allocating.
Control limits are not computed; their formulas are documented on the
`Diagnostics` fields.

`all_eigenvalues` returns the eigenvalues of every component the data allows,
the retained ones first. The discarded ones give the Jackson–Mudholkar limit for
Q, and divided by `total_variance` they show how many components the data
supports, even where fitting that many fails because preprocessing such as SNV
or detrending removed directions from the data.

Eigenvalues are score variances with divisor `samples - 1`, and
`explained_variance_ratio` divides them by the total variance of the centered
data, so the shares sum to one, up to rounding, when every component is kept.
The sign of a component is fixed by making its largest-magnitude loading
positive; loadings within a relative √ε of the largest count as equally large
and the first of them decides, so rounding cannot flip a component and results
stay reproducible. Components whose eigenvalues are nearly equal are not
determined by the data. A requested component that cannot be told apart from
rounding in the uncentered data gives `Error::NumericalFailure`.

Fitting takes O(samples × variables × min(samples, variables)) time and runs on
a single thread. At its peak it holds a few times the size of the data: the
centered copy, the decomposition's working copies and its factors. The data is
scaled by a power of two, so spectra of extreme magnitude stay stable;
magnitudes whose variances no longer fit into an `f64` are rejected. A spectrum
far larger or smaller than the training data still gets its scores and Q
residual wherever they fit into an `f64`. Results are not bit-identical across
CPU architectures, since the decomposition uses SIMD. Its working memory is
reserved like this crate's own buffers, but its matrix kernels may still
allocate internally and abort on failure instead of reporting
`Error::AllocationFailure`.

The feature costs about 50 additional crates through
[`faer`](https://crates.io/crates/faer), a pure-Rust linear algebra library that
needs no BLAS or LAPACK. `faer` is an internal implementation detail and does
not appear in this crate's API. `unsafe_code = "forbid"` applies to this crate
only; `faer` uses `unsafe` for SIMD. The default build remains dependency-free.
Cargo has no per-feature `rust-version`: the feature is tested on 1.85 with this
repository's `Cargo.lock`, but a fresh resolution may pick dependency versions
that require a newer toolchain.

## Signal and edge conventions

Window lengths are positive odd numbers of samples. The input must contain
at least one full window. The polynomial order must be below the window length.
No x-axis array is required: ascending and descending uniform sampling both work.
Uneven sampling is not supported.

Output length and sample alignment are preserved. At index `i`, the window
starts at `min(i.saturating_sub(window / 2), input.len() - window)`.
No padding is added. The moving average therefore repeats the first and last
full-window means at the edges. For `[0, 1, 2, 3, 4]` with window 3, it returns
`[1, 1, 2, 3, 3]`. Savitzky–Golay evaluates the polynomial or its derivative at
each sample position, including the edges (equivalent in intent to SciPy's
`mode="interp"`). Both methods can distort features; SG can also overshoot.

NaN and infinity are rejected with the first offending index. Length and input
errors leave an existing output buffer unchanged. Numerical failures may leave
it partially written. Derivative construction rejects overflow or complete
underflow of its coordinate-scaling factor and non-finite coefficients with
`NumericalFailure`. Individual coefficients and outputs may still underflow. Constructors and `apply` report `AllocationFailure` for
unrepresentable buffer sizes or failed memory reservations. This does not
guarantee recovery from operating-system termination under memory pressure.
Empty inputs are too short, including for window 1.

Filters can be reused across spectra. `apply` allocates a result; `apply_into`
uses a caller-owned buffer without heap allocation. Both take O(n × window)
time. SG construction prepares O(window²) coefficients using scaled
Householder QR. Numerically rank-deficient fits return an error. Use modest
polynomial orders (typically 2 or 3); arbitrary high-order fits are not promised.

## Roadmap

Planned areas of development, without a fixed release schedule or ordering:

- Further spectral normalization (vector, area, min–max)
- Further baseline correction (asymmetric least squares, rubberband)
- Peak detection
- PLS regression, autoscaling and control limits for T² and Q, extending the
  `pca` feature

The default functionality will remain dependency-free. Apart from principal
component analysis, these capabilities are not implemented yet.

## Extension boundaries

The public API uses slices and has no dependency on a file format. Applications
can pass intensities read by `spc-spectra` directly to these filters.

Additional preprocessing can add slice-based modules. Matrix dependencies stay
optional and internal, as `pca` shows, so the preprocessing API is unaffected by
them. f32, no_std, irregular sampling, alternate edge modes, in-place filtering
and parallel processing are outside version 0.2.

## Numerical validation

Tests compare selected Savitzky–Golay results against values generated with
SciPy 1.18.1's
[`savgol_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html),
using `deriv=0`, `delta=1.0` and `mode="interp"`. Cases include edge samples,
signals exactly one window long, and windows of 21, 51 and 101 samples with
polynomial orders 2 and 3. Agreement is checked per sample using
`abs(actual - reference) <= 1e-10 + 1e-10 * abs(reference)`.

The stored reference values are in `tests/fixtures/scipy.txt`; regenerate them
with `uv run tests/fixtures/generate.py`, which installs SciPy 1.18.1 and NumPy
2.5.2 as declared in the script, or run it with Python in an environment with
those versions. Rust tests read the stored values and require no Python.
Reference agreement applies to the tested cases, not every possible input.
Independent tests also check polynomial preservation, known filter
coefficients, moving-average results, input validation and numerical failures.

Derivative references in `tests/fixtures/scipy_derivatives.txt` use the same
versions, edge mode and tolerance. Regenerate with
`uv run tests/fixtures/generate_derivatives.py`. Analytic tests also verify
polynomial derivatives, spacing scaling and axis reversal.

SNV references in `tests/fixtures/scipy_snv.txt` use `scipy.stats.zscore` with
`ddof=1`, the same versions and tolerance. Cases include random signals,
large offsets, impulses and synthetic NIR-like absorbance spectra: Gaussian
overtone and combination bands with multiplicative scatter, offset, baseline
slope and noise. Regenerate with `uv run tests/fixtures/generate_normalization.py`.
Analytic tests also verify unit moments, removal of scaling and offsets on
NIR-like spectra, constant spectra and extreme values.

Detrend references in `tests/fixtures/scipy_detrend.txt` are the residuals of
`numpy.polynomial.polynomial.polyfit` on the same scaled coordinate, using the
same versions and tolerance, and are cross-checked against `scipy.signal.detrend`
for orders 0 and 1. Cases cover orders 0 to 3, spectra of exactly `order + 1`
samples, large offsets, strong trends and NIR-like spectra with sloping and
curved baselines. Regenerate with `uv run tests/fixtures/generate_baseline.py`.
Analytic tests also verify exact removal of polynomials up to the fitted order,
orthogonality of the residual, invariance, idempotence, axis reversal and
extreme values.

PCA references in `tests/fixtures/numpy_pca.txt` come from `numpy.linalg.svd` of
the centered data and are cross-checked inside the generator against
scikit-learn 1.9.1's `PCA(svd_solver="full")`. Cases cover square, tall and wide
data, a fully determined case, large offsets and extreme magnitudes, and
NIR-like mixtures after SNV, each with clearly separated eigenvalues, since
loadings of nearly equal eigenvalues are not determined. Regenerate with
`uv run tests/fixtures/generate_pca.py`. Analytic tests also verify orthonormal
loadings, score variances, invariance under offsets, scaling and sample order,
the sign convention, the mean of T² and the sum of Q over the training set, T²
and Q for components of equal variance, whose loadings are not determined, and
the separation of T² and Q outliers.
