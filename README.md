# chemometrics

Spectral preprocessing in Rust, dependency-free by default. Version 0.1 provides
moving average and Savitzky–Golay smoothing for uniformly sampled `f64` signals.

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

Run `cargo run --example smoothing` for an executable example.

## Signal and edge conventions

Window lengths are positive odd numbers of samples. The input must contain
at least one full window. The polynomial order must be below the window length.
No x-axis is required: ascending and descending uniform sampling both work.
Uneven sampling is not supported.

Output length and sample alignment are preserved. At index `i`, the window
starts at `min(i.saturating_sub(window / 2), input.len() - window)`.
No padding is added. The moving average therefore repeats the first and last
full-window means at the edges. For `[0, 1, 2, 3, 4]` with window 3, it returns
`[1, 1, 2, 3, 3]`. Savitzky–Golay instead evaluates the fitted polynomial at each
sample position, including edge positions (equivalent in intent to SciPy's
`mode="interp"`). Both methods can distort features; SG can also overshoot.

NaN and infinity are rejected with the first offending index. Length and input
errors leave an existing output buffer unchanged. Numerical failures may leave
it partially written. Constructors and `apply` report `AllocationFailure` for
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

- Numerical derivatives
- Spectral normalization
- Baseline correction
- Peak detection
- PCA and PLS through optional features

The default functionality will remain dependency-free. These capabilities are
not implemented in version 0.1.

## Extension boundaries

The public API uses slices and has no dependency on a file format. Applications
can pass intensities read by `spc-spectra` directly to these filters.

Future derivatives can reuse the private polynomial machinery and introduce
sample spacing in their own API. Additional preprocessing can add slice-based
modules. Matrix dependencies will be optional, preserving the smoothing API.
No placeholder APIs or feature flags are published yet. f32, no_std, irregular
sampling, alternate edge modes, in-place filtering and parallel processing are
outside version 0.1.

## Numerical validation

Tests compare selected Savitzky–Golay results against values generated with
[SciPy 1.17.1](https://docs.scipy.org/doc/scipy-1.17.1/reference/generated/scipy.signal.savgol_filter.html),
using `deriv=0`, `delta=1.0` and `mode="interp"`. Cases include edge samples,
signals exactly one window long, and windows of 21, 51 and 101 samples with
polynomial orders 2 and 3. Agreement is checked per sample using
`abs(actual - reference) <= 1e-10 + 1e-10 * abs(reference)`.

The stored reference values are in `tests/fixtures/scipy.txt`; regenerate them
with `tests/fixtures/generate.py` in a Python environment containing SciPy
1.17.1 and NumPy 2.5.2. Rust tests read the stored values and require no Python.
Reference agreement applies to the tested cases, not every possible input.
Independent tests also check polynomial preservation, known filter
coefficients, moving-average results, input validation and numerical failures.
