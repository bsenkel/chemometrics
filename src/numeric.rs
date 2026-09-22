// Private numerical helpers shared by the public modules: least squares, sample
// scaling, validation, buffers and the thin SVD adapter; intentionally not a
// general matrix API.
use crate::Error;

fn check_capacity<T>(length: usize) -> Result<(), Error> {
    let bytes = length
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(Error::AllocationFailure)?;
    if bytes > isize::MAX as usize {
        return Err(Error::AllocationFailure);
    }
    Ok(())
}

fn reserved<T>(length: usize) -> Result<Vec<T>, Error> {
    check_capacity::<T>(length)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Error::AllocationFailure)?;
    Ok(values)
}

pub(crate) fn zeros(length: usize) -> Result<Vec<f64>, Error> {
    let mut values = reserved(length)?;
    values.resize(length, 0.0);
    Ok(values)
}

pub(crate) fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let next = total + adjusted;
        correction = (next - total) - adjusted;
        total = next;
    }
    total
}

/// Largest power of two not above a positive, finite `magnitude`.
///
/// Division by a power of two is exact whenever the quotient is a normal
/// number. Only samples many orders of magnitude below the largest one can
/// round, and their contribution to the result is negligible.
fn power_of_two_floor(magnitude: f64) -> f64 {
    let bits = magnitude.to_bits();
    let exponent = bits & (0x7ff << 52);
    if exponent == 0 {
        // Subnormal: keep only the highest set mantissa bit.
        f64::from_bits(1 << (63 - bits.leading_zeros()))
    } else {
        f64::from_bits(exponent)
    }
}

/// Checks a spectrum and its output buffer for the per-spectrum transforms.
///
/// Reports too few samples, a buffer of different length and non-finite input,
/// in that order, before anything is written.
pub(crate) fn validate(input: &[f64], output_len: usize, minimum: usize) -> Result<(), Error> {
    if input.len() < minimum {
        return Err(Error::TooFewSamples {
            length: input.len(),
            minimum,
        });
    }
    if output_len != input.len() {
        return Err(Error::OutputLengthMismatch {
            expected: input.len(),
            actual: output_len,
        });
    }
    if let Some(index) = input.iter().position(|x| !x.is_finite()) {
        return Err(Error::NonFiniteInput { index });
    }
    Ok(())
}

/// Divides samples by a power of two near their largest magnitude and
/// subtracts the scaled first sample.
///
/// Scaled samples lie in (-2, 2) and their offsets in (-4, 4), so sums and
/// squares stay finite near `f64::MAX` and representable for subnormal
/// spectra. Samples within a factor of two of the first one differ exactly
/// (Sterbenz lemma), so a large offset cannot round away a small variation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Shift {
    pub(crate) scale: Scale,
    reference: f64,
}

/// Power-of-two divisor that maps the largest magnitude of a sample set into
/// [1, 2), or `1.0` when every sample is zero.
///
/// Dividing by it is exact, so scaled results are restored without error.
/// Restoring states the units: `restore` for a quantity in sample units,
/// `restore_squared` for one in their square, such as a variance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Scale {
    factor: f64,
}

impl Scale {
    /// Assumes `values` is finite.
    pub(crate) fn new(values: &[f64]) -> Self {
        let magnitude = values.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let factor = if magnitude == 0.0 {
            1.0
        } else {
            power_of_two_floor(magnitude)
        };
        Self { factor }
    }

    pub(crate) fn divide(self, x: f64) -> f64 {
        x / self.factor
    }

    pub(crate) fn restore(self, x: f64) -> f64 {
        x * self.factor
    }

    #[cfg(feature = "pca")]
    pub(crate) fn restore_squared(self, x: f64) -> f64 {
        (x * self.factor) * self.factor
    }
}

impl Shift {
    /// Assumes `input` is nonempty and finite.
    pub(crate) fn new(input: &[f64]) -> Self {
        let scale = Scale::new(input);
        Self {
            scale,
            reference: scale.divide(input[0]),
        }
    }

    pub(crate) fn apply(self, x: f64) -> f64 {
        self.scale.divide(x) - self.reference
    }
}

// All evaluation rows, flattened row-major: each row is one filter kernel.
pub(crate) fn kernels(
    window: usize,
    order: usize,
    derivative: usize,
    spacing: f64,
) -> Result<Vec<f64>, Error> {
    if window == 1 {
        let mut identity = zeros(1)?;
        identity[0] = 1.0;
        return Ok(identity);
    }
    let columns = order.checked_add(1).ok_or(Error::AllocationFailure)?;
    let size = window
        .checked_mul(columns)
        .ok_or(Error::AllocationFailure)?;
    let kernel_size = window.checked_mul(window).ok_or(Error::AllocationFailure)?;
    // Validate every buffer layout before attempting any large allocation.
    check_capacity::<f64>(size)?;
    check_capacity::<f64>(kernel_size)?;
    check_capacity::<Vec<f64>>(columns)?;
    let mut result = reserved(kernel_size)?;
    let mut a = zeros(size)?;
    let coordinate = |i: usize| 2.0 * i as f64 / (window - 1) as f64 - 1.0;
    for i in 0..window {
        let x = coordinate(i);
        let mut power = 1.0;
        for j in 0..columns {
            a[i * columns + j] = power;
            power *= x;
        }
    }
    let norm = a.iter().fold(0.0_f64, |norm, x| norm.hypot(*x));
    // Standard scale-dependent rank threshold, using the Frobenius norm.
    let tolerance = f64::EPSILON * window.max(columns) as f64 * norm;
    let mut reflectors = reserved(columns)?;
    for k in 0..columns {
        let norm = (k..window).fold(0.0_f64, |norm, i| norm.hypot(a[i * columns + k]));
        if !norm.is_finite() || norm <= tolerance {
            return Err(Error::NumericalFailure);
        }
        let alpha = -norm.copysign(a[k * columns + k]);
        let mut v = reserved(window - k)?;
        v.extend((k..window).map(|i| a[i * columns + k]));
        v[0] -= alpha;
        let vnorm = v.iter().fold(0.0_f64, |norm, x| norm.hypot(*x));
        for x in &mut v {
            *x /= vnorm;
        }
        for j in k..columns {
            let dot = sum(v
                .iter()
                .enumerate()
                .map(|(r, x)| x * a[(k + r) * columns + j]));
            for (r, x) in v.iter().enumerate() {
                a[(k + r) * columns + j] -= 2.0 * x * dot;
            }
        }
        a[k * columns + k] = alpha;
        reflectors.push(v);
    }
    // Divide in this order to avoid overflowing (window - 1) * spacing.
    let mut scale = 1.0;
    if derivative > 0 {
        let step = (2.0 / (window - 1) as f64) / spacing;
        for _ in 0..derivative {
            scale *= step;
        }
        if !scale.is_finite() || scale == 0.0 {
            return Err(Error::NumericalFailure);
        }
    }
    // Reused across positions; entries beyond `columns` must start at zero.
    let mut weights = zeros(window)?;
    for position in 0..window {
        // Solve R^T z = evaluation basis, then compute Q z.
        weights.fill(0.0);
        let x = coordinate(position);
        let mut power = 1.0;
        for j in 0..columns {
            let previous = sum((0..j).map(|k| a[k * columns + j] * weights[k]));
            let basis = if j < derivative {
                0.0
            } else {
                let factor = (0..derivative).fold(1.0, |value, k| value * (j - k) as f64);
                let basis = factor * power;
                power *= x;
                basis
            };
            weights[j] = (basis - previous) / a[j * columns + j];
        }
        for k in (0..columns).rev() {
            let v = &reflectors[k];
            let dot = sum(v.iter().enumerate().map(|(r, x)| x * weights[k + r]));
            for (r, x) in v.iter().enumerate() {
                weights[k + r] -= 2.0 * x * dot;
            }
        }
        if derivative > 0 {
            for weight in &mut weights {
                *weight *= scale;
            }
        }
        if weights.iter().any(|x| !x.is_finite()) {
            return Err(Error::NumericalFailure);
        }
        result.extend_from_slice(&weights);
    }
    Ok(result)
}

/// Thin singular value decomposition of a row-major matrix, truncated to the
/// leading `keep` components.
///
/// Singular values are nonnegative and sorted in nonincreasing order. This is
/// the only place that uses a matrix library, so another backend would replace
/// this function alone.
#[cfg(feature = "pca")]
pub(crate) struct ThinSvd {
    /// Leading `keep` singular values.
    pub(crate) values: Vec<f64>,
    /// Left factor, `rows` × `keep`, row-major.
    pub(crate) left: Vec<f64>,
    /// Right factor transposed, `keep` × `columns`, row-major.
    pub(crate) right: Vec<f64>,
}

/// Decomposes `matrix`, given row-major with `rows` × `columns` finite entries.
///
/// `keep` must not exceed `min(rows, columns)`. Returns
/// [`Error::NumericalFailure`] if the decomposition fails or produces
/// non-finite values. Allocations inside the backend are not fallible.
#[cfg(feature = "pca")]
pub(crate) fn thin_svd(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    keep: usize,
) -> Result<ThinSvd, Error> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::svd::{self, ComputeSvdVectors};
    debug_assert_eq!(matrix.len(), rows * columns);
    debug_assert!(keep <= rows.min(columns));
    let size = rows.min(columns);
    // Every product is bounded by `matrix.len()`, so it cannot overflow.
    let mut singular = zeros(size)?;
    let mut u = zeros(rows * size)?;
    let mut v = zeros(columns * size)?;
    // An explicit sequential `Par` keeps the decomposition independent of
    // faer's global parallelism, which other crates may enable or disable.
    let par = faer::Par::Seq;
    let thin = ComputeSvdVectors::Thin;
    let request = svd::svd_scratch::<f64>(rows, columns, thin, thin, par, Default::default());
    let mut workspace = MemBuffer::try_new(request).map_err(|_| Error::AllocationFailure)?;
    svd::svd(
        faer::MatRef::from_row_major_slice(matrix, rows, columns),
        faer::ColMut::from_slice_mut(&mut singular).as_diagonal_mut(),
        Some(faer::MatMut::from_column_major_slice_mut(
            &mut u, rows, size,
        )),
        Some(faer::MatMut::from_column_major_slice_mut(
            &mut v, columns, size,
        )),
        par,
        MemStack::new(&mut workspace),
        Default::default(),
    )
    .map_err(|_| Error::NumericalFailure)?;
    let mut values = zeros(keep)?;
    let mut left = zeros(rows * keep)?;
    let mut right = zeros(keep * columns)?;
    values.copy_from_slice(&singular[..keep]);
    for (i, row) in left.chunks_exact_mut(keep).enumerate() {
        for (a, value) in row.iter_mut().enumerate() {
            *value = u[a * rows + i];
        }
    }
    // Column `a` of the column-major V is row `a` of its transpose.
    right.copy_from_slice(&v[..keep * columns]);
    let finite = |slice: &[f64]| slice.iter().all(|x| x.is_finite());
    if !finite(&values) || !finite(&left) || !finite(&right) {
        return Err(Error::NumericalFailure);
    }
    Ok(ThinSvd {
        values,
        left,
        right,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn power_of_two_floor_boundaries() {
        let largest_subnormal = f64::from_bits((1 << 52) - 1);
        for (magnitude, expected) in [
            (1.0, 1.0),
            (1.5, 1.0),
            (0.75, 0.5),
            (3.0, 2.0),
            (f64::MAX, f64::from_bits(0x7fe << 52)),
            (f64::MIN_POSITIVE, f64::MIN_POSITIVE),
            (largest_subnormal, f64::from_bits(1 << 51)),
            (f64::from_bits(3), f64::from_bits(2)),
            (f64::from_bits(7), f64::from_bits(4)),
            (f64::from_bits(1 << 40), f64::from_bits(1 << 40)),
            (f64::from_bits(1), f64::from_bits(1)),
            (f64::from_bits((1 << 52) + 12_345), f64::MIN_POSITIVE),
            (f64::from_bits(0x3ff0_0000_0000_0001), 1.0),
            (
                f64::from_bits(0x7fef_ffff_ffff_fffe),
                f64::from_bits(0x7fe << 52),
            ),
        ] {
            assert_eq!(power_of_two_floor(magnitude), expected, "{magnitude:e}");
        }
    }

    #[test]
    fn shift_scales_by_a_power_of_two_and_subtracts_the_first_sample() {
        assert_eq!(Shift::new(&[0.0, -0.0]).scale.restore(1.0), 1.0);
        let shift = Shift::new(&[0.5, -3.0]);
        assert_eq!(shift.scale.restore(1.0), 2.0);
        #[cfg(feature = "pca")]
        assert_eq!(shift.scale.restore_squared(1.0), 4.0);
        assert_eq!(shift.apply(0.5), 0.0);
        assert_eq!(shift.apply(-3.0), -1.75);
    }

    #[test]
    fn rejects_impossible_byte_capacity_without_allocating() {
        assert_eq!(
            check_capacity::<f64>(isize::MAX as usize / 8 + 1),
            Err(Error::AllocationFailure)
        );
        assert_eq!(
            check_capacity::<f64>(usize::MAX),
            Err(Error::AllocationFailure)
        );
        // window² fits usize but its f64 byte layout exceeds isize::MAX.
        let window = 1_usize << (usize::BITS / 2 - 1);
        assert_eq!(
            kernels(window + 1, 0, 0, 1.0),
            Err(Error::AllocationFailure)
        );
        assert_eq!(
            kernels(usize::MAX, 0, 0, 1.0),
            Err(Error::AllocationFailure)
        );
    }

    #[test]
    fn known_center_kernel() {
        let k = kernels(5, 2, 0, 1.0).unwrap();
        for (actual, expected) in k[10..15].iter().zip([-3.0, 12.0, 17.0, 12.0, -3.0]) {
            assert!((actual - expected / 35.0).abs() < 1e-12);
        }
    }
    #[test]
    fn rejects_rank_deficient_high_order() {
        assert_eq!(kernels(101, 100, 0, 1.0), Err(Error::NumericalFailure));
    }
}
