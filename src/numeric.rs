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

pub(crate) fn reserved<T>(length: usize) -> Result<Vec<T>, Error> {
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

/// Checks a signal and its output buffer for the filters and per-spectrum
/// transforms.
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

/// Number of spectra in a row-major slice with `variables` values each, or a
/// shape error.
pub(crate) fn samples_of(length: usize, variables: usize) -> Result<usize, Error> {
    if variables == 0 || length == 0 || length % variables != 0 {
        return Err(Error::InvalidDataShape { length, variables });
    }
    Ok(length / variables)
}

/// Standard scale-dependent rank threshold: a norm or singular value at or
/// below it is rounding rather than information.
pub(crate) fn rank_tolerance(rows: usize, columns: usize, norm: f64) -> f64 {
    f64::EPSILON * rows.max(columns) as f64 * norm
}

/// Savitzky–Golay coefficients, flattened row-major: row `p` holds the
/// `window` weights that evaluate the `derivative`-th derivative of the
/// least-squares polynomial of degree `order` at window position `p`.
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
    // Positions scaled to [-1, 1] keep the design matrix well conditioned, so
    // higher orders still pass the rank check.
    let coordinate = |i: usize| 2.0 * i as f64 / (window - 1) as f64 - 1.0;
    // The design matrix, column-major so that each column is one slice: column
    // `j` holds `x^j` at every window position.
    let mut a = zeros(size)?;
    let mut norm = 0.0_f64;
    for i in 0..window {
        let x = coordinate(i);
        let mut power = 1.0;
        for j in 0..columns {
            a[j * window + i] = power;
            norm = norm.hypot(power);
            power *= x;
        }
    }
    let reflectors = householder_qr(
        &mut a,
        window,
        columns,
        rank_tolerance(window, columns, norm),
    )?;
    let scale = derivative_scale(window, derivative, spacing)?;
    // The derivative of x^j is j!/(j − d)! · x^(j − d). The factors do not
    // depend on the position, and entries of `basis` below `derivative` stay
    // zero.
    let mut factors = zeros(columns)?;
    for (j, factor) in factors.iter_mut().enumerate().skip(derivative) {
        *factor = (0..derivative).fold(1.0, |value, k| value * (j - k) as f64);
    }
    let mut basis = zeros(columns)?;
    // Reused across positions; entries beyond `columns` must start at zero.
    let mut weights = zeros(window)?;
    for position in 0..window {
        let x = coordinate(position);
        let mut power = 1.0;
        for (value, factor) in basis.iter_mut().zip(&factors).skip(derivative) {
            *value = factor * power;
            power *= x;
        }
        // The weights are Q R⁻ᵀ b: solve Rᵀ z = b, then apply Q to z.
        weights.fill(0.0);
        for j in 0..columns {
            let column = &a[j * window..j * window + j];
            let previous = sum(column.iter().zip(&weights).map(|(r, z)| r * z));
            weights[j] = (basis[j] - previous) / a[j * window + j];
        }
        for (k, v) in reflectors.iter().enumerate().rev() {
            apply_reflection(v, &mut weights[k..]);
        }
        for weight in &mut weights {
            *weight *= scale;
        }
        if weights.iter().any(|x| !x.is_finite()) {
            return Err(Error::NumericalFailure);
        }
        result.extend_from_slice(&weights);
    }
    Ok(result)
}

/// Why a QR factorization stopped.
#[derive(Debug, PartialEq)]
pub(crate) enum QrError {
    /// The column at this index adds no direction of its own to the preceding
    /// ones.
    DependentColumn(usize),
    /// Arithmetic became non-finite or memory could not be reserved.
    Other(Error),
}

impl From<Error> for QrError {
    fn from(error: Error) -> Self {
        Self::Other(error)
    }
}

impl From<QrError> for Error {
    fn from(failure: QrError) -> Self {
        match failure {
            QrError::DependentColumn(_) => Self::NumericalFailure,
            QrError::Other(error) => error,
        }
    }
}

/// Factorizes the column-major `rows` × `columns` matrix `a` as `Q R` with
/// Householder reflections. Afterwards `a` holds `R` in its upper triangle, and
/// `Q` is the product of the returned reflections, first to last.
///
/// Returns [`QrError::DependentColumn`] if a diagonal entry of `R` falls to
/// `tolerance`, where a column no longer adds a direction of its own,
/// [`Error::NumericalFailure`] if one is not finite, and
/// [`Error::AllocationFailure`] if the reflections cannot be reserved.
fn householder_qr(
    a: &mut [f64],
    rows: usize,
    columns: usize,
    tolerance: f64,
) -> Result<Vec<Vec<f64>>, QrError> {
    let mut reflectors = reserved(columns)?;
    for k in 0..columns {
        let column = &a[k * rows + k..(k + 1) * rows];
        let norm = column.iter().fold(0.0_f64, |norm, x| norm.hypot(*x));
        if !norm.is_finite() {
            return Err(Error::NumericalFailure.into());
        }
        if norm <= tolerance {
            return Err(QrError::DependentColumn(k));
        }
        // The sign opposite to the diagonal avoids cancellation in `v[0]`.
        let alpha = -norm.copysign(column[0]);
        let mut v = reserved(rows - k)?;
        v.extend_from_slice(column);
        v[0] -= alpha;
        let vnorm = v.iter().fold(0.0_f64, |norm, x| norm.hypot(*x));
        for x in &mut v {
            *x /= vnorm;
        }
        for j in k..columns {
            apply_reflection(&v, &mut a[j * rows + k..(j + 1) * rows]);
        }
        a[k * rows + k] = alpha;
        reflectors.push(v);
    }
    Ok(reflectors)
}

/// Thin QR factorization of a matrix with at least as many rows as columns.
pub(crate) struct ThinQr {
    /// `Q` transposed, `columns` × `rows`, row-major: row `j` is the `j`-th
    /// orthonormal column of `Q`.
    pub(crate) q: Vec<f64>,
    /// `R`, `columns` × `columns`, column-major, zero below the diagonal.
    pub(crate) r: Vec<f64>,
}

/// Factorizes the column-major `rows` × `columns` matrix `a` as `Q R`, with
/// orthonormal columns in `Q` and an upper triangular `R`.
///
/// `columns` must not exceed `rows`. Fails as [`householder_qr`] does.
pub(crate) fn thin_qr(
    mut a: Vec<f64>,
    rows: usize,
    columns: usize,
    tolerance: f64,
) -> Result<ThinQr, QrError> {
    debug_assert_eq!(a.len(), rows * columns);
    debug_assert!(columns <= rows);
    let reflectors = householder_qr(&mut a, rows, columns, tolerance)?;
    // The product is bounded by `a.len()`, so it cannot overflow.
    let mut r = zeros(columns * columns)?;
    for (j, column) in r.chunks_exact_mut(columns).enumerate() {
        column[..=j].copy_from_slice(&a[j * rows..=j * rows + j]);
    }
    // `R` is copied, so the buffer is free for `Q`, whose column `j` is `Q`
    // applied to the `j`-th unit vector.
    let mut q = a;
    q.fill(0.0);
    for (j, column) in q.chunks_exact_mut(rows).enumerate() {
        column[j] = 1.0;
        for (k, v) in reflectors.iter().enumerate().rev() {
            apply_reflection(v, &mut column[k..]);
        }
    }
    Ok(ThinQr { q, r })
}

/// Solves `min ‖a x − b‖` for the column-major `rows` × `columns` matrix `a`
/// with at least as many rows as columns. `a` is overwritten, and the solution
/// replaces the first `columns` entries of `b`.
///
/// Fails as [`householder_qr`] does.
pub(crate) fn least_squares(
    a: &mut [f64],
    rows: usize,
    columns: usize,
    tolerance: f64,
    b: &mut [f64],
) -> Result<(), QrError> {
    debug_assert_eq!(a.len(), rows * columns);
    debug_assert!(columns <= rows && b.len() == rows);
    let reflectors = householder_qr(a, rows, columns, tolerance)?;
    for (k, v) in reflectors.iter().enumerate() {
        apply_reflection(v, &mut b[k..]);
    }
    for j in (0..columns).rev() {
        let known = sum((j + 1..columns).map(|i| a[i * rows + j] * b[i]));
        b[j] = (b[j] - known) / a[j * rows + j];
    }
    Ok(())
}

/// Applies the reflection `I − 2 v vᵀ`, with `v` of unit length, to `values`.
fn apply_reflection(v: &[f64], values: &mut [f64]) {
    let dot = sum(v.iter().zip(values.iter()).map(|(x, y)| x * y));
    for (x, y) in v.iter().zip(values) {
        *y -= 2.0 * x * dot;
    }
}

/// Factor that turns derivatives along the fit coordinate in [-1, 1] into
/// derivatives per unit of x, where adjacent samples lie `spacing` apart.
fn derivative_scale(window: usize, derivative: usize, spacing: f64) -> Result<f64, Error> {
    // Divide in this order to avoid overflowing (window - 1) * spacing.
    let step = (2.0 / (window - 1) as f64) / spacing;
    let scale = (0..derivative).fold(1.0, |scale, _| scale * step);
    if !scale.is_finite() || scale == 0.0 {
        return Err(Error::NumericalFailure);
    }
    Ok(scale)
}

/// Thin singular value decomposition of a row-major matrix, with the factors
/// truncated to the leading `keep` components.
///
/// Singular values are nonnegative and sorted in nonincreasing order. This is
/// the only place that uses a matrix library, so another backend would replace
/// this function alone.
#[cfg(feature = "pca")]
pub(crate) struct ThinSvd {
    /// All `min(rows, columns)` singular values, not only the kept ones.
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
    let mut left = zeros(rows * keep)?;
    let mut right = zeros(keep * columns)?;
    for (i, row) in left.chunks_exact_mut(keep).enumerate() {
        for (a, value) in row.iter_mut().enumerate() {
            *value = u[a * rows + i];
        }
    }
    // Column `a` of the column-major V is row `a` of its transpose.
    right.copy_from_slice(&v[..keep * columns]);
    let finite = |slice: &[f64]| slice.iter().all(|x| x.is_finite());
    if !finite(&singular) || !finite(&left) || !finite(&right) {
        return Err(Error::NumericalFailure);
    }
    Ok(ThinSvd {
        values: singular,
        left,
        right,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn known_kernels_are_exact_to_rounding() {
        // Classic quadratic coefficients for window 5: the center and the left
        // edge of the smoother, and the center of the first derivative.
        let smooth = kernels(5, 2, 0, 1.0).unwrap();
        let derivative = kernels(5, 2, 1, 1.0).unwrap();
        for (row, expected) in [
            (
                &smooth[10..15],
                [-3.0, 12.0, 17.0, 12.0, -3.0].map(|x| x / 35.0),
            ),
            (
                &smooth[0..5],
                [31.0, 9.0, -3.0, -5.0, 3.0].map(|x| x / 35.0),
            ),
            (
                &derivative[10..15],
                [-2.0, -1.0, 0.0, 1.0, 2.0].map(|x| x / 10.0),
            ),
        ] {
            for (actual, expected) in row.iter().zip(expected) {
                let error = (actual - expected).abs();
                assert!(error <= 4.0 * f64::EPSILON, "{actual} != {expected}");
            }
        }
    }
    #[test]
    fn rejects_rank_deficient_high_order() {
        assert_eq!(kernels(101, 100, 0, 1.0), Err(Error::NumericalFailure));
    }

    #[test]
    fn thin_qr_reproduces_the_matrix_with_orthonormal_columns() {
        let (rows, columns) = (4, 3);
        let a = vec![1.0, 2.0, 0.5, -1.0, 0.0, 1.0, 3.0, 2.0, 2.0, -1.0, 1.0, 0.5];
        let qr = thin_qr(a.clone(), rows, columns, 0.0).unwrap();
        for i in 0..columns {
            for j in 0..columns {
                let dot = sum((0..rows).map(|k| qr.q[i * rows + k] * qr.q[j * rows + k]));
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() <= 8.0 * f64::EPSILON, "{i} {j}");
                if i > j {
                    assert_eq!(qr.r[j * columns + i], 0.0);
                }
            }
        }
        for j in 0..columns {
            for k in 0..rows {
                let product = sum((0..columns).map(|i| qr.q[i * rows + k] * qr.r[j * columns + i]));
                let error = (product - a[j * rows + k]).abs();
                assert!(error <= 16.0 * f64::EPSILON, "{j} {k}");
            }
        }
    }

    #[test]
    fn thin_qr_reports_the_first_dependent_column() {
        // The third column is the sum of the first two.
        let a = vec![1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 3.0];
        let tolerance = rank_tolerance(3, 3, 4.0);
        assert_eq!(
            thin_qr(a, 3, 3, tolerance).err(),
            Some(QrError::DependentColumn(2))
        );
    }
}
