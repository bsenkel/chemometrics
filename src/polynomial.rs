// Private least-squares machinery; intentionally not a general matrix API.
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
