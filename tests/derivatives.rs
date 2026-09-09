//! Analytic and independent reference checks for SG derivatives.
use chemometrics::{Error, smooth::SavitzkyGolay};

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 + 1e-10 * b.abs(),
            "sample {i}: {a} != {b}"
        );
    }
}

#[test]
fn analytic_polynomial_derivatives_including_edges() {
    for window in [3, 5, 9, 15] {
        for order in 0..=4.min(window - 1) {
            for derivative in 0..=order {
                for degree in 0..=order {
                    for spacing in [0.5_f64, 1.0, 2.0, -0.5, -2.0] {
                        for length in [window, window + 8] {
                            // Bounded amplitudes avoid making this a test of large offsets.
                            let radius = length as f64 * spacing.abs();
                            let coordinates: Vec<_> = (0..length)
                                .map(|i| (i as f64 - (length - 1) as f64 / 2.0) * spacing)
                                .collect();
                            let input: Vec<_> = coordinates
                                .iter()
                                .map(|x| (x / radius).powi(degree as i32))
                                .collect();
                            let expected: Vec<_> = coordinates
                                .iter()
                                .map(|x| {
                                    if degree < derivative {
                                        0.0
                                    } else {
                                        let factor = (0..derivative)
                                            .fold(1.0, |a, k| a * (degree - k) as f64);
                                        factor * (x / radius).powi((degree - derivative) as i32)
                                            / radius.powi(derivative as i32)
                                    }
                                })
                                .collect();
                            let filter =
                                SavitzkyGolay::new_derivative(window, order, derivative, spacing)
                                    .unwrap();
                            close(&filter.apply(&input).unwrap(), &expected);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn order_zero_is_exactly_the_existing_smoother() {
    let input: Vec<_> = (0..21).map(|i| (i as f64 * 0.7).sin()).collect();
    for (window, order) in [(1, 0), (5, 0), (5, 2), (9, 3), (21, 4)] {
        let expected = SavitzkyGolay::new(window, order)
            .unwrap()
            .apply(&input)
            .unwrap();
        for spacing in [
            1.0,
            0.5,
            -2.0,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
        ] {
            assert_eq!(
                SavitzkyGolay::new_derivative(window, order, 0, spacing)
                    .unwrap()
                    .apply(&input)
                    .unwrap(),
                expected
            );
        }
    }
}

#[test]
fn spacing_scaling_and_axis_reversal() {
    let input: Vec<_> = (0..23).map(|i| (i as f64 * 0.7).sin()).collect();
    let reversed: Vec<_> = input.iter().rev().copied().collect();
    for derivative in 1..=3 {
        let unit = SavitzkyGolay::new_derivative(7, 3, derivative, 1.0)
            .unwrap()
            .apply(&input)
            .unwrap();
        for spacing in [0.5_f64, 2.0, -0.5, -2.0] {
            let actual = SavitzkyGolay::new_derivative(7, 3, derivative, spacing)
                .unwrap()
                .apply(&input)
                .unwrap();
            let expected: Vec<_> = unit
                .iter()
                .map(|x| x / spacing.powi(derivative as i32))
                .collect();
            close(&actual, &expected);
            let backward = SavitzkyGolay::new_derivative(7, 3, derivative, -spacing)
                .unwrap()
                .apply(&reversed)
                .unwrap();
            close(&backward, &actual.iter().rev().copied().collect::<Vec<_>>());
        }
    }
}

#[test]
fn derivative_validation_and_precedence() {
    assert!(matches!(
        SavitzkyGolay::new_derivative(0, 0, 1, 0.0),
        Err(Error::InvalidWindowLength(0))
    ));
    assert!(matches!(
        SavitzkyGolay::new_derivative(3, 3, 4, 0.0),
        Err(Error::InvalidPolynomialOrder { .. })
    ));
    assert_eq!(
        SavitzkyGolay::new_derivative(3, 2, 3, 0.0).unwrap_err(),
        Error::InvalidDerivativeOrder {
            derivative_order: 3,
            polynomial_order: 2
        }
    );
    for derivative in [0, 1, 2] {
        for spacing in [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                SavitzkyGolay::new_derivative(5, 2, derivative, spacing).unwrap_err(),
                Error::InvalidSampleSpacing
            );
        }
    }
    assert!(matches!(
        SavitzkyGolay::new_derivative(1, 0, 1, 1.0),
        Err(Error::InvalidDerivativeOrder { .. })
    ));
    assert!(matches!(
        SavitzkyGolay::new_derivative(usize::MAX, 2, 1, 1.0),
        Err(Error::AllocationFailure)
    ));
    assert!(matches!(
        SavitzkyGolay::new_derivative(101, 100, 1, 1.0),
        Err(Error::NumericalFailure)
    ));
}

#[test]
fn derivatives_reuse_and_preserve_buffer_on_invalid_input() {
    let filter = SavitzkyGolay::new_derivative(5, 2, 1, 0.5).unwrap();
    let mut buffer = [42.0; 7];
    for input in [[1.0; 7], [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0], [1.0; 7]] {
        filter.apply_into(&input, &mut buffer).unwrap();
        assert_eq!(buffer.as_slice(), filter.apply(&input).unwrap());
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        buffer.fill(42.0);
        let input = [0.0, bad, f64::NAN, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(
            filter.apply_into(&input, &mut buffer),
            Err(Error::NonFiniteInput { index: 1 })
        );
        assert_eq!(
            filter.apply(&input),
            Err(Error::NonFiniteInput { index: 1 })
        );
        assert_eq!(buffer, [42.0; 7]);
    }
    for input in [&[][..], &[1.0; 4][..]] {
        assert!(matches!(
            filter.apply_into(input, &mut buffer),
            Err(Error::SignalTooShort { .. })
        ));
        assert!(matches!(
            filter.apply(input),
            Err(Error::SignalTooShort { .. })
        ));
        assert_eq!(buffer, [42.0; 7]);
    }
    assert!(matches!(
        filter.apply_into(&[1.0; 6], &mut buffer),
        Err(Error::OutputLengthMismatch { .. })
    ));
    assert_eq!(buffer, [42.0; 7]);
}

#[test]
fn extreme_spacing_reports_unrepresentable_scaling() {
    for spacing in [1e-200, -1e-200, 1e200, -1e200] {
        assert!(matches!(
            SavitzkyGolay::new_derivative(5, 2, 2, spacing),
            Err(Error::NumericalFailure)
        ));
    }
    assert!(matches!(
        SavitzkyGolay::new_derivative(3, 1, 1, f64::from_bits(1)),
        Err(Error::NumericalFailure)
    ));
    // A large finite spacing must not fail just because window * spacing overflows.
    for spacing in [f64::MAX, -f64::MAX] {
        let output = SavitzkyGolay::new_derivative(5, 2, 1, spacing)
            .unwrap()
            .apply(&[0.0, 1.0, 2.0, 3.0, 4.0])
            .unwrap();
        for value in output {
            assert!((value * spacing - 1.0).abs() < 1e-12);
        }
    }
    // A tiny representable spacing can have finite kernels but overflow on application.
    let filter = SavitzkyGolay::new_derivative(3, 1, 1, 1e-308).unwrap();
    assert_eq!(
        filter.apply(&[-10.0, 0.0, 10.0]),
        Err(Error::NumericalFailure)
    );
}

#[test]
fn scipy_derivative_reference() {
    let mut cases = 0;
    for (line_number, line) in include_str!("fixtures/scipy_derivatives.txt")
        .lines()
        .enumerate()
    {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<_> = line.split('|').collect();
        assert_eq!(parts.len(), 6, "fixture row {}", line_number + 1);
        let window = parts[0].parse().unwrap();
        let order = parts[1].parse().unwrap();
        let derivative = parts[2].parse().unwrap();
        let spacing = parts[3].parse().unwrap();
        let parse = |text: &str| {
            text.split(',')
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let actual = SavitzkyGolay::new_derivative(window, order, derivative, spacing)
            .unwrap()
            .apply(&parse(parts[4]))
            .unwrap();
        let expected = parse(parts[5]);
        assert_eq!(actual.len(), expected.len());
        for (index, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 + 1e-10 * b.abs(),
                "fixture row {}, window={window}, order={order}, derivative={derivative}, spacing={spacing}, sample {index}: {a} != {b}",
                line_number + 1
            );
        }
        cases += 1;
    }
    assert_eq!(cases, 56, "missing derivative reference cases");
}
