//! Public API regression tests.
use chemometrics::{
    Error,
    smooth::{MovingAverage, SavitzkyGolay},
};

fn close(actual: &[f64], expected: &[f64]) {
    close_with(actual, expected, "");
}

fn close_with(actual: &[f64], expected: &[f64], context: impl std::fmt::Display) {
    assert_eq!(actual.len(), expected.len(), "length{context}");
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 + 1e-10 * b.abs(),
            "index {i}{context}: {a} != {b}"
        );
    }
}

#[test]
fn moving_average_reference_and_edges() {
    close(
        &MovingAverage::new(3)
            .unwrap()
            .apply(&[0., 1., 2., 3., 4.])
            .unwrap(),
        &[1., 1., 2., 3., 3.],
    );
    let input: Vec<f64> = (0..41).map(|i| (i as f64 * 1.7).sin()).collect();
    for w in [1, 3, 5, 11, 41] {
        let expected: Vec<f64> = (0..input.len())
            .map(|i| {
                let start = i.saturating_sub(w / 2).min(input.len() - w);
                input[start..start + w].iter().sum::<f64>() / w as f64
            })
            .collect();
        close(
            &MovingAverage::new(w).unwrap().apply(&input).unwrap(),
            &expected,
        );
        close(
            &SavitzkyGolay::new(w, 0).unwrap().apply(&input).unwrap(),
            &expected,
        );
    }
}

#[test]
fn polynomial_preservation_including_edges() {
    for w in [1, 3, 5, 9, 15] {
        for order in 0..=4.min(w - 1) {
            for n in [w, w + 12] {
                let input: Vec<f64> = (0..n)
                    .map(|i| {
                        let x = i as f64 / n as f64 - 0.5;
                        (0..=order).map(|p| (p + 1) as f64 * x.powi(p as i32)).sum()
                    })
                    .collect();
                close(
                    &SavitzkyGolay::new(w, order).unwrap().apply(&input).unwrap(),
                    &input,
                );
            }
        }
    }
}

#[test]
fn identity_and_reuse() {
    let input = [1.0, -2.0, 8.0, 0.1, 0.0];
    close(
        &MovingAverage::new(1).unwrap().apply(&input).unwrap(),
        &input,
    );
    close(
        &SavitzkyGolay::new(1, 0).unwrap().apply(&input).unwrap(),
        &input,
    );
    let ma = MovingAverage::new(3).unwrap();
    let sg = SavitzkyGolay::new(3, 1).unwrap();
    let mut buffer = [0.; 5];
    for data in [input, [7.; 5], input] {
        ma.apply_into(&data, &mut buffer).unwrap();
        close(&buffer, &ma.apply(&data).unwrap());
        sg.apply_into(&data, &mut buffer).unwrap();
        close(&buffer, &sg.apply(&data).unwrap());
    }
}

#[test]
fn validation_and_buffer_preservation() {
    for w in [0, 2, 4] {
        assert!(matches!(
            MovingAverage::new(w),
            Err(Error::InvalidWindowLength(_))
        ));
        assert!(matches!(
            SavitzkyGolay::new(w, 0),
            Err(Error::InvalidWindowLength(_))
        ));
    }
    assert!(matches!(
        SavitzkyGolay::new(3, 3),
        Err(Error::InvalidPolynomialOrder { .. })
    ));
    let ma = MovingAverage::new(3).unwrap();
    let sg = SavitzkyGolay::new(3, 2).unwrap();
    for input in [&[][..], &[1., 2.][..]] {
        let too_short = Err(Error::TooFewSamples {
            length: input.len(),
            minimum: 3,
        });
        assert_eq!(ma.apply(input), too_short);
        assert_eq!(sg.apply(input), too_short);
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let input = [1., value, f64::NAN];
        let mut buffer = [42.; 3];
        assert_eq!(
            ma.apply_into(&input, &mut buffer),
            Err(Error::NonFiniteInput { index: 1 })
        );
        assert_eq!(buffer, [42.; 3]);
        assert_eq!(
            sg.apply_into(&input, &mut buffer),
            Err(Error::NonFiniteInput { index: 1 })
        );
        assert_eq!(buffer, [42.; 3]);
    }
    let mut buffer = [42.; 2];
    assert!(matches!(
        ma.apply_into(&[1.; 3], &mut buffer),
        Err(Error::OutputLengthMismatch { .. })
    ));
    assert_eq!(buffer, [42.; 2]);
    assert!(matches!(
        sg.apply_into(&[1.; 3], &mut buffer),
        Err(Error::OutputLengthMismatch { .. })
    ));
    assert_eq!(buffer, [42.; 2]);
    assert!(matches!(
        sg.apply_into(&[1.; 2], &mut buffer),
        Err(Error::TooFewSamples { .. })
    ));
    assert_eq!(buffer, [42.; 2]);
    assert!(matches!(
        ma.apply_into(&[1.; 2], &mut buffer),
        Err(Error::TooFewSamples { .. })
    ));
    assert_eq!(buffer, [42.; 2]);
}

#[test]
fn extreme_values_do_not_succeed_with_nonfinite_results() {
    let max = f64::MAX;
    assert_eq!(
        MovingAverage::new(3).unwrap().apply(&[max; 3]),
        Err(Error::NumericalFailure)
    );
    let input = [-max, max, max, max, -max];
    assert_eq!(
        SavitzkyGolay::new(5, 2).unwrap().apply(&input),
        Err(Error::NumericalFailure)
    );
}

#[test]
fn scipy_reference() {
    // Generated by fixtures/generate.py; each row: window|order|input|expected.
    let mut cases = 0;
    for (line_number, line) in include_str!("fixtures/scipy.txt").lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let row = line_number + 1;
        let parts: Vec<_> = line.split('|').collect();
        assert_eq!(parts.len(), 4, "fixture row {row}");
        let number = |text: &str| -> usize {
            text.parse()
                .unwrap_or_else(|e| panic!("fixture row {row}: {e}"))
        };
        let values = |text: &str| -> Vec<f64> {
            text.split(',')
                .map(|x| {
                    x.parse()
                        .unwrap_or_else(|e| panic!("fixture row {row}: {e}"))
                })
                .collect()
        };
        let (window, order) = (number(parts[0]), number(parts[1]));
        let actual = SavitzkyGolay::new(window, order)
            .unwrap()
            .apply(&values(parts[2]))
            .unwrap();
        close_with(
            &actual,
            &values(parts[3]),
            format_args!(", fixture row {row}, window={window}, order={order}"),
        );
        cases += 1;
    }
    assert_eq!(cases, 18, "missing SciPy reference cases");
}

#[test]
fn impossible_sg_window_returns_allocation_error() {
    // Order 0 overflows only the window² coefficients, order 2 already the
    // window × (order + 1) design matrix.
    for order in [0, 2] {
        assert!(matches!(
            SavitzkyGolay::new(usize::MAX, order),
            Err(Error::AllocationFailure)
        ));
    }
}
