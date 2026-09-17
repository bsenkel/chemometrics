//! Analytic, synthetic NIR and NumPy/SciPy reference checks for detrending.
use chemometrics::{
    Error, baseline::Detrend, normalize::StandardNormalVariate, smooth::SavitzkyGolay,
};
use std::f64::consts::LN_2;

fn close_with(actual: &[f64], expected: &[f64], context: impl std::fmt::Display) {
    assert_eq!(actual.len(), expected.len(), "length{context}");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 + 1e-10 * b.abs(),
            "sample {i}{context}: {a} != {b}"
        );
    }
}

fn close(actual: &[f64], expected: &[f64]) {
    close_with(actual, expected, "");
}

fn detrend(order: usize, input: &[f64]) -> Vec<f64> {
    Detrend::new(order).apply(input).unwrap()
}

fn coordinate(i: usize, length: usize) -> f64 {
    if length == 1 {
        0.0
    } else {
        2.0 * i as f64 / (length - 1) as f64 - 1.0
    }
}

/// `sum(coefficients[k] * t^k)` on the scaled sample coordinate.
fn polynomial(coefficients: &[f64], length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| {
            let t = coordinate(i, length);
            coefficients
                .iter()
                .rev()
                .fold(0.0, |value, c| value * t + c)
        })
        .collect()
}

fn wavy(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| (i as f64 * 0.37).sin() + 0.3 * (i as f64 * 0.11).cos())
        .collect()
}

/// Absorbance on 1100–2500 nm at 2 nm from Gaussian bands at typical C–H and
/// O–H overtone and combination positions (centre, FWHM in nm).
fn nir_like() -> Vec<f64> {
    const BANDS: [(f64, f64, f64); 6] = [
        (0.30, 1210.0, 60.0),
        (0.55, 1450.0, 90.0),
        (0.20, 1730.0, 50.0),
        (0.45, 1940.0, 110.0),
        (0.15, 2100.0, 80.0),
        (0.25, 2310.0, 45.0),
    ];
    (0..701)
        .map(|i| {
            let wavelength = 1100.0 + 2.0 * i as f64;
            BANDS
                .iter()
                .map(|&(amplitude, centre, fwhm)| {
                    amplitude * (-4.0 * LN_2 * ((wavelength - centre) / fwhm).powi(2)).exp()
                })
                .sum()
        })
        .collect()
}

fn largest(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |a, b| a.max(b.abs()))
}

#[test]
fn removes_polynomials_up_to_the_fitted_order() {
    for order in 0..=4 {
        for length in [order + 1, order + 2, 11, 251] {
            for degree in 0..=order {
                let mut coefficients = vec![0.0; degree + 1];
                coefficients[degree] = 1.5;
                if degree > 0 {
                    coefficients[0] = -0.75;
                }
                let input = polynomial(&coefficients, length);
                let result = detrend(order, &input);
                let context = format!(" (order {order}, length {length}, degree {degree})");
                close_with(&result, &vec![0.0; length], context);
            }
        }
    }
}

#[test]
fn fitting_order_plus_one_samples_gives_zeros() {
    for order in 0..=5 {
        let input = wavy(order + 1);
        close(&detrend(order, &input), &vec![0.0; order + 1]);
    }
}

#[test]
fn adding_a_fitted_polynomial_does_not_change_the_result() {
    let input = wavy(101);
    for order in 0..=3 {
        let expected = detrend(order, &input);
        for coefficients in [
            vec![5.0],
            vec![-2.0, 0.5],
            vec![0.0, 0.0, 3.0],
            vec![1.0, -1.0, 2.0, -0.5],
        ] {
            if coefficients.len() > order + 1 {
                continue;
            }
            let baseline = polynomial(&coefficients, input.len());
            let measured: Vec<_> = input.iter().zip(&baseline).map(|(x, b)| x + b).collect();
            close_with(
                &detrend(order, &measured),
                &expected,
                format!(" {coefficients:?}"),
            );
        }
        // Detrending an already corrected spectrum changes nothing.
        close(&detrend(order, &expected), &expected);
    }
}

#[test]
fn residual_is_orthogonal_to_the_fitted_basis() {
    let input: Vec<_> = wavy(301).iter().map(|x| x + 4.0).collect();
    for order in 0..=3 {
        let residual = detrend(order, &input);
        for degree in 0..=order {
            let inner: f64 = residual
                .iter()
                .enumerate()
                .map(|(i, r)| r * coordinate(i, input.len()).powi(degree as i32))
                .sum();
            assert!(
                inner.abs() < 1e-10,
                "order {order}, degree {degree}: {inner}"
            );
        }
    }
}

#[test]
fn reversed_spectra_give_reversed_results() {
    let input = wavy(97);
    for order in 0..=3 {
        let expected: Vec<_> = detrend(order, &input).into_iter().rev().collect();
        let reversed: Vec<_> = input.iter().rev().copied().collect();
        close(&detrend(order, &reversed), &expected);
    }
}

#[test]
fn removes_baselines_from_nir_like_spectra() {
    let bands = nir_like();
    let expected = detrend(2, &bands);
    for (slope, curvature) in [(0.0, 0.0), (2e-4, 0.0), (5e-4, -1e-7), (-3e-4, 2e-7)] {
        let measured: Vec<_> = bands
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let offset = 2.0 * i as f64;
                x + 0.4 + slope * offset + curvature * offset * offset
            })
            .collect();
        close_with(
            &detrend(2, &measured),
            &expected,
            format!(" ({slope}, {curvature})"),
        );
    }
    // The bands survive: the correction is a smooth baseline, not a filter.
    let difference: Vec<_> = bands.iter().zip(&expected).map(|(x, y)| x - y).collect();
    assert!(largest(&difference) < 0.3, "{}", largest(&difference));
    // The O–H band at 1450 nm still stands above the corrected spectrum.
    assert!(expected[175] > 0.2, "{}", expected[175]);
}

#[test]
fn snv_and_detrend_pipeline() {
    let bands = nir_like();
    let sloped: Vec<_> = bands
        .iter()
        .enumerate()
        .map(|(i, x)| 1.3 * x + 0.5 + 3e-4 * 2.0 * i as f64)
        .collect();
    // SNV removes the scatter and detrending the slope that SNV leaves behind.
    // The residual keeps the scale SNV gave it, so a closing SNV compares the
    // shapes the pipeline is meant to preserve.
    let pipeline = |spectrum: &[f64]| {
        let normalized = StandardNormalVariate.apply(spectrum).unwrap();
        let corrected = Detrend::new(2).apply(&normalized).unwrap();
        StandardNormalVariate.apply(&corrected).unwrap()
    };
    close(&pipeline(&sloped), &pipeline(&bands));

    // Smoothing first is the usual order and keeps the pipeline finite.
    let smoothed = SavitzkyGolay::new(11, 2).unwrap().apply(&sloped).unwrap();
    let result = Detrend::new(2).apply(&smoothed).unwrap();
    assert!(result.iter().all(|x| x.is_finite()));
    assert!(largest(&result) > 0.1);
}

#[test]
fn reuse_across_lengths_and_buffers() {
    let correction = Detrend::new(2);
    let mut buffer = vec![0.0; 64];
    for length in [3, 17, 64] {
        let input = wavy(length);
        let expected = correction.apply(&input).unwrap();
        correction
            .apply_into(&input, &mut buffer[..length])
            .unwrap();
        close(&buffer[..length], &expected);
    }
}

#[test]
fn validation_and_buffer_preservation() {
    let mut buffer = [42.0; 4];
    assert_eq!(
        Detrend::new(2).apply(&[1.0, 2.0]),
        Err(Error::TooFewSamples {
            length: 2,
            minimum: 3
        })
    );
    assert_eq!(
        Detrend::new(0).apply(&[]),
        Err(Error::TooFewSamples {
            length: 0,
            minimum: 1
        })
    );
    assert_eq!(
        Detrend::new(usize::MAX).apply(&[1.0, 2.0, 3.0]),
        Err(Error::TooFewSamples {
            length: 3,
            minimum: usize::MAX
        })
    );
    assert_eq!(
        Detrend::new(1).apply_into(&[1.0, 2.0, 3.0], &mut buffer),
        Err(Error::OutputLengthMismatch {
            expected: 3,
            actual: 4
        })
    );
    assert_eq!(
        Detrend::new(1).apply_into(&[1.0, 2.0, f64::NAN, 4.0], &mut buffer),
        Err(Error::NonFiniteInput { index: 2 })
    );
    assert_eq!(
        Detrend::new(1).apply(&[1.0, f64::INFINITY, 3.0]),
        Err(Error::NonFiniteInput { index: 1 })
    );
    // Too few samples is reported before a length mismatch, and both before
    // non-finite input.
    assert_eq!(
        Detrend::new(5).apply_into(&[f64::NAN, 1.0], &mut buffer),
        Err(Error::TooFewSamples {
            length: 2,
            minimum: 6
        })
    );
    assert_eq!(
        Detrend::new(1).apply_into(&[f64::NAN, 1.0, 2.0], &mut buffer),
        Err(Error::OutputLengthMismatch {
            expected: 3,
            actual: 4
        })
    );
    assert_eq!(buffer, [42.0; 4], "invalid input must not write");
}

#[test]
fn extreme_values_stay_finite_or_fail() {
    let tiny = f64::from_bits(1);
    for value in [f64::MAX, -f64::MAX, f64::MIN_POSITIVE, tiny] {
        for order in 0..=2 {
            let input: Vec<_> = (0..11).map(|i| value / (i + 1) as f64).collect();
            match Detrend::new(order).apply(&input) {
                Ok(result) => assert!(result.iter().all(|x| x.is_finite())),
                Err(error) => assert_eq!(error, Error::NumericalFailure),
            }
        }
    }
    // A constant spectrum has no trend to remove, at any magnitude.
    for value in [0.0, -0.0, 7.5, f64::MAX, tiny] {
        close(&detrend(2, &vec![value; 33]), &vec![0.0; 33]);
    }
    // A large offset must not swamp the variation the input still represents.
    // Subtracting the offset is exact here (Sterbenz lemma), so the shifted
    // spectrum holds exactly what the offset spectrum carries.
    let input: Vec<_> = wavy(101).iter().map(|x| 1e9 + x).collect();
    let shifted: Vec<_> = input.iter().map(|x| x - 1e9).collect();
    close(&detrend(2, &input), &detrend(2, &shifted));
}

#[test]
fn excessive_order_is_rejected() {
    assert_eq!(
        Detrend::new(60).apply(&wavy(101)),
        Err(Error::NumericalFailure)
    );
}

#[test]
fn numpy_reference() {
    let fixture = include_str!("fixtures/scipy_detrend.txt");
    let mut cases = 0;
    for line in fixture.lines().filter(|line| !line.starts_with('#')) {
        let mut parts = line.split('|');
        let order: usize = parts.next().unwrap().parse().unwrap();
        let values = |text: &str| -> Vec<f64> {
            text.split(',').map(|v| v.parse::<f64>().unwrap()).collect()
        };
        let input = values(parts.next().unwrap());
        let expected = values(parts.next().unwrap());
        assert!(parts.next().is_none());
        let context = format!(" (order {order}, length {}, case {cases})", input.len());
        close_with(&detrend(order, &input), &expected, context);
        cases += 1;
    }
    assert!(cases >= 30, "{cases} reference cases");
}
