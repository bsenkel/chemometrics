//! Analytic, synthetic NIR and SciPy reference checks for SNV.
use chemometrics::{Error, normalize::StandardNormalVariate};
use std::f64::consts::{FRAC_1_SQRT_2, LN_2};

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 + 1e-10 * b.abs(),
            "sample {i}: {a} != {b}"
        );
    }
}

fn snv(input: &[f64]) -> Vec<f64> {
    StandardNormalVariate.apply(input).unwrap()
}

fn mean_and_sample_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, variance.sqrt())
}

fn largest_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Absorbance on 1100–2500 nm at 2 nm from Gaussian bands at typical C–H and
/// O–H overtone and combination positions (centre, FWHM in nm).
fn nir_like(amplitudes: [f64; 6]) -> Vec<f64> {
    const BANDS: [(f64, f64); 6] = [
        (1210.0, 60.0),
        (1450.0, 90.0),
        (1730.0, 50.0),
        (1940.0, 110.0),
        (2100.0, 80.0),
        (2310.0, 45.0),
    ];
    (0..701)
        .map(|i| {
            let wavelength = 1100.0 + 2.0 * i as f64;
            BANDS
                .iter()
                .zip(amplitudes)
                .map(|(&(centre, fwhm), amplitude)| {
                    amplitude * (-4.0 * LN_2 * ((wavelength - centre) / fwhm).powi(2)).exp()
                })
                .sum()
        })
        .collect()
}

#[test]
fn exact_small_cases() {
    assert_eq!(snv(&[1.0, 2.0, 3.0]), vec![-1.0, 0.0, 1.0]);
    close(&snv(&[4.0, 9.0]), &[-FRAC_1_SQRT_2, FRAC_1_SQRT_2]);
    close(&snv(&[9.0, 4.0]), &[FRAC_1_SQRT_2, -FRAC_1_SQRT_2]);
}

#[test]
fn unit_moments_affine_invariance_and_reversal() {
    let input: Vec<_> = (0..57)
        .map(|i| (i as f64 * 0.37).sin() + 0.01 * i as f64)
        .collect();
    let expected = snv(&input);
    let (mean, sd) = mean_and_sample_sd(&expected);
    assert!(
        mean.abs() < 1e-12 && (sd - 1.0).abs() < 1e-12,
        "{mean}, {sd}"
    );
    for (a, b) in [(2.0, 0.0), (0.001, 5.0), (1e6, -3e6), (7.5, 1e-3)] {
        let scaled: Vec<_> = input.iter().map(|x| a * x + b).collect();
        close(&snv(&scaled), &expected);
        let flipped: Vec<_> = input.iter().map(|x| -a * x + b).collect();
        let negated: Vec<_> = expected.iter().map(|x| -x).collect();
        close(&snv(&flipped), &negated);
    }
    let reversed: Vec<_> = input.iter().rev().copied().collect();
    let expected_reversed: Vec<_> = expected.iter().rev().copied().collect();
    close(&snv(&reversed), &expected_reversed);
}

#[test]
fn removes_multiplicative_scatter_and_offset_from_nir_like_spectra() {
    let sample = nir_like([0.30, 0.55, 0.20, 0.45, 0.15, 0.25]);
    let expected = snv(&sample);
    for (a, b) in [(0.6, 0.1), (1.0, 0.8), (1.6, -0.05)] {
        let measured: Vec<_> = sample.iter().map(|x| a * x + b).collect();
        close(&snv(&measured), &expected);
    }
    // A weaker O–H band is chemistry, not scatter, and must survive SNV.
    let drier = snv(&nir_like([0.30, 0.35, 0.20, 0.45, 0.15, 0.25]));
    assert!(largest_difference(&expected, &drier) > 0.1);
}

#[test]
fn sloping_baseline_is_not_removed() {
    let sample = nir_like([0.30, 0.55, 0.20, 0.45, 0.15, 0.25]);
    let sloped: Vec<_> = sample
        .iter()
        .enumerate()
        .map(|(i, x)| x + 2e-4 * 2.0 * i as f64)
        .collect();
    assert!(largest_difference(&snv(&sample), &snv(&sloped)) > 0.1);
}

#[test]
fn constant_spectra_yield_zeros() {
    let tiny = f64::from_bits(1);
    for value in [
        0.0,
        -0.0,
        -3.5,
        2.0,
        f64::MAX,
        -f64::MAX,
        f64::MIN_POSITIVE,
        tiny,
    ] {
        for length in [2, 3, 701] {
            let mut buffer = vec![42.0; length];
            StandardNormalVariate
                .apply_into(&vec![value; length], &mut buffer)
                .unwrap();
            assert!(buffer.iter().all(|&x| x == 0.0), "{value} x {length}");
        }
    }
    assert_eq!(snv(&[0.0, -0.0]), vec![0.0, 0.0]);
}

#[test]
fn extreme_values_stay_finite_and_correct() {
    let max = f64::MAX;
    close(&snv(&[-max, max]), &[-FRAC_1_SQRT_2, FRAC_1_SQRT_2]);
    close(&snv(&[max, max / 2.0, 0.0]), &[1.0, 0.0, -1.0]);
    let tiny = f64::from_bits(1);
    close(&snv(&[0.0, tiny]), &[-FRAC_1_SQRT_2, FRAC_1_SQRT_2]);
    close(&snv(&[tiny, 2.0 * tiny, 3.0 * tiny]), &[-1.0, 0.0, 1.0]);
    // Subtracting 3 is exact here (Sterbenz lemma), so a tiny variation on a
    // large offset must normalize like the variation alone.
    let offset: Vec<_> = (0..101)
        .map(|i| 3.0 + 1e-9 * (i as f64 * 0.7).sin())
        .collect();
    let variation: Vec<_> = offset.iter().map(|x| x - 3.0).collect();
    close(&snv(&offset), &snv(&variation));
}

#[test]
fn ulp_spaced_samples_on_large_offsets_are_exact() {
    let sd = (5.0_f64 / 3.0).sqrt();
    let expected: Vec<_> = [-1.5, -0.5, 0.5, 1.5].iter().map(|x| x / sd).collect();
    // Each step is the spacing of adjacent doubles at the offset; 2^26 * EPSILON
    // is that spacing at 1e8.
    for (offset, step) in [
        (1.0, f64::EPSILON),
        (-1.0, f64::EPSILON),
        (1e8, 67_108_864.0 * f64::EPSILON),
    ] {
        let input: Vec<_> = (0..4).map(|k| offset + k as f64 * step).collect();
        close(&snv(&input), &expected);
    }
}

#[test]
fn validation_order_and_buffer_preservation() {
    for input in [&[][..], &[1.0][..]] {
        let expected = Err(Error::TooFewSamples {
            length: input.len(),
            minimum: 2,
        });
        assert_eq!(StandardNormalVariate.apply(input), expected);
        let mut buffer = vec![42.0; input.len()];
        assert_eq!(
            StandardNormalVariate.apply_into(input, &mut buffer),
            expected.map(|_: Vec<f64>| ())
        );
        assert!(buffer.iter().all(|&x| x == 42.0));
    }
    // Length is checked before output length and finiteness.
    let mut buffer = [42.0; 3];
    assert_eq!(
        StandardNormalVariate.apply_into(&[f64::NAN], &mut buffer),
        Err(Error::TooFewSamples {
            length: 1,
            minimum: 2
        })
    );
    // Output length is checked before finiteness.
    assert_eq!(
        StandardNormalVariate.apply_into(&[f64::NAN, 1.0, 2.0, 3.0], &mut buffer),
        Err(Error::OutputLengthMismatch {
            expected: 4,
            actual: 3
        })
    );
    assert_eq!(buffer, [42.0; 3]);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let input = [0.0, bad, f64::NAN, 1.0];
        let mut buffer = [42.0; 4];
        assert_eq!(
            StandardNormalVariate.apply_into(&input, &mut buffer),
            Err(Error::NonFiniteInput { index: 1 })
        );
        assert_eq!(
            StandardNormalVariate.apply(&input),
            Err(Error::NonFiniteInput { index: 1 })
        );
        assert_eq!(buffer, [42.0; 4]);
    }
}

#[test]
fn reuse_overwrites_the_buffer_and_matches_apply() {
    let transform = StandardNormalVariate;
    let mut buffer = vec![42.0; 701];
    let spectra = [
        nir_like([0.30, 0.55, 0.20, 0.45, 0.15, 0.25]),
        vec![0.4; 701],
        nir_like([0.05, 0.10, 0.60, 0.20, 0.50, 0.10]),
    ];
    for spectrum in &spectra {
        transform.apply_into(spectrum, &mut buffer).unwrap();
        assert_eq!(buffer, transform.apply(spectrum).unwrap());
    }
}

#[test]
fn scipy_snv_reference() {
    let mut cases = 0;
    for (line_number, line) in include_str!("fixtures/scipy_snv.txt").lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<_> = line.split('|').collect();
        assert_eq!(parts.len(), 2, "fixture row {}", line_number + 1);
        let parse = |text: &str| {
            text.split(',')
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let input = parse(parts[0]);
        let actual = snv(&input);
        let expected = parse(parts[1]);
        assert_eq!(actual.len(), expected.len());
        for (index, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 + 1e-10 * b.abs(),
                "fixture row {}, length {}, sample {index}: {a} != {b}",
                line_number + 1,
                input.len()
            );
        }
        cases += 1;
    }
    assert_eq!(cases, 16, "missing SNV reference cases");
}
