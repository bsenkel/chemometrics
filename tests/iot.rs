//! Analytic, brute-force, synthetic NIR and NumPy reference checks for
//! iterative optimization technology.
use chemometrics::{
    Error,
    baseline::Detrend,
    iot::{Iot, Prediction},
    smooth::SavitzkyGolay,
};
use std::f64::consts::LN_2;

/// Deterministic values in [-1, 1) from a linear congruential generator.
fn pseudo_random(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

/// Absorption band on 1100–2500 nm at 14 nm with its centre and full width
/// at half maximum in nm.
fn band(centre: f64, fwhm: f64) -> Vec<f64> {
    (0..101)
        .map(|i| {
            let wavelength = 1100.0 + 14.0 * i as f64;
            (-4.0 * LN_2 * ((wavelength - centre) / fwhm).powi(2)).exp()
        })
        .collect()
}

/// Absorbance of an active ingredient, two excipients and a lubricant, each
/// from bands (centre, FWHM, height) on a small offset.
fn nir_components() -> Vec<Vec<f64>> {
    let absorbance = |bands: &[(f64, f64, f64)]| -> Vec<f64> {
        (0..101)
            .map(|i| {
                let heights = bands.iter().map(|(c, w, h)| h * band(*c, *w)[i]);
                0.05 + heights.sum::<f64>()
            })
            .collect()
    };
    vec![
        absorbance(&[
            (1210.0, 60.0, 0.5),
            (1680.0, 50.0, 0.6),
            (2270.0, 60.0, 0.8),
        ]),
        absorbance(&[
            (1450.0, 60.0, 0.4),
            (1535.0, 50.0, 0.7),
            (1930.0, 40.0, 0.9),
            (2090.0, 60.0, 0.6),
        ]),
        absorbance(&[
            (1490.0, 90.0, 0.5),
            (1780.0, 70.0, 0.4),
            (2100.0, 80.0, 0.7),
        ]),
        absorbance(&[(1725.0, 30.0, 0.9), (2310.0, 30.0, 0.8)]),
    ]
}

/// Weighted sum of spectra.
fn blend(spectra: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    (0..spectra[0].len())
        .map(|i| spectra.iter().zip(weights).map(|(s, w)| w * s[i]).sum())
        .collect()
}

fn with_noise(spectrum: &[f64], amplitude: f64, seed: u64) -> Vec<f64> {
    let noise = pseudo_random(spectrum.len(), seed);
    spectrum
        .iter()
        .zip(noise)
        .map(|(x, n)| x + amplitude * n)
        .collect()
}

fn model(spectra: &[Vec<f64>], partial: bool) -> Result<Iot, Error> {
    let pure = spectra.concat();
    let variables = spectra[0].len();
    if partial {
        Iot::new_partial(&pure, variables)
    } else {
        Iot::new(&pure, variables)
    }
}

fn predict(spectra: &[Vec<f64>], partial: bool, mixture: &[f64]) -> Prediction {
    model(spectra, partial).unwrap().predict(mixture).unwrap()
}

fn close(actual: &[f64], expected: &[f64], tolerance: f64, context: impl std::fmt::Display) {
    assert_eq!(actual.len(), expected.len(), "length{context}");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= tolerance,
            "value {i}{context}: {a} != {b}"
        );
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The pure spectra, followed by the spectrum of zeros of the further
/// component of a partial model.
fn constituents(spectra: &[Vec<f64>], partial: bool) -> Vec<Vec<f64>> {
    let mut spectra = spectra.to_vec();
    if partial {
        spectra.push(vec![0.0; spectra[0].len()]);
    }
    spectra
}

fn squared_misfit(spectra: &[Vec<f64>], contributions: &[f64], mixture: &[f64]) -> f64 {
    let fitted = blend(spectra, contributions);
    fitted
        .iter()
        .zip(mixture)
        .map(|(f, x)| (x - f).powi(2))
        .sum()
}

/// Asserts the conditions that identify the best composition: contributions
/// lie in [0, 1] and sum to one, and the gradient of the squared misfit is the
/// same for every nonzero contribution and not smaller for the others.
fn assert_optimal(spectra: &[Vec<f64>], partial: bool, mixture: &[f64], contributions: &[f64]) {
    let spectra = constituents(spectra, partial);
    let mut contributions = contributions.to_vec();
    if partial {
        contributions.push(1.0 - contributions.iter().sum::<f64>());
    }
    assert!(
        contributions.iter().all(|c| (0.0..=1.0).contains(c)),
        "{contributions:?}"
    );
    let total: f64 = contributions.iter().sum();
    assert!((total - 1.0).abs() <= 1e-14, "{total}");
    let fitted = blend(&spectra, &contributions);
    let misfit: Vec<f64> = fitted.iter().zip(mixture).map(|(f, x)| f - x).collect();
    let gradient: Vec<f64> = spectra.iter().map(|s| dot(s, &misfit)).collect();
    let norm = |values: &[f64]| dot(values, values).sqrt();
    let size = norm(&spectra.concat());
    let tolerance = 1e-12 * size * (size + norm(mixture));
    // A share of the further component computed as one minus the others is
    // zero only up to rounding.
    let free: Vec<usize> = (0..spectra.len())
        .filter(|i| contributions[*i] > 1e-12)
        .collect();
    let level = free.iter().map(|i| gradient[*i]).sum::<f64>() / free.len() as f64;
    for (i, (g, c)) in gradient.iter().zip(&contributions).enumerate() {
        if free.contains(&i) {
            assert!(
                (g - level).abs() <= tolerance,
                "component {i}: {g} != {level}, {c}"
            );
        } else {
            assert!(*g >= level - tolerance, "component {i}: {g} < {level}");
        }
    }
}

/// Solves a small dense system by Gaussian elimination with partial pivoting.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|i, j| a[*i][c].abs().total_cmp(&a[*j][c].abs()))?;
        if a[p][c].abs() < 1e-13 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let factor = a[r][c] / a[c][c];
            let pivot_row = a[c].clone();
            for (value, pivot) in a[r][c..].iter_mut().zip(&pivot_row[c..]) {
                *value -= factor * pivot;
            }
            b[r] -= factor * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let known: f64 = (r + 1..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - known) / a[r][r];
    }
    Some(x)
}

/// The best composition found by trying every set of nonzero contributions,
/// each fitted through the optimality system of the sum of one. The normal
/// equations in this system are accurate enough for well-conditioned random
/// spectra.
fn brute_force(spectra: &[Vec<f64>], mixture: &[f64]) -> Vec<f64> {
    let count = spectra.len();
    let mut best: Option<(Vec<f64>, f64)> = None;
    for mask in 1_u32..1 << count {
        let support: Vec<usize> = (0..count).filter(|i| mask >> i & 1 == 1).collect();
        let n = support.len();
        let mut a = vec![vec![0.0; n + 1]; n + 1];
        let mut b = vec![0.0; n + 1];
        for (p, i) in support.iter().enumerate() {
            for (q, j) in support.iter().enumerate() {
                a[p][q] = 2.0 * dot(&spectra[*i], &spectra[*j]);
            }
            a[p][n] = 1.0;
            a[n][p] = 1.0;
            b[p] = 2.0 * dot(&spectra[*i], mixture);
        }
        b[n] = 1.0;
        let Some(solution) = solve_dense(a, b) else {
            continue;
        };
        if solution[..n].iter().any(|v| *v < -1e-11) {
            continue;
        }
        let mut x = vec![0.0; count];
        for (p, i) in support.iter().enumerate() {
            x[*i] = solution[p].max(0.0);
        }
        let value = squared_misfit(spectra, &x, mixture);
        if best.as_ref().is_none_or(|(_, v)| value < *v) {
            best = Some((x, value));
        }
    }
    best.unwrap().0
}

#[test]
fn exact_mixtures_are_recovered() {
    let pure = nir_components();
    // Contributions of exactly zero sit on the bound with nothing to gain
    // from releasing them, where rounding decides which side they fall on.
    for weights in [
        [0.10, 0.50, 0.39, 0.01],
        [0.25, 0.25, 0.25, 0.25],
        [0.70, 0.10, 0.15, 0.05],
        [0.40, 0.60, 0.00, 0.00],
        [0.00, 0.30, 0.00, 0.70],
    ] {
        let prediction = predict(&pure, false, &blend(&pure, &weights));
        close(&prediction.contributions, &weights, 1e-12, "");
        let total: f64 = prediction.contributions.iter().sum();
        assert!((total - 1.0).abs() <= 4.0 * f64::EPSILON, "{total}");
        assert!(prediction.residual < 1e-25, "{}", prediction.residual);
    }
    // The further component of a partial model takes the remaining share.
    let weights = [0.10, 0.50, 0.30];
    let prediction = predict(&pure[..3], true, &blend(&pure, &weights));
    close(&prediction.contributions, &weights, 1e-12, "");
    assert!(prediction.residual < 1e-25, "{}", prediction.residual);
}

#[test]
fn absent_components_report_positive_zero() {
    // A zero from the back substitution can carry a negative sign, which
    // would print as -0.0.
    let pure = [
        vec![0.05, 0.60, 0.10, 0.05, 0.40, 0.05, 0.10, 0.70],
        vec![0.30, 0.10, 0.50, 0.70, 0.10, 0.60, 0.20, 0.10],
        vec![0.05, 0.05, 0.10, 0.05, 0.80, 0.05, 0.60, 0.10],
    ];
    let mut negative = 0;
    for partial in [false, true] {
        for weights in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ] {
            let contributions = predict(&pure, partial, &blend(&pure, &weights)).contributions;
            negative += contributions
                .iter()
                .filter(|c| c.is_sign_negative())
                .count();
            assert_eq!(
                format!("{:.1}", contributions[2]),
                format!("{:.1}", weights[2])
            );
        }
    }
    assert_eq!(negative, 0);
}

#[test]
fn two_components_are_clamped_to_the_segment() {
    let a = [0.2, 0.9, 0.4, 0.1];
    let b = [0.7, 0.1, 0.3, 0.6];
    let pair = Iot::new(&[a, b].concat(), 4).unwrap();
    let single = Iot::new_partial(&a, 4).unwrap();
    let difference: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    for weight in [-0.5, 0.0, 0.3, 1.0, 1.7] {
        let mixture: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(x, y)| weight * x + (1.0 - weight) * y + 0.01)
            .collect();
        // The share of `a` is the position of the mixture's projection on
        // the line from `b` to `a`, kept within the segment.
        let from_b: Vec<f64> = mixture.iter().zip(&b).map(|(m, y)| m - y).collect();
        let along = dot(&from_b, &difference) / dot(&difference, &difference);
        let prediction = pair.predict(&mixture).unwrap();
        let expected = along.clamp(0.0, 1.0);
        close(
            &prediction.contributions,
            &[expected, 1.0 - expected],
            1e-14,
            format!(" at {weight}"),
        );
        // Alone, `a` is scaled towards the mixture, at most to its full size.
        let scale = dot(&mixture, &a) / dot(&a, &a);
        let prediction = single.predict(&mixture).unwrap();
        close(
            &prediction.contributions,
            &[scale.clamp(0.0, 1.0)],
            1e-14,
            format!(" at {weight}"),
        );
    }
}

#[test]
fn matches_the_best_of_every_set_of_nonzero_contributions() {
    let (mut cases, mut bounded, mut reference_zero) = (0, 0, 0);
    // Rarely a component fixed at zero on the way has to be released again;
    // this many seeds include such cases.
    for seed in 1..=1500_u64 {
        let components = 2 + (seed % 5) as usize;
        let variables = components + (seed % 7) as usize;
        let partial = seed % 3 == 0;
        let draws = pseudo_random((components + 1) * variables + components, seed);
        let spectra: Vec<Vec<f64>> = draws[..components * variables]
            .chunks_exact(variables)
            .map(|row| row.iter().map(|x| 0.6 + 0.5 * x).collect())
            .collect();
        let extra = &draws[components * variables..(components + 1) * variables];
        let raw_weights = &draws[(components + 1) * variables..];
        // Mixtures inside the allowed compositions with noise, outside them,
        // and unrelated spectra.
        let mixture: Vec<f64> = match seed % 4 {
            0 => extra.iter().map(|x| 0.6 + 0.5 * x).collect(),
            kind => {
                let mut weights: Vec<f64> = raw_weights
                    .iter()
                    .map(|w| if kind == 1 { w.abs() } else { 0.7 * w + 0.2 })
                    .collect();
                let total: f64 = weights.iter().sum();
                weights.iter_mut().for_each(|w| *w /= total);
                let mixture = blend(&spectra, &weights);
                let noise = if kind == 3 { 0.02 } else { 0.0 };
                mixture
                    .iter()
                    .zip(extra)
                    .map(|(m, e)| m + noise * e)
                    .collect()
            }
        };
        // Every rotation makes another component the reference.
        for rotation in 0..components {
            let mut rotated = spectra.clone();
            rotated.rotate_left(rotation);
            let Ok(model) = model(&rotated, partial) else {
                continue;
            };
            let contributions = model.predict(&mixture).unwrap().contributions;
            let expected = brute_force(&constituents(&rotated, partial), &mixture);
            close(
                &contributions,
                &expected[..components],
                1e-9,
                format!(" for seed {seed}, rotation {rotation}"),
            );
            cases += 1;
            bounded += usize::from(contributions.contains(&0.0));
            reference_zero += usize::from(!partial && contributions[0] == 0.0);
        }
    }
    assert!(
        cases > 4000 && bounded > 2000 && reference_zero > 500,
        "{cases} cases, {bounded} with a bound, {reference_zero} with the reference at zero"
    );
}

#[test]
fn trace_components_end_at_the_bound() {
    let pure = nir_components();
    let mut at_bound = 0;
    // The lubricant level below zero stands for noise or drift that pushes
    // the unconstrained fit there.
    for (seed, lubricant) in [(1, 0.01), (2, 0.0), (3, -0.005), (4, -0.02)] {
        let weights = [0.10, 0.50, 0.40 - lubricant, lubricant];
        let mixture = with_noise(&blend(&pure, &weights), 1e-3, seed);
        let prediction = predict(&pure, false, &mixture);
        assert_optimal(&pure, false, &mixture, &prediction.contributions);
        at_bound += usize::from(prediction.contributions[3] == 0.0);
        // Without its spectrum the lubricant belongs to the further component.
        let prediction = predict(&pure[..3], true, &mixture);
        assert_optimal(&pure[..3], true, &mixture, &prediction.contributions);
    }
    assert!(at_bound >= 2, "{at_bound}");
}

#[test]
fn order_of_the_pure_spectra_does_not_matter() {
    let pure = nir_components();
    // The lubricant ends at zero, so in one rotation of the full model the
    // reference is at the bound.
    let mixture = with_noise(&blend(&pure, &[0.10, 0.52, 0.40, -0.02]), 1e-4, 5);
    for partial in [false, true] {
        let count = if partial { 3 } else { 4 };
        let expected = predict(&pure[..count], partial, &mixture).contributions;
        if !partial {
            assert_eq!(expected[3], 0.0);
        }
        for rotation in 1..count {
            let mut rotated = pure[..count].to_vec();
            rotated.rotate_left(rotation);
            let mut contributions = predict(&rotated, partial, &mixture).contributions;
            contributions.rotate_right(rotation);
            close(
                &contributions,
                &expected,
                1e-12,
                format!(" for rotation {rotation}"),
            );
        }
    }
}

#[test]
fn shared_scaling_and_offset_change_nothing() {
    let pure = nir_components();
    let mixture = with_noise(&blend(&pure, &[0.10, 0.52, 0.40, -0.02]), 1e-4, 6);
    for partial in [false, true] {
        let expected = predict(&pure, partial, &mixture);
        for factor in [1e-3, 7.0, 1e3] {
            let scaled: Vec<Vec<f64>> = pure
                .iter()
                .map(|s| s.iter().map(|x| factor * x).collect())
                .collect();
            let mixture: Vec<f64> = mixture.iter().map(|x| factor * x).collect();
            let prediction = predict(&scaled, partial, &mixture);
            close(
                &prediction.contributions,
                &expected.contributions,
                1e-12,
                format!(" at {factor}"),
            );
            let residual = expected.residual * factor * factor;
            assert!((prediction.residual - residual).abs() <= 1e-10 * residual);
        }
    }
    // Contributions that sum to one carry a shared offset along; a partial
    // model's spectrum of zeros does not, so it has no such invariance.
    let expected = predict(&pure, false, &mixture);
    for offset in [0.5, 1e3] {
        let shifted: Vec<Vec<f64>> = pure
            .iter()
            .map(|s| s.iter().map(|x| x + offset).collect())
            .collect();
        let mixture: Vec<f64> = mixture.iter().map(|x| x + offset).collect();
        let prediction = predict(&shifted, false, &mixture);
        // Adding the offset rounds the inputs by about EPSILON times it.
        let tolerance = 1e-12 * (1.0 + offset);
        close(
            &prediction.contributions,
            &expected.contributions,
            tolerance,
            format!(" at {offset}"),
        );
        assert!((prediction.residual - expected.residual).abs() <= tolerance * expected.residual);
    }
}

#[test]
fn linear_preprocessing_keeps_the_contributions() {
    let pure = nir_components();
    let weights = [0.15, 0.45, 0.38, 0.02];
    let mixture = blend(&pure, &weights);
    let derivative = SavitzkyGolay::new_derivative(7, 2, 2, 14.0).unwrap();
    let detrend = Detrend::new(2);
    // The pure spectra followed by the mixture, after each preprocessing.
    let spectra = || pure.iter().chain([&mixture]);
    for mut treated in [
        spectra()
            .map(|s| derivative.apply(s).unwrap())
            .collect::<Vec<_>>(),
        spectra().map(|s| detrend.apply(s).unwrap()).collect(),
    ] {
        let mixture = treated.pop().unwrap();
        let prediction = predict(&treated, false, &mixture);
        close(&prediction.contributions, &weights, 1e-10, "");
    }
}

#[test]
fn partial_model_equals_a_full_model_with_a_spectrum_of_zeros() {
    let pure = nir_components();
    let mut padded = pure[..3].to_vec();
    padded.push(vec![0.0; 101]);
    let partial = model(&pure[..3], true).unwrap();
    let full = model(&padded, false).unwrap();
    for (seed, weights) in [(7, [0.1, 0.5, 0.3, 0.1]), (8, [0.2, 0.5, 0.35, -0.05])] {
        let mixture = with_noise(&blend(&pure, &weights), 1e-4, seed);
        let expected = full.predict(&mixture).unwrap();
        let prediction = partial.predict(&mixture).unwrap();
        close(
            &prediction.contributions,
            &expected.contributions[..3],
            1e-12,
            "",
        );
        let rest = 1.0 - prediction.contributions.iter().sum::<f64>();
        assert!((rest - expected.contributions[3]).abs() <= 1e-12);
        assert!((prediction.residual - expected.residual).abs() <= 1e-10 * expected.residual);
    }
}

#[test]
fn residual_is_the_misfit_of_the_contributions() {
    let pure = nir_components();
    for partial in [false, true] {
        let mixture = with_noise(&blend(&pure, &[0.10, 0.50, 0.39, 0.01]), 1e-3, 9);
        let count = if partial { 3 } else { 4 };
        let prediction = predict(&pure[..count], partial, &mixture);
        let misfit = squared_misfit(&pure[..count], &prediction.contributions, &mixture);
        assert!((prediction.residual - misfit).abs() <= 1e-12 * misfit);
        assert!(prediction.residual > 0.0);
    }
}

#[test]
fn rejects_invalid_inputs_in_order() {
    // The data contain a non-finite value, so each rejection of them shows
    // that its check comes before the one for non-finite values.
    let mut data = pseudo_random(12, 10);
    data[7] = f64::NAN;
    for (variables, length) in [(5, 12), (0, 12)] {
        let expected = Error::InvalidDataShape { length, variables };
        assert_eq!(Iot::new(&data, variables).unwrap_err(), expected);
        assert_eq!(Iot::new_partial(&data, variables).unwrap_err(), expected);
    }
    assert_eq!(
        Iot::new(&[], 3).unwrap_err(),
        Error::InvalidDataShape {
            length: 0,
            variables: 3
        }
    );
    assert_eq!(
        Iot::new(&data[..4], 4).unwrap_err(),
        Error::TooFewSpectra {
            count: 1,
            minimum: 2
        }
    );
    // Four spectra of two variables: `new` supports three, `new_partial` two.
    assert_eq!(
        Iot::new(&data[..8], 2).unwrap_err(),
        Error::InvalidComponentCount {
            requested: 4,
            maximum: 3
        }
    );
    assert_eq!(
        Iot::new_partial(&data[..6], 2).unwrap_err(),
        Error::InvalidComponentCount {
            requested: 3,
            maximum: 2
        }
    );
    assert_eq!(
        Iot::new(&data, 4).unwrap_err(),
        Error::NonFiniteInput { index: 7 }
    );
    data[7] = f64::INFINITY;
    assert_eq!(
        Iot::new_partial(&data, 4).unwrap_err(),
        Error::NonFiniteInput { index: 7 }
    );
    // Once every value is finite, the largest counts are accepted: three
    // spectra of two variables, and a single spectrum without a partner.
    data[7] = 0.5;
    assert!(Iot::new(&data[..6], 2).is_ok());
    assert!(Iot::new_partial(&data[..4], 2).is_ok());
    assert!(Iot::new_partial(&data[..4], 4).is_ok());

    // The spectrum length comes before non-finite values.
    let model = Iot::new(&data, 4).unwrap();
    assert_eq!(
        model.predict(&[f64::NAN; 5]).unwrap_err(),
        Error::InvalidSpectrumLength {
            expected: 4,
            actual: 5
        }
    );
    assert_eq!(
        model
            .predict(&[0.1, 0.2, f64::NEG_INFINITY, 0.3])
            .unwrap_err(),
        Error::NonFiniteInput { index: 2 }
    );
}

#[test]
fn rejects_pure_spectra_that_are_mixtures_of_the_preceding_ones() {
    let a = [1.0, 2.0, 3.0, 0.5];
    let b = [0.5, 0.1, 0.9, 1.5];
    let c = [0.2, 1.0, 0.0, 0.7];
    let mixture: Vec<f64> = a.iter().zip(&b).map(|(x, y)| 0.6 * x + 0.4 * y).collect();
    for (spectra, index) in [
        (vec![a.to_vec(), b.to_vec(), a.to_vec()], 2),
        (vec![a.to_vec(), a.to_vec(), b.to_vec()], 1),
        (vec![a.to_vec(), b.to_vec(), mixture.clone(), c.to_vec()], 2),
    ] {
        for partial in [false, true] {
            assert_eq!(
                model(&spectra, partial).unwrap_err(),
                Error::DependentSpectra { index },
                "{partial}"
            );
        }
    }
    // Twice a spectrum is no mixture, but it is a multiple, which a partial
    // model cannot tell apart from the spectrum plus the further component.
    let double: Vec<f64> = a.iter().map(|x| 2.0 * x).collect();
    assert!(model(&[a.to_vec(), double.clone()], false).is_ok());
    assert_eq!(
        model(&[a.to_vec(), double], true).unwrap_err(),
        Error::DependentSpectra { index: 1 }
    );
    // A spectrum of zeros is a component of its own, except in a partial
    // model, where it duplicates the further component.
    assert!(model(&[vec![0.0; 4], a.to_vec()], false).is_ok());
    assert_eq!(
        model(&[vec![0.0; 4], a.to_vec()], true).unwrap_err(),
        Error::DependentSpectra { index: 0 }
    );
    // Spectra that differ only by a straight baseline are equal after
    // detrending.
    let pure = nir_components();
    let sloped: Vec<f64> = pure[0]
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.2 + 0.003 * i as f64)
        .collect();
    let detrend = Detrend::new(1);
    let treated: Vec<Vec<f64>> = [&pure[0], &pure[1], &sloped]
        .iter()
        .map(|s| detrend.apply(s).unwrap())
        .collect();
    assert_eq!(
        model(&treated, false).unwrap_err(),
        Error::DependentSpectra { index: 2 }
    );
}

#[test]
fn spectra_of_very_different_size_or_close_shape() {
    let pure = nir_components();
    let weights = [0.15, 0.45, 0.38, 0.02];
    let sizes: Vec<Vec<f64>> = pure
        .iter()
        .zip([1.0, 1e-4, 1e4, 1.0])
        .map(|(s, factor)| s.iter().map(|x| factor * x).collect())
        .collect();
    let prediction = predict(&sizes, false, &blend(&sizes, &weights));
    close(&prediction.contributions, &weights, 1e-10, "");
    // Two spectra that differ by a millionth: the contributions are still
    // recovered, with the precision their difference allows.
    let mut close_pair = pure[..3].to_vec();
    close_pair[2] = pure[0]
        .iter()
        .zip(&pure[2])
        .map(|(x, y)| x + 1e-6 * y)
        .collect();
    let weights = [0.3, 0.5, 0.2];
    let prediction = predict(&close_pair, false, &blend(&close_pair, &weights));
    close(&prediction.contributions, &weights, 1e-8, "");
}

#[test]
fn handles_extreme_magnitudes() {
    let pure = nir_components();
    let mixture = with_noise(&blend(&pure, &[0.10, 0.52, 0.40, -0.02]), 1e-4, 11);
    let scaled =
        |values: &[f64], factor: f64| -> Vec<f64> { values.iter().map(|x| factor * x).collect() };
    for partial in [false, true] {
        let expected = predict(&pure, partial, &mixture).contributions;
        for factor in [1e-150, 1e150] {
            let spectra: Vec<Vec<f64>> = pure.iter().map(|s| scaled(s, factor)).collect();
            let prediction = predict(&spectra, partial, &scaled(&mixture, factor));
            close(
                &prediction.contributions,
                &expected,
                1e-12,
                format!(" at {factor}"),
            );
        }
        // Beyond the supported range the squares no longer fit into `f64`.
        for factor in [1e-170, 1e160] {
            let spectra: Vec<Vec<f64>> = pure.iter().map(|s| scaled(s, factor)).collect();
            assert_eq!(
                model(&spectra, partial).unwrap_err(),
                Error::NumericalFailure,
                "{factor}"
            );
        }
        let model = model(&pure, partial).unwrap();
        assert_eq!(
            model.predict(&scaled(&mixture, 1e200)).unwrap_err(),
            Error::NumericalFailure
        );
        // A mixture of almost nothing is closest to the smallest composition
        // the components allow.
        let faint = model.predict(&scaled(&mixture, 1e-200)).unwrap();
        assert_optimal(&pure, partial, &vec![0.0; 101], &faint.contributions);
    }
}

#[test]
fn model_is_reusable_and_debug_shows_the_shape() {
    let pure = nir_components();
    let model = model(&pure, false).unwrap();
    assert_eq!((model.components(), model.variables()), (4, 101));
    let mixtures: Vec<Vec<f64>> = (0..3)
        .map(|seed| with_noise(&blend(&pure, &[0.1, 0.5, 0.39, 0.01]), 1e-3, seed))
        .collect();
    let first: Vec<Prediction> = mixtures.iter().map(|m| model.predict(m).unwrap()).collect();
    let copy = model.clone();
    for (mixture, expected) in mixtures.iter().zip(&first) {
        assert_eq!(&copy.predict(mixture).unwrap(), expected);
        assert_eq!(&model.predict(mixture).unwrap(), expected);
    }
    let text = format!("{model:?}");
    assert!(
        text.contains("components: 4") && text.contains("variables: 101"),
        "{text}"
    );
    assert!(text.len() < 100, "{} characters", text.len());
}

#[test]
fn numpy_reference() {
    let fixture = include_str!("fixtures/numpy_iot.txt");
    let values =
        |field: &str| -> Vec<f64> { field.split(',').map(|v| v.parse().unwrap()).collect() };
    let mut current: Option<(Iot, Vec<f64>, f64)> = None;
    let (mut cases, mut predictions) = (0, 0);
    for line in fixture.lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<&str> = line.split('|').collect();
        match fields.as_slice() {
            ["case", partial, components, variables, pure] => {
                let (components, variables): (usize, usize) =
                    (components.parse().unwrap(), variables.parse().unwrap());
                let pure = values(pure);
                assert_eq!(pure.len(), components * variables);
                let model = if *partial == "1" {
                    Iot::new_partial(&pure, variables)
                } else {
                    Iot::new(&pure, variables)
                }
                .unwrap();
                assert_eq!(model.components(), components);
                // The tolerance comes from the data alone: rounding the inputs
                // costs digits in proportion to their magnitude over the
                // differences between the spectra.
                let first = pure[..variables].to_vec();
                let magnitude = pure.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                let spread = pure
                    .chunks_exact(variables)
                    .flat_map(|s| s.iter().zip(&first).map(|(x, y)| (x - y).abs()))
                    .fold(magnitude * f64::EPSILON, f64::max);
                let spread = if *partial == "1" { magnitude } else { spread };
                let precision = 1e-10 + 1e3 * f64::EPSILON * magnitude / spread;
                current = Some((model, first, precision));
                cases += 1;
            }
            ["predict", mixture, contributions, residual] => {
                let (model, first, precision) = current.as_ref().expect("case header");
                let mixture = values(mixture);
                let prediction = model.predict(&mixture).unwrap();
                let context = format!(" in case {cases}");
                close(
                    &prediction.contributions,
                    &values(contributions),
                    *precision,
                    &context,
                );
                // The residual is part of the squared distance of the mixture
                // from the first pure spectrum.
                let distance: f64 = mixture
                    .iter()
                    .zip(first)
                    .map(|(x, y)| (x - y).powi(2))
                    .sum();
                let residual = values(residual)[0];
                assert!(
                    (prediction.residual - residual).abs() <= precision * distance.max(residual),
                    "residual{context}: {} != {residual}",
                    prediction.residual
                );
                predictions += 1;
            }
            other => panic!("unknown row {other:?}"),
        }
    }
    assert!(
        cases >= 14 && predictions >= 56,
        "{cases} cases, {predictions} predictions"
    );
}
