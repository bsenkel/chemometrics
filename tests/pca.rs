//! Analytic, synthetic NIR and NumPy reference checks for principal components.
#![cfg(feature = "pca")]
use chemometrics::{Error, pca::Pca};
use std::f64::consts::LN_2;

fn largest(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |a, b| a.max(b.abs()))
}

/// Asserts each value lies within `floor` plus 1e-10 of its own magnitude.
///
/// The floor carries the scale of the quantity, not a fixed absolute value,
/// so spectra of any magnitude are checked equally strictly.
fn close_with(actual: &[f64], expected: &[f64], floor: f64, context: impl std::fmt::Display) {
    assert_eq!(actual.len(), expected.len(), "length{context}");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= floor + 1e-10 * b.abs(),
            "value {i}{context}: {a} != {b}"
        );
    }
}

/// Compares a quantity whose scale is its own largest value.
fn close(actual: &[f64], expected: &[f64]) {
    close_scaled(actual, expected, largest(expected));
}

/// Compares part of a quantity, such as the scores of one sample, against the
/// scale of the whole.
fn close_scaled(actual: &[f64], expected: &[f64], scale: f64) {
    close_with(actual, expected, 1e-10 * scale, "");
}

type Fields<'a> = std::str::Split<'a, char>;

fn values(parts: &mut Fields<'_>) -> Vec<f64> {
    parts
        .next()
        .unwrap()
        .split(',')
        .map(|v| v.parse::<f64>().unwrap())
        .collect()
}

fn number(parts: &mut Fields<'_>) -> usize {
    parts.next().unwrap().parse().unwrap()
}

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

/// Absorbance of one Gaussian band, 1100–2500 nm at 14 nm.
fn band(centre: f64, fwhm: f64) -> Vec<f64> {
    (0..101)
        .map(|i| {
            let wavelength = 1100.0 + 14.0 * i as f64;
            (-4.0 * LN_2 * ((wavelength - centre) / fwhm).powi(2)).exp()
        })
        .collect()
}

/// NIR-like mixtures: `samples` spectra of three bands with varying
/// concentrations, multiplicative scatter and an offset.
fn mixtures(samples: usize) -> Vec<f64> {
    let profiles = [band(1210.0, 60.0), band(1450.0, 90.0), band(1940.0, 110.0)];
    let draws = pseudo_random(samples * 5, 20260919);
    let mut data = Vec::with_capacity(samples * 101);
    for (i, draw) in draws.chunks_exact(5).enumerate() {
        let scatter = 1.0 + 0.3 * draw[3];
        let offset = 0.4 + 0.2 * draw[4];
        for j in 0..101 {
            let absorbance: f64 = profiles
                .iter()
                .zip(draw)
                .map(|(profile, concentration)| (0.5 + 0.4 * concentration) * profile[j])
                .sum();
            data.push(scatter * absorbance + offset + 1e-5 * ((i * 101 + j) as f64).sin());
        }
    }
    data
}

fn variance(values: impl Iterator<Item = f64> + Clone, count: usize) -> f64 {
    let mean = values.clone().sum::<f64>() / count as f64;
    values.map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64
}

#[test]
fn loadings_are_orthonormal() {
    let data = mixtures(12);
    let model = Pca::fit(&data, 101, 3).unwrap();
    for a in 0..model.components() {
        for b in 0..model.components() {
            let product: f64 = model
                .loading(a)
                .unwrap()
                .iter()
                .zip(model.loading(b).unwrap())
                .map(|(p, q)| p * q)
                .sum();
            let expected = if a == b { 1.0 } else { 0.0 };
            assert!((product - expected).abs() < 1e-12, "({a}, {b}): {product}");
        }
    }
}

#[test]
fn scores_carry_the_component_variance() {
    let data = mixtures(16);
    let model = Pca::fit(&data, 101, 3).unwrap();
    let column = |a: usize| model.scores().iter().skip(a).step_by(3).copied();
    for a in 0..3 {
        let mean: f64 = column(a).sum::<f64>() / 16.0;
        assert!(mean.abs() < 1e-12, "component {a} mean {mean}");
        let actual = variance(column(a), 16);
        let expected = model.eigenvalues()[a];
        assert!(
            (actual - expected).abs() <= 1e-12 * expected,
            "component {a}: {actual} != {expected}"
        );
        for b in (a + 1)..3 {
            let covariance: f64 = column(a).zip(column(b)).map(|(s, t)| s * t).sum::<f64>() / 15.0;
            assert!(
                covariance.abs() < 1e-12 * expected,
                "({a}, {b}) {covariance}"
            );
        }
    }
    // Eigenvalues are nonincreasing and sum to at most the total variance.
    assert!(model.eigenvalues().windows(2).all(|w| w[0] >= w[1]));
    assert!(model.eigenvalues().iter().sum::<f64>() <= model.total_variance() * (1.0 + 1e-12));
}

#[test]
fn training_q_sums_to_the_discarded_variance() {
    // Σ Q over the training samples is the squared norm of the discarded part
    // of the centred data, (n − 1) times the sum of the discarded eigenvalues.
    // The data has rank six, so every count here discards real variation.
    let data = mixtures(14);
    for components in [1, 2, 3, 5] {
        let model = Pca::fit(&data, 101, components).unwrap();
        let actual: f64 = data
            .chunks_exact(101)
            .map(|sample| model.project(sample).unwrap().diagnostics.q_residual)
            .sum();
        let discarded: f64 = model.all_eigenvalues()[components..].iter().sum();
        let expected = 13.0 * discarded;
        assert!(
            (actual - expected).abs() <= 1e-9 * expected,
            "{components} components: {actual} != {expected}"
        );
        let rest = model.total_variance() - model.eigenvalues().iter().sum::<f64>();
        assert!(
            (discarded - rest).abs() <= 1e-9 * model.total_variance(),
            "{components} components: {discarded} != {rest}"
        );
    }
}

#[test]
fn all_eigenvalues_cover_the_total_variance() {
    // Tall and wide data: at most min(samples − 1, variables) directions vary.
    for (samples, variables, components) in [(20, 5, 2), (6, 30, 1), (6, 30, 5)] {
        let data = pseudo_random(samples * variables, 5);
        let model = Pca::fit(&data, variables, components).unwrap();
        let all = model.all_eigenvalues();
        assert_eq!(all.len(), (samples - 1).min(variables));
        assert_eq!(&all[..components], model.eigenvalues());
        assert!(all.windows(2).all(|w| w[0] >= w[1]), "{all:?}");
        let sum: f64 = all.iter().sum();
        let total = model.total_variance();
        assert!((sum - total).abs() <= 1e-12 * total, "{sum} != {total}");
    }
}

#[test]
fn all_eigenvalues_show_the_supported_components() {
    // Forty spectra of twelve variables, each centred on its own mean as SNV
    // does, so one direction is removed and eleven components remain.
    let data: Vec<f64> = pseudo_random(40 * 12, 13)
        .chunks_exact(12)
        .flat_map(|row| {
            let mean = row.iter().sum::<f64>() / 12.0;
            row.iter().map(move |x| x - mean)
        })
        .collect();
    assert_eq!(
        Pca::fit(&data, 12, 12).unwrap_err(),
        Error::InsufficientRank {
            requested: 12,
            supported: 11
        }
    );
    let model = Pca::fit(&data, 12, 1).unwrap();
    let all = model.all_eigenvalues();
    assert_eq!(all.len(), 12);
    assert!(all[10] > 1e-3 * all[0], "{all:?}");
    assert!(all[11] < 1e-20 * all[0], "{all:?}");
    assert!(Pca::fit(&data, 12, 11).is_ok());
}

#[test]
fn diagnostics_do_not_depend_on_rotations_within_equal_components() {
    // The first two directions carry equal or nearly equal variance, so their
    // loadings may be any rotation within their plane, but the plane itself,
    // and with it Q and T², is determined by the gap to the third direction.
    // The directions are the orthonormal columns of a 4 × 4 Hadamard matrix
    // divided by two, so every expected value is exact.
    let hadamard = [
        [0.5, 0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
    ];
    let spectrum = |coordinates: [f64; 4]| -> Vec<f64> {
        (0..4)
            .map(|j| (0..4).map(|a| hadamard[j][a] * coordinates[a]).sum())
            .collect()
    };
    let first = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
    let second = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
    let third = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    for spread in [1.0, 1.0 + 1e-9] {
        let data: Vec<f64> = (0..8)
            .flat_map(|i| spectrum([first[i], spread * second[i], 0.1 * third[i], 0.0]))
            .collect();
        let model = Pca::fit(&data, 4, 2).unwrap();
        let variance = 8.0 / 7.0;
        close(model.eigenvalues(), &[variance * spread * spread, variance]);
        let (u, v, w, z) = (0.7, -1.3, 0.4, -0.2);
        let diagnostics = model.project(&spectrum([u, v, w, z])).unwrap().diagnostics;
        let q = w * w + z * z;
        assert!(
            (diagnostics.q_residual - q).abs() <= 1e-12 * q,
            "spread {spread}: {diagnostics:?}"
        );
        // Unequal variances make T² depend on the rotation by about their
        // relative difference.
        let t2 = (u * u + v * v) / variance;
        assert!(
            (diagnostics.hotelling_t2 - t2).abs() <= 1e-8 * t2,
            "spread {spread}: {diagnostics:?}"
        );
    }
}

#[test]
fn every_component_describes_the_data_completely() {
    // Six samples of four variables: four components span the centred data.
    let data = pseudo_random(24, 7);
    let model = Pca::fit(&data, 4, 4).unwrap();
    let ratios: f64 = model.explained_variance_ratio().iter().sum();
    assert!((ratios - 1.0).abs() < 1e-12, "{ratios}");
    for sample in data.chunks_exact(4) {
        let projection = model.project(sample).unwrap();
        assert!(
            projection.diagnostics.q_residual < 1e-24,
            "{:?}",
            projection.diagnostics
        );
    }
}

#[test]
fn training_projections_match_the_stored_scores() {
    let data = mixtures(10);
    let model = Pca::fit(&data, 101, 2).unwrap();
    let mut total = 0.0;
    for (i, sample) in data.chunks_exact(101).enumerate() {
        let projection = model.project(sample).unwrap();
        close_scaled(
            &projection.scores,
            model.score(i).unwrap(),
            largest(model.scores()),
        );
        total += projection.diagnostics.hotelling_t2;
    }
    // The mean of T² over the training samples is k (n − 1) / n.
    let expected = 2.0 * 9.0 / 10.0;
    let actual = total / 10.0;
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

#[test]
fn model_is_invariant_under_offsets_and_scaling() {
    let data = mixtures(8);
    let model = Pca::fit(&data, 101, 2).unwrap();

    // A constant per variable moves the mean only.
    let shifted: Vec<f64> = data
        .chunks_exact(101)
        .flat_map(|row| row.iter().enumerate().map(|(j, x)| x + 0.1 * j as f64))
        .collect();
    let moved = Pca::fit(&shifted, 101, 2).unwrap();
    close(moved.loadings(), model.loadings());
    close(moved.scores(), model.scores());
    close(moved.eigenvalues(), model.eigenvalues());
    let expected: Vec<f64> = model
        .mean()
        .iter()
        .enumerate()
        .map(|(j, m)| m + 0.1 * j as f64)
        .collect();
    close(moved.mean(), &expected);

    // Scaling by c keeps the loadings, scales scores by c and variances by c².
    let factor = 1024.0;
    let scaled: Vec<f64> = data.iter().map(|x| x * factor).collect();
    let bigger = Pca::fit(&scaled, 101, 2).unwrap();
    close(bigger.loadings(), model.loadings());
    let scores: Vec<f64> = model.scores().iter().map(|t| t * factor).collect();
    close(bigger.scores(), &scores);
    let eigenvalues: Vec<f64> = model
        .eigenvalues()
        .iter()
        .map(|v| v * factor * factor)
        .collect();
    close(bigger.eigenvalues(), &eigenvalues);
    close(
        bigger.explained_variance_ratio(),
        model.explained_variance_ratio(),
    );

    // Reordering the samples reorders the scores only.
    let swapped: Vec<f64> = data[101..202]
        .iter()
        .chain(&data[..101])
        .chain(&data[202..])
        .copied()
        .collect();
    let other = Pca::fit(&swapped, 101, 2).unwrap();
    close(other.loadings(), model.loadings());
    let scale = largest(model.scores());
    close_scaled(other.score(0).unwrap(), model.score(1).unwrap(), scale);
    close_scaled(other.score(1).unwrap(), model.score(0).unwrap(), scale);
}

#[test]
fn component_signs_are_deterministic() {
    let data = mixtures(6);
    let model = Pca::fit(&data, 101, 3).unwrap();
    for a in 0..model.components() {
        let loading = model.loading(a).unwrap();
        let bound = largest(loading) * (1.0 - f64::EPSILON.sqrt());
        let leading = loading.iter().find(|v| v.abs() >= bound).unwrap();
        assert!(*leading > 0.0, "component {a}: {leading}");
    }
    // Negating the data keeps the loadings and negates the scores.
    let negated: Vec<f64> = data.iter().map(|x| -x).collect();
    let mirrored = Pca::fit(&negated, 101, 3).unwrap();
    close(mirrored.loadings(), model.loadings());
    let scores: Vec<f64> = model.scores().iter().map(|t| -t).collect();
    close(mirrored.scores(), &scores);
}

#[test]
fn diagnostics_separate_the_two_kinds_of_outlier() {
    let data = mixtures(14);
    let model = Pca::fit(&data, 101, 5).unwrap();
    let training = data
        .chunks_exact(101)
        .map(|sample| model.project(sample).unwrap().diagnostics)
        .collect::<Vec<_>>();
    let worst_t2 = training.iter().fold(0.0_f64, |a, d| a.max(d.hotelling_t2));
    let worst_q = training.iter().fold(0.0_f64, |a, d| a.max(d.q_residual));

    // An unmodelled band leaves the component plane.
    let extra = band(2300.0, 40.0);
    let unmodelled: Vec<f64> = data[..101]
        .iter()
        .zip(&extra)
        .map(|(x, e)| x + 0.3 * e)
        .collect();
    let outside = model.project(&unmodelled).unwrap().diagnostics;
    assert!(
        outside.q_residual > 100.0 * worst_q,
        "{outside:?} against {worst_q}"
    );

    // A spectrum along a known direction stays in the plane but far out.
    let mean = model.mean();
    let along: Vec<f64> = data[..101]
        .iter()
        .zip(mean)
        .map(|(x, m)| m + 8.0 * (x - m))
        .collect();
    let far = model.project(&along).unwrap().diagnostics;
    assert!(
        far.hotelling_t2 > 10.0 * worst_t2 && far.q_residual < 100.0 * worst_q,
        "{far:?} against {worst_t2} and {worst_q}"
    );
}

#[test]
fn project_into_matches_project() {
    let data = mixtures(9);
    let model = Pca::fit(&data, 101, 2).unwrap();
    let mut scores = [0.0; 2];
    for sample in data.chunks_exact(101) {
        let diagnostics = model.project_into(sample, &mut scores).unwrap();
        let projection = model.project(sample).unwrap();
        close_scaled(&scores, &projection.scores, largest(model.scores()));
        assert_eq!(diagnostics, projection.diagnostics);
    }
    assert_eq!(model.samples(), 9);
    assert_eq!(model.variables(), 101);
    assert_eq!(model.components(), 2);
    assert_eq!(model.loading(2), None);
    assert_eq!(model.score(9), None);
}

#[test]
fn rejects_invalid_shapes_and_counts() {
    let data = pseudo_random(12, 3);
    assert_eq!(
        Pca::fit(&data, 5, 1).unwrap_err(),
        Error::InvalidDataShape {
            length: 12,
            variables: 5
        }
    );
    assert_eq!(
        Pca::fit(&data, 0, 1).unwrap_err(),
        Error::InvalidDataShape {
            length: 12,
            variables: 0
        }
    );
    assert_eq!(
        Pca::fit(&[], 3, 1).unwrap_err(),
        Error::InvalidDataShape {
            length: 0,
            variables: 3
        }
    );
    assert_eq!(
        Pca::fit(&data[..4], 4, 1).unwrap_err(),
        Error::TooFewSpectra {
            count: 1,
            minimum: 2
        }
    );
    // Three samples of four variables support at most two components.
    assert_eq!(
        Pca::fit(&data, 4, 3).unwrap_err(),
        Error::InvalidComponentCount {
            requested: 3,
            maximum: 2
        }
    );
    assert_eq!(
        Pca::fit(&data, 4, 0).unwrap_err(),
        Error::InvalidComponentCount {
            requested: 0,
            maximum: 2
        }
    );
    // Six samples of two variables support at most two components.
    assert!(Pca::fit(&data, 2, 2).is_ok());
    assert_eq!(
        Pca::fit(&data, 2, 3).unwrap_err(),
        Error::InvalidComponentCount {
            requested: 3,
            maximum: 2
        }
    );
}

#[test]
fn checks_inputs_in_order() {
    let mut data = pseudo_random(12, 11);
    data[7] = f64::NAN;
    // The shape, the number of spectra and the component count come first.
    assert_eq!(
        Pca::fit(&data, 5, 1).unwrap_err(),
        Error::InvalidDataShape {
            length: 12,
            variables: 5
        }
    );
    assert_eq!(
        Pca::fit(&data[..3], 3, 1).unwrap_err(),
        Error::TooFewSpectra {
            count: 1,
            minimum: 2
        }
    );
    assert_eq!(
        Pca::fit(&data, 4, 9).unwrap_err(),
        Error::InvalidComponentCount {
            requested: 9,
            maximum: 2
        }
    );
    assert_eq!(
        Pca::fit(&data, 4, 2).unwrap_err(),
        Error::NonFiniteInput { index: 7 }
    );
    data[7] = f64::INFINITY;
    assert_eq!(
        Pca::fit(&data, 4, 2).unwrap_err(),
        Error::NonFiniteInput { index: 7 }
    );

    // The spectrum length comes before the buffer length and non-finite
    // values, the buffer length before non-finite values.
    let model = Pca::fit(&mixtures(6), 101, 2).unwrap();
    let mut scores = [7.0; 2];
    let mut wrong = [7.0; 3];
    let mut short = vec![0.0; 100];
    short[40] = f64::NAN;
    assert_eq!(
        model.project_into(&short, &mut wrong),
        Err(Error::InvalidSpectrumLength {
            expected: 101,
            actual: 100
        })
    );
    let mut spectrum = vec![0.0; 101];
    spectrum[40] = f64::NAN;
    assert_eq!(
        model.project_into(&spectrum, &mut wrong),
        Err(Error::OutputLengthMismatch {
            expected: 2,
            actual: 3
        })
    );
    assert_eq!(
        model.project_into(&spectrum, &mut scores),
        Err(Error::NonFiniteInput { index: 40 })
    );
    // Rejected inputs leave the buffer untouched.
    assert_eq!(scores, [7.0; 2]);
    assert_eq!(wrong, [7.0; 3]);
}

#[test]
fn rejects_rank_deficient_data() {
    // Identical samples carry no variation at all.
    let constant = vec![0.5; 30];
    assert_eq!(
        Pca::fit(&constant, 3, 2).unwrap_err(),
        Error::InsufficientRank {
            requested: 2,
            supported: 0
        }
    );
    // Two directions of variation cannot support three components.
    let mut data = Vec::new();
    for i in 0..8 {
        let a = i as f64;
        let b = (i * i) as f64;
        for j in 0..5 {
            data.push(a * (j as f64) + b * ((j * j) as f64));
        }
    }
    assert!(Pca::fit(&data, 5, 2).is_ok());
    assert_eq!(
        Pca::fit(&data, 5, 3).unwrap_err(),
        Error::InsufficientRank {
            requested: 3,
            supported: 2
        }
    );
}

#[test]
fn rank_tolerance_scales_with_the_matrix_size() {
    // Four samples of forty variables with exactly orthogonal, centred
    // columns, so the singular values are 1 and `second`. The tolerance for
    // this size is 40 EPSILON, which the two cases bracket by a wide margin.
    let data = |second: f64| -> Vec<f64> {
        let first = [0.5, -0.5, 0.5, -0.5];
        let other = [0.5, 0.5, -0.5, -0.5];
        let mut data = vec![0.0; 4 * 40];
        for i in 0..4 {
            data[i * 40] = first[i];
            data[i * 40 + 1] = second * other[i];
        }
        data
    };
    assert_eq!(
        Pca::fit(&data(8.0 * f64::EPSILON), 40, 2).unwrap_err(),
        Error::InsufficientRank {
            requested: 2,
            supported: 1
        }
    );
    assert!(Pca::fit(&data(1e4 * f64::EPSILON), 40, 2).is_ok());
}

#[test]
fn fit_ignores_faer_global_parallelism() {
    // Applications may configure faer themselves; the decomposition runs on
    // one thread regardless. Before, this setting made the fit panic.
    faer::disable_global_parallelism();
    assert!(Pca::fit(&mixtures(6), 101, 2).is_ok());
}

#[test]
fn rounding_cannot_flip_a_component() {
    // The loadings of the single component are ±1/√2, which rounding turns
    // into magnitudes that differ in the last bit, differently for each data
    // set.
    let data = |factor: f64, offset: f64| -> Vec<f64> {
        (0..7)
            .flat_map(|i| {
                let t = 0.37 * i as f64 - 1.1;
                [factor * t + offset, -factor * t + offset]
            })
            .collect()
    };
    let reference = Pca::fit(&data(1.0, 0.0), 2, 1).unwrap();
    assert!(reference.loadings()[0] > 0.0);
    for (factor, offset) in [
        (3.0, 0.0),
        (1.0 / 3.0, 0.0),
        (11.0, 0.0),
        (13.0, 0.0),
        (1.0, 10.0),
    ] {
        let model = Pca::fit(&data(factor, offset), 2, 1).unwrap();
        close(model.loadings(), reference.loadings());
        assert_eq!(
            model.scores()[0].signum(),
            reference.scores()[0].signum(),
            "factor {factor}, offset {offset}"
        );
    }
}

#[test]
fn rejects_components_made_of_rounding() {
    // The mean of this constant is not exactly representable, so centring
    // leaves rounding instead of zeros.
    assert_eq!(
        Pca::fit(&[1.4673634523923815; 3], 1, 1).unwrap_err(),
        Error::InsufficientRank {
            requested: 1,
            supported: 0
        }
    );
    // One direction of variation on a growing offset: a second component
    // could only be rounding.
    for offset in [0.0, 1e3, 1e6, 1e9] {
        let data: Vec<f64> = (0..5)
            .flat_map(|i| [0.1, 0.3, 0.7].map(|v| offset + 0.1 * i as f64 * v))
            .collect();
        assert!(Pca::fit(&data, 3, 1).is_ok(), "offset {offset}");
        assert_eq!(
            Pca::fit(&data, 3, 2).unwrap_err(),
            Error::InsufficientRank {
                requested: 2,
                supported: 1
            },
            "offset {offset}"
        );
    }
}

#[test]
fn keeps_small_real_components_on_a_large_offset() {
    // Two uncorrelated directions, the second a million times weaker, so its
    // variance is 3e-13 of the first. Even on an offset of 1e6 it spans about
    // ten thousand units of rounding and must be kept.
    let first = [-3.0, -1.0, 1.0, 3.0];
    let second = [1.0, -1.0, -1.0, 1.0];
    let expected = (4.0 / 3.0 * 1e-12) / (20.0 / 3.0);
    for offset in [0.0, 1e3, 1e6] {
        let data: Vec<f64> = (0..4)
            .flat_map(|i| [offset + first[i], offset + 1e-6 * second[i], offset])
            .collect();
        let model = Pca::fit(&data, 3, 2).unwrap();
        let ratio = model.eigenvalues()[1] / model.eigenvalues()[0];
        assert!(
            (ratio - expected).abs() <= 1e-3 * expected,
            "offset {offset}: {ratio:e}"
        );
    }
}

#[test]
fn q_is_exact_far_from_the_training_magnitude() {
    // The model varies along the first variable only, so a spectrum at the
    // mean with a value in the second variable has exactly that value's
    // square as Q, whatever the magnitudes of the mean and the value.
    for (factor, off_plane) in [
        (2.0_f64.powi(-500), 1e5),
        (1e-100, 1e55),
        (1e100, 1e-70),
        (1.0, 1e-140),
        (1.0, 1e140),
    ] {
        let data: Vec<f64> = [1.0, 2.0, 3.5, 5.0]
            .iter()
            .flat_map(|x| [x * factor, 0.0])
            .collect();
        let model = Pca::fit(&data, 2, 1).unwrap();
        let projection = model.project(&[model.mean()[0], off_plane]).unwrap();
        let expected = off_plane * off_plane;
        let q = projection.diagnostics.q_residual;
        assert!(
            (q - expected).abs() <= 1e-12 * expected,
            "factor {factor:e}, value {off_plane:e}: {q:e}"
        );
    }
}

#[test]
fn handles_extreme_magnitudes() {
    let base = mixtures(6);
    let model = Pca::fit(&base, 101, 2).unwrap();
    // Very small and very large spectra keep the structure of the original.
    for exponent in [-200, 200] {
        let factor = 2.0_f64.powi(exponent);
        let shrunk: Vec<f64> = base.iter().map(|x| x * factor).collect();
        let other = Pca::fit(&shrunk, 101, 2).unwrap();
        close(other.loadings(), model.loadings());
        let scores: Vec<f64> = model.scores().iter().map(|t| t * factor).collect();
        close(other.scores(), &scores);
        close(
            other.explained_variance_ratio(),
            model.explained_variance_ratio(),
        );
    }

    // Beyond that the variances themselves leave the representable range.
    for exponent in [-1000, 1000] {
        let factor = 2.0_f64.powi(exponent);
        let extreme: Vec<f64> = base.iter().map(|x| x * factor).collect();
        assert_eq!(
            Pca::fit(&extreme, 101, 2).unwrap_err(),
            Error::NumericalFailure
        );
    }
}

/// Twelve spectra of forty variables whose component variances span many
/// orders of magnitude, so scaling them pushes one variance below the
/// representable range before the others.
fn wide_spread() -> Vec<f64> {
    let mut data = Vec::with_capacity(12 * 40);
    for i in 0..12 {
        for j in 0..40 {
            let position = j as f64 / 40.0;
            let first = (i as f64 * 0.7).sin();
            let second = (i as f64 * 1.3).cos();
            data.push(
                first * (1.0 + position)
                    + 1e-6 * second * (3.0 * position).cos()
                    + 1e-12 * ((i * j) as f64).sin(),
            );
        }
    }
    data
}

#[test]
fn shares_and_t2_stay_exact_for_tiny_data() {
    // Near 2^-497 the smallest eigenvalue in data units is subnormal and keeps
    // only a few bits, but the shares and T² do not depend on the scale.
    let base = wide_spread();
    let reference = Pca::fit(&base, 40, 3).unwrap();
    let tiny: Vec<f64> = base.iter().map(|x| x * 2.0_f64.powi(-497)).collect();
    let model = Pca::fit(&tiny, 40, 3).unwrap();
    assert!(model.eigenvalues()[2] < f64::MIN_POSITIVE);
    // Each share relative to itself: the smallest is 1e-25 of the largest.
    close_with(
        model.explained_variance_ratio(),
        reference.explained_variance_ratio(),
        0.0,
        "",
    );
    for (sample, original) in tiny.chunks_exact(40).zip(base.chunks_exact(40)) {
        let t2 = model.project(sample).unwrap().diagnostics.hotelling_t2;
        let expected = reference
            .project(original)
            .unwrap()
            .diagnostics
            .hotelling_t2;
        assert!(
            (t2 - expected).abs() <= 1e-10 * expected,
            "{t2} != {expected}"
        );
    }
}

#[test]
fn debug_shows_the_shape_not_the_buffers() {
    let model = Pca::fit(&mixtures(12), 101, 2).unwrap();
    let text = format!("{model:?}");
    assert!(
        text.contains("samples: 12") && text.contains("components: 2"),
        "{text}"
    );
    assert!(text.len() < 200, "{} characters", text.len());
}

#[test]
fn rejects_a_total_variance_that_overflows() {
    // Two orthogonal centred columns whose variances are each 0.75 f64::MAX:
    // the kept eigenvalue fits, the total does not, and the explained share
    // would otherwise silently become zero.
    let size = 1.5 * f64::MAX.sqrt();
    let first = [0.5, -0.5, 0.5, -0.5];
    let other = [0.5, 0.5, -0.5, -0.5];
    let data: Vec<f64> = (0..4)
        .flat_map(|i| [size * first[i], size * other[i]])
        .collect();
    assert_eq!(Pca::fit(&data, 2, 1).unwrap_err(), Error::NumericalFailure);
}

#[test]
fn projections_beyond_the_representable_range_fail() {
    let model = Pca::fit(&mixtures(6), 101, 2).unwrap();
    let mut scores = [0.0; 2];
    assert_eq!(
        model.project_into(&[f64::MAX; 101], &mut scores),
        Err(Error::NumericalFailure)
    );
}

#[test]
fn rejects_variances_that_underflow() {
    let base = wide_spread();
    let scaled = |exponent: i32| -> Vec<f64> {
        let factor = 2.0_f64.powi(exponent);
        base.iter().map(|x| x * factor).collect()
    };
    let usable = scaled(-480);
    let model = Pca::fit(&usable, 40, 3).unwrap();
    assert!(model.eigenvalues().iter().all(|v| *v > 0.0));
    assert!(model.project(&usable[..40]).is_ok());
    // The smallest variance underflows here while the total is still positive,
    // which would leave a model whose every projection fails.
    assert_eq!(
        Pca::fit(&scaled(-500), 40, 3).unwrap_err(),
        Error::NumericalFailure
    );
    assert!(Pca::fit(&scaled(-500), 40, 2).is_ok());
}

#[test]
fn numpy_reference() {
    let fixture = include_str!("fixtures/numpy_pca.txt");
    let mut model: Option<Pca> = None;
    let (mut magnitude, mut precision, mut score_scale) = (0.0, 0.0, f64::NAN);
    let mut centre = Vec::new();
    let mut case = 0;
    let mut projections = 0;
    for line in fixture.lines().filter(|line| !line.starts_with('#')) {
        let mut parts = line.split('|');
        let tag = parts.next().unwrap();
        let context = format!(" ({tag}, case {case})");
        match tag {
            "case" => {
                let (samples, variables, components) =
                    (number(&mut parts), number(&mut parts), number(&mut parts));
                let data = values(&mut parts);
                assert_eq!(data.len(), samples * variables);
                let fitted = Pca::fit(&data, variables, components).unwrap();
                assert_eq!(fitted.samples(), samples);
                assert_eq!(fitted.components(), components);
                // The tolerance scales come from the data alone, never from the
                // results under test.
                centre = (0..variables)
                    .map(|j| data.iter().skip(j).step_by(variables).sum::<f64>() / samples as f64)
                    .collect();
                // Centring loses relative precision in proportion to the
                // magnitude of the data over its spread, in any implementation.
                magnitude = largest(&data);
                let spread = data
                    .chunks_exact(variables)
                    .flat_map(|row| row.iter().zip(&centre).map(|(x, m)| (x - m).abs()))
                    .fold(0.0_f64, f64::max);
                precision = 1e-10 + 1e3 * f64::EPSILON * magnitude / spread;
                score_scale = f64::NAN;
                model = Some(fitted);
                case += 1;
            }
            "project" => {
                let model = model.as_ref().expect("case header");
                let spectrum = values(&mut parts);
                let projection = model.project(&spectrum).unwrap();
                let diagnostics = projection.diagnostics;
                let scores = values(&mut parts);
                assert!(
                    score_scale.is_finite(),
                    "scores row precedes projections{context}"
                );
                close_with(
                    &projection.scores,
                    &scores,
                    precision * score_scale,
                    &context,
                );
                // T² is dimensionless; about one per component is typical.
                let t2 = values(&mut parts);
                let typical = t2[0].max(model.components() as f64);
                close_with(
                    &[diagnostics.hotelling_t2],
                    &t2,
                    precision * typical,
                    &context,
                );
                // Q is part of the squared distance of the spectrum from the mean.
                let distance: f64 = spectrum
                    .iter()
                    .zip(&centre)
                    .map(|(x, m)| (x - m).powi(2))
                    .sum();
                let q = values(&mut parts);
                close_with(
                    &[diagnostics.q_residual],
                    &q,
                    precision * distance,
                    &context,
                );
                projections += 1;
            }
            "all" => {
                let model = model.as_ref().expect("case header");
                let expected = values(&mut parts);
                assert_eq!(
                    model.all_eigenvalues().len(),
                    expected.len(),
                    "length{context}"
                );
                // Perturbing the data by `precision` moves each singular value
                // by at most `precision` times the largest (Weyl), so each
                // eigenvalue by about 2 precision √(λ λ_max). A floor relative
                // to the largest alone would barely check the small ones that
                // Q limits depend on.
                let top = largest(&expected);
                for (i, (&a, &b)) in model.all_eigenvalues().iter().zip(&expected).enumerate() {
                    let floor = 2.0 * precision * (b * top).sqrt() + precision * precision * top;
                    close_with(&[a], &[b], floor, format!(" of eigenvalue {i}{context}"));
                }
            }
            other => {
                let model = model.as_ref().expect("case header");
                let expected = values(&mut parts);
                let (actual, floor) = match other {
                    // The mean is not affected by centring.
                    "mean" => (model.mean(), (1e-10 + 1e3 * f64::EPSILON) * magnitude),
                    "eigenvalues" => (model.eigenvalues(), precision * largest(&expected)),
                    "ratios" => (
                        model.explained_variance_ratio(),
                        precision * largest(&expected),
                    ),
                    "loadings" => (model.loadings(), precision),
                    "scores" => {
                        score_scale = largest(&expected);
                        (model.scores(), precision * score_scale)
                    }
                    unknown => panic!("unknown row {unknown}"),
                };
                close_with(actual, &expected, floor, &context);
            }
        }
        assert!(parts.next().is_none(), "trailing values{context}");
    }
    assert!(
        case >= 19 && projections >= 40,
        "{case} cases, {projections} projections"
    );
}
