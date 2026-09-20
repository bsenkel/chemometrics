//! Preprocess a set of NIR-like spectra, fit a PCA model and flag outliers.
use chemometrics::{baseline::Detrend, normalize::StandardNormalVariate, pca::Pca};
use std::f64::consts::LN_2;

const POINTS: usize = 101;

/// Absorbance of one Gaussian band on 1100–2500 nm at 14 nm.
fn band(centre: f64, fwhm: f64) -> Vec<f64> {
    (0..POINTS)
        .map(|i| {
            let wavelength = 1100.0 + 14.0 * i as f64;
            let width = (wavelength - centre) / fwhm;
            (-4.0 * LN_2 * width * width).exp()
        })
        .collect()
}

/// A mixture spectrum with multiplicative scatter and an offset baseline.
fn mixture(concentrations: [f64; 3], scatter: f64, offset: f64) -> Vec<f64> {
    let profiles = [band(1210.0, 60.0), band(1450.0, 90.0), band(1940.0, 110.0)];
    (0..POINTS)
        .map(|j| {
            let absorbance: f64 = profiles
                .iter()
                .zip(concentrations)
                .map(|(profile, concentration)| concentration * profile[j])
                .sum();
            scatter * absorbance + offset + 3e-4 * j as f64
        })
        .collect()
}

/// Scatter correction followed by a quadratic baseline, as Barnes proposed.
fn preprocess(spectrum: &[f64]) -> Result<Vec<f64>, chemometrics::Error> {
    Detrend::new(2).apply(&StandardNormalVariate.apply(spectrum)?)
}

fn main() -> Result<(), chemometrics::Error> {
    let recipes = [
        ([0.8, 0.3, 0.2], 1.00, 0.30),
        ([0.6, 0.5, 0.2], 1.15, 0.35),
        ([0.4, 0.7, 0.3], 0.90, 0.25),
        ([0.3, 0.6, 0.5], 1.05, 0.40),
        ([0.2, 0.4, 0.7], 1.20, 0.28),
        ([0.5, 0.5, 0.5], 0.95, 0.33),
        ([0.7, 0.2, 0.4], 1.10, 0.31),
        ([0.35, 0.55, 0.45], 1.02, 0.36),
    ];
    let mut data = Vec::with_capacity(recipes.len() * POINTS);
    for (concentrations, scatter, offset) in recipes {
        data.extend(preprocess(&mixture(concentrations, scatter, offset))?);
    }

    let model = Pca::fit(&data, POINTS, 3)?;
    for (a, ratio) in model.explained_variance_ratio().iter().enumerate() {
        println!(
            "Component {}: {:.2} % of the variance",
            a + 1,
            100.0 * ratio
        );
    }

    // Reuse one score buffer for every spectrum, here without allocating.
    let mut scores = vec![0.0; model.components()];
    let mut limit: f64 = 0.0;
    for (i, spectrum) in data.chunks_exact(POINTS).enumerate() {
        let diagnostics = model.project_into(spectrum, &mut scores)?;
        limit = limit.max(diagnostics.q_residual);
        println!(
            "Sample {i}: scores {:+.3?}, T² {:.2}, Q {:.2e}",
            scores, diagnostics.hotelling_t2, diagnostics.q_residual
        );
    }

    // A sample with an extra component leaves the plane of the model.
    let contaminant = band(2300.0, 40.0);
    let raw = mixture([0.6, 0.5, 0.2], 1.15, 0.35);
    let adulterated: Vec<f64> = raw
        .iter()
        .zip(&contaminant)
        .map(|(x, extra)| x + 0.25 * extra)
        .collect();
    let projection = model.project(&preprocess(&adulterated)?)?;
    println!("Largest Q among the training samples: {limit:.2e}");
    println!(
        "Adulterated sample: T² {:.2}, Q {:.2e}",
        projection.diagnostics.hotelling_t2, projection.diagnostics.q_residual
    );
    Ok(())
}
