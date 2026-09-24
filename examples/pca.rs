//! Incoming inspection of lactose monohydrate by near-infrared spectroscopy:
//! a PCA model of approved lots accepts a new lot or rejects it.
use chemometrics::pca::Pca;
use std::f64::consts::LN_2;

/// 1100–2500 nm in steps of 1 nm.
const WAVELENGTHS: usize = 1401;
/// Approved lots that define good material.
const LOTS: usize = 40;

/// Absorption band with its center and full width at half maximum in nm.
fn band(wavelength: f64, center: f64, width: f64) -> f64 {
    (-4.0 * LN_2 * ((wavelength - center) / width).powi(2)).exp()
}

/// Spectrum of a lot with a particle size relative to the usual one, surface
/// water and magnesium stearate in percent, and measurement noise.
fn measure(size: f64, water: f64, stearate: f64, random: &mut impl FnMut() -> f64) -> Vec<f64> {
    (0..WAVELENGTHS)
        .map(|i| {
            let w = 1100.0 + i as f64;
            let lactose = band(w, 1535.0, 50.0) + band(w, 2090.0, 60.0) + band(w, 2270.0, 50.0);
            size * lactose
                + 0.1 * water * band(w, 1940.0, 110.0)
                + 0.01 * stearate * band(w, 1725.0, 30.0)
                + 1e-4 * random()
        })
        .collect()
}

/// 99 % limit of T² for a new lot and two components:
/// `2(n²−1)/(n(n−2)) · F(2, n−2)`, whose F quantile has a closed form.
fn t2_limit(n: f64) -> f64 {
    let f = (n - 2.0) / 2.0 * (0.01_f64.powf(-2.0 / (n - 2.0)) - 1.0);
    2.0 * (n * n - 1.0) / (n * (n - 2.0)) * f
}

/// 99 % limit of Q after Jackson and Mudholkar, from the discarded eigenvalues.
fn q_limit(discarded: &[f64]) -> f64 {
    let theta = |power| discarded.iter().map(|l| l.powi(power)).sum::<f64>();
    let (t1, t2, t3) = (theta(1), theta(2), theta(3));
    let h = 1.0 - 2.0 * t1 * t3 / (3.0 * t2 * t2);
    let z = 2.326_347_874; // standard normal quantile for 99 %
    t1 * (z * (2.0 * t2 * h * h).sqrt() / t1 + 1.0 + t2 * h * (h - 1.0) / (t1 * t1)).powf(1.0 / h)
}

fn main() -> Result<(), chemometrics::Error> {
    // Pseudo-random numbers in [-1, 1) with a fixed seed, so every run prints
    // the same.
    let mut state = 5_u64;
    let mut random = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    // Approved lots differ in particle size and surface water.
    let mut references = Vec::with_capacity(LOTS * WAVELENGTHS);
    for _ in 0..LOTS {
        let (size, water) = (1.0 + 0.1 * random(), 0.2 + 0.1 * random());
        references.extend(measure(size, water, 0.0, &mut random));
    }
    let model = Pca::fit(&references, WAVELENGTHS, 2)?;
    println!(
        "Explained variance: {:.4?}",
        model.explained_variance_ratio()
    );
    let (t2_max, q_max) = (
        t2_limit(LOTS as f64),
        q_limit(&model.all_eigenvalues()[2..]),
    );
    println!("99 % limits: T² {t2_max:.1}, Q {q_max:.1e}\n");

    // T² flags a lot that varies in a known way, but too much; Q flags one
    // that contains something the approved lots do not.
    for (name, size, water, stearate) in [
        ("typical lot", 1.02, 0.21, 0.0),
        ("damp lot", 0.98, 0.45, 0.0),
        ("0.2 % Mg stearate", 1.01, 0.19, 0.2),
    ] {
        let d = model
            .project(&measure(size, water, stearate, &mut random))?
            .diagnostics;
        let verdict = if d.hotelling_t2 > t2_max {
            "reject: outside the known variation (T²)"
        } else if d.q_residual > q_max {
            "reject: contains something new (Q)"
        } else {
            "accept"
        };
        println!(
            "{name:<18} T² {:5.1}  Q {:.1e}  {verdict}",
            d.hotelling_t2, d.q_residual
        );
    }
    Ok(())
}
