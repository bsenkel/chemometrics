//! Preprocessing of a near-infrared spectrum: each step removes one kind of
//! disturbance, and the output shows how much of it remains.
use chemometrics::{
    baseline::Detrend,
    normalize::StandardNormalVariate,
    smooth::{MovingAverage, SavitzkyGolay},
};
use std::f64::consts::LN_2;

/// 1100–2500 nm in steps of 2 nm.
const WAVELENGTHS: usize = 701;
/// Distance between adjacent wavelengths in nm.
const SPACING: f64 = 2.0;

/// Absorption band with its center and full width at half maximum in nm.
fn band(wavelength: f64, center: f64, width: f64) -> f64 {
    (-4.0 * LN_2 * ((wavelength - center) / width).powi(2)).exp()
}

/// Root-mean-square difference between two spectra.
fn deviation(a: &[f64], b: &[f64]) -> f64 {
    let squares: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    (squares / a.len() as f64).sqrt()
}

fn main() -> Result<(), chemometrics::Error> {
    // Pseudo-random numbers in [-1, 1) with a fixed seed, so every run prints
    // the same: a linear congruential generator with Knuth's MMIX constants,
    // whose top 53 bits become the number.
    let mut state = 5_u64;
    let mut random = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    // The undisturbed spectrum: water bands at 1450 and 1940 nm.
    let spectrum: Vec<f64> = (0..WAVELENGTHS)
        .map(|i| {
            let wavelength = 1100.0 + SPACING * i as f64;
            band(wavelength, 1450.0, 90.0) + 0.6 * band(wavelength, 1940.0, 110.0)
        })
        .collect();

    // Noise: both filters reduce it. The filters and one output buffer are
    // reused for every measurement.
    let average = MovingAverage::new(11)?;
    let savitzky_golay = SavitzkyGolay::new(11, 2)?;
    let mut smoothed = vec![0.0; WAVELENGTHS];
    for measurement in 1..=3 {
        let noisy: Vec<f64> = spectrum.iter().map(|x| x + 0.01 * random()).collect();
        let raw = deviation(&noisy, &spectrum);
        average.apply_into(&noisy, &mut smoothed)?;
        let averaged = deviation(&smoothed, &spectrum);
        savitzky_golay.apply_into(&noisy, &mut smoothed)?;
        let fitted = deviation(&smoothed, &spectrum);
        println!(
            "Noise {measurement}: {raw:.4} raw, {averaged:.4} moving average, {fitted:.4} Savitzky–Golay"
        );
    }
    // The moving average also flattens the band peaks, which the polynomial
    // fit of Savitzky–Golay keeps.
    let peak = 175; // 1450 nm
    println!(
        "Peak at 1450 nm: {:.4} true, {:.4} moving average, {:.4} Savitzky–Golay",
        spectrum[peak],
        average.apply(&spectrum)?[peak],
        savitzky_golay.apply(&spectrum)?[peak]
    );

    // Scatter scales the spectrum and shifts it; SNV removes both.
    let scattered: Vec<f64> = spectrum.iter().map(|x| 1.3 * x + 0.2).collect();
    println!(
        "Scatter: {:.4} raw, {:.1e} after SNV",
        deviation(&scattered, &spectrum),
        deviation(
            &StandardNormalVariate.apply(&scattered)?,
            &StandardNormalVariate.apply(&spectrum)?
        )
    );

    // A sloping baseline: detrending removes it. A first derivative removes
    // only the offset and leaves the slope, 1.5e-4 per nm; a second derivative
    // removes both.
    let sloped: Vec<f64> = (0..WAVELENGTHS)
        .map(|i| spectrum[i] + 0.1 + 3e-4 * i as f64)
        .collect();
    let detrend = Detrend::new(2);
    let first = SavitzkyGolay::new_derivative(11, 2, 1, SPACING)?;
    let second = SavitzkyGolay::new_derivative(11, 2, 2, SPACING)?;
    println!(
        "Baseline: {:.4} raw, {:.1e} after detrending",
        deviation(&sloped, &spectrum),
        deviation(&detrend.apply(&sloped)?, &detrend.apply(&spectrum)?)
    );
    println!(
        "Baseline in derivatives: {:.1e} per nm in the first, {:.1e} per nm² in the second",
        deviation(&first.apply(&sloped)?, &first.apply(&spectrum)?),
        deviation(&second.apply(&sloped)?, &second.apply(&spectrum)?)
    );
    Ok(())
}
