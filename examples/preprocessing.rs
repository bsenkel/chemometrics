//! Apply smoothing, derivative, SNV and detrending steps and reuse an output buffer.
use chemometrics::{
    baseline::Detrend,
    normalize::StandardNormalVariate,
    smooth::{MovingAverage, SavitzkyGolay},
};

fn main() -> Result<(), chemometrics::Error> {
    let signal = [2.0, 2.0, 5.0, 2.0, 1.0, 0.0, 1.0, 4.0, 9.0];
    let average = MovingAverage::new(5)?;
    println!("Moving average: {:?}", average.apply(&signal)?);
    let sg = SavitzkyGolay::new(5, 2)?;
    let mut output = vec![0.0; signal.len()];
    sg.apply_into(&signal, &mut output)?;
    println!("Savitzky–Golay: {output:?}");
    // Reuse the same filter and output allocation for another spectrum.
    sg.apply_into(&[1.0; 9], &mut output)?;
    // The original spectrum is sampled every 0.5 axis units.
    let derivative = SavitzkyGolay::new_derivative(5, 2, 1, 0.5)?;
    derivative.apply_into(&signal, &mut output)?;
    println!("First derivative (intensity / axis unit): {output:?}");
    SavitzkyGolay::new_derivative(5, 2, 2, 0.5)?.apply_into(&signal, &mut output)?;
    println!("Second derivative (intensity / axis unit²): {output:?}");
    let normalized = StandardNormalVariate.apply(&output)?;
    println!("SNV of the second derivative: {normalized:?}");
    // SNV and Detrend: scatter correction, then a quadratic baseline.
    let corrected = Detrend::new(2).apply(&StandardNormalVariate.apply(&signal)?)?;
    println!("SNV and Detrend of the spectrum: {corrected:?}");
    Ok(())
}
