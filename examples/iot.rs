//! Composition of a powder blend from the near-infrared spectra of its pure
//! components alone, without calibration samples.
use chemometrics::iot::Iot;

fn percent(fractions: &[f64]) -> String {
    fractions
        .iter()
        .map(|f| format!("{:9.1}", 100.0 * f))
        .collect()
}

fn main() -> Result<(), chemometrics::Error> {
    // Absorbance of the pure components at eight wavelengths, one row each:
    // active ingredient, lactose and magnesium stearate. Each is measured
    // once and then serves for every blend.
    let wavelengths = 8;
    let pure = [
        [0.05, 0.60, 0.10, 0.05, 0.40, 0.05, 0.10, 0.70],
        [0.30, 0.10, 0.50, 0.70, 0.10, 0.60, 0.20, 0.10],
        [0.05, 0.05, 0.10, 0.05, 0.80, 0.05, 0.60, 0.10],
    ];
    let model = Iot::new(pure.as_flattened(), wavelengths)?;

    // In practice the blend spectra come from the spectrometer. Here they
    // are simulated from known fractions of the pure spectra, with fixed small
    // deviations in place of measurement noise, and optionally an impurity
    // that is not among the pure components and absorbs at the first
    // wavelength.
    let noise = [
        0.002, -0.001, 0.0015, -0.002, -0.0025, 0.001, -0.0015, 0.002,
    ];
    let impurity = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let simulate = |fractions: [f64; 3], impurity_share: f64| -> Vec<f64> {
        (0..wavelengths)
            .map(|i| {
                let blend: f64 = fractions.iter().zip(&pure).map(|(f, s)| f * s[i]).sum();
                blend + impurity_share * impurity[i] + noise[i]
            })
            .collect()
    };

    println!(
        "{:<20}{:>9}{:>9}{:>9}",
        "Mass fractions in %", "active", "lactose", "stearate"
    );
    for (name, fractions, impurity_share) in [
        // The noise weighs most on the trace of stearate.
        ("on target", [0.10, 0.895, 0.005], 0.0),
        // Without the constraints, the noise would put the stearate at
        // -0.3 % and the sum at 99.9 %; IOT reports zero and keeps the sum
        // at 100 %.
        ("stearate forgotten", [0.10, 0.90, 0.0], 0.0),
        // The impurity distorts the composition, and the residual, about 40
        // times that of the other blends, shows that the pure spectra do not
        // describe this one.
        (
            "with 3 % of an unknown impurity",
            [0.097, 0.868, 0.005],
            0.03,
        ),
    ] {
        let prediction = model.predict(&simulate(fractions, impurity_share))?;
        println!("{name}");
        println!("  {:<18}{}", "true", percent(&fractions));
        println!(
            "  {:<18}{}   residual {:.1e}",
            "estimated by IOT",
            percent(&prediction.contributions),
            prediction.residual
        );
    }
    Ok(())
}
