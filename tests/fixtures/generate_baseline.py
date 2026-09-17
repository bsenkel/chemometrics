# /// script
# requires-python = ">=3.12"
# dependencies = ["scipy==1.18.1", "numpy==2.5.2"]
# ///
"""Generate detrend references; run with `uv run tests/fixtures/generate_baseline.py`."""
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial import polynomial as P
from scipy.signal import detrend

assert scipy.__version__ == "1.18.1"
assert np.__version__ == "2.5.2"
rows = [
    f"# SciPy {scipy.__version__}; numpy {np.__version__}; residual of polyfit on 2 i / (n - 1) - 1",
    "# order|input|expected",
]


def encode(values):
    return ",".join(format(float(v), ".17g") for v in values)


def add_case(y, order):
    y = np.asarray(y, dtype=np.float64)
    assert len(y) > order
    coordinate = 2.0 * np.arange(len(y)) / (len(y) - 1) - 1.0 if len(y) > 1 else np.zeros(1)
    expected = y - P.polyval(coordinate, P.polyfit(coordinate, y, order))
    assert np.isfinite(expected).all()
    # Guards against a wrong coordinate or degree in the reference fit itself.
    if order in (0, 1):
        kind = "constant" if order == 0 else "linear"
        assert np.allclose(expected, detrend(y, type=kind), rtol=1e-11, atol=1e-11 * np.abs(y).max())
    rows.append(f"{order}|{encode(y)}|{encode(expected)}")


rows.append("# Random signals: PCG64 seed 20260917.")
rng = np.random.Generator(np.random.PCG64(20260917))
for order in (0, 1, 2, 3):
    for length in (order + 1, order + 2, 11, 101, 250):
        add_case(rng.standard_normal(length), order)

rows.append("# Sinusoids on a large offset and with a strong trend.")
for order in (0, 1, 2):
    add_case(1000.0 + np.sin(np.arange(101) * 0.7), order)
    add_case(np.sin(np.arange(101) * 0.3) + 0.05 * np.arange(101), order)

rows.append("# NIR-like absorbance, 1100-2500 nm at 14 nm: Gaussian bands (centre, FWHM in nm)")
rows.append("# with uniform amplitudes in [0.05, 0.6], then a * bands + b + slope * (nm - 1100)")
rows.append("# + curvature * (nm - 1100)^2 + noise, a in [0.6, 1.6], b in [0.1, 0.8].")
wavelengths = np.arange(1100.0, 2501.0, 14.0)
assert len(wavelengths) == 101
bands = ((1210.0, 60.0), (1450.0, 90.0), (1730.0, 50.0),
         (1940.0, 110.0), (2100.0, 80.0), (2310.0, 45.0))
for noise in (0.0, 1e-4):
    for slope, curvature in ((0.0, 0.0), (2e-4, 0.0), (2e-4, 1e-7)):
        amplitudes = rng.uniform(0.05, 0.6, len(bands))
        absorbance = sum(
            amplitude * np.exp(-4.0 * np.log(2.0) * ((wavelengths - centre) / fwhm) ** 2)
            for amplitude, (centre, fwhm) in zip(amplitudes, bands)
        )
        offset = wavelengths - 1100.0
        measured = (rng.uniform(0.6, 1.6) * absorbance + rng.uniform(0.1, 0.8)
                    + slope * offset + curvature * offset ** 2
                    + noise * rng.standard_normal(len(wavelengths)))
        for order in (1, 2):
            add_case(measured, order)

Path(__file__).with_name("scipy_detrend.txt").write_text("\n".join(rows) + "\n")
