# /// script
# requires-python = ">=3.12"
# dependencies = ["scipy==1.18.1", "numpy==2.5.2"]
# ///
"""Generate SNV references; run with `uv run tests/fixtures/generate_normalization.py`."""
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import zscore

assert scipy.__version__ == "1.18.1"
assert np.__version__ == "2.5.2"
rows = [
    f"# SciPy {scipy.__version__}; numpy {np.__version__}; zscore(x, ddof=1)",
    "# input|expected",
]


def encode(values):
    return ",".join(format(float(v), ".17g") for v in values)


def add_case(y):
    y = np.asarray(y, dtype=np.float64)
    expected = zscore(y, ddof=1)
    assert np.isfinite(expected).all()
    # Guards against a wrong ddof or axis in the reference call itself.
    explicit = (y - y.mean()) / y.std(ddof=1)
    assert np.allclose(expected, explicit, rtol=1e-12, atol=1e-12)
    rows.append(f"{encode(y)}|{encode(expected)}")


rows.append("# Random signals: PCG64 seed 20260915; the last is shifted negative.")
rng = np.random.Generator(np.random.PCG64(20260915))
for length in (2, 3, 7, 101, 1000):
    add_case(rng.standard_normal(length))
add_case(rng.standard_normal(101) - 5.0)

rows.append("# A sinusoid on a moderate and on a large offset.")
for offset, lengths in ((2.5, (3, 101)), (1000.0, (7, 101))):
    for length in lengths:
        add_case(offset + np.sin(np.arange(length) * 0.7))

rows.append("# Impulses.")
for position in (0, 3):
    add_case(np.eye(7)[position])

rows.append("# NIR-like absorbance, 1100-2500 nm at 14 nm: Gaussian bands (centre, FWHM in nm)")
rows.append("# with uniform amplitudes in [0.05, 0.6], then a * bands + b + slope * (nm - 1100)")
rows.append("# + noise, a in [0.6, 1.6], b in [0.1, 0.8]; same generator state as above.")
wavelengths = np.arange(1100.0, 2501.0, 14.0)
assert len(wavelengths) == 101
bands = ((1210.0, 60.0), (1450.0, 90.0), (1730.0, 50.0),
         (1940.0, 110.0), (2100.0, 80.0), (2310.0, 45.0))
for noise in (0.0, 1e-4):
    for slope in (0.0, 2e-4):
        amplitudes = rng.uniform(0.05, 0.6, len(bands))
        absorbance = sum(
            amplitude * np.exp(-4.0 * np.log(2.0) * ((wavelengths - centre) / fwhm) ** 2)
            for amplitude, (centre, fwhm) in zip(amplitudes, bands)
        )
        scale = rng.uniform(0.6, 1.6)
        offset = rng.uniform(0.1, 0.8)
        add_case(scale * absorbance + offset + slope * (wavelengths - 1100.0)
                 + noise * rng.standard_normal(len(wavelengths)))

Path(__file__).with_name("scipy_snv.txt").write_text("\n".join(rows) + "\n")
