"""Generate derivative references with scipy==1.18.1 and numpy==2.5.2."""
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import savgol_filter

assert scipy.__version__ == "1.18.1"
assert np.__version__ == "2.5.2"
rows = [
    f"# SciPy {scipy.__version__}; numpy {np.__version__}; mode=interp",
    "# window|polynomial_order|derivative_order|sample_spacing|input|expected",
]


def add_case(y, window, order, derivative, spacing):
    expected = savgol_filter(y, window, order, deriv=derivative,
                             delta=spacing, mode="interp")
    assert np.isfinite(expected).all()

    def encode(values):
        return ",".join(format(float(v), ".17g") for v in values)

    rows.append(f"{window}|{order}|{derivative}|{spacing}|{encode(y)}|{encode(expected)}")


# Include small and large windows, full-window spectra, and interior samples.
settings = [(5, 2, 1), (5, 2, 2), (7, 3, 3), (9, 4, 4),
            (21, 3, 1), (51, 3, 2), (101, 3, 3)]
for window, order, derivative in settings:
    for length in (window, 2 * window + 7):
        for spacing in (1.0, 0.5, 2.0, -0.5):
            x = np.arange(length, dtype=np.float64)
            y = np.sin(x * 0.7) + 0.03 * x * x + np.cos(x * 2.1) * 0.2
            add_case(y, window, order, derivative, spacing)

rows.append("# Impulses at every position: each row checks one operator column.")
for position in range(13):
    y = np.zeros(13, dtype=np.float64)
    y[position] = 1.0
    for derivative in (1, 2):
        add_case(y, 5, 2, derivative, 1.0)
for position in range(17):
    y = np.zeros(17, dtype=np.float64)
    y[position] = 1.0
    add_case(y, 7, 3, 2, -0.5)

rows.append("# Random signals: PCG64 seed 20260909; the last has overlapping edge windows.")
rng = np.random.Generator(np.random.PCG64(20260909))
for length, window, derivative, spacing in ((7, 7, 1, 0.5),
                                           (49, 21, 2, 1.0),
                                           (109, 51, 3, -0.5),
                                           (22, 21, 1, 2.0)):
    add_case(rng.standard_normal(length), window, 3, derivative, spacing)

Path(__file__).with_name("scipy_derivatives.txt").write_text("\n".join(rows) + "\n")
