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
# Include small and large windows, full-window spectra, and interior samples.
settings = [(5, 2, 1), (5, 2, 2), (7, 3, 3), (9, 4, 4),
            (21, 3, 1), (51, 3, 2), (101, 3, 3)]
for window, order, derivative in settings:
    for length in (window, 2 * window + 7):
        for spacing in (1.0, 0.5, 2.0, -0.5):
            x = np.arange(length, dtype=np.float64)
            y = np.sin(x * 0.7) + 0.03 * x * x + np.cos(x * 2.1) * 0.2
            expected = savgol_filter(y, window, order, deriv=derivative,
                                     delta=spacing, mode="interp")
            assert np.isfinite(expected).all()
            def encode(values):
                return ",".join(format(float(v), ".17g") for v in values)
            rows.append(f"{window}|{order}|{derivative}|{spacing}|{encode(y)}|{encode(expected)}")
Path(__file__).with_name("scipy_derivatives.txt").write_text("\n".join(rows) + "\n")
