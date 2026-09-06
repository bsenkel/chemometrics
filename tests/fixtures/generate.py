"""Regenerate with Python + scipy==1.17.1 + numpy==2.5.2; unnecessary for Rust tests."""
from pathlib import Path
import numpy as np
import scipy
from scipy.signal import savgol_filter

assert scipy.__version__ == "1.17.1"
assert np.__version__ == "2.5.2"
rows = [f"# SciPy {scipy.__version__}; numpy {np.__version__}; deriv=0; delta=1; mode=interp"]
cases = [(9,5,2),(17,7,3),(21,11,4),(7,7,2),(9,3,0),(3,1,0)]
# Cover full-window signals and repeated centered windows at larger sizes.
cases += [(length, window, order)
          for window in (21, 51, 101)
          for order in (2, 3)
          for length in (window, 2 * window + 7)]
for length, window, order in cases:
    x = np.arange(length, dtype=np.float64)
    y = np.sin(x * 0.7) + 0.03 * x * x + np.cos(x * 2.1) * 0.2
    expected = savgol_filter(y, window, order, deriv=0, delta=1.0, mode="interp")
    encode = lambda values: ",".join(format(float(v), ".17g") for v in values)
    rows.append(f"{window}|{order}|{encode(y)}|{encode(expected)}")
Path(__file__).with_name("scipy.txt").write_text("\n".join(rows) + "\n")
