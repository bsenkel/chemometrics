# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.2", "scipy==1.18.1"]
# ///
"""Generate IOT references; run with `uv run tests/fixtures/generate_iot.py`."""
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.signal import savgol_filter

assert np.__version__ == "2.5.2"
assert scipy.__version__ == "1.18.1"
rows = [
    f"# numpy {np.__version__}; exact optimum by trying every set of nonzero contributions, checked",
    f"# against the optimality conditions and SciPy {scipy.__version__} SLSQP",
    "# case|partial|components|variables|pure spectra, then one line per mixture:",
    "# predict|mixture|contributions|residual",
]
# Every reference keeps a clear margin, so that it is well determined: nonzero
# contributions stay away from zero, and releasing a zero contribution would
# clearly raise the misfit. Relative to one, or to the scale of the gradient.
MARGIN = 1e-6


def encode(values):
    return ",".join(format(float(v), ".17g") for v in np.ravel(values))


def optimum(constituents, mixture):
    """Contributions that are not negative and sum to one with the least misfit.

    Returns None if the optimum is not well determined.
    """
    # The sum of one makes the misfit invariant under a shift shared by all
    # spectra; centring on the mean constituent keeps a large offset from
    # costing digits. The crate shifts by a constituent instead.
    centre = constituents.mean(axis=0)
    a = (constituents - centre).T
    b = mixture - centre
    count = len(constituents)
    best = None
    for size in range(1, count + 1):
        for support in combinations(range(count), size):
            columns = a[:, support]
            weights = np.full(size, 1.0 / size)
            if size > 1:
                # Parameterize the plane of weights that sum to one by an
                # orthonormal basis of directions that keep the sum.
                basis = null_space(np.ones((1, size)))
                step, *_ = np.linalg.lstsq(columns @ basis, b - columns @ weights, rcond=None)
                weights = weights + basis @ step
            if weights.min() <= 0.0:
                continue
            r = np.zeros(count)
            r[list(support)] = weights
            misfit = a @ r - b
            value = misfit @ misfit
            if best is None or value < best[1]:
                best = (r, value, support)
    r, value, support = best
    assert np.isclose(r.sum(), 1.0, rtol=0.0, atol=1e-14)
    # Optimality: the gradient is the same on the support and larger elsewhere.
    scale = np.linalg.norm(a) * (np.linalg.norm(a) + np.linalg.norm(b))
    gradient = a.T @ (a @ r - b)
    level = gradient[list(support)].mean()
    others = [i for i in range(count) if i not in support]
    assert np.abs(gradient[list(support)] - level).max() <= 1e-10 * scale
    if r[list(support)].min() < MARGIN or (others and (gradient[others] - level).min() < MARGIN * scale):
        return None
    # Independent check by a general-purpose solver, which is only accurate to
    # its own stopping threshold.
    result = minimize(
        lambda x: np.sum((a @ x - b) ** 2) / scale,
        np.full(count, 1.0 / count),
        jac=lambda x: 2.0 * a.T @ (a @ x - b) / scale,
        bounds=[(0.0, 1.0)] * count,
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(count)}],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    assert result.success, result.message
    assert np.abs(result.x - r).max() < 1e-6, (result.x, r)
    assert np.isfinite(r).all() and np.isfinite(value)
    return r, value


def add_case(pure, draws, partial=False):
    """Adds pure spectra and one mixture per draw, drawing again while the
    optimum is not well determined."""
    pure = np.asarray(pure, dtype=np.float64)
    components, variables = pure.shape
    # The further component of a partial model has a spectrum of zeros.
    constituents = np.vstack([pure, np.zeros(variables)]) if partial else pure
    assert len(constituents) - 1 <= variables
    rows.append(f"case|{int(partial)}|{components}|{variables}|{encode(pure)}")
    for draw in draws:
        for _ in range(100):
            mixture = np.asarray(draw(), dtype=np.float64)
            found = optimum(constituents, mixture)
            if found is not None:
                break
        else:
            raise AssertionError("no well determined mixture")
        r, value = found
        rows.append(f"predict|{encode(mixture)}|{encode(r[:components])}|{encode([value])}")


def blend(spectra, weights, noise=0.0):
    """Weighted sum of spectra with optional Gaussian noise."""
    return lambda: np.asarray(weights) @ spectra + noise * rng.standard_normal(spectra.shape[1])


rng = np.random.Generator(np.random.PCG64(20261003))


def weights_inside(count, total=1.0):
    weights = rng.uniform(0.2, 1.0, count)
    return total * weights / weights.sum()


def mixtures(pure, partial):
    """Exact and noisy mixtures, one outside the allowed compositions, and an
    unrelated spectrum.

    Outside means the first component below zero, and for a partial model
    also more than all of the given components.
    """
    components, variables = pure.shape
    total = 0.8 if partial else 1.0
    outside = rng.uniform(0.2, 1.0, components)
    if components > 1:
        outside[0] = -0.4
    outside *= (1.3 if partial else 1.0) / outside.sum()
    return [
        lambda: weights_inside(components, total) @ pure,
        lambda: weights_inside(components, total) @ pure + 0.05 * rng.standard_normal(variables),
        blend(pure, outside),
        lambda: pure.mean(axis=0) + rng.standard_normal(variables),
    ]


rows.append("# Random data: PCG64 seed 20261003; the last shape of each kind has as many")
rows.append("# components as the variables support.")
for components, variables, partial in (
    (2, 4, False), (3, 6, False), (4, 9, False), (6, 20, False), (4, 3, False),
    (1, 4, True), (2, 5, True), (4, 10, True), (3, 3, True),
):
    pure = rng.standard_normal((components, variables))
    add_case(pure, mixtures(pure, partial), partial)

rows.append("# A shared offset of 1e6 and magnitudes of 1e-8 and 1e8.")
pure = rng.standard_normal((3, 7))
draws = mixtures(pure, False)
base = [draw() for draw in draws]
for factor, offset in ((1.0, 1e6), (1e-8, 0.0), (1e8, 0.0)):
    add_case(factor * pure + offset, [lambda m=m: factor * m + offset for m in base])

rows.append("# NIR-like absorbance, 1100-2500 nm at 14 nm: Gaussian bands (centre, FWHM in nm)")
rows.append("# of an active ingredient, two excipients and a lubricant on a small offset; blends")
rows.append("# with noise of 1e-4, then the same after a Savitzky-Golay second derivative")
rows.append("# (window 7, order 2, spacing 14 nm) without a spectrum of the lubricant.")
wavelengths = np.arange(1100.0, 2501.0, 14.0)
assert len(wavelengths) == 101


def absorbance(bands):
    return 0.05 + sum(
        height * np.exp(-4.0 * np.log(2.0) * ((wavelengths - centre) / fwhm) ** 2)
        for centre, fwhm, height in bands
    )


nir = np.array([
    absorbance(((1210.0, 60.0, 0.5), (1680.0, 50.0, 0.6), (2270.0, 60.0, 0.8))),
    absorbance(((1450.0, 60.0, 0.4), (1535.0, 50.0, 0.7), (1930.0, 40.0, 0.9), (2090.0, 60.0, 0.6))),
    absorbance(((1490.0, 90.0, 0.5), (1780.0, 70.0, 0.4), (2100.0, 80.0, 0.7))),
    absorbance(((1725.0, 30.0, 0.9), (2310.0, 30.0, 0.8))),
])
blends = [
    blend(nir, [0.10, 0.50, 0.39, 0.01], 1e-4),
    blend(nir, [0.15, 0.45, 0.39, 0.01], 1e-4),
    blend(nir, [0.05, 0.60, 0.34, 0.01], 1e-4),
    # A lubricant level below zero stands for noise or drift that pushes it
    # there; its contribution ends at the bound.
    blend(nir, [0.10, 0.52, 0.40, -0.02], 1e-4),
]
add_case(nir, blends)


def second_derivative(spectra):
    return savgol_filter(spectra, 7, 2, deriv=2, delta=14.0, mode="interp", axis=-1)


add_case(second_derivative(nir[:3]), [lambda draw=draw: second_derivative(draw()) for draw in blends], True)

Path(__file__).with_name("numpy_iot.txt").write_text("\n".join(rows) + "\n")
