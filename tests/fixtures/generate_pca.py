# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.2", "scikit-learn==1.9.1"]
# ///
"""Generate PCA references; run with `uv run tests/fixtures/generate_pca.py`."""
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA

assert np.__version__ == "2.5.2"
assert sklearn.__version__ == "1.9.1"
rows = [
    f"# numpy {np.__version__}; checked against scikit-learn {sklearn.__version__} PCA(svd_solver='full')",
    "# case|samples|variables|components|data, then mean, eigenvalues, ratios, loadings,",
    "# scores and one project line per spectrum: project|spectrum|scores|t2|q",
]
# Loadings and scores of a component are only determined as far as the gap to
# the neighbouring eigenvalue allows, so every case keeps a clear gap.
MINIMUM_GAP = 1e-3


def encode(values):
    return ",".join(format(float(v), ".17g") for v in np.ravel(values))


def signs(loadings):
    """Signs that make the largest-magnitude loading of each component positive."""
    extreme = loadings[np.arange(len(loadings)), np.argmax(np.abs(loadings), axis=1)]
    return np.where(extreme < 0.0, -1.0, 1.0)


def add_case(data, components, spectra):
    data = np.asarray(data, dtype=np.float64)
    samples, variables = data.shape
    assert 1 <= components <= min(samples - 1, variables)
    mean = data.mean(axis=0)
    centered = data - mean
    # Centring loses absolute precision proportional to the data magnitude.
    magnitude = 1e3 * np.finfo(np.float64).eps * np.abs(data).max()
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    gaps = np.diff(np.append(singular, 0.0)) / -singular[0]
    assert gaps[:components].min() > MINIMUM_GAP, gaps
    eigenvalues = singular**2 / (samples - 1)
    total = eigenvalues.sum()
    sign = signs(vt[:components])
    loadings = sign[:, None] * vt[:components]
    scores = sign * (u * singular)[:, :components]
    ratios = eigenvalues[:components] / total
    # The total variance is the variance of the data, not only of the kept part.
    assert np.isclose(total, centered.var(axis=0, ddof=1).sum(), rtol=1e-12)
    assert np.allclose(centered, (u * singular) @ vt, rtol=1e-12, atol=1e-12 * np.abs(centered).max())
    # Independent reference: scikit-learn, compared up to the sign convention.
    reference = PCA(n_components=components, svd_solver="full").fit(data)
    assert np.allclose(reference.mean_, mean, rtol=1e-12)
    assert np.allclose(reference.explained_variance_, eigenvalues[:components], rtol=1e-11)
    assert np.allclose(reference.explained_variance_ratio_, ratios, rtol=1e-11)
    other_sign = signs(reference.components_)
    assert np.allclose(other_sign[:, None] * reference.components_, loadings, rtol=1e-9, atol=1e-11)
    assert np.allclose(other_sign * reference.transform(data), scores,
                       rtol=1e-9, atol=1e-9 * np.abs(scores).max() + magnitude)
    lines = [
        f"case|{samples}|{variables}|{components}|{encode(data)}",
        f"mean|{encode(mean)}",
        f"eigenvalues|{encode(eigenvalues[:components])}",
        f"ratios|{encode(ratios)}",
        f"loadings|{encode(loadings)}",
        f"scores|{encode(scores)}",
    ]
    for spectrum in np.asarray(spectra, dtype=np.float64):
        projected = (spectrum - mean) @ loadings.T
        residual = spectrum - mean - projected @ loadings
        t2 = float((projected**2 / eigenvalues[:components]).sum())
        q = float(residual @ residual)
        assert np.allclose(other_sign * reference.transform(spectrum[None, :])[0], projected,
                           rtol=1e-9, atol=1e-9 * np.abs(projected).max() + magnitude)
        for values in (projected, [t2, q]):
            assert np.isfinite(values).all()
        lines.append(f"project|{encode(spectrum)}|{encode(projected)}|{encode([t2])}|{encode([q])}")
    for values in (mean, eigenvalues[:components], ratios, loadings, scores):
        assert np.isfinite(values).all()
    rows.extend(lines)


rng = np.random.Generator(np.random.PCG64(20260919))

rows.append("# Random data: PCG64 seed 20260919; square, tall and wide shapes.")
for samples, variables in ((6, 4), (20, 5), (5, 12), (40, 3)):
    data = rng.standard_normal((samples, variables))
    for components in (1, min(samples - 1, variables) // 2 or 1, min(samples - 1, variables)):
        add_case(data, components, [data[0], data[-1], rng.standard_normal(variables)])

rows.append("# Exactly determined case: three components describe the data completely.")
data = rng.standard_normal((4, 3))
add_case(data, 3, [data[1], data[2]])

rows.append("# Large offset and small magnitude, testing the internal scaling.")
data = rng.standard_normal((8, 5))
for factor, offset in ((1.0, 1e6), (1e-8, 0.0), (1e8, 0.0)):
    shifted = factor * data + offset
    add_case(shifted, 2, [shifted[0], factor * rng.standard_normal(5) + offset])

rows.append("# NIR-like absorbance, 1100-2500 nm at 14 nm: Gaussian bands (centre, FWHM in nm)")
rows.append("# with sample-specific concentrations, multiplicative scatter and an offset,")
rows.append("# followed by SNV. The extra spectra add an unmodelled band and an extreme")
rows.append("# concentration.")
wavelengths = np.arange(1100.0, 2501.0, 14.0)
assert len(wavelengths) == 101
bands = ((1210.0, 60.0), (1450.0, 90.0), (1730.0, 50.0), (1940.0, 110.0))


def gaussian(centre, fwhm):
    return np.exp(-4.0 * np.log(2.0) * ((wavelengths - centre) / fwhm) ** 2)


def snv(spectra):
    spectra = np.atleast_2d(spectra)
    return (spectra - spectra.mean(axis=1, keepdims=True)) / spectra.std(axis=1, ddof=1, keepdims=True)


profiles = np.array([gaussian(centre, fwhm) for centre, fwhm in bands])
concentrations = rng.uniform(0.1, 0.9, (16, len(bands)))
raw = (rng.uniform(0.7, 1.4, (16, 1)) * (concentrations @ profiles)
       + rng.uniform(0.1, 0.6, (16, 1))
       + 1e-4 * rng.standard_normal((16, len(wavelengths))))
spectra = snv(raw)
outlier = snv(0.9 * (concentrations[0] @ profiles) + 0.4 * gaussian(2300.0, 40.0) + 0.3)[0]
extreme = snv(1.1 * (np.array([2.5, 0.1, 0.1, 0.1]) @ profiles) + 0.2)[0]
for components in (1, 2, 3):
    add_case(spectra, components, [spectra[0], outlier, extreme])

Path(__file__).with_name("numpy_pca.txt").write_text("\n".join(rows) + "\n")
