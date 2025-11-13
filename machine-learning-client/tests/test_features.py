import numpy as np
from app.features import cosine_sim, extract_features


def test_cosine_sim_basic():
    """cosine_sim should return 1.0 for identical vectors."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    sim = cosine_sim(a, b)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_sim_orthogonal():
    """cosine_sim should return 0 for orthogonal vectors."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    sim = cosine_sim(a, b)
    assert abs(sim - 0.0) < 1e-6


def test_extract_features_shape():
    """extract_features should produce a fixed-length numeric vector."""
    y = np.zeros(22050)  # 1 second of silence
    sr = 22050

    vec = extract_features(y, sr)

    # Should be a NumPy array
    assert isinstance(vec, np.ndarray)

    # Should have finite length — 26 or 34 depending on MFCC setup
    assert vec.ndim == 1
    assert len(vec) in (26, 34)

    # Should contain real numbers (not NaN)
    assert np.all(np.isfinite(vec))
