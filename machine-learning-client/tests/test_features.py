# pylint: skip-file
"""
Tests for feature extraction utilities.
"""

import numpy as np
from app.features import extract_features, cosine_sim


def test_extract_features_basic():
    y = np.ones(22050)
    sr = 22050
    vec = extract_features(y, sr)
    assert vec.shape == (26,)
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_extract_features_zero_norm():
    y = np.zeros(22050)
    sr = 22050
    vec = extract_features(y, sr)
    assert np.all(vec == 0)


def test_cosine_similarity():
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    c = np.array([0, 1, 0])
    assert cosine_sim(a, b) == 1.0
    assert cosine_sim(a, c) == 0.0
