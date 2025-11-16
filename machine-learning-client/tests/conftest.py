# pylint: skip-file
"""Pytest fixtures for machine-learning-client tests."""

import numpy as np
import pytest


@pytest.fixture
def fake_audio():
    """Return simple fake audio waveform and sample rate."""
    return np.ones(22050), 22050


