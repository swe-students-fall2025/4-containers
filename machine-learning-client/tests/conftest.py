# pylint: skip-file
"""
Shared pytest fixtures for ML client tests.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def fake_db():
    """Return a fake MongoDB-like object for tests."""
    return MagicMock()


@pytest.fixture
def fake_fs():
    """Return fake GridFS-like object."""
    return MagicMock()


@pytest.fixture
def fake_audio():
    """Return simple fake audio waveform and sample rate."""
    return np.ones(22050), 22050

