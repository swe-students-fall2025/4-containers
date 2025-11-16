# pylint: skip-file
"""
Tests for reference seeding functionality.
"""

import numpy as np
from unittest.mock import patch, MagicMock
from app.seed_reference import seed_reference_tracks


@patch("time.time", return_value=12345)
@patch("app.seed_reference.extract_features_audio", return_value=np.ones(8))
def test_seed_reference_tracks_basic(mock_extract, mock_time):
    db = MagicMock()
    fs = MagicMock()

    seed_reference_tracks(db, fs)

    assert db.reference_tracks.insert_one.called
    call_args = db.reference_tracks.insert_one.call_args[0][0]
    assert "genre" in call_args
    assert "fp" in call_args
    assert len(call_args["fp"]) == 8
