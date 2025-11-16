# pylint: skip-file
"""Tests for worker.process_one."""

from unittest.mock import MagicMock, patch
import numpy as np
from app.worker import process_one


@patch("app.worker.extract_features")
@patch("app.worker.read_audio_file")
def test_process_one_success(mock_read_func, mock_extract_func):
    """Test that process_one processes one task and stores a result."""

    db = MagicMock()
    db.tasks.find_one_and_update.return_value = {
        "_id": "task1",
        "gridfs_id": "audio123",
        "status": "pending",
    }

    db.reference_tracks.find.return_value = [
        {"genre": "vocal", "fp": np.ones(26).tolist()}
    ]

    mock_read_func.return_value = (np.ones(22050), 22050)
    mock_extract_func.return_value = np.ones(26)

    fs = MagicMock()

    result = process_one(db, fs)
    assert result is True
    db.results.insert_one.assert_called_once()
    db.tasks.update_one.assert_called()


def test_process_one_no_task():
    """Test that process_one returns False when no pending task is available."""
    db = MagicMock()
    db.tasks.find_one_and_update.return_value = None
    fs = MagicMock()

    result = process_one(db, fs)
    assert result is False



