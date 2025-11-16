# pylint: skip-file
"""
Tests for worker task processing.
"""

import numpy as np
from unittest.mock import MagicMock, patch
from app.worker import process_one


@patch("app.worker.extract_features", return_value=np.ones(26))
@patch("app.worker._read_gridfs_audio", return_value=(np.ones(22050), 22050))
def test_process_one_with_task(mock_read, mock_extract):
    db = MagicMock()
    db.tasks.find_one_and_update.return_value = {
        "_id": "task1",
        "gridfs_id": "audio123",
        "status": "pending",
    }

    db.reference_tracks.find.return_value = [
        {"genre": "rock", "fp": np.ones(26).tolist()}
    ]

    db.results.insert_one.return_value = True
    fs = MagicMock()

    result = process_one(db, fs)
    assert result is True
    db.results.insert_one.assert_called_once()
    db.tasks.update_one.assert_called()


def test_process_one_no_task():
    db = MagicMock()
    db.tasks.find_one_and_update.return_value = None
    fs = MagicMock()

    result = process_one(db, fs)
    assert result is False

