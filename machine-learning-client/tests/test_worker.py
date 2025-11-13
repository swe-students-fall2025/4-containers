import numpy as np
from unittest.mock import MagicMock, patch
import app.worker as w


@patch("app.worker.extract_features", return_value=np.ones(26))
def test_process_one_minimal(mock_extract):
    """process_one should return True when a task is handled."""

    # Fake DB
    db = MagicMock()
    db.tasks.find_one_and_update.return_value = {
        "_id": "t1",
        "gridfs_id": "fake_audio",
        "status": "pending",
    }

    # Fake reference track (vocal + electronic)
    db.reference_tracks.find.return_value = [
        {"genre": "vocal", "fp": np.ones(26).tolist()},
        {"genre": "electronic", "fp": (-np.ones(26)).tolist()},
    ]

    # Fake results insertion
    db.results.insert_one.return_value = True

    # Fake GridFS file read → fake audio bytes
    fs = MagicMock()
    fs.get.return_value.read.return_value = b"0000"

    # Fake audio load
    with patch(
        "app.worker._read_audio_from_gridfs",
        return_value=(np.ones(22050), 22050),
    ):
        worked = w.process_one(db, fs)

    assert worked is True
    db.results.insert_one.assert_called_once()
