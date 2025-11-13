#!/usr/bin/env python3
"""
Background worker for the ML client.

Reads one pending audio classification task from MongoDB, loads the
corresponding audio bytes from GridFS, extracts features, and classifies
the track as either "vocal" or "electronic" using the reference fingerprints
stored in the database.
"""
# pylint: disable=import-error
import os
import time
from typing import Iterable, Tuple, Any

import numpy as np
from gridfs import GridFS
from pymongo import MongoClient

from app.features import extract_features


# ---------------------------------------------------------
#   RENAMED to match tests: _read_audio_from_gridfs
# ---------------------------------------------------------
def _read_audio_from_gridfs(
    gridfs_bucket: GridFS, gridfs_id: Any
) -> Tuple[np.ndarray, int]:
    """
    Read raw audio data from GridFS and decode to mono float32 numpy array.
    """
    import io
    import soundfile as sf  # local import so worker imports even w/o SF

    raw_bytes = gridfs_bucket.get(gridfs_id).read()
    with sf.SoundFile(io.BytesIO(raw_bytes)) as sound_file:
        audio = sound_file.read(dtype="float32")
        sr = sound_file.samplerate

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sr


def _predict_binary_genre(
    feature_vector: np.ndarray, references: Iterable[dict]
) -> str:
    """Predict 'vocal' or 'electronic' from reference fingerprints."""
    ref_fps: list[np.ndarray] = []
    labels: list[str] = []

    for doc in references:
        fp = np.asarray(doc.get("fp", []), dtype=float)
        if fp.size == 0:
            continue
        norm = np.linalg.norm(fp)
        if norm == 0:
            continue

        fp = fp / norm
        ref_fps.append(fp)
        labels.append(str(doc.get("genre", "")).lower())

    if not ref_fps:
        return "vocal"

    ref_matrix = np.vstack(ref_fps)
    similarities = ref_matrix @ feature_vector
    best_index = int(np.argmax(similarities))
    raw_label = labels[best_index]

    if "vocal" in raw_label or "voice" in raw_label or "singer" in raw_label:
        return "vocal"
    if "electronic" in raw_label or "edm" in raw_label or "electro" in raw_label:
        return "electronic"

    return "vocal"


def process_one(db, gridfs_bucket: GridFS) -> bool:
    """
    Process a single pending classification task.
    """
    task = db.tasks.find_one_and_update(
        {"status": "pending"},
        {"$set": {"status": "processing"}},
    )

    if task is None:
        return False

    try:
        # ---------------------------------------------------------
        #   Updated name here to match the renamed function
        # ---------------------------------------------------------
        audio, sample_rate = _read_audio_from_gridfs(gridfs_bucket, task["gridfs_id"])

        feature_vector = extract_features(audio, sample_rate)
        references = db.reference_tracks.find()
        predicted_genre = _predict_binary_genre(feature_vector, references)

        db.results.insert_one(
            {
                "task_id": task["_id"],
                "predicted_genre": predicted_genre,
                "created_at": time.time(),
            }
        )

        db.tasks.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "done", "predicted_genre": predicted_genre}},
        )
        return True

    except Exception as exc:  # pylint: disable=broad-except
        db.tasks.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error_message": str(exc)}},
        )
        return False


def process_loop(db, gridfs_bucket: GridFS, poll_interval: float = 1.0) -> None:
    """Loop forever, processing tasks."""
    while True:
        worked = process_one(db, gridfs_bucket)
        if not worked:
            time.sleep(poll_interval)


def create_mongo_client(uri: str) -> MongoClient:
    """Factory for MongoClient so tests can patch easily."""
    return MongoClient(uri)


def main() -> None:
    """Entry point."""
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")
    client = create_mongo_client(mongo_uri)
    database = client["ml_audio"]
    gridfs_bucket = GridFS(database)
    process_loop(database, gridfs_bucket)


if __name__ == "__main__":
    main()
