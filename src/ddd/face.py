from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import face_recognition
import numpy as np


def extract_embedding(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    rgb = frame_bgr[:, :, ::-1]
    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        return None
    encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)
    return encodings[0] if encodings else None


def identify_driver(
    embedding: np.ndarray,
    known_drivers: list[Dict[str, Any]],
    distance_threshold: float,
) -> Tuple[Optional[Dict[str, Any]], float]:
    if not known_drivers:
        return None, 999.0

    known_embeddings = np.array([d["embedding"] for d in known_drivers], dtype=np.float32)
    distances = np.linalg.norm(known_embeddings - embedding.astype(np.float32), axis=1)
    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])

    if best_dist <= distance_threshold:
        return known_drivers[best_idx], best_dist
    return None, best_dist
