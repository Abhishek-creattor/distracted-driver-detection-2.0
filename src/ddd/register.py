from __future__ import annotations

from datetime import datetime, timezone

import cv2

from ddd.config import Settings
from ddd.db import MongoRepository
from ddd.face import extract_embedding


def register_driver(settings: Settings, driver_id: str, name: str) -> None:
    repo = MongoRepository(settings.mongo_uri, settings.mongo_db)
    cap = cv2.VideoCapture(settings.camera_index)

    print("Press 'c' to capture face and register. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imshow("Register Driver", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            embedding = extract_embedding(frame)
            if embedding is None:
                print("No face detected. Try again.")
                continue

            repo.upsert_driver(
                {
                    "driver_id": driver_id,
                    "name": name,
                    "embedding": embedding.tolist(),
                    "thresholds": {"ear": 0.22, "mar": 0.70},
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            print(f"Driver {name} ({driver_id}) registered.")
            break

    cap.release()
    cv2.destroyAllWindows()
