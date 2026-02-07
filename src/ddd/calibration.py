from __future__ import annotations

import cv2
import numpy as np

from ddd.config import Settings
from ddd.db import MongoRepository
from ddd.distraction import FaceAnalyzer


def calibrate_driver(settings: Settings, driver_id: str) -> None:
    repo = MongoRepository(settings.mongo_uri, settings.mongo_db)
    driver = repo.get_driver(driver_id)
    if not driver:
        raise ValueError(f"Unknown driver_id: {driver_id}")

    analyzer = FaceAnalyzer()
    cap = cv2.VideoCapture(settings.camera_index)
    ear_vals: list[float] = []
    mar_vals: list[float] = []

    print("Calibration started. Keep face centered and eyes open. Press q to cancel.")
    while len(ear_vals) < settings.calibration_frames:
        ok, frame = cap.read()
        if not ok:
            break

        metrics = analyzer.analyze(frame)
        if metrics:
            ear_vals.append(metrics.ear)
            mar_vals.append(metrics.mar)

        cv2.putText(
            frame,
            f"Collecting calibration {len(ear_vals)}/{settings.calibration_frames}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not ear_vals:
        raise RuntimeError("Calibration failed: no face landmarks collected.")

    ear_mean, ear_std = float(np.mean(ear_vals)), float(np.std(ear_vals))
    mar_mean, mar_std = float(np.mean(mar_vals)), float(np.std(mar_vals))

    ear_threshold = max(0.12, ear_mean - 1.5 * ear_std)
    mar_threshold = min(1.2, mar_mean + 1.5 * mar_std)
    repo.update_thresholds(driver_id, ear_threshold=ear_threshold, mar_threshold=mar_threshold)

    print(
        f"Calibration complete. EAR threshold={ear_threshold:.3f}, "
        f"MAR threshold={mar_threshold:.3f}"
    )
