from __future__ import annotations

from collections import defaultdict

import cv2

from ddd.config import Settings
from ddd.db import MongoRepository
from ddd.distraction import FaceAnalyzer
from ddd.face import extract_embedding, identify_driver
from ddd.phone_detection import PhoneDetector


def monitor_driver(settings: Settings) -> None:
    repo = MongoRepository(settings.mongo_uri, settings.mongo_db)
    analyzer = FaceAnalyzer()
    phone_detector = PhoneDetector(settings.yolov8_model_path, settings.phone_confidence)

    cap = cv2.VideoCapture(settings.camera_index)
    eye_closed_counter = defaultdict(int)

    print("Monitoring started. Press q to exit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        alert_text = []
        embedding = extract_embedding(frame)
        active_driver = None

        if embedding is not None:
            drivers = repo.get_all_drivers()
            active_driver, distance = identify_driver(
                embedding,
                known_drivers=drivers,
                distance_threshold=settings.face_match_threshold,
            )
            if active_driver:
                cv2.putText(
                    frame,
                    f"Driver: {active_driver['name']} ({distance:.2f})",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        metrics = analyzer.analyze(frame)
        if active_driver and metrics:
            thresholds = active_driver.get("thresholds", {"ear": 0.22, "mar": 0.7})
            d_id = active_driver["driver_id"]

            if metrics.ear < thresholds["ear"]:
                eye_closed_counter[d_id] += 1
            else:
                eye_closed_counter[d_id] = 0

            if eye_closed_counter[d_id] >= settings.min_eye_closed_frames:
                alert_text.append("DROWSINESS")
                repo.log_event({"driver_id": d_id, "event_type": "drowsiness", "score": metrics.ear})

            if abs(metrics.yaw_deg) > settings.head_away_angle_deg:
                alert_text.append("LOOKING AWAY")
                repo.log_event({"driver_id": d_id, "event_type": "head_away", "score": metrics.yaw_deg})

            if metrics.mar > thresholds["mar"]:
                alert_text.append("YAWNING")
                repo.log_event({"driver_id": d_id, "event_type": "yawning", "score": metrics.mar})

            phone = phone_detector.detect(frame)
            if phone.detected:
                alert_text.append("PHONE USAGE")
                repo.log_event({"driver_id": d_id, "event_type": "phone_usage", "score": phone.confidence})

            cv2.putText(
                frame,
                f"EAR:{metrics.ear:.2f} MAR:{metrics.mar:.2f} YAW:{metrics.yaw_deg:.1f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

        if alert_text:
            cv2.putText(
                frame,
                " | ".join(alert_text),
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
            )
            print("ALERT:", ", ".join(alert_text))

        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
