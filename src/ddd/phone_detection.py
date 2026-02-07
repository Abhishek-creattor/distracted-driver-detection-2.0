from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # noqa: BLE001
    YOLO = None


@dataclass
class PhoneDetection:
    detected: bool
    confidence: float


class PhoneDetector:
    def __init__(self, model_path: str, confidence: float = 0.35) -> None:
        self.confidence = confidence
        self.model = None
        if YOLO and Path(model_path).exists():
            self.model = YOLO(model_path)

    def detect(self, frame_bgr: np.ndarray) -> PhoneDetection:
        if self.model is None:
            return PhoneDetection(False, 0.0)

        results = self.model.predict(source=frame_bgr, verbose=False, conf=self.confidence)
        for r in results:
            if r.boxes is None:
                continue
            for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if int(cls_id) == 67:  # COCO class id for cell phone
                    return PhoneDetection(True, float(conf))
        return PhoneDetection(False, 0.0)
