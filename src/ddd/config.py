from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Settings:
    mongo_uri: str
    mongo_db: str
    camera_index: int
    face_match_threshold: float
    min_eye_closed_frames: int
    head_away_angle_deg: float
    phone_confidence: float
    calibration_frames: int
    yolov8_model_path: str


def load_config(config_path: str | Path = "config/config.yaml") -> Settings:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw: Dict[str, Any] = yaml.safe_load(file)

    return Settings(
        mongo_uri=raw["mongo"]["uri"],
        mongo_db=raw["mongo"]["db"],
        camera_index=raw["camera"]["index"],
        face_match_threshold=float(raw["face_recognition"]["distance_threshold"]),
        min_eye_closed_frames=int(raw["alerts"]["min_eye_closed_frames"]),
        head_away_angle_deg=float(raw["alerts"]["head_away_angle_deg"]),
        phone_confidence=float(raw["phone_detection"]["confidence"]),
        calibration_frames=int(raw["calibration"]["frames"]),
        yolov8_model_path=raw["phone_detection"]["yolov8_model_path"],
    )
