from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 308, 14]


@dataclass
class DistractionMetrics:
    ear: float
    mar: float
    yaw_deg: float


class FaceAnalyzer:
    def __init__(self) -> None:
        self.mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    @staticmethod
    def _eye_aspect_ratio(points: np.ndarray) -> float:
        return (euclidean(points[1], points[5]) + euclidean(points[2], points[4])) / (
            2.0 * euclidean(points[0], points[3])
        )

    @staticmethod
    def _mouth_aspect_ratio(points: np.ndarray) -> float:
        return euclidean(points[1], points[3]) / euclidean(points[0], points[2])

    def _head_yaw(self, landmarks: np.ndarray, image_shape: Tuple[int, int, int]) -> float:
        h, w = image_shape[:2]
        face_2d = np.array([
            [landmarks[1][0] * w, landmarks[1][1] * h],
            [landmarks[152][0] * w, landmarks[152][1] * h],
            [landmarks[263][0] * w, landmarks[263][1] * h],
            [landmarks[33][0] * w, landmarks[33][1] * h],
            [landmarks[61][0] * w, landmarks[61][1] * h],
            [landmarks[291][0] * w, landmarks[291][1] * h],
        ], dtype=np.float64)
        face_3d = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -63.6, -12.5],
            [43.3, 32.7, -26.0],
            [-43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1],
            [28.9, -28.9, -24.1],
        ], dtype=np.float64)

        focal = w
        cam_matrix = np.array([[focal, 0, h / 2], [0, focal, w / 2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)
        _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist)
        rot_mtx, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rot_mtx)
        return float(angles[1] * 360)

    def analyze(self, frame_bgr: np.ndarray) -> Optional[DistractionMetrics]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        all_pts = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32)

        left_eye = all_pts[LEFT_EYE]
        right_eye = all_pts[RIGHT_EYE]
        mouth = all_pts[MOUTH]

        ear = (self._eye_aspect_ratio(left_eye) + self._eye_aspect_ratio(right_eye)) / 2.0
        mar = self._mouth_aspect_ratio(mouth)
        yaw = self._head_yaw(all_pts, frame_bgr.shape)
        return DistractionMetrics(ear=ear, mar=mar, yaw_deg=yaw)
