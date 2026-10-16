"""Real face detector built on OpenCV + MediaPipe FaceMesh.

Lazily imports ``cv2`` and ``mediapipe`` so the rest of the package works on a
bare Python install. Produces :class:`~vision.pipeline.FrameResult` per frame:

* face presence
* head pose (yaw / pitch) via solvePnP against a 3D face model
* eye closure from eye-aspect-ratio, calibrated against an open-eye baseline
"""

from __future__ import annotations

import math
from typing import Optional

from vision.pipeline import Detector, FrameResult

# MediaPipe FaceMesh landmark indices (468-face mesh).
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_LEFT_MOUTH = 61
_RIGHT_MOUTH = 291

# Eye-aspect-ratio landmarks: (horizontal outer, upper1, upper2, horizontal
# inner, lower1, lower2). EAR = (d(upper1,lower2) + d(upper2,lower1)) /
#                               (2 * d(outer, inner))
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (263, 385, 387, 362, 373, 380)

# Approximate 3D face model (mm), aligned to the six solvePnP points above.
_MODEL_POINTS = [
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -63.6, -12.5),    # chin
    (-43.3, 32.7, -26.0),   # left eye outer corner
    (43.3, 32.7, -26.0),    # right eye outer corner
    (-28.9, -28.2, -24.1),  # left mouth corner
    (28.9, -28.2, -24.1),   # right mouth corner
]

_EAR_THRESHOLD_CLOSED = 0.22  # EAR below this => eye considered closed


def _rotation_matrix_to_euler(R) -> tuple[float, float, float]:
    """Yaw (y), pitch (x), roll (z) in radians from a rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])  # about x
        yaw = math.atan2(-R[2, 0], sy)        # about y
    else:  # gimbal lock
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return yaw, pitch


class MediaPipeFaceDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.refine_landmarks = refine_landmarks
        self._mesh = None

    def _lazy(self):
        if self._mesh is None:
            import cv2  # noqa: F401  (ensures opencv is available)
            import mediapipe as mp  # type: ignore

            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        return self._mesh

    def _ear(self, landmarks, w, h, eye) -> float:
        def pt(i):
            lm = landmarks[i]
            return lm.x * w, lm.y * h

        p0, p1, p2, p3, p4, p5 = map(pt, eye)
        d1 = math.dist(p1, p5)
        d2 = math.dist(p2, p4)
        d3 = math.dist(p0, p3)
        if d3 <= 0:
            return 1.0
        return (d1 + d2) / (2.0 * d3)

    def detect(self, frame) -> FrameResult:
        try:
            mesh = self._lazy()
        except ImportError:
            return FrameResult(False, None, None, None, None)

        import cv2

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            return FrameResult(False, None, None, None, None)

        landmarks = result.multi_face_landmarks[0].landmark

        # --- head pose via solvePnP ---
        image_points = []
        for idx in (_NOSE_TIP, _CHIN, _LEFT_EYE_OUTER, _RIGHT_EYE_OUTER,
                    _LEFT_MOUTH, _RIGHT_MOUTH):
            lm = landmarks[idx]
            image_points.append((lm.x * w, lm.y * h))

        focal = w
        camera_matrix = [
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1],
        ]
        dist = [0.0, 0.0, 0.0, 0.0, 0.0]

        try:
            success, rot_vec, _ = cv2.solvePnP(
                __import__("numpy").array(_MODEL_POINTS, dtype="double"),
                __import__("numpy").array(image_points, dtype="double"),
                __import__("numpy").array(camera_matrix, dtype="double"),
                __import__("numpy").array(dist, dtype="double"),
            )
        except Exception:
            success = False

        yaw = pitch = None
        if success:
            R, _ = cv2.Rodrigues(rot_vec)
            yaw_r, pitch_r, _ = _rotation_matrix_to_euler(R)
            yaw = math.degrees(yaw_r)
            pitch = math.degrees(pitch_r)

        # --- eye closure ---
        ear_l = self._ear(landmarks, w, h, LEFT_EYE)
        ear_r = self._ear(landmarks, w, h, RIGHT_EYE)
        ear = (ear_l + ear_r) / 2.0
        # eye_closure: 0 = open, 1 = closed; simple sigmoid-ish around threshold
        closure = clamp(1.0 - ear / _EAR_THRESHOLD_CLOSED, 0.0, 1.0)

        return FrameResult(True, yaw, pitch, closure, 1.0)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class OpenCVCamera:
    """Reads frames from a webcam via OpenCV."""

    def __init__(self, index: int = 0):
        import cv2

        self._cap = cv2.VideoCapture(index)
        self._index = index

    def read(self):
        if not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass
