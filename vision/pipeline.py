"""AAC vision pipeline — webcam.

Produces one row per second to ``vision.csv`` per the aacdata schema:

| column | meaning |
|---|---|
| ts_ms | PC clock, epoch ms |
| face_present | 1 if a face was detected this second |
| head_yaw / head_pitch | head rotation in deg (neutral = 0 after calibration) |
| eye_closure | 0 = open, 1 = closed (relative to calibrated open-eye baseline) |
| conf | detection confidence 0..1 |

Privacy by construction: only numbers are saved — **no video, no frames**.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from aacdata.columns import VISION

HEADER = list(VISION.column_names)


@dataclass(frozen=True)
class FrameResult:
    """Result of analysing a single camera frame."""

    face_present: bool
    head_yaw: float | None
    head_pitch: float | None
    eye_closure: float | None
    conf: float | None


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame) -> FrameResult:
        """Analyse one BGR frame and return a FrameResult."""
        ...


@runtime_checkable
class Camera(Protocol):
    def read(self):
        """Return one BGR frame, or None if none is available."""
        ...


class NullDetector:
    """No face ever detected. Useful when no camera/detector is available."""

    def detect(self, frame) -> FrameResult:
        return FrameResult(False, None, None, None, None)


class SyntheticDetector:
    """Scripted detections for tests and demos."""

    def __init__(self, results):
        self.results = list(results)
        self.i = 0

    def detect(self, frame) -> FrameResult:
        r = self.results[min(self.i, len(self.results) - 1)]
        self.i += 1
        return r


class SyntheticCamera:
    """Returns dummy frames (never None) so the pipeline runs without a camera."""

    def read(self):
        return object()  # content irrelevant for a SyntheticDetector


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass
class CalibrationResult:
    neutral_yaw: float
    neutral_pitch: float
    neutral_ear: float
    duration_s: float


class Calibration:
    """10-second neutral-face calibration at the start of a session.

    Asks the user to look straight ahead with eyes open while we record the
    baseline head pose and eye aspect ratio. These baselines neutralise the
    per-person anatomy so yaw/pitch/eye_closure are comparable across users.
    """

    def __init__(
        self,
        detector: Detector,
        camera: Camera,
        duration_s: float = 10.0,
        clock=time.time,
        sleep=time.sleep,
    ):
        self.detector = detector
        self.camera = camera
        self.duration_s = duration_s
        self.clock = clock
        self.sleep = sleep

    def run(self) -> CalibrationResult:
        yaws, pitches, ears = [], [], []
        start = self.clock()
        while self.clock() - start < self.duration_s:
            frame = self.camera.read()
            if frame is None:
                self.sleep(0.03)
                continue
            res = self.detector.detect(frame)
            if res.face_present and res.head_yaw is not None and res.eye_closure is not None:
                yaws.append(res.head_yaw)
                pitches.append(res.head_pitch)
                # eye_closure here is raw (0..1); invert to an openness proxy
                ears.append(1.0 - res.eye_closure)
            self.sleep(0.03)
        if not yaws:
            # No face seen during calibration: neutral defaults.
            return CalibrationResult(0.0, 0.0, 0.5, self.duration_s)
        return CalibrationResult(
            neutral_yaw=sum(yaws) / len(yaws),
            neutral_pitch=sum(pitches) / len(pitches),
            neutral_ear=sum(ears) / len(ears),
            duration_s=self.duration_s,
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class VisionPipeline:
    def __init__(
        self,
        detector: Detector,
        writer,
        camera: Camera,
        calibration: CalibrationResult | None = None,
        clock=time.time,
        sleep=time.sleep,
        emit_hz: float = 1.0,
    ):
        self.detector = detector
        self.writer = writer
        self.camera = camera
        self.calibration = calibration or CalibrationResult(0.0, 0.0, 0.5, 0.0)
        self.clock = clock
        self.sleep = sleep
        self.emit_hz = emit_hz
        self.rows = 0
        self._stop = False
        self._latest: FrameResult | None = None

    def stop(self) -> None:
        self._stop = True

    def _process_frame(self, frame) -> FrameResult:
        res = self.detector.detect(frame)
        if not res.face_present:
            return FrameResult(False, None, None, None, None)
        neu = self.calibration
        yaw = 0.0 if res.head_yaw is None else res.head_yaw - neu.neutral_yaw
        pitch = 0.0 if res.head_pitch is None else res.head_pitch - neu.neutral_pitch
        closure = 0.5 if res.eye_closure is None else clamp(res.eye_closure, 0.0, 1.0)
        return FrameResult(True, yaw, pitch, closure, res.conf if res.conf is not None else 1.0)

    def _emit(self, ts_ms: int) -> None:
        res = self._latest or FrameResult(False, None, None, None, None)
        self.writer.write(
            {
                "ts_ms": ts_ms,
                "face_present": 1 if res.face_present else 0,
                "head_yaw": "" if res.head_yaw is None else round(res.head_yaw, 2),
                "head_pitch": "" if res.head_pitch is None else round(res.head_pitch, 2),
                "eye_closure": "" if res.eye_closure is None else round(res.eye_closure, 4),
                "conf": "" if res.conf is None else round(res.conf, 3),
            }
        )
        self.rows += 1
        self._latest = None

    def run(self, duration_s: float | None = None, max_rows: int | None = None) -> int:
        """Run the pipeline.

        The first ``calibration.duration_s`` seconds are assumed to have been
        calibrated before this call (see :meth:`run_with_calibration`). Loops:
        read frame -> detect -> buffer latest -> every 1 s write one row.
        """
        start = self.clock()
        next_emit = start + 1.0 / self.emit_hz
        while not self._stop:
            frame = self.camera.read()
            if frame is not None:
                self._latest = self._process_frame(frame)
            now = self.clock()
            if now >= next_emit:
                self._emit(int(now * 1000))
                next_emit += 1.0 / self.emit_hz
            self.sleep(0.01)
            if max_rows is not None and self.rows >= max_rows:
                break
            if duration_s is not None and now - start >= duration_s:
                break
        return self.rows

    def run_with_calibration(self, duration_s: float | None = None) -> CalibrationResult:
        """Calibrate for 10 s, then run the pipeline."""
        cal = Calibration(self.detector, self.camera, clock=self.clock, sleep=self.sleep).run()
        self.calibration = cal
        self.run(duration_s=duration_s)
        return cal


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------
class VisionCsvWriter:
    def __init__(self, path):
        from pathlib import Path

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8", newline="")
        import csv

        self._writer = csv.writer(self._fh)
        self._writer.writerow(HEADER)
        self._fh.flush()
        self.count = 0

    def write(self, row: dict) -> None:
        import csv

        self._writer.writerow([row.get(name, "") for name in HEADER])
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
