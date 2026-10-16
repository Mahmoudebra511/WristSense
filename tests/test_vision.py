"""Vision pipeline tests. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aacdata.columns import VISION  # noqa: E402
from aacdata.session import Session  # noqa: E402
from vision.detector import clamp, MediaPipeFaceDetector  # noqa: E402
from vision.pipeline import (  # noqa: E402
    Calibration,
    CalibrationResult,
    FrameResult,
    NullDetector,
    SyntheticCamera,
    SyntheticDetector,
    VisionCsvWriter,
    VisionPipeline,
)

from tests.test_logger import StepClock  # noqa: E402


class _ListWriter:
    def __init__(self):
        self.rows = []
        self.count = 0

    def write(self, row: dict) -> None:
        self.rows.append(dict(row))
        self.count += 1

    def close(self) -> None:
        pass


class TestFrameResult(unittest.TestCase):
    def test_defaults(self):
        r = FrameResult(False, None, None, None, None)
        self.assertFalse(r.face_present)
        self.assertIsNone(r.head_yaw)


class TestClamp(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(clamp(-5, 0, 1), 0)
        self.assertEqual(clamp(0.5, 0, 1), 0.5)
        self.assertEqual(clamp(2, 0, 1), 1)


class TestCalibration(unittest.TestCase):
    def test_neutral_baselines(self):
        clock = StepClock(start=0.0, step=0.5)
        det = SyntheticDetector(
            [FrameResult(True, 10.0, 5.0, 0.1, 1.0)] * 4
        )  # eye_closure 0.1 -> ear 0.9
        cam = SyntheticCamera()
        cal = Calibration(det, cam, duration_s=1.5, clock=clock, sleep=lambda _s: None).run()
        self.assertAlmostEqual(cal.neutral_yaw, 10.0)
        self.assertAlmostEqual(cal.neutral_pitch, 5.0)
        self.assertAlmostEqual(cal.neutral_ear, 0.9)

    def test_no_face_defaults(self):
        clock = StepClock(start=0.0, step=0.5)
        det = SyntheticDetector([FrameResult(False, None, None, None, None)])
        cam = SyntheticCamera()
        cal = Calibration(det, cam, duration_s=0.6, clock=clock, sleep=lambda _s: None).run()
        self.assertEqual(cal.neutral_yaw, 0.0)
        self.assertEqual(cal.neutral_pitch, 0.0)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.clock = StepClock(start=1_000.0, step=1.0)
        self.writer = _ListWriter()

    def _pipeline(self, results, calibration=None):
        return VisionPipeline(
            SyntheticDetector(results),
            self.writer,
            SyntheticCamera(),
            calibration=calibration,
            clock=self.clock,
            sleep=lambda _s: None,
            emit_hz=1.0,
        )

    def test_rows_match_schema(self):
        results = [FrameResult(True, 2.0, -3.0, 0.1, 1.0)] * 3
        pipe = self._pipeline(results)
        n = pipe.run(max_rows=3)
        self.assertEqual(n, 3)
        self.assertEqual(self.writer.count, 3)
        for row in self.writer.rows:
            self.assertEqual(set(row), set(VISION.column_names))
            self.assertEqual(row["face_present"], 1)
            self.assertAlmostEqual(row["head_yaw"], 2.0)
            self.assertAlmostEqual(row["head_pitch"], -3.0)
            self.assertAlmostEqual(row["eye_closure"], 0.1)
            self.assertAlmostEqual(row["conf"], 1.0)

    def test_no_face_writes_empty_angles(self):
        results = [FrameResult(False, None, None, None, None)] * 2
        pipe = self._pipeline(results)
        pipe.run(max_rows=2)
        self.assertEqual(self.writer.rows[0]["face_present"], 0)
        self.assertEqual(self.writer.rows[0]["head_yaw"], "")
        self.assertEqual(self.writer.rows[0]["eye_closure"], "")

    def test_ts_advance_by_one_second(self):
        pipe = self._pipeline([FrameResult(True, 0.0, 0.0, 0.0, 1.0)] * 3)
        pipe.run(max_rows=3)
        self.assertEqual(self.writer.rows[1]["ts_ms"] - self.writer.rows[0]["ts_ms"], 1000)

    def test_calibration_neutralises_head_pose(self):
        cal = CalibrationResult(neutral_yaw=10.0, neutral_pitch=5.0, neutral_ear=0.8, duration_s=10.0)
        results = [FrameResult(True, 15.0, 10.0, 0.1, 1.0)]  # ear=0.9
        pipe = self._pipeline(results, calibration=cal)
        pipe.run(max_rows=1)
        row = self.writer.rows[0]
        self.assertAlmostEqual(row["head_yaw"], 5.0)  # 15 - 10
        self.assertAlmostEqual(row["head_pitch"], 5.0)  # 10 - 5


class TestSessionIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.session = Session.at(Path(self._tmp.name) / "p00_20261016T090000_laptop")

    def tearDown(self):
        self._tmp.cleanup()

    def _write_vision_csv(self, rows: int = 3):
        start_ms = 1_792_141_200_000
        with VisionCsvWriter(self.session.path / VISION.filename) as writer:
            for i in range(rows):
                writer.write({
                    "ts_ms": start_ms + i * 1000,
                    "face_present": 1 if i % 2 == 0 else 0,
                    "head_yaw": 1.5 if i % 2 == 0 else "",
                    "head_pitch": -2.0 if i % 2 == 0 else "",
                    "eye_closure": 0.1 if i % 2 == 0 else "",
                    "conf": 1.0 if i % 2 == 0 else "",
                })

    def _declare_vision(self, partial: bool):
        meta = self.session.load_meta()
        meta.streams = {"vision": "vision.csv"}
        meta.sampling_rate_hz = {"vision": 1.0}
        meta.row_counts = {"vision": 3}
        meta.partial = partial
        self.session.write_meta(meta)

    def test_vision_only_partial_session_is_valid(self):
        self._write_vision_csv()
        self._declare_vision(partial=True)
        report = self.session.validate()
        self.assertTrue(report.ok, report.format())

    def test_vision_only_full_session_errors(self):
        self._write_vision_csv()
        self._declare_vision(partial=False)
        report = self.session.validate()
        self.assertFalse(report.ok)
        self.assertIn("meta.streams", {i.code for i in report.errors})


if __name__ == "__main__":
    unittest.main()
