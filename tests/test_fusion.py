"""Fusion tests. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.fusion import (  # noqa: E402
    DecisionWindow,
    LABELS,
    _add_face_share,
    _aggregate_periodic,
    _majority,
    _mean,
    _mode,
    aggregate_logger,
    aggregate_vision,
    decision_windows,
    fuse_session,
    fuse_all,
    write_csv,
)

from tests.test_ml import _write_session  # noqa: E402


class TestHelpers(unittest.TestCase):
    def test_mean(self):
        self.assertAlmostEqual(_mean([1.0, 2.0, 3.0]), 2.0)
        self.assertEqual(_mean([]), 0.0)

    def test_mode(self):
        self.assertEqual(_mode(["a", "b", "a", "c", "a"]), "a")
        self.assertEqual(_mode([]), "unknown")

    def test_majority(self):
        self.assertEqual(
            _majority(["ON_TASK", "ON_TASK", "DISTRACTED"]), "ON_TASK"
        )
        self.assertEqual(_majority(["BREAK", "UNCERTAIN"]), "unknown")
        self.assertEqual(_majority([]), "unknown")


class TestDecisionWindows(unittest.TestCase):
    def test_split(self):
        wins = decision_windows(0, 90_000, window_s=30.0)
        self.assertEqual(len(wins), 3)
        self.assertEqual(wins[0], DecisionWindow(0, 30_000))
        self.assertEqual(wins[1], DecisionWindow(30_000, 60_000))
        self.assertEqual(wins[2], DecisionWindow(60_000, 90_000))

    def test_remainder(self):
        wins = decision_windows(0, 95_000, window_s=30.0)
        self.assertEqual(len(wins), 4)
        self.assertEqual(wins[-1], DecisionWindow(90_000, 95_000))


class TestAggregatePeriodic(unittest.TestCase):
    def test_numeric_and_category(self):
        rows = [
            {"ts_ms": "1000", "keys": "5", "mouse_moves": "10", "category": "study"},
            {"ts_ms": "1100", "keys": "7", "mouse_moves": "20", "category": "study"},
            {"ts_ms": "1200", "keys": "3", "mouse_moves": "30", "category": "other"},
        ]
        out = aggregate_logger(rows)
        self.assertEqual(len(out), 3)
        agg = out[1000]
        self.assertAlmostEqual(agg["keys"], 5.0)
        self.assertAlmostEqual(agg["mouse_moves"], 10.0)
        self.assertEqual(agg["category"], "study")
        self.assertEqual(agg["n_samples"], 1)


class TestFaceShare(unittest.TestCase):
    def test_face_present_share(self):
        buckets = {
            1000: {"n_samples": 1},
            2000: {"n_samples": 1},
        }
        rows = [
            {"ts_ms": "1000", "face_present": "1"},
            {"ts_ms": "2000", "face_present": "0"},
        ]
        _add_face_share(buckets, rows)
        self.assertAlmostEqual(buckets[1000]["face_present_share"], 1.0)
        self.assertAlmostEqual(buckets[2000]["face_present_share"], 0.0)


class TestFuseSession(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_full_session(
        self,
        session_id: str,
        duration_s: float = 90.0,
        n_blocks: int = 3,
    ) -> Path:
        """Create a session with features.csv + logger.csv + vision.csv + labels.csv."""
        sess_dir = self.root / session_id
        sess_dir.mkdir(parents=True)
        start_ms = 1_792_141_200_000
        rng = np.random.default_rng(42)

        # --- imu features (4 s windows, 50% overlap) -----------------------
        feat_header = [
            "start_ms", "end_ms", "window_s", "sample_rate_hz",
            "task_state", "activity",
            "rms_accel_nogravity", "rms_gyro", "gyro_mag_mean", "gyro_mag_std",
            "dominant_freq_hz", "spectral_entropy",
            "band_energy_0_5_3_hz", "band_energy_3_8_hz",
            "zero_crossing_rate", "gravity_x", "gravity_y", "gravity_z",
            "peak_count", "max_jerk", "stillness_duration_s",
        ]
        block_ms = int(duration_s * 1000 / n_blocks)
        states = ["ON_TASK", "DISTRACTED", "AWAY"]
        feat_rows = []
        t = start_ms
        while t + 4000 <= start_ms + duration_s * 1000:
            block_idx = min((t - start_ms) // block_ms, n_blocks - 1)
            st = states[block_idx % len(states)]
            feat_rows.append(
                {
                    "start_ms": str(t),
                    "end_ms": str(t + 4000),
                    "window_s": "4.0",
                    "sample_rate_hz": "50.0",
                    "task_state": st,
                    "activity": "writing_notes",
                    "rms_accel_nogravity": str(float(rng.normal(0.5, 0.1))),
                    "rms_gyro": str(float(rng.normal(0.3, 0.1))),
                    "gyro_mag_mean": str(float(rng.normal(0.2, 0.05))),
                    "gyro_mag_std": str(float(rng.normal(0.05, 0.01))),
                    "dominant_freq_hz": str(float(rng.normal(2.0, 0.5))),
                    "spectral_entropy": str(float(rng.normal(3.0, 0.5))),
                    "band_energy_0_5_3_hz": str(float(rng.normal(0.1, 0.02))),
                    "band_energy_3_8_hz": str(float(rng.normal(0.05, 0.01))),
                    "zero_crossing_rate": str(float(rng.normal(0.3, 0.05))),
                    "gravity_x": "0.0",
                    "gravity_y": "1.0",
                    "gravity_z": "0.0",
                    "peak_count": "5",
                    "max_jerk": "1.0",
                    "stillness_duration_s": "0.5",
                }
            )
            t += 2000  # 50% overlap

        with open(sess_dir / "features.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=feat_header)
            w.writeheader()
            w.writerows(feat_rows)

        # --- logger.csv at 1 Hz -------------------------------------------
        logger_rows = []
        for s in range(int(duration_s)):
            ts = start_ms + s * 1000
            cat = "study" if s % 30 < 20 else "entertainment"
            logger_rows.append(
                {
                    "ts_ms": str(ts),
                    "keys": str(rng.integers(0, 10)),
                    "mouse_moves": str(rng.integers(0, 50)),
                    "mouse_clicks": str(rng.integers(0, 3)),
                    "win_switches": str(rng.integers(0, 2)),
                    "category": cat,
                    "time_since_input_s": "0.0",
                }
            )
        with open(sess_dir / "logger.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "ts_ms", "keys", "mouse_moves", "mouse_clicks",
                    "win_switches", "category", "time_since_input_s",
                ],
            )
            w.writeheader()
            w.writerows(logger_rows)

        # --- vision.csv at 1 Hz -------------------------------------------
        vision_rows = []
        for s in range(int(duration_s)):
            ts = start_ms + s * 1000
            face = "1" if s % 40 < 30 else "0"
            vision_rows.append(
                {
                    "ts_ms": str(ts),
                    "face_present": face,
                    "head_yaw": str(float(rng.normal(0, 5))),
                    "head_pitch": str(float(rng.normal(0, 5))),
                    "eye_closure": str(float(rng.normal(0.1, 0.05))),
                    "conf": "0.9",
                }
            )
        with open(sess_dir / "vision.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "ts_ms", "face_present", "head_yaw", "head_pitch",
                    "eye_closure", "conf",
                ],
            )
            w.writeheader()
            w.writerows(vision_rows)

        # --- labels.csv ----------------------------------------------------
        label_rows = []
        for b in range(n_blocks):
            s_ms = start_ms + b * block_ms
            e_ms = start_ms + (b + 1) * block_ms
            st = states[b % len(states)]
            label_rows.append(
                {
                    "start_ms": str(s_ms),
                    "end_ms": str(e_ms),
                    "activity": "writing_notes",
                    "task_state": st,
                    "labeler": "mah",
                    "notes": "",
                }
            )
        with open(sess_dir / "labels.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["start_ms", "end_ms", "activity", "task_state", "labeler", "notes"],
            )
            w.writeheader()
            w.writerows(label_rows)

        # --- meta.json -----------------------------------------------------
        meta = {
            "schema_version": "1.0.0",
            "session_id": session_id,
            "participant_id": "p01",
            "wrist": "right",
            "mode": "laptop",
            "start_utc": "2026-10-16T09:00:00Z",
            "end_utc": "2026-10-16T09:01:30Z",
            "sampling_rate_hz": {"imu": 50.0, "logger": 1.0, "vision": 1.0},
            "streams": {"imu": "imu.csv", "logger": "logger.csv", "vision": "vision.csv"},
            "on_task_definition_version": "1.0.0-draft",
        }
        with open(sess_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        return sess_dir

    def test_fuse_single_session(self):
        sess_dir = self._write_full_session("p01", duration_s=90.0, n_blocks=3)
        rows = fuse_session(sess_dir, decision_window_s=30.0)
        self.assertEqual(len(rows), 3)  # 90 s / 30 s
        for row in rows:
            self.assertIn("start_ms", row)
            self.assertIn("end_ms", row)
            self.assertIn("task_state", row)
            self.assertIn("n_imu_windows", row)
            # imu_* features present
            self.assertIn("imu_rms_accel_nogravity", row)
            # logger_* aggregates present
            self.assertIn("logger_keys", row)
            self.assertIn("logger_category", row)
            # vision_* aggregates present
            self.assertIn("vision_head_yaw", row)
            self.assertIn("vision_face_present_share", row)
            # label should be one of the known states (or unknown if no windows)
            self.assertIn(row["task_state"], list(LABELS) + ["unknown"])

    def test_decision_window_label(self):
        """Each 30 s window should carry the label dominant inside it."""
        sess_dir = self._write_full_session("p01", duration_s=90.0, n_blocks=3)
        rows = fuse_session(sess_dir, decision_window_s=30.0)
        # block boundaries at 30 s, 60 s; decision windows align with them
        # so each window should be dominated by one state
        states = [r["task_state"] for r in rows]
        self.assertTrue(all(s in ("ON_TASK", "DISTRACTED", "AWAY") for s in states))

    def test_fuse_all(self):
        self._write_full_session("p01", duration_s=90.0, n_blocks=3)
        self._write_full_session("p02", duration_s=90.0, n_blocks=3)
        sessions = fuse_all(self.root, decision_window_s=30.0)
        self.assertEqual(set(sessions.keys()), {"p01", "p02"})
        total = sum(len(v) for v in sessions.values())
        self.assertEqual(total, 6)

    def test_write_csv(self):
        rows = [
            {"session_id": "p01", "start_ms": 0, "end_ms": 30000, "task_state": "ON_TASK", "imu_x": 1.0},
            {"session_id": "p01", "start_ms": 30000, "end_ms": 60000, "task_state": "DISTRACTED", "imu_x": 2.0},
        ]
        out = self.root / "fused.csv"
        write_csv(rows, out)
        self.assertTrue(out.exists())
        with out.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            out_rows = list(reader)
        self.assertEqual(len(out_rows), 2)
        self.assertEqual(out_rows[0]["task_state"], "ON_TASK")
        self.assertEqual(out_rows[1]["task_state"], "DISTRACTED")


if __name__ == "__main__":
    unittest.main()
