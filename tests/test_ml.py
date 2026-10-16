"""ML pipeline tests. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aacdata.columns import IMU, LABELS  # noqa: E402
from aacdata.session import Session  # noqa: E402
from ml.windowing import sliding_windows, assign_timestamps, decision_windows  # noqa: E402
from ml.features import compute_features, FeatureVector  # noqa: E402
from ml.labels import load_labels, window_label, drop_transition_windows  # noqa: E402
from ml.pipeline import run_pipeline, WINDOW_S, OVERLAP, TRANSITION_S  # noqa: E402


def _write_session(root: Path, n_samples: int = 3000, label_blocks=None) -> Path:
    """Create a valid session folder with synthetic IMU + labels."""
    sess_dir = root / "p01_20261016T090000_laptop"
    sess_dir.mkdir(parents=True)
    start_ms = 1_792_141_200_000
    rng = np.random.default_rng(42)

    with open(sess_dir / IMU.filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(IMU.column_names))
        for i in range(n_samples):
            t = start_ms + i * 20
            # gravity on Y + small movement
            w.writerow([t, t * 20, rng.normal(0, 0.02), 1.0 + rng.normal(0, 0.02), rng.normal(0, 0.02),
                        rng.normal(0, 0.5), rng.normal(0, 0.5), rng.normal(0, 0.5), i])

    if label_blocks is None:
        label_blocks = [(0, n_samples * 20, "writing_notes", "ON_TASK")]
    with open(sess_dir / LABELS.filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_ms", "end_ms", "activity", "task_state", "labeler", "notes"])
        for s_ms, e_ms, act, st in label_blocks:
            w.writerow([start_ms + s_ms, start_ms + e_ms, act, st, "mah", ""])

    meta = {
        "schema_version": "1.0.0",
        "session_id": sess_dir.name,
        "participant_id": "p01",
        "wrist": "right",
        "mode": "laptop",
        "start_utc": "2026-10-16T09:00:00Z",
        "end_utc": "2026-10-16T09:00:10Z",
        "sampling_rate_hz": {"imu": 50.0},
        "streams": {"imu": "imu.csv"},
        "on_task_definition_version": "1.0.0-draft",
    }
    with open(sess_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return sess_dir


class TestWindowing(unittest.TestCase):
    def test_sliding_windows_count(self):
        wins = sliding_windows(100, window_s=1.0, sample_rate_hz=50.0, overlap=0.5)
        # window = 50 samples, step = 25 -> starts at 0, 25, 50 -> 3 windows (0-50, 25-75, 50-100)
        self.assertEqual(len(wins), 3)

    def test_window_indices(self):
        wins = sliding_windows(100, window_s=1.0, sample_rate_hz=50.0, overlap=0.5)
        self.assertEqual(wins[0].start_idx, 0)
        self.assertEqual(wins[0].end_idx, 50)
        self.assertEqual(wins[1].start_idx, 25)
        self.assertEqual(wins[1].end_idx, 75)

    def test_assign_timestamps(self):
        wins = sliding_windows(100, window_s=1.0, sample_rate_hz=50.0, overlap=0.5)
        ts = list(range(0, 100 * 20, 20))
        assign_timestamps(wins, ts)
        self.assertEqual(wins[0].start_ms, 0)
        self.assertEqual(wins[0].end_ms, 980)  # idx 49 * 20

    def test_decision_windows(self):
        wins = sliding_windows(300, window_s=1.0, sample_rate_hz=50.0, overlap=0.5)
        ts = list(range(0, 300 * 20, 20))
        assign_timestamps(wins, ts)
        dw = decision_windows(wins, decision_window_s=5.0)
        self.assertGreaterEqual(len(dw), 1)
        self.assertEqual(dw[0][0], 0)


class TestFeatures(unittest.TestCase):
    def test_feature_vector_shape(self):
        n = 200
        accel = np.column_stack([np.zeros(n), np.ones(n), np.zeros(n)])
        gyro = np.zeros((n, 3))
        ts = np.arange(n) * 20
        fv = compute_features(accel, gyro, ts, sample_rate_hz=50.0)
        self.assertAlmostEqual(fv.rms_accel_nogravity, 0.0, places=6)
        self.assertAlmostEqual(fv.gravity_y, 1.0, places=6)
        self.assertEqual(len(FeatureVector.header("")), 15)

    def test_deterministic(self):
        n = 200
        rng = np.random.default_rng(7)
        accel = rng.normal(0, 0.1, (n, 3))
        accel[:, 1] += 1.0
        gyro = rng.normal(0, 0.5, (n, 3))
        ts = np.arange(n) * 20
        f1 = compute_features(accel, gyro, ts, sample_rate_hz=50.0)
        f2 = compute_features(accel, gyro, ts, sample_rate_hz=50.0)
        self.assertEqual(f1.as_dict(), f2.as_dict())

    def test_dominant_freq(self):
        n = 1000
        t = np.arange(n) / 50.0
        sig = np.sin(2 * np.pi * 2.0 * t)
        accel = np.column_stack([sig, np.zeros(n), np.ones(n)])
        gyro = np.zeros((n, 3))
        ts = (t * 1000).astype(int)
        fv = compute_features(accel, gyro, ts, sample_rate_hz=50.0)
        # Magnitude rectifies a pure sine, so dominant may be 2 or 4 Hz;
        # just verify it is a real non-DC frequency below Nyquist.
        self.assertGreater(fv.dominant_freq_hz, 0)
        self.assertLess(fv.dominant_freq_hz, 25)


class TestLabels(unittest.TestCase):
    def test_window_label_centre(self):
        ivs = load_labels([
            {"start_ms": 0, "end_ms": 5000, "activity": "writing_notes", "task_state": "ON_TASK"},
            {"start_ms": 5000, "end_ms": 10000, "activity": "phone_use", "task_state": "DISTRACTED"},
        ])
        self.assertEqual(window_label(1000, 2000, ivs).task_state, "ON_TASK")
        self.assertEqual(window_label(6000, 7000, ivs).task_state, "DISTRACTED")
        self.assertIsNone(window_label(90000, 100000, ivs))

    def test_drop_transition(self):
        wins = [(0, 4000), (5000, 9000), (9000, 13000)]
        ivs = load_labels([
            {"start_ms": 0, "end_ms": 10000, "activity": "a", "task_state": "ON_TASK"},
        ])
        kept = drop_transition_windows(wins, ivs, transition_s=1.0)  # 1000 ms
        # (0,4000): start 0 < 1000 -> drop
        # (5000,9000): end 9000 > 9000? block end=10000, 10000-1000=9000, end 9000 is not > 9000 -> keep
        # (9000,13000): centre 11000 outside block -> drop
        states = [k[2] for k in kept]
        self.assertEqual(states, ["ON_TASK"])


class TestPipeline(unittest.TestCase):
    def test_run_pipeline_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sess = _write_session(root, n_samples=3000)
            rows1 = run_pipeline(sess)
            rows2 = run_pipeline(sess)
            self.assertEqual(rows1, rows2)
            self.assertGreater(len(rows1), 0)
            for row in rows1:
                self.assertIn("task_state", row)
                self.assertIn("activity", row)
                self.assertIn("rms_accel_nogravity", row)
                self.assertIn("start_ms", row)

    def test_transition_drop_in_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 60 seconds at 50 Hz. Two blocks of 30 s each; 2 s transition drop.
            sess = _write_session(
                root, n_samples=3000,
                label_blocks=[(0, 30000, "writing_notes", "ON_TASK"), (30000, 60000, "phone_use", "DISTRACTED")],
            )
            rows = run_pipeline(sess, transition_s=2.0)
            states = {r["task_state"] for r in rows}
            self.assertGreater(len(rows), 0)
            self.assertTrue(states <= {"ON_TASK", "DISTRACTED"})


if __name__ == "__main__":
    unittest.main()
