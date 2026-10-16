"""Baseline model tests. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.baseline import (  # noqa: E402
    LABELS,
    BaselineModel,
    EvaluationReport,
    FoldResult,
    _safe_div,
    _compute_metrics,
    evaluate_leave_one_session_out,
    feature_columns,
    load_all_sessions,
    load_session_features,
    train_model,
)

from tests.test_ml import _write_session  # noqa: E402


def _make_session_features(
    root: Path,
    session_id: str,
    n_per_class: int = 40,
    seed: int = 0,
) -> Path:
    """Write a features.csv with a linearly separable-ish pattern per class."""
    sess_dir = root / session_id
    sess_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)

    rows = []
    for cls_idx, cls_name in enumerate(LABELS):
        for _ in range(n_per_class):
            # Each class has a different mean on the first 3 features.
            mean = np.zeros(8)
            mean[cls_idx % 3] = 1.5 * (cls_idx + 1)
            x = rng.normal(mean, 0.5, 8)
            rows.append(
                {
                    "start_ms": 0,
                    "end_ms": 4000,
                    "window_s": 4.0,
                    "sample_rate_hz": 50.0,
                    "task_state": cls_name,
                    "activity": "writing_notes",
                    "rms_accel_nogravity": float(x[0]),
                    "rms_gyro": float(x[1]),
                    "gyro_mag_mean": float(x[2]),
                    "gyro_mag_std": float(x[3]),
                    "dominant_freq_hz": float(x[4]),
                    "spectral_entropy": float(x[5]),
                    "band_energy_0_5_3_hz": float(x[6]),
                    "band_energy_3_8_hz": float(x[7]),
                    "zero_crossing_rate": 0.1,
                    "gravity_x": 0.0,
                    "gravity_y": 1.0,
                    "gravity_z": 0.0,
                    "peak_count": 5.0,
                    "max_jerk": 1.0,
                    "stillness_duration_s": 0.5,
                }
            )

    header = [
        "start_ms",
        "end_ms",
        "window_s",
        "sample_rate_hz",
        "task_state",
        "activity",
        "rms_accel_nogravity",
        "rms_gyro",
        "gyro_mag_mean",
        "gyro_mag_std",
        "dominant_freq_hz",
        "spectral_entropy",
        "band_energy_0_5_3_hz",
        "band_energy_3_8_hz",
        "zero_crossing_rate",
        "gravity_x",
        "gravity_y",
        "gravity_z",
        "peak_count",
        "max_jerk",
        "stillness_duration_s",
    ]
    import csv

    with open(sess_dir / "features.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    return sess_dir


class TestSafeDiv(unittest.TestCase):
    def test_safe_div(self):
        self.assertEqual(_safe_div(1.0, 2.0), 0.5)
        self.assertEqual(_safe_div(1.0, 0.0), 0.0)


class TestComputeMetrics(unittest.TestCase):
    def test_perfect(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        m = _compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["macro_f1"], 1.0)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertEqual(m["confusion"].trace(), 6)

    def test_worst(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        m = _compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["accuracy"], 0.0)


class TestFeatureColumns(unittest.TestCase):
    def test_excludes_id_cols(self):
        header = ["start_ms", "task_state", "feat_a", "feat_b", "activity"]
        cols = feature_columns(header)
        self.assertEqual(cols, ["feat_a", "feat_b"])


class TestLoadSessionFeatures(unittest.TestCase):
    def test_load_and_drop_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sess_dir = root / "p01"
            sess_dir.mkdir()
            import csv

            with open(sess_dir / "features.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    ["start_ms", "end_ms", "task_state", "activity", "f1", "f2"]
                )
                w.writerow([0, 4000, "ON_TASK", "writing_notes", 1.0, 2.0])
                w.writerow([0, 4000, "BREAK", "writing_notes", 3.0, 4.0])  # dropped
                w.writerow([0, 4000, "DISTRACTED", "phone", 5.0, 6.0])

            sess = load_session_features(sess_dir / "features.csv")
            self.assertIsNotNone(sess)
            self.assertEqual(sess.session_id, "p01")
            self.assertEqual(len(sess.y), 2)
            self.assertEqual(set(sess.labels), {"ON_TASK", "DISTRACTED"})


class TestTrainModel(unittest.TestCase):
    def test_random_forest(self):
        X = np.random.default_rng(1).normal(size=(60, 5))
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        model = train_model(X, y, use_lightgbm=False, seed=42)
        self.assertIsNotNone(model)
        preds = model.predict(X)
        self.assertEqual(len(preds), 60)

    def test_lightgbm_not_installed(self):
        # Should either work if lightgbm is installed or raise a clear error.
        X = np.random.default_rng(1).normal(size=(30, 5))
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        try:
            train_model(X, y, use_lightgbm=True, seed=42)
        except RuntimeError as exc:
            self.assertIn("lightgbm", str(exc).lower())


class TestLeaveOneSessionOut(unittest.TestCase):
    def test_three_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, sid in enumerate(["p01", "p02", "p03"]):
                _make_session_features(root, sid, n_per_class=30, seed=100 + i)

            sessions = load_all_sessions(root)
            self.assertEqual(len(sessions), 3)

            report = evaluate_leave_one_session_out(sessions, use_lightgbm=False, seed=42)
            self.assertEqual(len(report.folds), 3)
            self.assertGreater(report.macro_f1_mean, 0.0)
            self.assertGreater(report.accuracy_mean, 0.0)
            self.assertEqual(report.confusion_total.shape, (3, 3))

            # Each session should appear exactly once as held-out.
            held_out = {f.session_id for f in report.folds}
            self.assertEqual(held_out, {"p01", "p02", "p03"})


class TestBaselineModelSaveLoad(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_session_features(root, "p01", n_per_class=30, seed=7)
            sessions = load_all_sessions(root)
            X = np.vstack([s.X for s in sessions.values()])
            y = np.concatenate([s.y for s in sessions.values()])
            model = train_model(X, y, use_lightgbm=False, seed=42)

            bm = BaselineModel(model=model, feature_names=sessions["p01"].feature_names)
            model_path = Path(tmp) / "baseline.joblib"
            bm.save(model_path)

            bm2 = BaselineModel.load(model_path)
            X_one = sessions["p01"].X
            np.testing.assert_array_equal(bm.predict(X_one), bm2.predict(X_one))


class TestEvaluationReportDict(unittest.TestCase):
    def test_to_dict_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, sid in enumerate(["p01", "p02"]):
                _make_session_features(root, sid, n_per_class=20, seed=3 + i)
            sessions = load_all_sessions(root)
            report = evaluate_leave_one_session_out(sessions, use_lightgbm=False, seed=42)
            d = report.to_dict()
            self.assertIn("macro_f1_mean", d)
            self.assertIn("confusion_matrix", d)
            self.assertIn("per_class", d)
            self.assertEqual(set(d["per_class"].keys()), set(LABELS))


if __name__ == "__main__":
    unittest.main()
