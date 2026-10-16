"""Evaluation tests. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluate import (  # noqa: E402
    LABELS,
    _compute_confusion,
    _per_class_from_confusion,
    accuracy_from_confusion,
    ablation_study,
    compute_subject_metrics,
    evaluate_leave_one_subject_out,
    macro_f1_from_confusion,
)


class TestConfusion(unittest.TestCase):
    def test_perfect(self):
        y_true = ["ON_TASK", "DISTRACTED", "AWAY", "ON_TASK", "DISTRACTED", "AWAY"]
        y_pred = list(y_true)
        cm = _compute_confusion(y_true, y_pred)
        self.assertEqual(cm, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    def test_wrong(self):
        y_true = ["ON_TASK", "ON_TASK", "ON_TASK"]
        y_pred = ["DISTRACTED", "DISTRACTED", "DISTRACTED"]
        cm = _compute_confusion(y_true, y_pred)
        self.assertEqual(cm, [[0, 3, 0], [0, 0, 0], [0, 0, 0]])

    def test_ignores_unknown(self):
        y_true = ["ON_TASK", "BREAK", "UNCERTAIN"]
        y_pred = ["ON_TASK", "ON_TASK", "ON_TASK"]
        cm = _compute_confusion(y_true, y_pred)
        self.assertEqual(cm, [[1, 0, 0], [0, 0, 0], [0, 0, 0]])


class TestMetrics(unittest.TestCase):
    def test_macro_f1_perfect(self):
        cm = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        self.assertAlmostEqual(macro_f1_from_confusion(cm), 1.0)

    def test_accuracy_perfect(self):
        cm = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        self.assertAlmostEqual(accuracy_from_confusion(cm), 1.0)

    def test_per_class(self):
        cm = [[2, 0, 0], [0, 1, 1], [0, 0, 0]]
        pc = _per_class_from_confusion(cm)
        self.assertAlmostEqual(pc["ON_TASK"]["f1"], 1.0)
        self.assertAlmostEqual(pc["DISTRACTED"]["precision"], 1.0)
        self.assertAlmostEqual(pc["DISTRACTED"]["recall"], 0.5)
        self.assertAlmostEqual(pc["AWAY"]["support"], 0)


class TestComputeSubjectMetrics(unittest.TestCase):
    def test_basic(self):
        rows = [
            {"task_state": "ON_TASK", "pred": "ON_TASK"},
            {"task_state": "DISTRACTED", "pred": "DISTRACTED"},
            {"task_state": "ON_TASK", "pred": "DISTRACTED"},  # false alert
        ]
        res = compute_subject_metrics(rows, duration_s=3600.0)
        self.assertAlmostEqual(res["false_alerts_per_hour"], 1.0)
        self.assertEqual(res["n_windows"], 3)
        self.assertIn("ON_TASK", res["per_class"])


class TestLeaveOneSubjectOut(unittest.TestCase):
    def test_two_subjects(self):
        sessions = {
            "p01": [
                {"task_state": "ON_TASK", "pred": "ON_TASK"},
                {"task_state": "DISTRACTED", "pred": "DISTRACTED"},
            ],
            "p02": [
                {"task_state": "AWAY", "pred": "AWAY"},
                {"task_state": "ON_TASK", "pred": "ON_TASK"},
            ],
        }
        report = evaluate_leave_one_subject_out(sessions, duration_s_per_session=3600.0)
        self.assertEqual(report["n_subjects"], 2)
        self.assertGreaterEqual(report["macro_f1_mean"], 0.0)
        self.assertIn("p01", report["per_subject"])
        self.assertIn("p02", report["per_subject"])


class TestAblation(unittest.TestCase):
    def test_ablation_structure(self):
        sessions = {
            "p01": [
                {
                    "task_state": "ON_TASK",
                    "imu_rms_accel_nogravity": "0.5",
                    "logger_keys": "5",
                    "vision_head_yaw": "0.0",
                },
                {
                    "task_state": "DISTRACTED",
                    "imu_rms_accel_nogravity": "",
                    "logger_keys": "",
                    "vision_head_yaw": "",
                },
            ]
        }
        abl = ablation_study(sessions)
        self.assertIn("imu_only", abl)
        self.assertIn("imu_logger", abl)
        self.assertIn("imu_logger_vision", abl)
        self.assertEqual(abl["imu_only"]["total_windows"], 2)
        self.assertEqual(abl["imu_only"]["windows_with_source"], 1)


if __name__ == "__main__":
    unittest.main()
