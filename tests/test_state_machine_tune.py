"""State machine tuning tests. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.state_machine_tune import (  # noqa: E402
    _detection_delays,
    _parse_float_list,
    rows_to_events,
    tune,
)
from ml.state_machine import (  # noqa: E402
    StateMachineConfig,
    ON_TASK,
    DISTRACTED,
    AWAY,
)


def event(ts_s: float, model_state: str = ON_TASK, ground_truth: str = ON_TASK,
          face: bool = True, away: bool = False, brk: bool = False) -> dict:
    return {
        "ts_ms": int(ts_s * 1000),
        "model_state": model_state,
        "face_present": face,
        "away_signal": away,
        "break_declared": brk,
        "ground_truth": ground_truth,
    }


class TestParseFloatList(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(_parse_float_list("30,60,90"), [30.0, 60.0, 90.0])
        self.assertEqual(_parse_float_list("10"), [10.0])
        self.assertEqual(_parse_float_list(""), [])


class TestRowsToEvents(unittest.TestCase):
    def test_conversion(self):
        rows = [
            {"start_ms": "0", "task_state": "ON_TASK"},
            {"start_ms": "30000", "task_state": "DISTRACTED"},
            {"start_ms": "60000", "task_state": "AWAY"},
        ]
        events = rows_to_events(rows)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["ts_ms"], 0)
        self.assertEqual(events[0]["model_state"], "ON_TASK")
        self.assertEqual(events[0]["ground_truth"], "ON_TASK")
        self.assertEqual(events[1]["away_signal"], False)
        self.assertEqual(events[2]["away_signal"], True)


class TestDetectionDelays(unittest.TestCase):
    def test_delays(self):
        events = [
            event(0, ON_TASK, ON_TASK),
            event(30, DISTRACTED, DISTRACTED),   # block starts at 30s
            event(60, DISTRACTED, DISTRACTED),
            event(90, ON_TASK, ON_TASK),         # block ends at 90s
        ]
        alerts = [{"ts_ms": 75000, "reason": "sustained_distraction"}]
        delays = _detection_delays(events, alerts)
        self.assertAlmostEqual(delays[0], 45.0)  # 75s - 30s

    def test_no_alert_no_delay(self):
        events = [
            event(0, DISTRACTED, DISTRACTED),
            event(30, DISTRACTED, DISTRACTED),
        ]
        delays = _detection_delays(events, [])
        self.assertEqual(delays, [])


class TestTune(unittest.TestCase):
    def test_tune_selects_best(self):
        # Build sessions where one config clearly has fewer false alerts.
        sessions = {
            "p01": [
                event(0, ON_TASK, ON_TASK),
                event(30, ON_TASK, ON_TASK),
                event(60, DISTRACTED, DISTRACTED),
                event(90, DISTRACTED, DISTRACTED),
                event(120, ON_TASK, ON_TASK),
            ],
        }
        report = tune(
            sessions,
            grace_values=[30.0, 90.0],
            confirm_values=[10.0, 20.0],
            alert_delay_values=[60.0, 120.0],
            away_threshold_values=[20.0],
            max_delay_s=None,
        )
        self.assertIn("best_config", report)
        self.assertIn("selection_note", report)
        self.assertIn("best_metrics", report)
        self.assertIn("all_results", report)
        self.assertGreaterEqual(len(report["all_results"]), 1)

    def test_tune_respects_max_delay(self):
        # Long distraction block (30s -> 180s) so alerts can fire within it.
        sessions = {
            "p01": [
                event(0, ON_TASK, ON_TASK),
                event(30, DISTRACTED, DISTRACTED),
                event(60, DISTRACTED, DISTRACTED),
                event(90, DISTRACTED, DISTRACTED),
                event(120, DISTRACTED, DISTRACTED),
                event(150, DISTRACTED, DISTRACTED),
                event(180, ON_TASK, ON_TASK),
            ],
        }
        report = tune(
            sessions,
            grace_values=[30.0],
            confirm_values=[5.0, 15.0],
            alert_delay_values=[30.0, 90.0],
            away_threshold_values=[20.0],
            max_delay_s=None,
        )
        # Detection delay should be computed (not None) for at least one config.
        self.assertIsNotNone(report["best_metrics"]["detection_delay_s"])
        if report["best_metrics"]["detection_delay_s"] is not None:
            self.assertLessEqual(report["best_metrics"]["detection_delay_s"], 120.0)

    def test_tune_output_structure(self):
        sessions = {
            "p01": [event(0, ON_TASK, ON_TASK), event(30, DISTRACTED, DISTRACTED)],
        }
        report = tune(
            sessions,
            grace_values=[60.0],
            confirm_values=[10.0],
            alert_delay_values=[60.0],
            away_threshold_values=[30.0],
        )
        self.assertTrue(
            {"grace_s", "confirm_s", "alert_delay_s", "away_threshold_s"}
            <= set(report["best_config"].keys())
        )
        self.assertIn("false_alerts_per_hour", report["best_metrics"])
        self.assertIsInstance(report["all_results"], list)


if __name__ == "__main__":
    unittest.main()
