"""Replay + state machine tests. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.state_machine import (  # noqa: E402
    StateMachine,
    StateMachineConfig,
    ON_TASK,
    UNCERTAIN,
    DISTRACTED,
    AWAY,
    BREAK,
)
from ml.replay import synthetic_events, replay  # noqa: E402


def event(ts_s: float, model_state: str = ON_TASK, face: bool = True,
          away: bool = False, brk: bool = False) -> dict:
    return {
        "ts_ms": int(ts_s * 1000),
        "model_state": model_state,
        "face_present": face,
        "away_signal": away,
        "break_declared": brk,
    }


class TestStateMachine(unittest.TestCase):
    def test_stays_on_task(self):
        sm = StateMachine()
        evs = [event(t, ON_TASK, face=True) for t in range(0, 300, 30)]
        res = sm.run(evs)
        self.assertTrue(all(s["state"] == ON_TASK for s in res.states))

    def test_confirmed_distraction(self):
        # confirm_s=120: windows at 0,30,60,90 never reach 120s continuous
        sm = StateMachine(StateMachineConfig(confirm_s=120.0))
        evs = [event(t, DISTRACTED, face=True) for t in range(0, 120, 30)]
        res = sm.run(evs)
        states = {s["state"] for s in res.states}
        self.assertNotIn(DISTRACTED, states)

    def test_confirmed_distraction_switches(self):
        sm = StateMachine(StateMachineConfig(confirm_s=20.0))
        evs = [event(t, DISTRACTED, face=True) for t in range(0, 150, 30)]
        res = sm.run(evs)
        states = {s["state"] for s in res.states}
        self.assertIn(DISTRACTED, states)

    def test_grace_period(self):
        # stillness up to 90 s stays ON_TASK (grace=90 s, strictly greater to switch)
        sm = StateMachine(StateMachineConfig(grace_s=90.0))
        evs = [event(t, ON_TASK, face=False) for t in range(0, 120, 30)]  # 0,30,60,90
        res = sm.run(evs)
        self.assertTrue(all(s["state"] == ON_TASK for s in res.states))

    def test_grace_exceeded_becomes_uncertain(self):
        sm = StateMachine(StateMachineConfig(grace_s=60.0, confirm_s=5.0))
        evs = [event(t, ON_TASK, face=False) for t in range(0, 200, 30)]
        res = sm.run(evs)
        states = {s["state"] for s in res.states}
        self.assertIn(UNCERTAIN, states)

    def test_away(self):
        sm = StateMachine(StateMachineConfig(away_threshold_s=60.0))
        evs = [event(t, ON_TASK, face=True, away=True) for t in range(0, 120, 30)]
        res = sm.run(evs)
        states = {s["state"] for s in res.states}
        self.assertIn(AWAY, states)

    def test_break(self):
        sm = StateMachine()
        evs = [
            event(0, ON_TASK, face=True),
            event(30, ON_TASK, face=True, brk=True),
            event(60, ON_TASK, face=True),
        ]
        res = sm.run(evs)
        states = [s["state"] for s in res.states]
        self.assertIn(BREAK, states)

    def test_alerts(self):
        sm = StateMachine(StateMachineConfig(confirm_s=5.0, alert_delay_s=20.0))
        evs = [event(t, DISTRACTED, face=True) for t in range(0, 200, 30)]
        res = sm.run(evs)
        self.assertGreater(len(res.alerts), 0)

    def test_transitions_recorded(self):
        sm = StateMachine(StateMachineConfig(confirm_s=5.0))
        evs = [event(t, ON_TASK, face=True) for t in range(0, 60, 30)] + \
              [event(t, DISTRACTED, face=True) for t in range(60, 150, 30)]
        res = sm.run(evs)
        self.assertGreaterEqual(len(res.transitions), 1)


class TestReplay(unittest.TestCase):
    def test_synthetic_replay(self):
        events = synthetic_events(duration_s=300.0, window_s=30.0, seed=0)
        result = replay(events, StateMachineConfig())
        self.assertIn("agreement", result)
        self.assertIn("false_alerts_per_hour", result)
        self.assertGreaterEqual(result["agreement"], 0.0)

    def test_replay_deterministic(self):
        e1 = synthetic_events(duration_s=300.0, window_s=30.0, seed=7)
        e2 = synthetic_events(duration_s=300.0, window_s=30.0, seed=7)
        r1 = replay(e1, StateMachineConfig())
        r2 = replay(e2, StateMachineConfig())
        self.assertEqual(r1["agreement"], r2["agreement"])
        self.assertEqual(r1["false_alerts_per_hour"], r2["false_alerts_per_hour"])

    def test_config_tuning_affects_alerts(self):
        events = synthetic_events(duration_s=600.0, window_s=30.0, seed=1)
        r_strict = replay(events, StateMachineConfig(alert_delay_s=30.0))
        r_lenient = replay(events, StateMachineConfig(alert_delay_s=300.0))
        # longer alert delay -> fewer or equal alerts
        self.assertGreaterEqual(
            r_strict["false_alerts_per_hour"],
            r_lenient["false_alerts_per_hour"],
        )


if __name__ == "__main__":
    unittest.main()
