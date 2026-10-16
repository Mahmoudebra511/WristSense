"""AAC state machine (project document, Sections 2 & 5.6).

Implements the on-task state machine with hysteresis timers:

* ON_TASK  — student is doing the study task (or a short pause ≤ grace)
* UNCERTAIN — long stillness with no clear signal (reading vs zoning out)
* DISTRACTED — confirmed sustained distraction (≥ confirmation time)
* AWAY — away from the desk (> away threshold, no face / walking)
* BREAK — user-declared break (pauses alerts)

Timers (all configurable, tuned by replay on pilot data):

* grace_s — stillness up to this stays ON_TASK (default 90 s)
* confirm_s — a new signal must persist this long before switching (default 20 s)
* alert_delay_s — sustained distraction must last this long before alert (default ~120 s)
* away_threshold_s — away from desk longer than this → AWAY (default 30 s)

Designed to be driven by the replay tool with synthetic or recorded sessions,
so it can be unit-tested without hardware.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
ON_TASK = "ON_TASK"
UNCERTAIN = "UNCERTAIN"
DISTRACTED = "DISTRACTED"
AWAY = "AWAY"
BREAK = "BREAK"

ALL_STATES = (ON_TASK, UNCERTAIN, DISTRACTED, AWAY, BREAK)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class StateMachineConfig:
    grace_s: float = 90.0
    confirm_s: float = 20.0
    alert_delay_s: float = 120.0
    away_threshold_s: float = 30.0
    adaptive: bool = True  # reduce alert frequency when ignored

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StateMachineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
@dataclass
class _TimerState:
    """Tracks how long the current candidate signal has persisted."""
    active: bool = False
    start_ms: int = 0
    value: Any = None


@dataclass
class StateTransition:
    timestamp_ms: int
    from_state: str
    to_state: str
    trigger: str  # human-readable reason


@dataclass
class StateMachineResult:
    states: list[dict]            # per-window: {ts_ms, state, alert}
    transitions: list[StateTransition]
    alerts: list[dict]            # {ts_ms, state, reason}
    false_alerts_per_hour: float
    detection_delay_s: float | None


class StateMachine:
    def __init__(self, config: StateMachineConfig | None = None):
        self.config = config or StateMachineConfig()
        self.state = ON_TASK
        self.timer = _TimerState()
        self.away_timer = _TimerState()
        self.distraction_start_ms: int | None = None
        self.alerts_ignored = 0
        self._transitions: list[StateTransition] = []
        self._alerts: list[dict] = []
        self._states: list[dict] = []
        self._last_alert_ms: int | None = None

    # ------------------------------------------------------------------
    def _switch(self, new_state: str, ts_ms: int, trigger: str) -> None:
        if new_state == self.state:
            return
        self._transitions.append(
            StateTransition(ts_ms, self.state, new_state, trigger)
        )
        self.state = new_state
        self.timer = _TimerState()

    # ------------------------------------------------------------------
    def step(
        self,
        ts_ms: int,
        model_state: str,
        face_present: bool | None,
        away_signal: bool,
        break_declared: bool,
    ) -> str:
        """Advance one decision window.

        Parameters
        ----------
        ts_ms:
            Decision-window start time (PC clock, epoch ms).
        model_state:
            The ML model prediction for this window (ON_TASK / DISTRACTED /
            AWAY / unknown).
        face_present:
            Whether the webcam saw a face (None = no vision data).
        away_signal:
            Whether the band/logger indicates the user is away from desk.
        break_declared:
            Whether the user pressed the break button this window.

        Returns
        -------
        str
            The state after this step.
        """
        cfg = self.config

        # --- BREAK has highest priority --------------------------------
        if break_declared:
            self._switch(BREAK, ts_ms, "break_declared")
            self.distraction_start_ms = None
            self._record(ts_ms)
            return self.state

        if self.state == BREAK:
            # Break ends when the button is released (break_declared False)
            self._switch(ON_TASK, ts_ms, "break_ended")
            self.distraction_start_ms = None

        # --- AWAY detection -------------------------------------------
        if away_signal:
            if not self.away_timer.active:
                self.away_timer = _TimerState(active=True, start_ms=ts_ms, value=True)
            elif ts_ms - self.away_timer.start_ms >= cfg.away_threshold_s * 1000:
                self._switch(AWAY, ts_ms, "away_threshold")
                self.distraction_start_ms = None
                self._record(ts_ms)
                return self.state
        else:
            self.away_timer = _TimerState()
            if self.state == AWAY:
                self._switch(ON_TASK, ts_ms, "returned_to_desk")

        # --- AWAY short-circuits everything else ----------------------
        if self.state == AWAY:
            self._record(ts_ms)
            return self.state

        # --- Distraction confirmation timer ---------------------------
        is_distracted = model_state == DISTRACTED
        if is_distracted:
            if self.state == DISTRACTED:
                # Already confirmed distracted: just check alert delay.
                if self.distraction_start_ms is not None:
                    elapsed = ts_ms - self.distraction_start_ms
                    if elapsed >= cfg.alert_delay_s * 1000:
                        self._maybe_alert(ts_ms, "sustained_distraction")
                self._record(ts_ms)
                return self.state
            # Not yet in DISTRACTED state: need to confirm first.
            if not self.timer.active or self.timer.value != DISTRACTED:
                # Start / restart the distraction confirmation timer.
                self.timer = _TimerState(active=True, start_ms=ts_ms, value=DISTRACTED)
                self._record(ts_ms)
                return self.state  # still ON_TASK, confirming
            # Timer already running on DISTRACTED: check confirmation.
            if ts_ms - self.timer.start_ms <= cfg.confirm_s * 1000:
                self._record(ts_ms)
                return self.state  # still confirming
            # Confirmed! Enter DISTRACTED state.
            self.distraction_start_ms = ts_ms
            self._switch(DISTRACTED, ts_ms, "confirmed_distraction")
            self._record(ts_ms)
            return self.state
        # Not distracted this window: require confirmation to clear the timer.
        if self.timer.active and self.timer.value == DISTRACTED:
            if ts_ms - self.timer.start_ms >= cfg.confirm_s * 1000:
                self.timer = _TimerState()
                self.distraction_start_ms = None
                if self.state == DISTRACTED:
                    self._switch(ON_TASK, ts_ms, "distraction_ended")
        elif self.timer.active and self.timer.value == UNCERTAIN:
            # Leave the UNCERTAIN timer running for the section below.
            pass
        else:
            self.timer = _TimerState()
            self.distraction_start_ms = None

        # --- UNCERTAIN: long stillness with no clear signal -----------
        # Stillness proxy: model says ON_TASK but no face and no input
        still = model_state == ON_TASK and face_present is False
        if still:
            if not self.timer.active or self.timer.value != UNCERTAIN:
                # Start / restart the uncertainty confirmation timer.
                self.timer = _TimerState(active=True, start_ms=ts_ms, value=UNCERTAIN)
                self._record(ts_ms)
                return self.state
            # Timer already running on UNCERTAIN: check grace period.
            if ts_ms - self.timer.start_ms > cfg.grace_s * 1000:
                self._switch(UNCERTAIN, ts_ms, "long_stillness")
                self._record(ts_ms)
                return self.state
            # Still still, but grace not yet exceeded: keep waiting.
            self._record(ts_ms)
            return self.state
        # Not still this window: require confirmation to clear the timer.
        if self.timer.active and self.timer.value == UNCERTAIN:
            if ts_ms - self.timer.start_ms >= cfg.confirm_s * 1000:
                self.timer = _TimerState()
                if self.state == UNCERTAIN:
                    self._switch(ON_TASK, ts_ms, "uncertainty_ended")
        else:
            self.timer = _TimerState()

        # --- Default: ON_TASK -----------------------------------------
        self.timer = _TimerState()
        if self.state != ON_TASK:
            self._switch(ON_TASK, ts_ms, "returned_to_task")
        self._record(ts_ms)
        return self.state

    # ------------------------------------------------------------------
    def _maybe_alert(self, ts_ms: int, reason: str) -> None:
        cfg = self.config
        # Adaptive: reduce alert frequency when ignored.
        if cfg.adaptive and self._last_alert_ms is not None:
            # simple backoff: ignore alerts that come too close together
            if ts_ms - self._last_alert_ms < cfg.alert_delay_s * 1000:
                self.alerts_ignored += 1
                return
        self._alerts.append({"ts_ms": ts_ms, "reason": reason})
        self._last_alert_ms = ts_ms

    # ------------------------------------------------------------------
    def _record(self, ts_ms: int) -> None:
        self._states.append({"ts_ms": ts_ms, "state": self.state, "alert": False})

    def mark_alert_at(self, ts_ms: int) -> None:
        """Mark that an alert was shown at this timestamp (for the record)."""
        for s in reversed(self._states):
            if s["ts_ms"] <= ts_ms:
                s["alert"] = True
                break

    # ------------------------------------------------------------------
    def run(self, events: list[dict]) -> StateMachineResult:
        """Run the state machine over a list of decision-window events.

        Each event dict: {ts_ms, model_state, face_present, away_signal,
        break_declared}
        """
        self.state = ON_TASK
        self.timer = _TimerState()
        self.away_timer = _TimerState()
        self.distraction_start_ms = None
        self.alerts_ignored = 0
        self._transitions = []
        self._alerts = []
        self._states = []
        self._last_alert_ms = None

        for ev in events:
            self.step(
                ts_ms=int(ev["ts_ms"]),
                model_state=str(ev.get("model_state", "ON_TASK")),
                face_present=ev.get("face_present"),
                away_signal=bool(ev.get("away_signal", False)),
                break_declared=bool(ev.get("break_declared", False)),
            )

        # --- metrics ----------------------------------------------------
        duration_s = (events[-1]["ts_ms"] - events[0]["ts_ms"]) / 1000.0 if events else 0
        false_alerts_per_hour = (
            len(self._alerts) / (duration_s / 3600.0) if duration_s > 0 else 0.0
        )

        # Detection delay: time from distraction start to alert.
        detection_delays: list[float] = []
        for al in self._alerts:
            if self.distraction_start_ms is not None:
                detection_delays.append((al["ts_ms"] - self.distraction_start_ms) / 1000.0)
        detection_delay_s = (
            sum(detection_delays) / len(detection_delays) if detection_delays else None
        )

        return StateMachineResult(
            states=self._states,
            transitions=self._transitions,
            alerts=self._alerts,
            false_alerts_per_hour=false_alerts_per_hour,
            detection_delay_s=detection_delay_s,
        )
