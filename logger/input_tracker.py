"""Keyboard and mouse activity counters.

Uses ``pynput`` when available. Characters are never recorded — only *how many*
key presses happened in the last second, plus mouse movement/click counts and
the time since the last input.

Three implementations are provided:

* :class:`InputTracker` — the real one (pynput);
* :class:`NullInputTracker` — counts stay zero (no pynput, or headless);
* :class:`SyntheticInputTracker` — deterministic fake counts, for ``--simulate``
  and for tests.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Snapshot:
    keys: int
    mouse_moves: int
    mouse_clicks: int
    time_since_input_s: float


class _BaseTracker:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def snapshot(self) -> Snapshot:
        raise NotImplementedError


class NullInputTracker(_BaseTracker):
    """No counting. ``time_since_input`` still grows, so the column is valid."""

    def __init__(self, clock=time.time):
        self._clock = clock
        self._last_input = clock()

    def snapshot(self) -> Snapshot:
        return Snapshot(0, 0, 0, max(0.0, self._clock() - self._last_input))


class SyntheticInputTracker(_BaseTracker):
    """Seeded fake activity, so the logger can be demonstrated without pynput."""

    def __init__(self, seed: int = 0, clock=time.time):
        self._rng = random.Random(seed)
        self._clock = clock
        self._last_input = clock()

    def snapshot(self) -> Snapshot:
        keys = self._rng.choice([0, 0, 1, 2, 3, 5, 8, 12])
        moves = self._rng.randint(0, 250)
        clicks = self._rng.choice([0, 0, 0, 1, 1, 2])
        now = self._clock()
        if keys or moves or clicks:
            self._last_input = now
        return Snapshot(keys, moves, clicks, max(0.0, now - self._last_input))


class InputTracker(_BaseTracker):
    """Real counters backed by ``pynput`` global listeners."""

    def __init__(self, clock=time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._keys = 0
        self._moves = 0
        self._clicks = 0
        self._last_input = clock()
        self._listeners: list = []

    def start(self) -> None:
        from pynput import keyboard, mouse

        def on_press(_key) -> None:
            with self._lock:
                self._keys += 1
                self._last_input = self._clock()

        def on_move(_x, _y) -> None:
            with self._lock:
                self._moves += 1
                self._last_input = self._clock()

        def on_click(_x, _y, _button, _pressed) -> None:
            with self._lock:
                self._clicks += 1
                self._last_input = self._clock()

        kb = keyboard.Listener(on_press=on_press)
        ms = mouse.Listener(on_move=on_move, on_click=on_click)
        kb.daemon = True
        ms.daemon = True
        kb.start()
        ms.start()
        self._listeners = [kb, ms]

    def stop(self) -> None:
        for listener in self._listeners:
            try:
                listener.stop()
            except Exception:
                pass
        self._listeners = []

    def snapshot(self) -> Snapshot:
        with self._lock:
            keys, moves, clicks = self._keys, self._moves, self._clicks
            last = self._last_input
            self._keys = self._moves = self._clicks = 0
        return Snapshot(keys, moves, clicks, max(0.0, self._clock() - last))


def pynput_available() -> bool:
    try:
        import pynput  # noqa: F401

        return True
    except Exception:
        return False


def make_input_tracker(simulate: bool = False, seed: int = 0) -> tuple[_BaseTracker, str]:
    """Return (tracker, note). Falls back to a null tracker with an explanation."""
    if simulate:
        return SyntheticInputTracker(seed=seed), "synthetic"
    if not pynput_available():
        return NullInputTracker(), "pynput not installed: keyboard/mouse counts are zero"
    tracker = InputTracker()
    return tracker, "pynput"
