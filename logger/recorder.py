"""The 1 Hz aggregation loop.

``Recorder.tick()`` produces exactly one ``logger.csv`` row per second. It is
written so the loop can be driven with a fake clock, a fake window source and a
no-op sleep, which is how the tests exercise it without touching the OS.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from .config import Categorizer, LoggerConfig


class WindowSource(Protocol):
    def __call__(self): ...


class InputTrackerLike(Protocol):
    def snapshot(self): ...


class Recorder:
    def __init__(
        self,
        config: LoggerConfig,
        writer,
        window_source: WindowSource,
        input_tracker: InputTrackerLike,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        poll_hz: int | None = None,
    ):
        self.config = config
        self.writer = writer
        self.window_source = window_source
        self.input_tracker = input_tracker
        self.clock = clock
        self.sleep = sleep
        self.poll_hz = poll_hz or config.poll_hz
        self.categorizer = Categorizer(config)
        self._last_window_key: tuple | None = None
        self._current_category: str = config.default_category
        self.rows = 0

    def tick(self) -> dict:
        """Aggregate one second of activity and write one row."""
        switches = 0
        seen_category: str | None = None
        interval = 1.0 / self.poll_hz

        for _ in range(self.poll_hz):
            window = self.window_source()
            if window is not None:
                category = self.categorizer.categorize(window.process, window.title)
                if category is not None:
                    seen_category = category
                key = (window.handle or window.pid, window.title)
                if key != self._last_window_key:
                    switches += 1
                    self._last_window_key = key
            self.sleep(interval)

        if seen_category is not None:
            self._current_category = seen_category

        snap = self.input_tracker.snapshot()
        now = self.clock()
        row = {
            "ts_ms": int(now * 1000),
            "keys": snap.keys,
            "mouse_moves": snap.mouse_moves,
            "mouse_clicks": snap.mouse_clicks,
            "win_switches": switches,
            "category": self._current_category,
            "time_since_input_s": round(snap.time_since_input_s, 3),
        }
        self.writer.write(row)
        self.rows += 1
        return row

    def run(self, duration_s: float | None = None, max_rows: int | None = None,
            on_row=None) -> int:
        """Tick until ``duration_s`` elapse or ``max_rows`` rows are written.

        ``tick`` already spends about one second polling, so this loop runs at
        roughly 1 Hz without an extra sleep. ``on_row`` is called with each row.
        """
        start = self.clock()
        while True:
            row = self.tick()
            if on_row is not None:
                on_row(row)
            if max_rows is not None and self.rows >= max_rows:
                break
            if duration_s is not None and self.clock() - start >= duration_s:
                break
        return self.rows
