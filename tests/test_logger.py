"""Logger tests. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aacdata.session import Session  # noqa: E402
from logger.active_window import ActiveWindow  # noqa: E402
from logger.config import Categorizer, LoggerConfig  # noqa: E402
from logger.input_tracker import Snapshot  # noqa: E402
from logger.recorder import Recorder  # noqa: E402
from logger.writer import CsvWriter, MemoryWriter  # noqa: E402

CONFIG_PATH = REPO_ROOT / "logger" / "config.json"


def win(process: str, title: str, h: int = 1) -> ActiveWindow:
    return ActiveWindow(handle=h, pid=h, process=process, title=title)


class ScriptedWindow:
    def __init__(self, frames):
        self.frames = list(frames)
        self.i = 0

    def __call__(self):
        frame = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return frame


class ScriptedInput:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.i = 0

    def snapshot(self):
        snap = self.snapshots[min(self.i, len(self.snapshots) - 1)]
        self.i += 1
        return snap


class StepClock:
    def __init__(self, start: float = 1_000.0, step: float = 1.0):
        self.t = start
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


class TestCategorizer(unittest.TestCase):
    def setUp(self):
        self.config = LoggerConfig.load(CONFIG_PATH)
        self.cat = Categorizer(self.config)

    def test_process_match(self):
        self.assertEqual(self.cat.categorize("code.exe", "anything"), "study")

    def test_title_match_on_unlisted_process(self):
        self.assertEqual(self.cat.categorize("chrome.exe", "YouTube - cats"), "entertainment")

    def test_process_beats_title(self):
        # code.exe is a study process; a misleading title must not override it.
        self.assertEqual(self.cat.categorize("code.exe", "YouTube"), "study")

    def test_unknown_falls_back_to_default(self):
        self.assertEqual(self.cat.categorize("notepad.exe", "Untitled"), "other")

    def test_exclude_returns_none(self):
        cfg = LoggerConfig.from_dict(
            {"categories": {"study": {"processes": ["code\\.exe"]}},
             "exclude_processes": ["code\\.exe"]}
        )
        self.assertIsNone(Categorizer(cfg).categorize("code.exe", "x"))

    def test_rejects_unknown_category(self):
        with self.assertRaises(ValueError):
            LoggerConfig.from_dict({"categories": {"gaming": {}}})

    def test_hash_is_stable(self):
        self.assertEqual(LoggerConfig.load(CONFIG_PATH).hash, self.config.hash)


class TestRecorder(unittest.TestCase):
    def setUp(self):
        self.config = LoggerConfig.load(CONFIG_PATH)

    def _recorder(self, frames, snapshots):
        writer = MemoryWriter()
        recorder = Recorder(
            self.config,
            writer,
            ScriptedWindow(frames),
            ScriptedInput(snapshots),
            clock=StepClock(),
            sleep=lambda _s: None,
            poll_hz=4,
        )
        return recorder, writer

    def test_rows_have_schema_columns(self):
        recorder, writer = self._recorder([win("code.exe", "main.py")], [Snapshot(1, 2, 0, 0.5)])
        recorder.tick()
        self.assertEqual(set(writer.rows[0]), set(writer.header))

    def test_category_change_and_switch_count(self):
        frames = [win("code.exe", "main.py")] * 4 + [win("chrome.exe", "YouTube")] * 4
        snaps = [Snapshot(3, 10, 1, 0.0), Snapshot(0, 0, 0, 1.5)]
        recorder, writer = self._recorder(frames, snaps)
        recorder.tick()
        recorder.tick()
        self.assertEqual(writer.rows[0]["category"], "study")
        self.assertEqual(writer.rows[1]["category"], "entertainment")
        self.assertEqual(writer.rows[0]["keys"], 3)
        self.assertEqual(writer.rows[1]["keys"], 0)
        # first tick sees a window for the first time; second sees a real switch
        self.assertEqual(writer.rows[0]["win_switches"], 1)
        self.assertEqual(writer.rows[1]["win_switches"], 1)

    def test_timestamps_advance_by_one_second(self):
        recorder, writer = self._recorder([win("code.exe", "main.py")], [Snapshot(0, 0, 0, 0.0)])
        recorder.tick()
        recorder.tick()
        self.assertEqual(writer.rows[1]["ts_ms"] - writer.rows[0]["ts_ms"], 1000)


class TestSessionIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.session = Session.at(Path(self._tmp.name) / "p00_20261016T090000_laptop")

    def tearDown(self):
        self._tmp.cleanup()

    def _write_logger_csv(self, rows: int = 3):
        start_ms = 1_792_141_200_000
        with CsvWriter(self.session.path / "logger.csv") as writer:
            for i in range(rows):
                writer.write({
                    "ts_ms": start_ms + i * 1000, "keys": 1, "mouse_moves": 2,
                    "mouse_clicks": 0, "win_switches": 0, "category": "study",
                    "time_since_input_s": 0.0,
                })

    def _declare_logger(self, partial: bool):
        meta = self.session.load_meta()
        meta.streams = {"logger": "logger.csv"}
        meta.sampling_rate_hz = {"logger": 1.0}
        meta.row_counts = {"logger": 3}
        meta.partial = partial
        self.session.write_meta(meta)

    def test_logger_only_partial_session_is_valid(self):
        self._write_logger_csv()
        self._declare_logger(partial=True)
        report = self.session.validate()
        self.assertTrue(report.ok, report.format())

    def test_logger_only_full_session_errors(self):
        self._write_logger_csv()
        self._declare_logger(partial=False)
        report = self.session.validate()
        self.assertFalse(report.ok)
        self.assertIn("meta.streams", {i.code for i in report.errors})


if __name__ == "__main__":
    unittest.main()
