#!/usr/bin/env python
"""Laptop logger CLI (project document, Section 5.3).

Writes one row per second to ``logger.csv`` inside a session folder.

    # run a 2-minute logger-only session under data/raw/
    python -m logger.run --participant p01 --mode laptop --duration 120

    # write into a session the hub already created
    python -m logger.run --session-dir data/raw/p01_20261020T141530_laptop

    # no OS access needed: synthetic activity, useful to see the output
    python -m logger.run --simulate --duration 10 --print

Stop a long run with Ctrl-C; the file and ``meta.json`` are always finalised.
"""

from __future__ import annotations

import argparse
import platform
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aacdata.session import Session, iso_utc  # noqa: E402
from logger.active_window import ActiveWindow, active_window  # noqa: E402
from logger.config import LoggerConfig  # noqa: E402
from logger.input_tracker import make_input_tracker  # noqa: E402
from logger.recorder import Recorder  # noqa: E402
from logger.writer import CsvWriter  # noqa: E402


class SyntheticWindowSource:
    """Scripted foreground windows, so ``--simulate`` needs no OS access."""

    def __init__(self, dwell: int = 8, seed: int = 0):
        self._dwell = max(1, dwell)
        self._i = 0
        self._rng = random.Random(seed)
        self._frames = [
            ("chrome.exe", "Overleaf - graduation report.tex"),
            ("code.exe", "main.py - Visual Studio Code"),
            ("chrome.exe", "YouTube - LoFi study mix"),
            ("Discord.exe", "friends - Discord"),
            ("Acrobat.exe", "lecture-notes.pdf"),
        ]

    def __call__(self) -> ActiveWindow:
        idx = (self._i // self._dwell) % len(self._frames)
        self._i += 1
        process, title = self._frames[idx]
        return ActiveWindow(handle=1000 + idx, pid=2000 + idx, process=process, title=title)


def _resolve_session(args: argparse.Namespace) -> Session:
    if args.session_dir:
        return Session.at(
            args.session_dir,
            participant_id=args.participant,
            wrist=args.wrist,
            mode=args.mode,
        )
    return Session.create(Path(args.out), args.participant, args.wrist, args.mode)


def _finalise(session: Session, config: LoggerConfig, rows: int) -> None:
    meta = session.load_meta()
    now = datetime.now(timezone.utc)
    meta.streams = {**meta.streams, "logger": "logger.csv"}
    meta.sampling_rate_hz = {**meta.sampling_rate_hz, "logger": 1.0}
    meta.row_counts = {**meta.row_counts, "logger": rows}
    meta.app_whitelist_hash = config.hash
    meta.end_utc = iso_utc(now)
    meta.pc.os = platform.system()
    meta.pc.host = socket.gethostname()
    try:
        start = datetime.fromisoformat(meta.start_utc.replace("Z", "+00:00"))
        meta.duration_s = max(0.0, (now - start).total_seconds())
    except ValueError:
        meta.duration_s = None
    # A session with no IMU stream is a partial capture (logger-only, for now).
    meta.partial = "imu" not in meta.streams
    session.write_meta(meta)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", help="attach to this session folder (created if missing)")
    ap.add_argument("--out", default="data/raw", help="root for a new session (default: data/raw)")
    ap.add_argument("--participant", default="p00", help="pseudonymous participant id")
    ap.add_argument("--wrist", default="right", choices=["left", "right"])
    ap.add_argument("--mode", default="laptop", choices=["laptop", "paper"])
    ap.add_argument("--config", default=None, help="category rules JSON (default: logger/config.json)")
    ap.add_argument("--duration", type=float, default=0, help="seconds to log; 0 = until Ctrl-C")
    ap.add_argument("--simulate", action="store_true", help="synthetic input/window, no OS access")
    ap.add_argument("--seed", type=int, default=0, help="seed for --simulate")
    ap.add_argument("--print", dest="echo", action="store_true", help="echo each row to stdout")
    args = ap.parse_args(argv)

    config = LoggerConfig.load(args.config)
    session = _resolve_session(args)
    out_path = session.path / "logger.csv"

    writer = CsvWriter(out_path)
    tracker, note = make_input_tracker(simulate=args.simulate, seed=args.seed)
    window_source = SyntheticWindowSource(seed=args.seed) if args.simulate else active_window
    recorder = Recorder(config, writer, window_source, tracker, poll_hz=config.poll_hz)

    print(f"session : {session.path}")
    print(f"output  : {out_path}")
    print(f"input   : {note}")
    print(f"window  : {'synthetic' if args.simulate else 'operating system'}")
    print(f"config  : {config.source_path} (sha256 {config.hash[:12]}...)")
    print(f"rules   : {len(config.rules)} categories, poll {config.poll_hz} Hz")
    if not args.simulate and note.startswith("pynput"):
        print("WARNING : pynput is not installed, so only the active window is logged.")
        print("          install it with:  pip install pynput")
    print()

    on_row = (lambda row: print(row)) if args.echo else None
    tracker.start()
    interrupted = False
    try:
        recorder.run(duration_s=args.duration or None, on_row=on_row)
    except KeyboardInterrupt:
        interrupted = True
        print("\nstopped by user")
    finally:
        tracker.stop()
        writer.close()

    _finalise(session, config, writer.count)

    print(f"\nwrote {writer.count} rows to {out_path}")
    report = session.validate()
    print(report.format())
    if interrupted:
        return 0
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
