"""Hub CLI.

    # with the synthetic (no-hardware) source
    python -m hub.run --simulate --duration 60 --participant p01 --wrist left

    # with a real band
    python -m hub.run --address AA:BB:CC:DD:EE:FF --char 0000ffe1-0000-1000-8000-00805f9b34fb

    Ctrl-C finalises meta.json and validates the session.
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aacdata.session import Session, iso_utc  # noqa: E402
from hub.packet import (  # noqa: E402
    Hub,
    SyncMarkerDetector,
    SyntheticPacketSource,
)
from hub.writer import ImuCsvWriter  # noqa: E402

try:  # pragma: no cover - hardware path needs bleak
    from hub.packet import BleakPacketSource  # noqa: F401
    _HAVE_BLEAK = True
except ImportError:
    _HAVE_BLEAK = False


def _resolve_session(args: argparse.Namespace) -> Session:
    if args.session_dir:
        return Session.at(
            args.session_dir,
            participant_id=args.participant,
            wrist=args.wrist,
            mode=args.mode,
        )
    return Session.create(
        Path(args.out),
        args.participant,
        args.wrist,
        args.mode,
        sampling_rate_hz={"imu": 50.0},
    )


def _finalise(session: Session, hub: Hub) -> None:
    meta = hub.finalize()
    meta.pc.os = platform.system()
    meta.pc.host = socket.gethostname()
    session.write_meta(meta)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", help="attach to this session folder (created if missing)")
    ap.add_argument("--out", default="data/raw", help="root for a new session (default: data/raw)")
    ap.add_argument("--participant", default="p00", help="pseudonymous participant id")
    ap.add_argument("--wrist", default="right", choices=["left", "right"])
    ap.add_argument("--mode", default="laptop", choices=["laptop", "paper"])
    ap.add_argument("--duration", type=float, default=0, help="seconds to stream; 0 = until Ctrl-C")
    ap.add_argument("--simulate", action="store_true", help="synthetic IMU source, no hardware")
    ap.add_argument("--address", help="BLE MAC address of the band (real hardware)")
    ap.add_argument("--char", default="0000ffe1-0000-1000-8000-00805f9b34fb", help="BLE notify characteristic UUID")
    ap.add_argument("--seed", type=int, default=0, help="seed for --simulate")
    ap.add_argument("--shake-every", type=float, default=None, help="(--simulate) seconds between fake shake spikes")
    args = ap.parse_args(argv)

    if args.address and not _HAVE_BLEAK:
        print("ERROR: --address requires bleak (pip install bleak)", file=sys.stderr)
        return 2
    if args.address and args.simulate:
        print("ERROR: choose either --address or --simulate", file=sys.stderr)
        return 2

    session = _resolve_session(args)
    writer = ImuCsvWriter(session.path / "imu.csv")

    if args.simulate:
        source = SyntheticPacketSource(rate_hz=50.0, batch=10, shake_every_s=args.shake_every, seed=args.seed)
        source_kind = "synthetic"
    else:
        try:
            from hub.packet import BleakPacketSource

            source = BleakPacketSource(args.address, args.char)
            source_kind = "bleak"
        except ImportError:
            print("ERROR: bleak not installed", file=sys.stderr)
            return 2

    hub = Hub(session, source, writer=writer)
    print(f"session : {session.path}")
    print(f"output  : {session.path / 'imu.csv'}")
    print(f"source  : {source_kind}")
    print(f"wrist   : {args.wrist}   mode : {args.mode}")
    print("streaming IMU... (Ctrl-C to stop)")
    try:
        asyncio.run(hub.stream(duration_s=args.duration or None))
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        hub.stop()
        writer.close()
    _finalise(session, hub)
    print(f"\nwrote {hub.rows} IMU rows")
    print(hub.report().format())
    return 0 if hub.report().ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
