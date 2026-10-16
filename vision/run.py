"""Vision CLI.

    # no hardware: synthetic face, no OpenCV/MediaPipe needed
    python -m vision.run --simulate --duration 20 --participant p01

    # real webcam with calibration
    python -m vision.run --camera-index 0 --duration 120

    # skip the 10 s calibration (baseline = neutral defaults)
    python -m vision.run --no-calibration --duration 60

    # attach to a session the hub/logger already created
    python -m vision.run --session-dir data/raw/p01_20261020T141530_laptop

Ctrl-C finalises meta.json and validates the session.
"""

from __future__ import annotations

import argparse
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aacdata.session import Session, iso_utc  # noqa: E402
from vision.pipeline import (  # noqa: E402
    CalibrationResult,
    FrameResult,
    SyntheticCamera,
    SyntheticDetector,
    VisionCsvWriter,
    VisionPipeline,
)
from vision.detector import MediaPipeFaceDetector, OpenCVCamera  # noqa: E402


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
        sampling_rate_hz={"vision": 1.0},
    )


def _finalise(session: Session, pipeline: VisionPipeline) -> None:
    meta = session.load_meta()
    now = datetime.now(timezone.utc)
    meta.streams = {**meta.streams, "vision": "vision.csv"}
    meta.sampling_rate_hz = {**meta.sampling_rate_hz, "vision": 1.0}
    meta.row_counts = {**meta.row_counts, "vision": pipeline.rows}
    meta.end_utc = iso_utc(now)
    try:
        start = datetime.fromisoformat(meta.start_utc.replace("Z", "+00:00"))
        meta.duration_s = max(0.0, (now - start).total_seconds())
    except ValueError:
        meta.duration_s = None
    meta.pc.os = platform.system()
    meta.pc.host = socket.gethostname()
    # calibration is a one-time per-session baseline; store its key numbers
    cal = pipeline.calibration
    meta.notes = (
        f"calibration neutral_yaw={cal.neutral_yaw:.2f} "
        f"neutral_pitch={cal.neutral_pitch:.2f} "
        f"neutral_ear={cal.neutral_ear:.3f} "
        f"duration_s={cal.duration_s:.1f}"
    )
    meta.partial = ("imu" not in meta.streams) and ("logger" not in meta.streams)
    session.write_meta(meta)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", help="attach to this session folder (created if missing)")
    ap.add_argument("--out", default="data/raw", help="root for a new session (default: data/raw)")
    ap.add_argument("--participant", default="p00", help="pseudonymous participant id")
    ap.add_argument("--wrist", default="right", choices=["left", "right"])
    ap.add_argument("--mode", default="laptop", choices=["laptop", "paper"])
    ap.add_argument("--duration", type=float, default=0, help="seconds to run; 0 = until Ctrl-C")
    ap.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--simulate", action="store_true", help="synthetic face/camera, no OpenCV/MediaPipe")
    ap.add_argument("--no-calibration", action="store_true", help="skip the 10 s calibration")
    ap.add_argument("--seed", type=int, default=0, help="seed for --simulate")
    args = ap.parse_args(argv)

    session = _resolve_session(args)
    writer = VisionCsvWriter(session.path / "vision.csv")

    if args.simulate:
        camera = SyntheticCamera()
        detector = SyntheticDetector(
            [
                # neutral face, then looking away, then no face
                FrameResult(True, 0.0, 0.0, 0.05, 1.0),
                FrameResult(True, 15.0, -10.0, 0.1, 1.0),
                FrameResult(False, None, None, None, None),
            ]
        )
        source_kind = "synthetic"
    else:
        camera = OpenCVCamera(args.camera_index)
        detector = MediaPipeFaceDetector()
        source_kind = "webcam"

    pipeline = VisionPipeline(detector, writer, camera)
    print(f"session : {session.path}")
    print(f"output  : {session.path / 'vision.csv'}")
    print(f"source  : {source_kind}")
    print(f"wrist   : {args.wrist}   mode : {args.mode}")
    print("running vision pipeline... (Ctrl-C to stop)")
    try:
        if args.no_calibration:
            pipeline.run(duration_s=args.duration or None)
        else:
            pipeline.run_with_calibration(duration_s=args.duration or None)
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        pipeline.stop()
        writer.close()
        if hasattr(camera, "release"):
            camera.release()
    _finalise(session, pipeline)
    print(f"\nwrote {pipeline.rows} vision rows")
    from aacdata.validate import validate_session

    print(validate_session(session.path).format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
