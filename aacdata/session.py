"""Session-folder helpers shared by the hub, labeler and ML pipeline.

Naming is the one place where the producers (hub/logger/vision) and the
consumers (features/replay) have to agree, so it lives here and nowhere else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .constants import ON_TASK_DEFINITION_VERSION, SCHEMA_VERSION
from .models import Device, Meta, PC
from .validate import Report, validate_session

DATA_DIR_RAW = Path("data") / "raw"


def make_session_id(participant_id: str, start: datetime, mode: str) -> str:
    """Build the canonical session id, e.g. ``p01_20261020T141530_laptop``."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    stamp = start.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{participant_id}_{stamp}_{mode}"


def session_dir(root: str | Path, session_id: str) -> Path:
    return Path(root) / session_id


def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Session:
    """A session folder plus its parsed metadata."""

    path: Path

    @property
    def meta_path(self) -> Path:
        return self.path / "meta.json"

    @property
    def session_id(self) -> str:
        return self.path.name

    def load_meta(self) -> Meta:
        return Meta.from_dict(json.loads(self.meta_path.read_text(encoding="utf-8")))

    def write_meta(self, meta: Meta) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(
            json.dumps(meta.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def stream_path(self, name: str) -> Path:
        from .columns import STREAMS

        return self.path / STREAMS[name].filename

    def validate(self, require_labels: bool = False) -> Report:
        return validate_session(self.path, require_labels=require_labels)

    @classmethod
    def create(cls, root: str | Path, participant_id: str, wrist: str, mode: str,
               start: datetime | None = None, sampling_rate_hz: dict[str, float] | None = None,
               device: dict | None = None) -> "Session":
        """Create an empty session folder with a valid ``meta.json``."""
        start = start or datetime.now(timezone.utc)
        sid = make_session_id(participant_id, start, mode)
        sess = cls(session_dir(root, sid))
        sess.write_meta(
            cls._blank_meta(sid, participant_id, wrist, mode, start, sampling_rate_hz, device)
        )
        return sess

    @classmethod
    def at(cls, path: str | Path, participant_id: str | None = None, wrist: str = "right",
           mode: str | None = None, start: datetime | None = None,
           sampling_rate_hz: dict[str, float] | None = None,
           device: dict | None = None) -> "Session":
        """Use the session at exactly ``path``, creating ``meta.json`` if absent.

        Unlike :meth:`create`, the folder name is taken as-is (it must already
        follow the session-id format). This is what a tool that attaches to a
        session the hub already created uses, and also what a standalone tool
        uses when the user points it at a folder.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sess = cls(path)
        if not sess.meta_path.exists():
            sid = path.name
            parts = sid.split("_")
            pid = participant_id or (parts[0] if parts else "p00")
            guessed_mode = parts[-1] if parts and parts[-1] in ("laptop", "paper") else "laptop"
            sess.write_meta(
                cls._blank_meta(sid, pid, wrist, mode or guessed_mode,
                                start or datetime.now(timezone.utc),
                                sampling_rate_hz, device)
            )
        return sess

    @staticmethod
    def _blank_meta(session_id: str, participant_id: str, wrist: str, mode: str,
                    start: datetime, sampling_rate_hz: dict[str, float] | None,
                    device: dict | None) -> Meta:
        return Meta(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            participant_id=participant_id,
            wrist=wrist,
            mode=mode,
            start_utc=iso_utc(start),
            sampling_rate_hz=sampling_rate_hz or {},
            streams={},
            on_task_definition_version=ON_TASK_DEFINITION_VERSION,
            device=Device.from_dict(device),
            pc=PC(),
        )


def discover(root: str | Path) -> list[Session]:
    """Return every session folder under ``root`` (contains a meta.json)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return [Session(p) for p in sorted(root.glob("*/meta.json"))]
