"""Typed views over ``meta.json``.

The validator works on plain dicts (so it can report *why* a file is malformed
instead of raising), but producers/consumers use these dataclasses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Device:
    board: str = ""
    firmware_version: str = ""
    packet_version: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Device":
        d = d or {}
        return cls(
            board=str(d.get("board", "")),
            firmware_version=str(d.get("firmware_version", "")),
            packet_version=int(d.get("packet_version", 0) or 0),
        )


@dataclass
class PC:
    os: str = ""
    host: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PC":
        d = d or {}
        return cls(os=str(d.get("os", "")), host=str(d.get("host", "")))


@dataclass
class Meta:
    """Metadata for one recording session (``meta.json``)."""

    schema_version: str
    session_id: str
    participant_id: str
    wrist: str
    mode: str
    start_utc: str
    device: Device = field(default_factory=Device)
    pc: PC = field(default_factory=PC)
    end_utc: str | None = None
    duration_s: float | None = None
    timezone: str | None = None
    sampling_rate_hz: dict[str, float] = field(default_factory=dict)
    streams: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    sync_markers_ms: list[int] = field(default_factory=list)
    on_task_definition_version: str = ""
    app_whitelist_hash: str | None = None
    partial: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Meta":
        return cls(
            schema_version=str(d.get("schema_version", "")),
            session_id=str(d.get("session_id", "")),
            participant_id=str(d.get("participant_id", "")),
            wrist=str(d.get("wrist", "")),
            mode=str(d.get("mode", "")),
            start_utc=str(d.get("start_utc", "")),
            device=Device.from_dict(d.get("device")),
            pc=PC.from_dict(d.get("pc")),
            end_utc=d.get("end_utc"),
            duration_s=d.get("duration_s"),
            timezone=d.get("timezone"),
            sampling_rate_hz=dict(d.get("sampling_rate_hz") or {}),
            streams=dict(d.get("streams") or {}),
            row_counts=dict(d.get("row_counts") or {}),
            sync_markers_ms=list(d.get("sync_markers_ms") or []),
            on_task_definition_version=str(d.get("on_task_definition_version", "")),
            app_whitelist_hash=d.get("app_whitelist_hash"),
            partial=bool(d.get("partial", False)),
            notes=str(d.get("notes", "")),
        )
