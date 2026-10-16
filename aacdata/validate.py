"""Validation of a session folder against the AAC data contract.

Design goals:

* **Explain, never crash.** Every problem becomes an :class:`Issue` with a
  stable ``code``; callers decide whether an error is fatal.
* **No third-party dependencies.** This must run on a bare Python install,
  before the ML stack is set up.
* **Cheap enough to run at the end of every recording** so bad data is caught
  on the day it is collected, not at training time.

Usage::

    from aacdata.validate import validate_session
    report = validate_session("data/raw/p01_20261020T141530_laptop")
    if not report.ok:
        print(report.format())
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .columns import ALL_FILES, LABELS, STREAMS, Stream
from .constants import SCHEMA_VERSION
from .enums import AppCategory, Mode, PROVISIONAL_ACTIVITIES, Wrist

SESSION_ID_RE = re.compile(r"^p[0-9A-Za-z]{1,16}_\d{8}T\d{6}_(laptop|paper)$")

# Cap on value-level errors reported per file, so one broken column cannot
# produce a 50 000-line report.
_MAX_VALUE_ERRORS = 25


@dataclass
class Issue:
    level: str  # "error" | "warning"
    code: str
    message: str
    where: str = ""

    def __str__(self) -> str:
        tag = "ERROR" if self.level == "error" else "WARN "
        loc = f" [{self.where}]" if self.where else ""
        return f"{tag} {self.code}{loc}: {self.message}"


@dataclass
class Report:
    session: str
    issues: list[Issue] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, code: str, message: str, where: str = "") -> None:
        self.issues.append(Issue(level, code, message, where))

    def error(self, code: str, message: str, where: str = "") -> None:
        self.add("error", code, message, where)

    def warn(self, code: str, message: str, where: str = "") -> None:
        self.add("warning", code, message, where)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = [f"session: {self.session}"]
        lines += [str(i) for i in self.issues]
        if not self.issues:
            lines.append("OK: no problems found")
        if self.info:
            lines.append("info:")
            for k, v in self.info.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "ok": self.ok,
            "info": self.info,
            "issues": [
                {"level": i.level, "code": i.code, "message": i.message, "where": i.where}
                for i in self.issues
            ],
        }


# ---------------------------------------------------------------------------
# dtype coercion
# ---------------------------------------------------------------------------
_TRUE = {"1", "true", "t", "yes", "y"}
_FALSE = {"0", "false", "f", "no", "n"}
_NULLISH = {"", "nan", "na", "null", "none"}


def _coerce(raw: str, dtype: str) -> tuple[bool, Any]:
    """Return (is_valid, value). ``None`` means a legitimate missing value."""
    s = raw.strip()
    if s.lower() in _NULLISH:
        return True, None
    try:
        if dtype == "int":
            try:
                return True, int(s)
            except ValueError:
                f = float(s)
                if f.is_integer():
                    return True, int(f)
                return False, None
        if dtype == "float":
            f = float(s)
            if math.isnan(f) or math.isinf(f):
                return True, None
            return True, f
        if dtype == "bool":
            low = s.lower()
            if low in _TRUE:
                return True, True
            if low in _FALSE:
                return True, False
            return False, None
        if dtype == "str":
            return True, s
    except (ValueError, TypeError):
        return False, None
    return False, None


def _read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    return header, rows


# ---------------------------------------------------------------------------
# meta.json
# ---------------------------------------------------------------------------
def validate_meta(meta: dict[str, Any], session_dir: Path, report: Report) -> None:
    def req(key: str, typ: type) -> Any:
        if key not in meta or meta[key] in (None, ""):
            report.error("meta.missing", f"required key '{key}' is missing or empty", "meta.json")
            return None
        if not isinstance(meta[key], typ):
            report.error(
                "meta.type",
                f"'{key}' should be {typ.__name__}, got {type(meta[key]).__name__}",
                "meta.json",
            )
            return None
        return meta[key]

    sv = req("schema_version", str)
    if sv and not sv.startswith(SCHEMA_VERSION.split(".")[0] + "."):
        report.warn("meta.schema_version", f"schema_version '{sv}' differs from current '{SCHEMA_VERSION}'")

    sid = req("session_id", str)
    if sid:
        if not SESSION_ID_RE.match(sid):
            report.error(
                "meta.session_id",
                f"session_id '{sid}' does not match 'p<id>_YYYYMMDDTHHMMSS_<laptop|paper>'",
                "meta.json",
            )
        if sid != session_dir.name:
            report.error(
                "meta.session_id",
                f"session_id '{sid}' does not match its folder name '{session_dir.name}'",
                "meta.json",
            )

    req("participant_id", str)

    wrist = req("wrist", str)
    if wrist and wrist not in {w.value for w in Wrist}:
        report.error("meta.wrist", f"wrist '{wrist}' not in {[w.value for w in Wrist]}", "meta.json")

    mode = req("mode", str)
    if mode and mode not in {m.value for m in Mode}:
        report.error("meta.mode", f"mode '{mode}' not in {[m.value for m in Mode]}", "meta.json")

    start = req("start_utc", str)
    if start and not _is_iso(start):
        report.warn("meta.start_utc", f"start_utc '{start}' is not ISO-8601 UTC", "meta.json")

    if "end_utc" in meta and meta["end_utc"] and not _is_iso(str(meta["end_utc"])):
        report.warn("meta.end_utc", f"end_utc '{meta['end_utc']}' is not ISO-8601 UTC", "meta.json")

    rates = meta.get("sampling_rate_hz")
    if not isinstance(rates, dict) or not rates:
        report.error("meta.sampling_rate_hz", "sampling_rate_hz must be a non-empty object", "meta.json")
        rates = {}
    else:
        for name, value in rates.items():
            if not isinstance(value, (int, float)) or value <= 0:
                report.error(
                    "meta.sampling_rate_hz",
                    f"sampling_rate_hz['{name}'] must be a positive number",
                    "meta.json",
                )
        if "imu" not in rates:
            report.warn("meta.sampling_rate_hz", "no 'imu' entry in sampling_rate_hz", "meta.json")

    streams = meta.get("streams")
    if not isinstance(streams, dict) or not streams:
        report.error("meta.streams", "streams must map stream name -> filename", "meta.json")
        streams = {}
    for name, filename in streams.items():
        if name not in STREAMS:
            report.warn("meta.streams", f"unknown stream '{name}' (known: {sorted(STREAMS)})", "meta.json")
        if not isinstance(filename, str) or not filename:
            report.error("meta.streams", f"streams['{name}'] must be a filename", "meta.json")
        elif not (session_dir / filename).exists():
            report.error("meta.streams", f"streams['{name}'] -> '{filename}' not found on disk", "meta.json")

    if "row_counts" in meta and not isinstance(meta["row_counts"], dict):
        report.warn("meta.row_counts", "row_counts should be an object", "meta.json")

    markers = meta.get("sync_markers_ms")
    if markers is not None and not isinstance(markers, list):
        report.warn("meta.sync_markers_ms", "sync_markers_ms should be a list", "meta.json")

    if not meta.get("on_task_definition_version"):
        report.warn(
            "meta.on_task_definition_version",
            "no on_task_definition_version recorded; labels are not traceable to a definition",
            "meta.json",
        )


def _is_iso(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# periodic CSV streams
# ---------------------------------------------------------------------------
def validate_stream(path: Path, stream: Stream, report: Report) -> None:
    where = stream.filename
    if not path.exists():
        report.error("file.missing", f"'{stream.filename}' not found", where)
        return

    header, rows = _read_rows(path)
    if not header:
        report.error("csv.empty", "file is empty", where)
        return

    missing = [c.name for c in stream.columns if c.required and c.name not in header]
    if missing:
        report.error("csv.columns", f"missing required columns: {missing}", where)
        return
    unknown = [c for c in header if c not in stream.column_names]
    if unknown:
        report.warn("csv.columns", f"unexpected extra columns: {unknown}", where)

    index = {c.name: header.index(c.name) for c in stream.columns if c.name in header}
    col = {c.name: c for c in stream.columns}

    if not rows:
        report.error("csv.rows", "no data rows", where)
        return

    value_errors = 0
    ts_values: list[int] = []
    dev_values: list[int] = []

    for row_no, row in enumerate(rows, start=2):
        for cname, c in col.items():
            if cname not in index:
                continue
            raw = row[index[cname]] if index[cname] < len(row) else ""
            ok, value = _coerce(raw, c.dtype)
            if not ok:
                value_errors += 1
                if value_errors <= _MAX_VALUE_ERRORS:
                    report.error(
                        "csv.value",
                        f"line {row_no}: column '{cname}' value '{raw}' is not a valid {c.dtype}",
                        where,
                    )
                continue
            if value is None and c.required and not c.nullable:
                value_errors += 1
                if value_errors <= _MAX_VALUE_ERRORS:
                    report.error(
                        "csv.null", f"line {row_no}: required column '{cname}' is empty", where
                    )
            if cname == "ts_ms" and value is not None:
                ts_values.append(value)
            elif cname == "t_dev_ms" and value is not None:
                dev_values.append(value)
            elif cname == "category" and value is not None:
                if value not in {a.value for a in AppCategory}:
                    report.warn(
                        "logger.category",
                        f"line {row_no}: category '{value}' is not a known AppCategory",
                        where,
                    )

    if value_errors > _MAX_VALUE_ERRORS:
        report.error("csv.value", f"{value_errors - _MAX_VALUE_ERRORS} more value errors suppressed", where)

    if stream.name in ("imu", "logger", "vision", "phone"):
        _check_time_order(ts_values, "ts_ms", where, report)
    if stream.name == "imu" and dev_values:
        _check_time_order(dev_values, "t_dev_ms", where, report)

    if stream.rate_hz and len(ts_values) >= 2:
        _report_rate(ts_values, stream, report)


def _check_time_order(values: list[int], name: str, where: str, report: Report) -> None:
    # Count how many times the series moves backwards (tolerate +1 jitter).
    backwards = sum(1 for a, b in zip(values, values[1:]) if b < a - 1)
    if backwards:
        report.error(
            "csv.time_order",
            f"'{name}' is not monotonic non-decreasing ({backwards} backward steps)",
            where,
        )


def _report_rate(ts_values: list[int], stream: Stream, report: Report) -> None:
    span_ms = ts_values[-1] - ts_values[0]
    if span_ms <= 0:
        report.error("csv.time_span", "timestamp span is zero; cannot measure rate", stream.filename)
        return
    measured = (len(ts_values) - 1) / (span_ms / 1000.0)
    gaps = [b - a for a, b in zip(ts_values, ts_values[1:])]
    max_gap = max(gaps) if gaps else 0
    median_dt = statistics.median(gaps) if gaps else 0

    key = f"{stream.name}_rate_hz"
    report.info[key] = round(measured, 3)
    report.info[f"{stream.name}_max_gap_ms"] = max_gap
    report.info[f"{stream.name}_rows"] = len(ts_values)

    nominal = stream.rate_hz
    if abs(measured - nominal) > 0.15 * nominal:
        report.warn(
            "csv.rate",
            f"measured rate {measured:.2f} Hz differs from nominal {nominal} Hz",
            stream.filename,
        )
    # The 100 ms gap budget is for the high-rate IMU stream. For a slow stream
    # (1 Hz) the expected gap is ~1000 ms, so the budget scales with the rate.
    expected_gap_ms = 1000.0 / nominal
    budget_ms = max(100.0, 3.0 * expected_gap_ms)
    if max_gap > budget_ms:
        report.warn(
            "csv.gap",
            f"largest gap between samples is {max_gap} ms (> {budget_ms:.0f} ms budget)",
            stream.filename,
        )
    if measured > nominal and median_dt == 0:
        report.warn("csv.rate", "duplicate timestamps detected", stream.filename)


# ---------------------------------------------------------------------------
# labels.csv
# ---------------------------------------------------------------------------
def validate_labels(path: Path, report: Report) -> None:
    where = LABELS.filename
    if not path.exists():
        report.warn("file.missing", "labels.csv not found (session is unlabelled)", where)
        return

    header, rows = _read_rows(path)
    missing = [c for c in ("start_ms", "end_ms", "activity", "task_state") if c not in header]
    if missing:
        report.error("csv.columns", f"labels.csv missing required columns: {missing}", where)
        return

    idx = {name: header.index(name) for name in header}
    allowed_states = {"ON_TASK", "DISTRACTED", "AWAY", "BREAK"}
    intervals: list[tuple[int, int]] = []
    for row_no, row in enumerate(rows, start=2):
        def get(name: str) -> str:
            return row[idx[name]] if idx[name] < len(row) else ""

        ok_s, start = _coerce(get("start_ms"), "int")
        ok_e, end = _coerce(get("end_ms"), "int")
        state = get("task_state").strip()
        activity = get("activity").strip()
        if not ok_s or start is None or not ok_e or end is None:
            report.error("labels.value", f"line {row_no}: start_ms/end_ms not integers", where)
            continue
        if end <= start:
            report.error("labels.range", f"line {row_no}: end_ms ({end}) <= start_ms ({start})", where)
        else:
            intervals.append((start, end))
        if state and state not in allowed_states:
            report.error(
                "labels.task_state",
                f"line {row_no}: task_state '{state}' not in {sorted(allowed_states)} (UNCERTAIN is runtime-only)",
                where,
            )
        if activity and activity not in PROVISIONAL_ACTIVITIES:
            report.warn(
                "labels.activity",
                f"line {row_no}: activity '{activity}' not in the provisional vocabulary",
                where,
            )

    intervals.sort()
    overlaps = sum(1 for a, b in zip(intervals, intervals[1:]) if b[0] < a[1])
    if overlaps:
        report.warn("labels.overlap", f"{overlaps} overlapping label intervals", where)
    report.info["label_intervals"] = len(intervals)

    # Coverage versus the session duration, when both are known.
    duration_s = report.info.get("duration_s")
    if intervals and duration_s:
        covered = sum(e - s for s, e in intervals) / 1000.0
        report.info["label_coverage_percent"] = round(100.0 * covered / duration_s, 1)
        if covered / duration_s > 1.05:
            report.error("labels.coverage", "labels cover more time than the session duration", where)


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------
def validate_session(session_dir: str | Path, require_labels: bool = False) -> Report:
    session_dir = Path(session_dir)
    report = Report(session=session_dir.name)

    if not session_dir.is_dir():
        report.error("session.missing", f"'{session_dir}' is not a directory")
        return report

    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        report.error("meta.missing", "meta.json not found", "meta.json")
        return report
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.error("meta.json", f"invalid JSON: {exc}", "meta.json")
        return report
    if not isinstance(meta, dict):
        report.error("meta.json", "top-level value must be an object", "meta.json")
        return report

    validate_meta(meta, session_dir, report)

    duration = meta.get("duration_s")
    if isinstance(duration, (int, float)):
        report.info["duration_s"] = float(duration)

    partial = bool(meta.get("partial", False))

    streams = meta.get("streams") if isinstance(meta.get("streams"), dict) else {}
    # Validate every stream that is declared.
    for name in streams:
        stream = STREAMS.get(name)
        if stream is None:
            continue
        validate_stream(session_dir / stream.filename, stream, report)

    # A required stream that is not declared at all.
    for name in ("imu",):
        if name not in streams:
            if partial:
                report.warn(
                    "meta.streams",
                    f"stream '{name}' is not declared (session marked partial)",
                    "meta.json",
                )
            else:
                report.error("meta.streams", f"required stream '{name}' is not declared", "meta.json")

    if require_labels and not (session_dir / LABELS.filename).exists():
        report.error("file.missing", "labels.csv required but missing", LABELS.filename)
    validate_labels(session_dir / LABELS.filename, report)

    # Files present but not declared in meta.streams.
    declared = set(streams.values()) | {"meta.json", LABELS.filename}
    for entry in session_dir.glob("*.csv"):
        if entry.name not in declared:
            report.warn("session.undeclared", f"'{entry.name}' is present but not declared in meta.streams")

    # Cross-check row counts and duration from the IMU stream.
    row_counts = meta.get("row_counts") if isinstance(meta.get("row_counts"), dict) else {}
    imu_rows = report.info.get("imu_rows")
    declared_imu = row_counts.get("imu")
    if imu_rows is not None and isinstance(declared_imu, int) and declared_imu != imu_rows:
        report.warn(
            "meta.row_counts",
            f"meta.row_counts['imu']={declared_imu} but imu.csv has {imu_rows} rows",
            "meta.json",
        )

    if isinstance(duration, (int, float)) and report.info.get("imu_rate_hz"):
        report.info["imu_expected_rows"] = round(float(duration) * report.info["imu_rate_hz"])

    return report


def validate_all(root: str | Path, require_labels: bool = False) -> list[Report]:
    """Validate every session folder under ``root``."""
    root = Path(root)
    reports: list[Report] = []
    for meta in sorted(root.glob("*/meta.json")):
        reports.append(validate_session(meta.parent, require_labels=require_labels))
    if not reports and root.is_dir():
        # Treat root itself as a single session if it holds a meta.json.
        if (root / "meta.json").exists():
            reports.append(validate_session(root, require_labels=require_labels))
    return reports
