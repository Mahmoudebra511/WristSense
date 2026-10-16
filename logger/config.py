"""User-editable category rules for the laptop logger (Section 5.3).

The window **process name** and **title** are read locally and matched against
these rules to produce a coarse category. Only the category is written to disk;
the process/title are never stored. This is the "whitelist defined by the user"
of the project document.

Matching is case-insensitive and uses regular expressions. Rules are evaluated
in two passes, so a process match always beats a title match:

1. every category's ``processes`` list, in the order declared in the JSON;
2. every category's ``titles`` list, in the same order.

If nothing matches, ``default_category`` is used (``other`` by default).
If the process matches ``exclude_processes`` the sample is *ignored*: the
previous known category is kept, which is useful for the terminal that hosts
the logger itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from aacdata.enums import AppCategory

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass(frozen=True)
class Rule:
    category: str
    processes: tuple[re.Pattern, ...]
    titles: tuple[re.Pattern, ...]


@dataclass(frozen=True)
class LoggerConfig:
    rules: tuple[Rule, ...]
    exclude_processes: tuple[re.Pattern, ...]
    default_category: str
    poll_hz: int
    hash: str
    source_path: str

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LoggerConfig":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        text = p.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"{p}: config must be a JSON object")
        return cls.from_dict(data, text=text, source_path=str(p))

    @classmethod
    def from_dict(cls, data: dict, text: str | None = None,
                  source_path: str = "<dict>") -> "LoggerConfig":
        rules: list[Rule] = []
        for category, spec in (data.get("categories") or {}).items():
            if category not in {c.value for c in AppCategory}:
                raise ValueError(f"unknown category '{category}' in config")
            spec = spec or {}
            rules.append(
                Rule(
                    category=category,
                    processes=tuple(re.compile(p, re.I) for p in spec.get("processes", [])),
                    titles=tuple(re.compile(p, re.I) for p in spec.get("titles", [])),
                )
            )
        excludes = tuple(re.compile(p, re.I) for p in data.get("exclude_processes", []))
        default = data.get("default_category", AppCategory.OTHER.value)
        if default not in {c.value for c in AppCategory}:
            raise ValueError(f"unknown default_category '{default}'")
        poll_hz = int(data.get("poll_hz", 4))
        if poll_hz < 1:
            raise ValueError("poll_hz must be >= 1")
        payload = text if text is not None else json.dumps(data, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return cls(
            rules=tuple(rules),
            exclude_processes=excludes,
            default_category=default,
            poll_hz=poll_hz,
            hash=digest,
            source_path=source_path,
        )


class Categorizer:
    """Turn (process, title) into an :class:`AppCategory` value, or ``None``."""

    def __init__(self, config: LoggerConfig):
        self.config = config

    def is_excluded(self, process: str | None) -> bool:
        if not process:
            return False
        return any(p.search(process) for p in self.config.exclude_processes)

    def categorize(self, process: str | None, title: str | None) -> str | None:
        """Return a category value, or ``None`` to mean "keep the previous one"."""
        if self.is_excluded(process):
            return None
        proc = process or ""
        ttl = title or ""
        for rule in self.config.rules:
            if any(p.search(proc) for p in rule.processes):
                return rule.category
        for rule in self.config.rules:
            if any(p.search(ttl) for p in rule.titles):
                return rule.category
        return self.config.default_category
