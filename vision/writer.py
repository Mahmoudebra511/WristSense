"""CSV sink for ``vision.csv`` matching :mod:`aacdata.columns.VISION`."""

from __future__ import annotations

import csv
from pathlib import Path

from aacdata.columns import VISION

HEADER = list(VISION.column_names)


class VisionCsvWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(HEADER)
        self._fh.flush()
        self.count = 0

    def write(self, row: dict) -> None:
        self._writer.writerow([row.get(name, "") for name in HEADER])
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "VisionCsvWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
