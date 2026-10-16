"""Row sinks for the logger. One implements the schema, one is for tests."""

from __future__ import annotations

import csv
from pathlib import Path

from aacdata.columns import LOGGER

HEADER = list(LOGGER.column_names)


class CsvWriter:
    """Append-ready writer for ``logger.csv``. Flushes every row on purpose."""

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

    def __enter__(self) -> "CsvWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MemoryWriter:
    """Keeps rows in a list instead of a file, for tests."""

    def __init__(self) -> None:
        self.header = HEADER
        self.rows: list[dict] = []

    def write(self, row: dict) -> None:
        self.rows.append(dict(row))

    def close(self) -> None:
        pass

    @property
    def count(self) -> int:
        return len(self.rows)
