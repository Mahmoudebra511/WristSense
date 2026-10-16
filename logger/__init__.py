"""Laptop activity logger.

Counts keyboard/mouse activity and maps the active window to a coarse
category, writing one row per second to ``logger.csv`` in the AAC schema.

Privacy by construction: characters are never recorded, window titles are read
locally only to decide a category, and only the category name is stored.
"""

from .config import Categorizer, LoggerConfig
from .recorder import Recorder
from .writer import CsvWriter, MemoryWriter

__all__ = ["Categorizer", "LoggerConfig", "Recorder", "CsvWriter", "MemoryWriter"]
