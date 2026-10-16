"""aacdata — the AAC session data contract.

Single source of truth for the *on-disk* format of a recording session:
the per-stream CSV columns, the ``labels.csv`` table and ``meta.json``.

Nothing about the machine-learning model lives here on purpose. This package
only answers the question "what does a valid session look like on disk?", so
that training and runtime can never silently drift apart.
"""

from .constants import SCHEMA_VERSION, ON_TASK_DEFINITION_VERSION
from . import columns, enums, models, validate, session

__all__ = [
    "SCHEMA_VERSION",
    "ON_TASK_DEFINITION_VERSION",
    "columns",
    "enums",
    "models",
    "validate",
    "session",
]
