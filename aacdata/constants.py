"""Version numbers that pin the data contract.

Bump ``SCHEMA_VERSION`` on any breaking change to column names, units or the
``meta.json`` shape. Old sessions stay readable by declaring their own
``schema_version`` inside ``meta.json``.
"""

# On-disk session format version. See docs/schema.md.
SCHEMA_VERSION = "1.0.0"

# Version of the frozen on-task definition. See docs/on_task_definition.md.
# A recording must state which definition was in force when it was labelled;
# this number is what makes that traceable.
ON_TASK_DEFINITION_VERSION = "1.0.0-draft"
