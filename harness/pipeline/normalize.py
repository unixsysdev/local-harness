"""Stage 0: task normalization.

Tasks are stored as JSON files that already include most of the normalized
fields (so the harness can be run deterministically without invoking a model
for classification). If a field is missing, sensible defaults are applied.
"""
from __future__ import annotations
import json
from pathlib import Path

from ..types import Task, VerifierSpec


def load_task(path: str | Path) -> Task:
    data = json.loads(Path(path).read_text())
    verifier = data.pop("verifier", None)
    task = Task(**data)
    if verifier is not None:
        task.verifier = VerifierSpec(**verifier)
    return task
