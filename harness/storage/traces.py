"""JSONL trace store. Append-only; one file per run."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


class TraceWriter:
    """Append-only JSONL writer scoped to a single run."""

    def __init__(self, trace_dir: str | Path, run_id: str | None = None):
        self.run_id = run_id or uuid4().hex[:12]
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.run_id}.jsonl"
        self._fh = open(self.path, "a", buffering=1)  # line-buffered
        self.started_at = time.time()

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        rec = {
            "ts": time.time(),
            "run_id": self.run_id,
            "kind": kind,
            "payload": payload,
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
