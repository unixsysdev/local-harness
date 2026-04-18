"""Stage 3a: deterministic validation.

Dispatches to the validator named by the task's VerifierSpec.
Runs synchronously in a thread pool to avoid blocking the async loop.
"""
from __future__ import annotations
import asyncio
import time
from typing import Any

from ..types import Candidate, Task, ValidatorResult
from ..validation.json_schema import validate as json_schema_validate
from ..validation.tests import run_python_tests
from .jsonutil import parse_json_lenient


def _validate_one_sync(candidate: Candidate, task: Task) -> ValidatorResult:
    kind = task.verifier.kind
    inline = task.verifier.inline or {}
    t0 = time.perf_counter()

    if kind == "tests":
        result = run_python_tests(candidate.solution, inline)
        passed = result.get("status") == "ok"
        score = 1.0 if passed else 0.0
        # Near-miss ranking: import error is closer than timeout; fewer failures -> higher partial
        if not passed:
            status = result.get("status")
            if status == "fail":
                n_fail = len(result.get("failures") or [])
                score = max(0.0, 0.5 - 0.1 * n_fail)
            elif status == "import_error":
                score = 0.1
            elif status == "timeout":
                score = 0.0
            elif status == "harness_error":
                score = 0.0
        return ValidatorResult(
            candidate_id=candidate.candidate_id,
            kind=kind, passed=passed, score=score,
            signals=result,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )

    if kind == "json_schema":
        schema = inline.get("schema") or {}
        try:
            instance = parse_json_lenient(candidate.solution)
        except ValueError as e:
            return ValidatorResult(
                candidate_id=candidate.candidate_id,
                kind=kind, passed=False, score=0.0,
                signals={"parse_error": str(e)},
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
        ok, errors = json_schema_validate(instance, schema)
        # Near-miss: fewer errors -> higher partial score
        score = 1.0 if ok else max(0.0, 0.6 - 0.1 * len(errors))
        return ValidatorResult(
            candidate_id=candidate.candidate_id,
            kind=kind, passed=ok, score=score,
            signals={"errors": errors, "instance": instance},
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )

    if kind == "regex":
        import re
        pattern = inline.get("pattern") or ".*"
        m = re.search(pattern, candidate.solution, re.DOTALL)
        passed = m is not None
        return ValidatorResult(
            candidate_id=candidate.candidate_id,
            kind=kind, passed=passed, score=1.0 if passed else 0.0,
            signals={"matched": bool(m)},
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )

    # "none" or unknown -> inert pass (will not gate anything)
    return ValidatorResult(
        candidate_id=candidate.candidate_id,
        kind="none", passed=True, score=0.5, signals={},
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


async def validate_candidates(candidates: list[Candidate], task: Task) -> list[ValidatorResult]:
    if task.verifier.kind == "none":
        return []
    loop = asyncio.get_running_loop()
    return await asyncio.gather(*(
        loop.run_in_executor(None, _validate_one_sync, c, task) for c in candidates
    ))
