"""Run a candidate's solution under a Python test harness in a subprocess.

The task's verifier.inline carries:
  {
    "entrypoint": "is_prime",       # function name to import
    "tests": "<python code calling entrypoint + asserting>",
    "timeout_sec": 10
  }

The candidate's `solution` is written to a temp file; the test code is then
executed in a separate `python3` process with the solution on sys.path.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any

_FENCE_MATCHED_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*(.*?)```", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Extract Python code from a string that may be fenced or prose-wrapped.

    Order of attempts:
      1. matched ```lang ... ``` block  -> inner content
      2. unmatched leading ``` (truncation) -> drop first line, drop trailing ```
      3. otherwise return as-is
    """
    if not text:
        return ""
    m = _FENCE_MATCHED_RE.search(text)
    if m:
        return m.group(1).strip()
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence line
        nl = t.find("\n")
        if nl >= 0:
            t = t[nl + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def run_python_tests(solution: str, verifier: dict[str, Any]) -> dict[str, Any]:
    entrypoint = verifier.get("entrypoint", "")
    tests = verifier.get("tests", "")
    timeout = float(verifier.get("timeout_sec", 10))

    # Strip markdown fences that small models frequently emit around code
    code = strip_code_fences(solution)

    with tempfile.TemporaryDirectory() as tmp:
        solfile = os.path.join(tmp, "solution.py")
        with open(solfile, "w") as fh:
            fh.write(code + "\n")

        harness = textwrap.dedent(f"""
            import json, sys, traceback
            sys.path.insert(0, {tmp!r})
            try:
                from solution import {entrypoint}  # type: ignore
            except Exception as e:
                print(json.dumps({{
                    "status": "import_error",
                    "error": repr(e),
                    "trace": traceback.format_exc(),
                }}))
                sys.exit(0)

            failures = []
            try:
{textwrap.indent(tests, " " * 16)}
            except AssertionError as e:
                failures.append({{"type": "assertion", "msg": str(e)}})
            except Exception as e:
                failures.append({{"type": "exception", "msg": repr(e),
                                  "trace": traceback.format_exc()}})
            print(json.dumps({{
                "status": "ok" if not failures else "fail",
                "failures": failures,
            }}))
        """).strip()

        harnessfile = os.path.join(tmp, "harness.py")
        with open(harnessfile, "w") as fh:
            fh.write(harness)

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, harnessfile],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
            duration = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "duration_ms": int(timeout * 1000)}

        out = (proc.stdout or "").strip().splitlines()
        last = out[-1] if out else ""
        try:
            parsed = json.loads(last)
        except json.JSONDecodeError:
            return {
                "status": "harness_error",
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
                "duration_ms": int(duration * 1000),
            }
        parsed["duration_ms"] = int(duration * 1000)
        parsed["stderr_tail"] = (proc.stderr or "")[-1000:]
        return parsed
