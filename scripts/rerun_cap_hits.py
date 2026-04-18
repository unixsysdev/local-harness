"""Rerun specific (task, model) pairs with max_tokens effectively unlimited
(capped only by the server's context). Uses the streaming adapter so there's
no total-request timeout either — only a per-chunk idle timeout.

Targets: entries where the original run hit the max_tokens cap (so the model
ran out of budget mid-generation, not because it couldn't solve the task).
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
from pathlib import Path

from harness.backends.openai_compat import OpenAICompatBackend
from harness.baseline import run_baseline
from harness.config import Config
from harness.evaluator.core import extract_solution_for_verifier
from harness.pipeline.normalize import load_task
from harness.pipeline.validate import _validate_one_sync
from harness.types import Candidate


# Server context is 49152; baseline prompt ~400 tokens. 48000 leaves safe margin.
UNLIMITED_MAX_TOKENS = 48000

# Specific (task_filename, model) pairs to rerun
TARGETS = [
    ("coding_word_break.json", "model_a"),
    ("coding_valid_number.json", "model_a"),
]


async def one(task, backend, model_key, cfg):
    t0 = time.perf_counter()
    bl = await run_baseline(
        task, backend,
        model_key=model_key,
        temperature=cfg.baseline.temperature,
        max_tokens=UNLIMITED_MAX_TOKENS,
        thinking=cfg.baseline.thinking,
    )
    wall = int((time.perf_counter() - t0) * 1000)
    text = bl.get("text", "") or ""
    extracted = extract_solution_for_verifier(text, task.verifier.kind)
    pseudo = Candidate(
        model=f"baseline_{model_key}", strategy="baseline_uncapped",
        temperature=0.0, solution=extracted,
    )
    vr = _validate_one_sync(pseudo, task)
    return {
        "model": model_key,
        "passed": vr.passed,
        "wall_ms": wall,
        "output_tokens": bl.get("output_tokens"),
        "finish_reason": bl.get("finish_reason"),
        "truncated": bl.get("truncated"),
        "validator_summary": {
            k: v for k, v in (vr.signals or {}).items()
            if k in ("status", "failures", "error", "errors", "parse_error")
        },
        "output_preview": text[:280] if text else "",
        "_rerun_uncapped": True,
    }


async def main():
    cfg = Config.load("config_r3.yaml")
    triage_path = Path("baseline_triage.json")
    triage = json.loads(triage_path.read_text())

    backend = OpenAICompatBackend(
        endpoints={k: v.endpoint for k, v in cfg.models.items()},
        served_names={k: v.served_name for k, v in cfg.models.items()},
        thinking_defaults={k: v.thinking_default for k, v in cfg.models.items()},
    )

    print(f"Rerunning {len(TARGETS)} cap-hit entries with max_tokens={UNLIMITED_MAX_TOKENS} "
          f"(streaming, no total-request timeout)",
          file=sys.stderr, flush=True)
    for tn, mk in TARGETS:
        print(f"  - {tn} [{mk}]", file=sys.stderr)

    tasks_by_name = {tn: load_task(Path("tasks_r3") / tn) for (tn, _) in TARGETS}

    async def run_pair(tn, mk):
        t = tasks_by_name[tn]
        r = await one(t, backend, mk, cfg)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [uncapped {status}] {tn:40s} {mk:10s} "
              f"wall={r['wall_ms']/1000:.1f}s out={r['output_tokens']}tok "
              f"finish={r.get('finish_reason')}",
              file=sys.stderr, flush=True)
        return (tn, mk, r)

    try:
        results = await asyncio.gather(*(run_pair(tn, mk) for (tn, mk) in TARGETS))
    finally:
        await backend.close()

    # merge into triage
    new_results = {(tn, mk): r for (tn, mk, r) in results}
    for entry in triage:
        for i, mr in enumerate(entry["model_results"]):
            key = (entry["task"], mr["model"])
            if key in new_results:
                entry["model_results"][i] = new_results[key]
        mrs = entry["model_results"]
        passes = [r["model"] for r in mrs if r["passed"]]
        fails = [r["model"] for r in mrs if not r["passed"]]
        n_models = len(mrs)
        if len(passes) == n_models:
            entry["disposition"] = "BOTH_PASS (drop - too easy)"
        elif len(fails) == n_models:
            entry["disposition"] = "BOTH_FAIL (KEEP - sweet spot)"
        else:
            entry["disposition"] = f"ASYMMETRIC (KEEP - test if harness rescues {fails[0]})"

    triage_path.write_text(json.dumps(triage, indent=2, default=str))
    print("\n=== POST-UNCAPPED-RERUN FINAL ===", file=sys.stderr)
    for e in triage:
        d = e["disposition"].split(" ")[0]
        print(f"  {e['task']:40s} {d}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
