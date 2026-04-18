"""Baseline triage for BOTH models: for each task, run a single-shot baseline
on model_a AND model_b (both with thinking on), then run the task's validator
on both outputs. Report pass/fail per (task, model) plus wall time.

The harness can only demonstrate value on tasks where BOTH models fail single-
shot; if either model already solves the task, running 12 candidates and cross-
evaluating is wasted compute.
"""
from __future__ import annotations
import argparse
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


async def baseline_and_validate(task, backend, model_key: str, cfg: Config):
    t0 = time.perf_counter()
    bl = await run_baseline(
        task, backend,
        model_key=model_key,
        temperature=cfg.baseline.temperature,
        max_tokens=cfg.baseline.max_tokens,
        thinking=cfg.baseline.thinking,
    )
    wall = int((time.perf_counter() - t0) * 1000)

    text = bl.get("text", "") or ""
    extracted = extract_solution_for_verifier(text, task.verifier.kind)
    pseudo = Candidate(
        model=f"baseline_{model_key}", strategy="baseline", temperature=0.0,
        solution=extracted,
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
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="tasks_r3")
    p.add_argument("--config", default="config_r3.yaml")
    p.add_argument("--output", default="baseline_triage.json")
    args = p.parse_args()

    cfg = Config.load(args.config)
    model_keys = list(cfg.models.keys())
    backend = OpenAICompatBackend(
        endpoints={k: v.endpoint for k, v in cfg.models.items()},
        served_names={k: v.served_name for k, v in cfg.models.items()},
        thinking_defaults={k: v.thinking_default for k, v in cfg.models.items()},
    )

    results = []
    try:
        files = sorted(Path(args.tasks).glob("*.json"))
        print(f"Triaging {len(files)} tasks with BOTH models "
              f"(thinking={cfg.baseline.thinking})  "
              f"— firing all {len(files)*len(model_keys)} calls concurrently",
              file=sys.stderr, flush=True)

        # Build the full (task, model) cross-product and fire ALL calls concurrently.
        # Server --parallel 6 means each server runs 6 slots in parallel and queues
        # the rest — so 12 tasks -> ~2 sequential batches of 6 per server.
        tasks_loaded = [(f, load_task(f)) for f in files]
        flat = [(f, task, mk) for (f, task) in tasks_loaded for mk in model_keys]

        async def one(f, task, mk):
            r = await baseline_and_validate(task, backend, mk, cfg)
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {f.name:40s} {mk:10s} "
                  f"wall={r['wall_ms']/1000:.1f}s out={r['output_tokens']}tok",
                  file=sys.stderr, flush=True)
            return (f, task, mk, r)

        all_results = await asyncio.gather(*(one(f, t, mk) for (f, t, mk) in flat))

        # Regroup by task
        by_task: dict[str, dict] = {}
        for f, task, mk, r in all_results:
            by_task.setdefault(f.name, {"task_path": str(f), "model_results": []})
            by_task[f.name]["model_results"].append(r)

        # Disposition per task
        for tname, entry in by_task.items():
            mrs = entry["model_results"]
            passes = [r["model"] for r in mrs if r["passed"]]
            fails = [r["model"] for r in mrs if not r["passed"]]
            if len(passes) == len(model_keys):
                disposition = "BOTH_PASS (drop - too easy)"
            elif len(fails) == len(model_keys):
                disposition = "BOTH_FAIL (KEEP - sweet spot)"
            else:
                disposition = f"ASYMMETRIC (KEEP - test if harness rescues {fails[0]})"
            entry["disposition"] = disposition
            results.append({"task": tname, **entry})
    finally:
        await backend.close()

    Path(args.output).write_text(json.dumps(results, indent=2, default=str))

    print(f"\n=== TRIAGE SUMMARY ===", file=sys.stderr)
    both_pass = [r for r in results if "BOTH_PASS" in r["disposition"]]
    both_fail = [r for r in results if "BOTH_FAIL" in r["disposition"]]
    asym = [r for r in results if "ASYMMETRIC" in r["disposition"]]
    print(f"  total tasks:          {len(results)}", file=sys.stderr)
    print(f"  both_pass (drop):     {len(both_pass)}", file=sys.stderr)
    print(f"  asymmetric (drop):    {len(asym)}", file=sys.stderr)
    print(f"  both_fail (KEEP):     {len(both_fail)}  <- sweet spot for harness experiment",
          file=sys.stderr)
    print(f"\n  Sweet-spot tasks (run harness on these):", file=sys.stderr)
    for r in both_fail + asym:
        print(f"    - {r['task']:40s} [{r['disposition'].split(' ')[0]}]",
              file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
