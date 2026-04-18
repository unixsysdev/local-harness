"""Run ONLY the harness phase of Run 3, reading the already-computed Phase 1
baseline N=3 results from r3_baseline_n3.json. Writes r3_harness_results.json
and r3_summary.json.

Used when Phase 1 of the full experiment already completed but Phase 2 needs
to rerun (e.g. after a config fix for context-size exceeded).
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
from pathlib import Path

from harness.app import run as run_pipeline
from harness.config import Config
from harness.evaluator.core import extract_solution_for_verifier
from harness.pipeline.normalize import load_task
from harness.pipeline.validate import _validate_one_sync
from harness.types import Candidate


async def harness_one(path, cfg):
    print(f"\n  === {path.name} ===", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    out = await run_pipeline(cfg, str(path),
                             do_baseline=True, do_harness=True, quiet=False)
    wall_total = int((time.perf_counter() - t0) * 1000)
    task = load_task(path)
    final_text = (out.get("final") or {}).get("text", "") or ""
    extracted = extract_solution_for_verifier(final_text, task.verifier.kind)
    pseudo = Candidate(
        model="harness_final", strategy="harness",
        temperature=0.0, solution=extracted,
    )
    vr = _validate_one_sync(pseudo, task)
    verdict = "PASS" if vr.passed else "FAIL"
    print(f"  harness verdict: {verdict}  wall={wall_total/1000:.1f}s",
          file=sys.stderr, flush=True)
    return {
        "task": path.name,
        "harness_passed": vr.passed,
        "wall_ms_total": wall_total,
        "baseline_inside_run": out.get("baseline"),
        "harness_final_text": final_text[:2000],
        "run_id": out.get("run_id"),
    }


def summarize(baseline_n3, harness_results):
    by_tm: dict[tuple[str, str], list[bool]] = {}
    for r in baseline_n3:
        by_tm.setdefault((r["task"], r["model"]), []).append(r["passed"])

    per_task = {}
    for r in harness_results:
        t = r["task"]
        per_task[t] = {
            "harness_passed": r["harness_passed"],
            "harness_wall_s": round(r["wall_ms_total"] / 1000, 1),
            "baseline_pass_rate_by_model": {},
        }
        for (tt, mk), samples in by_tm.items():
            if tt == t:
                per_task[t]["baseline_pass_rate_by_model"][mk] = {
                    "pass_count": sum(samples),
                    "total": len(samples),
                    "pass_rate": round(sum(samples) / len(samples), 3),
                }
    return per_task


async def main():
    cfg = Config.load("config_r3_unlimited.yaml")
    task_paths = sorted(Path("tasks_r3_sweet").glob("*.json"))
    baseline_n3 = json.loads(Path("r3_baseline_n3.json").read_text())

    print(f"Running harness on {len(task_paths)} tasks (sequential; each is "
          f"heavy)", file=sys.stderr, flush=True)

    harness_results = []
    for p in task_paths:
        r = await harness_one(p, cfg)
        harness_results.append(r)
        Path("r3_harness_results.json").write_text(
            json.dumps(harness_results, indent=2, default=str))

    summary = summarize(baseline_n3, harness_results)
    Path("r3_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print("\n=== FINAL SUMMARY ===", file=sys.stderr)
    for t, s in summary.items():
        print(f"\n  {t}", file=sys.stderr)
        print(f"    harness: {'PASS' if s['harness_passed'] else 'FAIL'}  "
              f"({s['harness_wall_s']}s)", file=sys.stderr)
        for mk, rate in s["baseline_pass_rate_by_model"].items():
            print(f"    baseline {mk}: {rate['pass_count']}/{rate['total']} "
                  f"= {rate['pass_rate']*100:.0f}%", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
