"""Run 3 experiment driver:
  Phase 1: baseline N=3 per (task, model) — fires all 18 calls concurrently.
  Phase 2: harness on each of 3 tasks — baseline counted from the N=3 pool;
           harness runs with default pipeline (repair gated, kv-unified, etc).

Writes:
  r3_baseline_n3.json     -- per (task, model) list of 3 runs with pass/fail
  r3_harness_results.json -- per task, harness final output + pass/fail
  r3_summary.json         -- combined summary + baseline_pass_rate per model

All calls use the streaming adapter (per-chunk idle timeout only).
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
from pathlib import Path
from statistics import mean

from harness.app import run as run_pipeline
from harness.backends.openai_compat import OpenAICompatBackend
from harness.baseline import run_baseline
from harness.config import Config
from harness.evaluator.core import extract_solution_for_verifier
from harness.pipeline.normalize import load_task
from harness.pipeline.validate import _validate_one_sync
from harness.types import Candidate


N_SAMPLES = 3


async def baseline_sample(task, backend, model_key: str, cfg: Config, run_idx: int):
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
        model=f"baseline_{model_key}", strategy=f"sample_{run_idx}",
        temperature=0.0, solution=extracted,
    )
    vr = _validate_one_sync(pseudo, task)
    return {
        "task": None,           # filled by caller
        "model": model_key,
        "sample_idx": run_idx,
        "passed": vr.passed,
        "wall_ms": wall,
        "output_tokens": bl.get("output_tokens"),
        "finish_reason": bl.get("finish_reason"),
        "output_preview": text[:240],
    }


async def phase1_baseline_n3(task_paths, cfg):
    """Fire all (task, model, sample) triples concurrently."""
    model_keys = list(cfg.models.keys())
    backend = OpenAICompatBackend(
        endpoints={k: v.endpoint for k, v in cfg.models.items()},
        served_names={k: v.served_name for k, v in cfg.models.items()},
        thinking_defaults={k: v.thinking_default for k, v in cfg.models.items()},
    )

    tasks_loaded = [(p, load_task(p)) for p in task_paths]
    plan = []
    for p, task in tasks_loaded:
        for mk in model_keys:
            for i in range(N_SAMPLES):
                plan.append((p, task, mk, i))

    print(f"Phase 1: {len(plan)} baseline calls "
          f"({len(task_paths)} tasks × {len(model_keys)} models × {N_SAMPLES} samples)",
          file=sys.stderr, flush=True)

    async def one(p, task, mk, i):
        r = await baseline_sample(task, backend, mk, cfg, i)
        r["task"] = p.name
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {p.name:30s} {mk:10s} sample={i} "
              f"wall={r['wall_ms']/1000:.1f}s out={r['output_tokens']}tok",
              file=sys.stderr, flush=True)
        return r

    try:
        results = await asyncio.gather(*(one(p, t, mk, i) for (p, t, mk, i) in plan))
    finally:
        await backend.close()
    return results


async def phase2_harness(task_paths, cfg):
    """Run the harness (baseline + full pipeline) once per task."""
    print(f"\nPhase 2: harness on {len(task_paths)} tasks", file=sys.stderr, flush=True)
    results = []
    for p in task_paths:
        print(f"\n  === {p.name} ===", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        out = await run_pipeline(cfg, str(p),
                                 do_baseline=True, do_harness=True, quiet=False)
        wall_total = int((time.perf_counter() - t0) * 1000)
        # Validate final harness output against the task validator
        task = load_task(p)
        final_text = (out.get("final") or {}).get("text", "") or ""
        extracted = extract_solution_for_verifier(final_text, task.verifier.kind)
        pseudo = Candidate(
            model="harness_final", strategy="harness",
            temperature=0.0, solution=extracted,
        )
        vr = _validate_one_sync(pseudo, task)
        results.append({
            "task": p.name,
            "harness_passed": vr.passed,
            "wall_ms_total": wall_total,
            "baseline_inside_run": out.get("baseline"),
            "harness_final_text": final_text[:2000],
            "run_id": out.get("run_id"),
        })
        print(f"  harness verdict: {'PASS' if vr.passed else 'FAIL'}  "
              f"wall={wall_total/1000:.1f}s", file=sys.stderr, flush=True)
    return results


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

    baseline_n3 = await phase1_baseline_n3(task_paths, cfg)
    Path("r3_baseline_n3.json").write_text(
        json.dumps(baseline_n3, indent=2, default=str))

    harness_results = await phase2_harness(task_paths, cfg)
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
