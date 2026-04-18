"""CLI: python -m harness.evaluator [--tasks DIR] [--config FILE] [--output FILE]

Walks a tasks directory, runs baseline + harness on each, evaluates per-task,
writes a JSON report. Produces per-task verdicts and an aggregate summary.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from ..app import run as run_pipeline
from ..backends.openai_compat import OpenAICompatBackend
from ..config import Config
from ..pipeline.normalize import load_task
from .core import pairwise_compare, score_verifiable


def _find_tasks(tasks_dir: Path) -> list[Path]:
    return sorted(p for p in tasks_dir.glob("*.json") if p.is_file())


async def evaluate_one(task_path: Path, cfg: Config,
                       judge_backend: OpenAICompatBackend,
                       judges: list[str], quiet: bool) -> dict:
    t0 = time.perf_counter()
    out = await run_pipeline(cfg, str(task_path), do_baseline=True,
                             do_harness=True, quiet=quiet)
    wall_ms_total = int((time.perf_counter() - t0) * 1000)

    task = load_task(task_path)

    baseline_text = (out.get("baseline") or {}).get("text", "") or ""
    baseline_wall = (out.get("baseline") or {}).get("wall_ms", 0) or 0

    final = out.get("final") or {}
    harness_text = final.get("text", "") or ""
    harness_wall = final.get("wall_ms", 0) or 0

    record: dict = {
        "task_file": str(task_path),
        "task_id": out.get("task_id"),
        "run_id": out.get("run_id"),
        "task_class": task.task_class,
        "baseline_wall_ms": baseline_wall,
        "harness_wall_ms_total": wall_ms_total,  # full run including rubric/propose/eval
        "baseline_output_preview": baseline_text[:240],
        "harness_output_preview": harness_text[:240],
    }

    if task.task_class == "verifiable":
        v = score_verifiable(
            task=task, baseline_text=baseline_text,
            baseline_wall_ms=baseline_wall,
            harness_final_text=harness_text, harness_wall_ms=harness_wall,
            task_file=str(task_path),
        )
        record["mode"] = "verifiable"
        record["baseline_pass"] = v.baseline_pass
        record["harness_pass"] = v.harness_pass
        record["verdict"] = v.verdict
        record["baseline_signals"] = v.baseline_signals
        record["harness_signals"] = v.harness_signals
    else:
        v = await pairwise_compare(
            task=task, baseline_text=baseline_text,
            harness_final_text=harness_text,
            backend=judge_backend, judges=judges,
            max_tokens=cfg.sampling.eval_max_tokens,
            task_file=str(task_path),
            baseline_wall_ms=baseline_wall, harness_wall_ms=harness_wall,
            judge_weights=cfg.judge_calibration.weights,
            non_unanimous_penalty=cfg.judge_calibration.non_unanimous_penalty,
        )
        record["mode"] = "pairwise"
        record["harness_wins"] = v.harness_wins
        record["baseline_wins"] = v.baseline_wins
        record["ties"] = v.ties
        record["total_calls"] = v.total_calls
        record["order_consistent_per_judge"] = v.order_consistent_per_judge
        record["weighted_harness"] = v.weighted_harness
        record["weighted_baseline"] = v.weighted_baseline
        record["judge_weights_used"] = v.judge_weights_used
        record["verdict"] = v.verdict
        record["calls"] = [asdict(c) for c in v.calls]

    return record


def _summarize(records: list[dict]) -> dict:
    n = len(records)
    harness_wins = sum(1 for r in records if r.get("verdict") == "harness_win")
    baseline_wins = sum(1 for r in records if r.get("verdict") == "baseline_win")
    ties = sum(1 for r in records if r.get("verdict") in ("tie", "both_pass", "both_fail"))
    unresolved = sum(1 for r in records if r.get("verdict") == "unresolved")
    errors = sum(1 for r in records if r.get("verdict") == "error")

    baseline_walls = [r.get("baseline_wall_ms") or 0 for r in records]
    harness_walls = [r.get("harness_wall_ms_total") or 0 for r in records]

    # Judge bias aggregation: across all non-verifiable tasks, how often was
    # each judge's two orderings positionally consistent, and how often did
    # it vote for harness vs baseline on each side?
    judge_stats: dict[str, dict] = {}
    for rec in records:
        if rec.get("mode") != "pairwise":
            continue
        consistency = rec.get("order_consistent_per_judge") or {}
        for judge, consistent in consistency.items():
            s = judge_stats.setdefault(judge, {"tasks": 0, "consistent": 0,
                                               "harness_votes": 0, "baseline_votes": 0,
                                               "tie_votes": 0, "parse_errors": 0,
                                               "A_votes": 0, "B_votes": 0})
            s["tasks"] += 1
            if consistent:
                s["consistent"] += 1
        for c in rec.get("calls") or []:
            j = c.get("judge")
            if not j:
                continue
            s = judge_stats.setdefault(j, {"tasks": 0, "consistent": 0,
                                          "harness_votes": 0, "baseline_votes": 0,
                                          "tie_votes": 0, "parse_errors": 0,
                                          "A_votes": 0, "B_votes": 0})
            hw = c.get("harness_won")
            if hw is True:
                s["harness_votes"] += 1
            elif hw is False:
                s["baseline_votes"] += 1
            elif c.get("winner_label") == "TIE":
                s["tie_votes"] += 1
            else:
                s["parse_errors"] += 1
            wl = c.get("winner_label")
            if wl == "A":
                s["A_votes"] += 1
            elif wl == "B":
                s["B_votes"] += 1

    # Derived metrics
    for j, s in judge_stats.items():
        total_votes = s["A_votes"] + s["B_votes"]
        s["consistency_rate"] = round(s["consistent"] / s["tasks"], 3) if s["tasks"] else 0.0
        s["positional_bias_A"] = round(s["A_votes"] / total_votes, 3) if total_votes else 0.0

    return {
        "total_tasks": n,
        "harness_wins": harness_wins,
        "baseline_wins": baseline_wins,
        "ties_or_both": ties,
        "unresolved": unresolved,
        "errors": errors,
        "harness_win_rate": round(harness_wins / n, 3) if n else 0.0,
        "baseline_win_rate": round(baseline_wins / n, 3) if n else 0.0,
        "avg_baseline_wall_s": round(sum(baseline_walls) / (1000 * n), 2) if n else 0.0,
        "avg_harness_wall_s": round(sum(harness_walls) / (1000 * n), 2) if n else 0.0,
        "judge_stats": judge_stats,
    }


async def amain(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    tasks_dir = Path(args.tasks)
    files = _find_tasks(tasks_dir)
    if not files:
        print(f"No task JSON files found in {tasks_dir}", file=sys.stderr)
        return 2

    # Judge backend (for non-verifiable pairwise calls) -- shared client
    judge_backend = OpenAICompatBackend(
        endpoints={k: v.endpoint for k, v in cfg.models.items()},
        served_names={k: v.served_name for k, v in cfg.models.items()},
        thinking_defaults={k: v.thinking_default for k, v in cfg.models.items()},
    )
    judges = list(cfg.models.keys())

    records: list[dict] = []
    try:
        for f in files:
            print(f"\n=== evaluating {f.name} ===", file=sys.stderr, flush=True)
            try:
                rec = await evaluate_one(f, cfg, judge_backend, judges, quiet=args.quiet)
                records.append(rec)
                print(f"  -> verdict: {rec['verdict']}", file=sys.stderr, flush=True)
            except Exception as e:
                import traceback
                err_rec = {
                    "task_file": str(f),
                    "verdict": "error",
                    "error": repr(e),
                    "trace": traceback.format_exc()[-2000:],
                }
                records.append(err_rec)
                print(f"  -> ERROR: {e}", file=sys.stderr, flush=True)
            # persist incrementally so a later crash doesn't lose work
            Path(args.output).write_text(json.dumps(
                {"tasks": records, "summary": _summarize(records)},
                indent=2, default=str))
    finally:
        await judge_backend.close()

    summary = _summarize(records)
    report = {
        "config_file": args.config,
        "tasks_dir": str(tasks_dir),
        "summary": summary,
        "tasks": records,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, default=str))

    print("\n=== SUMMARY ===", file=sys.stderr)
    for k, v in summary.items():
        print(f"  {k:24} {v}", file=sys.stderr)
    print(f"\nReport: {args.output}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="harness.evaluator")
    p.add_argument("--tasks", default="tasks", help="directory of task JSON files")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default="eval_report.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
