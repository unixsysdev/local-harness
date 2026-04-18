"""Re-run baseline triage entries that produced 0 output tokens (client timeouts).

Reads baseline_triage.json, identifies (task, model) pairs where the original
run failed with out_tokens == 0 (i.e. the httpx 600s timeout fired before the
model produced anything). Re-runs each with timeout=1800s and max_tokens=16384,
then writes a merged baseline_triage.json with the same shape.
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


async def rerun_one(task, backend, model_key: str, max_tokens: int, thinking: bool,
                    temperature: float):
    t0 = time.perf_counter()
    bl = await run_baseline(
        task, backend, model_key=model_key,
        temperature=temperature, max_tokens=max_tokens, thinking=thinking,
    )
    wall = int((time.perf_counter() - t0) * 1000)
    text = bl.get("text", "") or ""
    extracted = extract_solution_for_verifier(text, task.verifier.kind)
    pseudo = Candidate(
        model=f"baseline_{model_key}", strategy="baseline_rerun", temperature=0.0,
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
        "_rerun": True,
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triage", default="baseline_triage.json")
    p.add_argument("--tasks", default="tasks_r3")
    p.add_argument("--config", default="config_r3.yaml")
    p.add_argument("--output", default="baseline_triage.json")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--max-tokens", type=int, default=16384)
    args = p.parse_args()

    cfg = Config.load(args.config)
    triage = json.loads(Path(args.triage).read_text())

    # find fake-fail pairs: (task_file_name, model)
    fake = []
    for entry in triage:
        for mr in entry["model_results"]:
            if (not mr["passed"]) and (mr.get("output_tokens") or 0) == 0:
                fake.append((entry["task"], mr["model"]))
    if not fake:
        print("No fake failures (0-token entries) found.", file=sys.stderr)
        return

    print(f"Re-running {len(fake)} fake failures with "
          f"timeout={args.timeout}s, max_tokens={args.max_tokens}",
          file=sys.stderr, flush=True)
    for tn, mk in fake:
        print(f"  - {tn} [{mk}]", file=sys.stderr)

    backend = OpenAICompatBackend(
        endpoints={k: v.endpoint for k, v in cfg.models.items()},
        served_names={k: v.served_name for k, v in cfg.models.items()},
        thinking_defaults={k: v.thinking_default for k, v in cfg.models.items()},
        timeout=args.timeout,
    )

    # Load all affected tasks once
    tasks_by_name = {}
    for tn, _ in fake:
        if tn not in tasks_by_name:
            tasks_by_name[tn] = load_task(Path(args.tasks) / tn)

    # Fire ALL reruns concurrently (server --parallel 6 will queue as needed)
    async def one(tn, mk):
        t = tasks_by_name[tn]
        r = await rerun_one(t, backend, mk, args.max_tokens,
                            cfg.baseline.thinking, cfg.baseline.temperature)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [rerun {status}] {tn:40s} {mk:10s} "
              f"wall={r['wall_ms']/1000:.1f}s out={r['output_tokens']}tok",
              file=sys.stderr, flush=True)
        return (tn, mk, r)

    try:
        done = await asyncio.gather(*(one(tn, mk) for (tn, mk) in fake))
    finally:
        await backend.close()

    # merge: for each (task, model), replace the original mr with the new one
    new_results = {(tn, mk): r for (tn, mk, r) in done}
    for entry in triage:
        for i, mr in enumerate(entry["model_results"]):
            key = (entry["task"], mr["model"])
            if key in new_results:
                entry["model_results"][i] = new_results[key]
        # re-compute disposition
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

    Path(args.output).write_text(json.dumps(triage, indent=2, default=str))

    print("\n=== POST-RERUN SUMMARY ===", file=sys.stderr)
    by_disp = {}
    for e in triage:
        d = e["disposition"].split(" ")[0]
        by_disp[d] = by_disp.get(d, 0) + 1
    for d, c in by_disp.items():
        print(f"  {d:15s} {c}", file=sys.stderr)
    print(f"\n  KEEP list (for harness experiment):", file=sys.stderr)
    for e in triage:
        if e["disposition"].split(" ")[0] in ("BOTH_FAIL", "ASYMMETRIC"):
            print(f"    - {e['task']:40s} [{e['disposition'].split(' ')[0]}]",
                  file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
