"""CLI entry: wire every stage together."""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from .backends.openai_compat import OpenAICompatBackend
from .baseline import run_baseline
from .config import Config
from .pipeline.evaluate import cross_evaluate
from .pipeline.finalize import synthesize
from .pipeline.normalize import load_task
from .pipeline.propose import generate_proposals
from .pipeline.prune import prune, score_all
from .pipeline.repair import (
    repair_noop_rate, revalidate_repaired, run_repair, select_repair_candidates,
)
from .pipeline.rubric import build_rubric
from .pipeline.validate import validate_candidates
from .storage.traces import TraceWriter
from .validation.soft import score_soft


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--task", required=True, help="Path to task JSON")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--harness-only", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


async def run(cfg: Config, task_path: str, do_baseline: bool, do_harness: bool,
              quiet: bool) -> dict:
    task = load_task(task_path)

    backend = OpenAICompatBackend(
        endpoints={k: v.endpoint for k, v in cfg.models.items()},
        served_names={k: v.served_name for k, v in cfg.models.items()},
        thinking_defaults={k: v.thinking_default for k, v in cfg.models.items()},
    )
    model_keys = list(cfg.models.keys())

    trace = TraceWriter(cfg.run.trace_dir)
    trace.event("run_start", {"task_id": task.task_id, "task_path": task_path,
                              "config_name": cfg.run.name,
                              "task_class": task.task_class})

    out: dict = {"run_id": trace.run_id, "task_id": task.task_id,
                 "task": task.model_dump()}

    try:
        # ---------- baseline ----------
        if do_baseline:
            _log("[baseline] running single-shot…", quiet)
            t0 = time.perf_counter()
            bl = await run_baseline(
                task, backend,
                model_key=cfg.baseline.model,
                temperature=cfg.baseline.temperature,
                max_tokens=cfg.baseline.max_tokens,
                thinking=cfg.baseline.thinking,
            )
            bl["wall_ms"] = int((time.perf_counter() - t0) * 1000)
            out["baseline"] = bl
            trace.event("baseline", bl)
            _log(f"[baseline] done in {bl['wall_ms']} ms, "
                 f"{bl['output_tokens']} out tokens", quiet)

        if not do_harness:
            return out

        # ---------- short-circuit: if baseline already passes validator, skip harness ----------
        if (cfg.short_circuit.enabled
                and task.task_class == "verifiable"
                and task.verifier.kind != "none"
                and do_baseline):
            from .evaluator.core import extract_solution_for_verifier
            from .pipeline.validate import _validate_one_sync
            from .types import Candidate as _C
            pseudo = _C(model="baseline", strategy="baseline", temperature=0.0,
                        solution=extract_solution_for_verifier(
                            (out.get("baseline") or {}).get("text", "") or "",
                            task.verifier.kind))
            bl_val = _validate_one_sync(pseudo, task)
            trace.event("short_circuit_probe", {
                "baseline_passes": bl_val.passed,
                "signals": bl_val.signals,
            })
            if bl_val.passed:
                out["short_circuited"] = True
                out["final"] = {
                    "finalist_id": "baseline",
                    "finalist_model": cfg.baseline.model,
                    "finalist_strategy": "baseline_short_circuit",
                    "text": (out.get("baseline") or {}).get("text", ""),
                    "wall_ms": 0,
                }
                _log("[short_circuit] baseline passed validator; skipping harness", quiet)
                return out

        # ---------- rubric ----------
        _log("[rubric] building…", quiet)
        t0 = time.perf_counter()
        rubric = await build_rubric(
            task, backend,
            model_key=cfg.baseline.model,   # reuse baseline model for rubric
            max_tokens=cfg.sampling.eval_max_tokens,
        )
        trace.event("rubric", {"rubric": rubric.model_dump(),
                               "wall_ms": int((time.perf_counter() - t0) * 1000)})
        out["rubric"] = rubric.model_dump()
        _log(f"[rubric] done in {int((time.perf_counter()-t0)*1000)} ms", quiet)

        # ---------- proposals ----------
        verifiable = task.task_class == "verifiable"
        s = cfg.search.verifiable if verifiable else cfg.search.non_verifiable
        per_model = s.proposals_per_model
        keep = s.post_score_keep

        _log(f"[propose] generating {per_model}×{len(model_keys)} candidates "
             f"(verifiable={verifiable})…", quiet)
        t0 = time.perf_counter()
        candidates = await generate_proposals(
            task=task, rubric=rubric, backend=backend,
            model_keys=model_keys,
            strategies=cfg.sampling.strategies,
            temperatures=cfg.sampling.temperatures,
            per_model=per_model,
            max_tokens=cfg.sampling.proposal_max_tokens,
            top_p=cfg.sampling.top_p,
        )
        wall = int((time.perf_counter() - t0) * 1000)
        tot_in = sum(c.input_tokens for c in candidates)
        tot_out = sum(c.output_tokens for c in candidates)
        tot_cached = sum(c.cached_tokens for c in candidates)
        trace.event("proposals", {
            "wall_ms": wall,
            "count": len(candidates),
            "sum_input_tokens": tot_in,
            "sum_output_tokens": tot_out,
            "sum_cached_tokens": tot_cached,
            "aggregate_out_tps": round(tot_out / (wall / 1000), 1) if wall else 0,
            "cache_hit_rate": round(tot_cached / tot_in, 3) if tot_in else 0,
            "candidates": [c.model_dump() for c in candidates],
        })
        _log(f"[propose] {len(candidates)} candidates in {wall} ms "
             f"→ {tot_out} out tok ({round(tot_out/(wall/1000),1) if wall else 0} tps), "
             f"cache {tot_cached}/{tot_in}", quiet)

        if not candidates:
            trace.event("abort", {"reason": "no_candidates"})
            out["error"] = "no candidates produced"
            return out

        # ---------- deterministic validation ----------
        validator_results = await validate_candidates(candidates, task)
        if validator_results:
            trace.event("validation", {
                "results": [v.model_dump() for v in validator_results],
                "passers": sum(1 for v in validator_results if v.passed),
            })
            _log(f"[validate] {sum(1 for v in validator_results if v.passed)}/"
                 f"{len(validator_results)} passed", quiet)

        # ---------- cross-eval ----------
        # In verifiable path: skip cross-eval if exactly one candidate passes
        # (we have a clear winner). Run it to tiebreak among >=2 passers.
        passers = [v for v in validator_results if v.passed]
        run_eval = (not verifiable) or (len(passers) != 1)
        evaluations = []
        if run_eval:
            _log("[evaluate] cross-scoring…", quiet)
            t0 = time.perf_counter()
            evaluations = await cross_evaluate(
                candidates=candidates, task=task, rubric=rubric,
                backend=backend,
                max_tokens=cfg.sampling.eval_max_tokens,
                model_keys=model_keys,
            )
            wall = int((time.perf_counter() - t0) * 1000)
            trace.event("evaluations", {
                "wall_ms": wall,
                "count": len(evaluations),
                "evaluations": [e.model_dump() for e in evaluations],
            })
            _log(f"[evaluate] {len(evaluations)} evaluations in {wall} ms", quiet)

        # ---------- soft constraints (word count etc) ----------
        soft_scores = [score_soft(c, task.soft_constraints) for c in candidates]
        if task.soft_constraints is not None:
            trace.event("soft_constraints", {
                "count_violating": sum(1 for s in soft_scores if s.violations),
                "scores": [s.model_dump() for s in soft_scores],
            })
            n_viol = sum(1 for s in soft_scores if s.violations)
            _log(f"[soft] {n_viol}/{len(soft_scores)} candidates violate constraints", quiet)

        # ---------- prune ----------
        scoring = cfg.scoring.verifiable if verifiable else cfg.scoring.non_verifiable
        scored = score_all(
            candidates=candidates,
            evaluations=evaluations,
            validator_results=validator_results,
            soft_scores=soft_scores,
            cross_weight=scoring.cross_eval_weight,
            deterministic_weight=scoring.deterministic_weight,
        )
        kept = prune(scored, keep=keep, verifiable_path=verifiable)
        trace.event("prune", {
            "kept": [s.candidate.candidate_id for s in kept],
            "ranked": [
                {"id": s.candidate.candidate_id,
                 "model": s.candidate.model,
                 "strategy": s.candidate.strategy,
                 "combined": s.combined,
                 "cross_mean": s.cross_mean,
                 "deterministic": s.deterministic,
                 "soft": s.soft,
                 "soft_violations": s.soft_violations,
                 "passed": s.passed_validator}
                for s in sorted(scored, key=lambda s: s.combined, reverse=True)
            ],
        })
        if not kept:
            out["error"] = "no candidates survived pruning"
            return out
        _log(f"[prune] kept {len(kept)} of {len(scored)}", quiet)

        # ---------- gated adversarial repair (v2 \u00a77.7) ----------
        if cfg.repair.enabled and evaluations:
            picks = select_repair_candidates(
                scored=scored,
                evaluations=evaluations,
                validator_results=validator_results,
                disagreement_tau=cfg.disagreement.tau,
                viability_floor=cfg.repair.viability_floor,
                max_to_repair=cfg.repair.max_to_repair,
            )
            if picks:
                _log(f"[repair] {len(picks)} candidates flagged for repair", quiet)
                t0 = time.perf_counter()
                repaired = await run_repair(
                    repair_picks=picks,
                    task_raw=task.raw_input,
                    task_constraints=task.constraints,
                    backend=backend,
                    model_keys=model_keys,
                    max_tokens=cfg.repair.max_tokens,
                )
                noop = repair_noop_rate(len(picks), len(repaired))
                r_val, r_soft = await revalidate_repaired(repaired, task, task.soft_constraints)
                wall = int((time.perf_counter() - t0) * 1000)
                trace.event("repair", {
                    "picked": len(picks),
                    "produced": len(repaired),
                    "noop_rate": noop,
                    "wall_ms": wall,
                    "repaired_candidates": [c.model_dump() for c in repaired],
                })
                _log(f"[repair] produced {len(repaired)}/{len(picks)} new candidates "
                     f"(noop_rate={noop:.2f}) in {wall} ms", quiet)

                # Merge repaired candidates: inherit cross_mean from originals (0 for
                # re-scoring would be unfair since we haven't cross-eval'd them). Use
                # heritage cross_mean of their parent.
                if repaired:
                    # Build a lookup from parent strategy-model to parent cross_mean.
                    # We treat each repaired candidate as having its parent's cross_mean
                    # as a heritage signal (optimistic but bounded by the multiplicative
                    # soft penalty + fresh deterministic score).
                    parent_cross = {}
                    for s in scored:
                        parent_cross[(s.candidate.model, s.candidate.strategy)] = s.cross_mean
                    # Build synthetic evaluations for repaired candidates at their parent's
                    # cross_mean so score_all can treat them uniformly.
                    from .types import Evaluation as _E, ScoreBreakdown as _SB
                    synth_evals = []
                    for rc in repaired:
                        # strategy is "repair:<parent_strategy>" \u2014 extract parent strategy
                        parent_strat = rc.strategy.split(":", 1)[1] if ":" in rc.strategy else rc.strategy
                        # For repaired-candidates the parent model is the OPPOSITE of rc.model
                        parent_model = next((m for m in model_keys if m != rc.model), rc.model)
                        parent_cm = parent_cross.get((parent_model, parent_strat), 0.5)
                        synth_evals.append(_E(
                            candidate_id=rc.candidate_id,
                            evaluator_model="repair_heritage",
                            stage="repair_heritage",
                            scores=_SB(), aggregate=parent_cm,
                        ))
                    # Fold repaired into pools
                    all_candidates = candidates + repaired
                    all_evals = evaluations + synth_evals
                    all_val = validator_results + r_val
                    all_soft = soft_scores + r_soft

                    scored2 = score_all(
                        candidates=all_candidates,
                        evaluations=all_evals,
                        validator_results=all_val,
                        soft_scores=all_soft,
                        cross_weight=scoring.cross_eval_weight,
                        deterministic_weight=scoring.deterministic_weight,
                    )
                    kept = prune(scored2, keep=keep, verifiable_path=verifiable)
                    trace.event("prune_post_repair", {
                        "kept": [s.candidate.candidate_id for s in kept],
                        "ranked": [
                            {"id": s.candidate.candidate_id,
                             "model": s.candidate.model,
                             "strategy": s.candidate.strategy,
                             "combined": s.combined,
                             "cross_mean": s.cross_mean,
                             "deterministic": s.deterministic,
                             "soft": s.soft,
                             "passed": s.passed_validator}
                            for s in sorted(scored2, key=lambda s: s.combined, reverse=True)
                        ],
                    })
                    _log(f"[repair] post-repair pool: {len(kept)} kept of "
                         f"{len(scored2)} total", quiet)

        # ---------- finalize ----------
        finalist = kept[0]
        _log(f"[synth] finalist={finalist.candidate.model}/"
             f"{finalist.candidate.strategy} combined={finalist.combined:.3f}", quiet)
        t0 = time.perf_counter()
        final_text = await synthesize(
            task=task,
            finalist=finalist.candidate,
            validator_passed=finalist.passed_validator,
            backend=backend,
            model_key=cfg.synthesis.model,
            temperature=cfg.synthesis.temperature,
            max_tokens=cfg.sampling.final_max_tokens,
            thinking=cfg.synthesis.thinking,
        )
        wall = int((time.perf_counter() - t0) * 1000)
        trace.event("final", {
            "finalist_id": finalist.candidate.candidate_id,
            "final_text": final_text,
            "wall_ms": wall,
        })
        out["final"] = {
            "finalist_id": finalist.candidate.candidate_id,
            "finalist_model": finalist.candidate.model,
            "finalist_strategy": finalist.candidate.strategy,
            "text": final_text,
            "wall_ms": wall,
        }
        _log(f"[synth] done in {wall} ms", quiet)
        return out

    finally:
        trace.event("run_end", {})
        trace.close()
        await backend.close()


def main() -> int:
    args = _parse_args()
    cfg = Config.load(args.config)
    do_baseline = not args.harness_only
    do_harness = not args.baseline_only
    out = asyncio.run(run(cfg, args.task, do_baseline, do_harness, args.quiet))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
