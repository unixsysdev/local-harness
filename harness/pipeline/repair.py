"""Stage 5: adversarial repair (gated, v2 spec \u00a77.7).

A candidate is repaired only when at least one of:
  - it has a specific flagged flaw (fatal_issues or validator signals)
  - the two evaluators disagree above tau (positional or substantive)
  - it is below median combined score AND above a minimum viability threshold

The OPPOSITE model repairs the candidate (model_a repairs model_b's, and vice
versa). The repair prompt surfaces the specific flaws from the evaluation
trace; generic rewriting is discouraged.

Repaired candidates are added to the pool and re-validated (deterministic) +
re-scored (soft constraints). They are NOT re-cross-evaluated in Phase 2 \u2014
that's deferred to keep the repair stage within a tight compute budget. The
repaired version keeps the original cross_mean as its heritage signal.
"""
from __future__ import annotations
import asyncio
import statistics
from typing import Optional

from ..backends.openai_compat import GenRequest, OpenAICompatBackend
from ..types import Candidate, Evaluation, SoftScore, ValidatorResult
from ..validation.soft import score_soft
from .jsonutil import parse_json_lenient
from .prune import ScoredCandidate
from .validate import _validate_one_sync


SYS_REPAIR = (
    "You repair a candidate solution by addressing SPECIFIC flaws that were "
    "identified. Do not rewrite the whole thing. Preserve what works. "
    "Output JSON only."
)

REPAIR_TEMPLATE = """Task:
<<<{task}>>>

HARD CONSTRAINTS (must be satisfied):
{constraints}

Original candidate (from {orig_model}):
---
{solution}
---

Specific flaws detected by evaluators (address these, do not invent new ones):
FATAL:
{fatal}

MINOR:
{minor}

VALIDATOR SIGNALS:
{validator}

Produce a repaired solution that fixes the flagged flaws while preserving the
rest of the candidate. If the flaws are truly unfixable without a full rewrite,
return the original with an assumption noting why.

Return JSON:
{{
  "repaired_solution": "string - the fixed deliverable",
  "changes_made": ["string", ...],
  "issues_addressed": ["string", ...],
  "remaining_uncertainties": ["string", ...]
}}
"""


def select_repair_candidates(
    scored: list[ScoredCandidate],
    evaluations: list[Evaluation],
    validator_results: list[ValidatorResult],
    disagreement_tau: float,
    viability_floor: float = 0.1,
    max_to_repair: int = 4,
) -> list[tuple[ScoredCandidate, list[str], list[str], dict]]:
    """Return (scored_candidate, fatal_list, minor_list, validator_signals)."""
    eval_by_cand: dict[str, list[Evaluation]] = {}
    for e in evaluations:
        eval_by_cand.setdefault(e.candidate_id, []).append(e)

    val_by_cand = {v.candidate_id: v for v in validator_results}

    # median combined for "below median" gate
    combined_values = [s.combined for s in scored]
    median_combined = statistics.median(combined_values) if combined_values else 0.0

    picks: list[tuple[ScoredCandidate, list[str], list[str], dict]] = []
    for s in scored:
        cid = s.candidate.candidate_id
        evs = eval_by_cand.get(cid, [])
        fatal: list[str] = []
        minor: list[str] = []
        for e in evs:
            fatal.extend(e.fatal_issues)
            minor.extend(e.minor_issues)
        # dedupe while preserving order
        fatal = list(dict.fromkeys(fatal))[:6]
        minor = list(dict.fromkeys(minor))[:6]

        val_signals = {}
        vr = val_by_cand.get(cid)
        if vr and not vr.passed:
            val_signals = {k: v for k, v in (vr.signals or {}).items()
                           if k in ("failures", "errors", "status", "error")}

        # disagreement: max |a - b| across evaluator pairs
        aggs = [e.aggregate for e in evs]
        disagreement = (max(aggs) - min(aggs)) if len(aggs) >= 2 else 0.0

        gates = []
        if fatal or val_signals:
            gates.append("has_flagged_flaw")
        if disagreement > disagreement_tau:
            gates.append("disagreement")
        if (s.combined < median_combined) and (s.combined > viability_floor):
            gates.append("below_median_viable")

        if gates:
            picks.append((s, fatal, minor, val_signals))

    # Respect the budget: prioritize candidates with specific flaws, then disagreement,
    # then below-median. Sort by combined desc to keep the most salvageable first.
    picks.sort(key=lambda x: (-len(x[1] + x[2]), -x[0].combined))
    return picks[:max_to_repair]


async def run_repair(
    repair_picks: list[tuple[ScoredCandidate, list[str], list[str], dict]],
    task_raw: str,
    task_constraints: list[str],
    backend: OpenAICompatBackend,
    model_keys: list[str],
    max_tokens: int,
) -> list[Candidate]:
    """Fire repair requests in parallel. The OTHER model repairs.

    Returns a list of new repaired candidates (same parent_id semantics:
    `candidate.strategy = "repair:<original_strategy>"`).
    """
    if not repair_picks or len(model_keys) < 2:
        return []

    # Map each original model to its "opposite" for repair
    other_of: dict[str, str] = {}
    for mk in model_keys:
        other_of[mk] = next((x for x in model_keys if x != mk), mk)

    requests: list[GenRequest] = []
    plan: list[tuple[ScoredCandidate, str]] = []

    constraints_block = ("\n- " + "\n- ".join(task_constraints)) if task_constraints else "(none)"

    for (s, fatal, minor, val_sig) in repair_picks:
        orig = s.candidate
        repair_model = other_of[orig.model]
        prompt = REPAIR_TEMPLATE.format(
            task=task_raw.strip(),
            constraints=constraints_block,
            orig_model=orig.model,
            solution=orig.solution,
            fatal=("\n- " + "\n- ".join(fatal)) if fatal else "(none)",
            minor=("\n- " + "\n- ".join(minor)) if minor else "(none)",
            validator=str(val_sig) if val_sig else "(none)",
        )
        requests.append(GenRequest(
            request_id=f"repair:{repair_model}:{orig.candidate_id}",
            model_key=repair_model,
            prompt=prompt,
            system=SYS_REPAIR,
            temperature=0.3,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        ))
        plan.append((s, repair_model))

    responses = await backend.generate(requests)

    new_candidates: list[Candidate] = []
    for (s, repair_model), resp in zip(plan, responses):
        if resp.error:
            continue
        try:
            data = parse_json_lenient(resp.text)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        sol = str(data.get("repaired_solution") or "").strip()
        if not sol:
            continue
        # If the repair is identical to the original, it's a no-op \u2014 skip
        if sol.strip() == s.candidate.solution.strip():
            continue
        new_candidates.append(Candidate(
            model=repair_model,
            strategy=f"repair:{s.candidate.strategy}",
            temperature=0.3,
            solution=sol,
            assumptions=[str(x) for x in (data.get("remaining_uncertainties") or [])],
            known_risks=[str(x) for x in (data.get("issues_addressed") or [])],
            self_confidence=0.6,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cached_tokens=resp.cached_tokens,
            latency_ms=resp.latency_ms,
        ))
    return new_candidates


async def revalidate_repaired(
    repaired: list[Candidate], task, soft_constraints
) -> tuple[list[ValidatorResult], list[SoftScore]]:
    """Run deterministic validator + soft-constraint score on repaired candidates."""
    loop = asyncio.get_running_loop()
    val_results: list[ValidatorResult] = []
    if task.verifier.kind != "none" and repaired:
        val_results = await asyncio.gather(*(
            loop.run_in_executor(None, _validate_one_sync, c, task) for c in repaired
        ))
    soft_scores = [score_soft(c, soft_constraints) for c in repaired]
    return val_results, soft_scores


def repair_noop_rate(picks_count: int, produced_count: int) -> float:
    """Fraction of repair attempts that produced no (new) candidate.

    repair_noop_rate > 0.5 is a signal to auto-disable repair for this task class.
    """
    if picks_count <= 0:
        return 0.0
    return max(0.0, 1.0 - (produced_count / picks_count))
