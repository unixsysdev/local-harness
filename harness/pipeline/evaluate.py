"""Stage 3b: cross-model evaluation.

Each candidate is scored by the *other* model (cross-eval). Self-eval is
off by default in Phase 1. Scores are absolute on the rubric; ordering is
randomized in the prompt to mitigate positional bias, but since each call
contains a single candidate, order bias is moot.
"""
from __future__ import annotations

from ..backends.openai_compat import GenRequest, OpenAICompatBackend
from ..types import Candidate, Evaluation, Rubric, ScoreBreakdown, Task
from .jsonutil import parse_json_lenient

SYS = (
    "You are a strict evaluator. You score candidate solutions against a rubric. "
    "Output JSON only. No prose."
)

EVAL_TEMPLATE = """Task:
<<<{task}>>>

Rubric weights: {weights}
Failure modes to watch: {failures}

Candidate solution (do NOT execute, just judge):
---
{solution}
---

Score each dimension in [0.0, 1.0]. Be stingy. A 1.0 means perfect on
that dimension. Cite concrete flaws in fatal_issues and minor_issues.

Return JSON:
{{
  "scores": {{
    "correctness": 0.0,
    "completeness": 0.0,
    "constraint_compliance": 0.0,
    "clarity": 0.0,
    "efficiency": 0.0,
    "risk": 0.0
  }},
  "fatal_issues": ["string", ...],
  "minor_issues": ["string", ...],
  "repair_suggestions": ["string", ...]
}}
"""


def _aggregate(scores: ScoreBreakdown, weights: dict[str, float]) -> float:
    total_w = sum(weights.values()) or 1.0
    acc = 0.0
    for k, w in weights.items():
        acc += (getattr(scores, k, 0.0) or 0.0) * (w / total_w)
    return acc


async def cross_evaluate(
    candidates: list[Candidate],
    task: Task,
    rubric: Rubric,
    backend: OpenAICompatBackend,
    max_tokens: int,
    model_keys: list[str],
) -> list[Evaluation]:
    """Each candidate is scored by the model that did NOT generate it.

    If there are >2 models this still works (candidate is scored by any other).
    For Phase 1 there are exactly two.
    """
    requests: list[GenRequest] = []
    meta: list[tuple[Candidate, str]] = []

    for c in candidates:
        evaluators = [mk for mk in model_keys if mk != c.model]
        for ek in evaluators:
            req_id = f"eval:{ek}:{c.candidate_id}"
            prompt = EVAL_TEMPLATE.format(
                task=task.raw_input.strip(),
                weights=rubric.weights,
                failures="; ".join(rubric.failure_modes) or "(none)",
                solution=c.solution,
            )
            requests.append(GenRequest(
                request_id=req_id,
                model_key=ek,
                prompt=prompt,
                system=SYS,
                temperature=0.1,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            ))
            meta.append((c, ek))

    responses = await backend.generate(requests)

    evaluations: list[Evaluation] = []
    for (c, ek), resp in zip(meta, responses):
        if resp.error:
            continue
        try:
            data = parse_json_lenient(resp.text)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        raw_scores = data.get("scores") or {}
        if not isinstance(raw_scores, dict):
            raw_scores = {}
        scores = ScoreBreakdown(
            correctness=float(raw_scores.get("correctness", 0) or 0),
            completeness=float(raw_scores.get("completeness", 0) or 0),
            constraint_compliance=float(raw_scores.get("constraint_compliance", 0) or 0),
            clarity=float(raw_scores.get("clarity", 0) or 0),
            efficiency=float(raw_scores.get("efficiency", 0) or 0),
            risk=float(raw_scores.get("risk", 0) or 0),
        )
        agg = _aggregate(scores, rubric.weights)
        evaluations.append(Evaluation(
            candidate_id=c.candidate_id,
            evaluator_model=ek,
            stage="cross_eval",
            scores=scores,
            aggregate=agg,
            fatal_issues=[str(x) for x in (data.get("fatal_issues") or [])][:8],
            minor_issues=[str(x) for x in (data.get("minor_issues") or [])][:8],
            repair_suggestions=[str(x) for x in (data.get("repair_suggestions") or [])][:8],
            raw=resp.text[:4000],
        ))
    return evaluations
