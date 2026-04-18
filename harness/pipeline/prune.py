"""Stage 4: pruning.

Combines deterministic validator score and cross-model aggregate into a single
combined score per candidate, then keeps top-K. In the verifiable path,
candidates that pass the validator are prioritized unconditionally.
"""
from __future__ import annotations
from dataclasses import dataclass

from ..types import Candidate, Evaluation, SoftScore, ValidatorResult


@dataclass
class ScoredCandidate:
    candidate: Candidate
    combined: float
    cross_mean: float
    deterministic: float
    soft: float
    soft_violations: list[str]
    passed_validator: bool | None
    evaluator_scores: dict[str, float]  # evaluator_model -> aggregate


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_all(
    candidates: list[Candidate],
    evaluations: list[Evaluation],
    validator_results: list[ValidatorResult],
    soft_scores: list[SoftScore],
    cross_weight: float,
    deterministic_weight: float,
    soft_penalty_weight: float = 0.30,
) -> list[ScoredCandidate]:
    """Combined score = (cross_eval + deterministic) * (1 - soft_penalty_weight * (1 - soft))

    Soft constraints apply multiplicatively so severe violations reduce the
    combined score proportionally without eliminating the candidate.
    """
    eval_by_cand: dict[str, list[Evaluation]] = {}
    for e in evaluations:
        eval_by_cand.setdefault(e.candidate_id, []).append(e)
    val_by_cand = {v.candidate_id: v for v in validator_results}
    soft_by_cand = {s.candidate_id: s for s in soft_scores}

    has_validator = len(validator_results) > 0
    if not has_validator:
        det_w = 0.0
        cr_w = 1.0
    else:
        total = cross_weight + deterministic_weight or 1.0
        cr_w = cross_weight / total
        det_w = deterministic_weight / total

    scored: list[ScoredCandidate] = []
    for c in candidates:
        evs = eval_by_cand.get(c.candidate_id, [])
        aggs = [e.aggregate for e in evs]
        cross_mean = _mean(aggs)
        evaluator_scores = {e.evaluator_model: e.aggregate for e in evs}

        vr = val_by_cand.get(c.candidate_id)
        det_score = vr.score if vr else 0.5
        passed = vr.passed if vr else None

        ss = soft_by_cand.get(c.candidate_id)
        soft_val = ss.score if ss else 1.0
        soft_viols = ss.violations if ss else []

        base = cr_w * cross_mean + det_w * det_score
        # multiplicative penalty: soft=1 -> no effect; soft=0 with weight 0.3 -> 0.7x
        combined = base * (1.0 - soft_penalty_weight * (1.0 - soft_val))

        scored.append(ScoredCandidate(
            candidate=c,
            combined=combined,
            cross_mean=cross_mean,
            deterministic=det_score,
            soft=soft_val,
            soft_violations=soft_viols,
            passed_validator=passed,
            evaluator_scores=evaluator_scores,
        ))
    return scored


def prune(scored: list[ScoredCandidate], keep: int, verifiable_path: bool) -> list[ScoredCandidate]:
    if not scored:
        return []

    if verifiable_path:
        passers = [s for s in scored if s.passed_validator]
        if passers:
            # Rank passers by cross_mean as tiebreak among passing candidates
            passers.sort(key=lambda s: (s.cross_mean, s.combined), reverse=True)
            return passers[:keep]
        # No passers: keep top near-misses by validator score, fall back to combined
        near = sorted(scored, key=lambda s: (s.deterministic, s.combined), reverse=True)
        return near[:max(1, min(keep, 2))]

    # Non-verifiable path: pure combined-score ranking
    scored.sort(key=lambda s: s.combined, reverse=True)
    return scored[:keep]
