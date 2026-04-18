"""Soft-constraint scoring: deterministic, non-blocking checks applied to any task.

Returns SoftScore in [0, 1] plus a list of violations. Used as a penalty on
the combined score so candidates violating stated constraints are deprioritized
but not eliminated outright.
"""
from __future__ import annotations
import re
from typing import Optional

from ..types import Candidate, SoftConstraints, SoftScore


_WORD_RE = re.compile(r"\b\w+\b")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _strip_code_for_writing(text: str) -> str:
    """For writing tasks, strip fenced code blocks so they don't inflate word count."""
    return re.sub(r"```.*?```", "", text or "", flags=re.DOTALL)


def score_soft(candidate: Candidate, soft: Optional[SoftConstraints]) -> SoftScore:
    if soft is None:
        return SoftScore(candidate_id=candidate.candidate_id, score=1.0)

    violations: list[str] = []
    signals: dict = {}
    penalty = 0.0  # accumulate additive penalties; clamp to [0,1] at end

    text = candidate.solution or ""

    # --- word count ---
    if soft.word_count is not None:
        wc_text = _strip_code_for_writing(text)
        wc = count_words(wc_text)
        signals["word_count"] = wc
        lo, hi = soft.word_count.min, soft.word_count.max
        if wc < lo:
            pct = (lo - wc) / max(lo, 1)
            # up to 0.6 penalty for being far under
            penalty += min(0.6, pct * 1.0)
            violations.append(f"word_count {wc} < min {lo}")
        elif wc > hi:
            pct = (wc - hi) / max(hi, 1)
            penalty += min(0.6, pct * 1.0)
            violations.append(f"word_count {wc} > max {hi}")

    # --- must_contain ---
    text_lower = text.lower()
    for needle in soft.must_contain:
        if needle.lower() not in text_lower:
            penalty += 0.1
            violations.append(f"missing required: {needle!r}")

    # --- must_not_contain ---
    for needle in soft.must_not_contain:
        if needle.lower() in text_lower:
            penalty += 0.1
            violations.append(f"forbidden present: {needle!r}")

    score = max(0.0, 1.0 - penalty)
    return SoftScore(
        candidate_id=candidate.candidate_id,
        score=score,
        violations=violations,
        signals=signals,
    )
