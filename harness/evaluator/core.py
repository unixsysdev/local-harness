"""Core evaluator helpers: code/JSON extraction, verifiable scoring, pairwise judging."""
from __future__ import annotations
import asyncio
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..backends.openai_compat import GenRequest, OpenAICompatBackend
from ..pipeline.jsonutil import parse_json_lenient
from ..pipeline.validate import _validate_one_sync
from ..types import Candidate, Task, ValidatorResult


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull out the first fenced code block; fall back to the whole text."""
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def extract_solution_for_verifier(text: str, verifier_kind: str) -> str:
    """Normalize the baseline's raw reply into something validators can chew on.

    - tests: strip markdown fences, keep Python code
    - json_schema: the lenient parser already handles fences + prose; return as-is
    - regex/other: return as-is
    """
    if verifier_kind == "tests":
        return extract_code(text)
    return text or ""


# ---------- verifiable path ----------

@dataclass
class VerifiableVerdict:
    task_id: str
    task_file: str
    task_class: str
    baseline_pass: bool
    harness_pass: bool
    verdict: str                      # harness_win | baseline_win | both_pass | both_fail
    baseline_signals: dict[str, Any] = field(default_factory=dict)
    harness_signals: dict[str, Any] = field(default_factory=dict)
    baseline_wall_ms: int = 0
    harness_wall_ms: int = 0


def score_verifiable(task: Task, baseline_text: str, baseline_wall_ms: int,
                     harness_final_text: str, harness_wall_ms: int,
                     task_file: str) -> VerifiableVerdict:
    """Run the task's validator on both baseline and harness final output."""
    def _run(text: str) -> ValidatorResult:
        pseudo = Candidate(
            model="eval", strategy="eval", temperature=0.0,
            solution=extract_solution_for_verifier(text, task.verifier.kind),
        )
        return _validate_one_sync(pseudo, task)

    bl = _run(baseline_text or "")
    hr = _run(harness_final_text or "")

    if bl.passed and hr.passed:
        verdict = "both_pass"
    elif hr.passed and not bl.passed:
        verdict = "harness_win"
    elif bl.passed and not hr.passed:
        verdict = "baseline_win"
    else:
        verdict = "both_fail"

    return VerifiableVerdict(
        task_id=task.task_id, task_file=task_file, task_class=task.task_class,
        baseline_pass=bl.passed, harness_pass=hr.passed, verdict=verdict,
        baseline_signals=bl.signals, harness_signals=hr.signals,
        baseline_wall_ms=baseline_wall_ms, harness_wall_ms=harness_wall_ms,
    )


# ---------- non-verifiable (pairwise) path ----------

PAIRWISE_SYS = (
    "You are a strict pairwise judge. You are given a task and two candidate "
    "answers. Pick the better one against the task's success criteria. "
    "Return JSON only."
)

PAIRWISE_TEMPLATE = """Task:
<<<{task}>>>

Success criteria:
{success_criteria}

Constraints:
{constraints}

Answer A:
---
{left}
---

Answer B:
---
{right}
---

Decide which is better under the criteria and constraints above.
Output JSON:
{{
  "winner": "A" | "B" | "tie",
  "confidence": 0.0,
  "reason": "one sentence"
}}
"""


@dataclass
class PairwiseCall:
    judge: str               # model_a | model_b
    side_of_harness: str     # A | B  (order presented this round)
    winner_label: str        # A | B | tie (judge's raw answer)
    harness_won: Optional[bool]  # True/False/None(tie or parse-fail)
    confidence: float = 0.0
    reason: str = ""
    raw: str = ""
    error: Optional[str] = None


@dataclass
class PairwiseVerdict:
    task_id: str
    task_file: str
    task_class: str
    calls: list[PairwiseCall]
    harness_wins: int                # number of non-tie harness wins
    baseline_wins: int
    ties: int
    total_calls: int
    order_consistent_per_judge: dict[str, bool]  # judge -> both orderings agree?
    verdict: str                     # harness_win | baseline_win | tie | unresolved
    weighted_harness: float = 0.0    # calibrated harness wins (sum of judge weights)
    weighted_baseline: float = 0.0
    judge_weights_used: dict[str, float] = field(default_factory=dict)
    baseline_wall_ms: int = 0
    harness_wall_ms: int = 0


async def pairwise_compare(
    task: Task,
    baseline_text: str,
    harness_final_text: str,
    backend: OpenAICompatBackend,
    judges: list[str],
    max_tokens: int,
    task_file: str,
    baseline_wall_ms: int,
    harness_wall_ms: int,
    judge_weights: Optional[dict[str, float]] = None,
    non_unanimous_penalty: float = 0.5,
) -> PairwiseVerdict:
    """For each judge model, run two comparisons (harness-as-A, harness-as-B).

    Total = 2 * len(judges) LLM calls per task. Fires concurrently.
    """
    requests: list[GenRequest] = []
    plan: list[tuple[str, str]] = []  # (judge_model_key, side_of_harness: 'A'|'B')

    for j in judges:
        for side in ("A", "B"):
            left = harness_final_text if side == "A" else baseline_text
            right = baseline_text if side == "A" else harness_final_text
            prompt = PAIRWISE_TEMPLATE.format(
                task=task.raw_input.strip(),
                success_criteria="\n- " + "\n- ".join(task.success_criteria) if task.success_criteria else "(none)",
                constraints="\n- " + "\n- ".join(task.constraints) if task.constraints else "(none)",
                left=(left or "").strip(),
                right=(right or "").strip(),
            )
            requests.append(GenRequest(
                request_id=f"pw:{j}:{side}",
                model_key=j,
                prompt=prompt,
                system=PAIRWISE_SYS,
                temperature=0.1,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            ))
            plan.append((j, side))

    responses = await backend.generate(requests)

    calls: list[PairwiseCall] = []
    wins_h = 0
    wins_b = 0
    ties = 0

    # Per-judge list of raw winner labels normalized to "harness" / "baseline" / "tie"
    per_judge: dict[str, list[str]] = {}

    for (j, side), resp in zip(plan, responses):
        call = PairwiseCall(judge=j, side_of_harness=side, winner_label="",
                            harness_won=None, raw=(resp.text or "")[:2000])
        if resp.error:
            call.error = resp.error
            calls.append(call)
            per_judge.setdefault(j, []).append("parse_error")
            continue
        try:
            data = parse_json_lenient(resp.text)
            if not isinstance(data, dict):
                raise ValueError(f"non-object JSON: {type(data).__name__}")
            winner = str(data.get("winner", "")).strip().upper()
            conf = float(data.get("confidence", 0.5) or 0.5)
            reason = str(data.get("reason", ""))[:300]
        except Exception as e:
            call.error = f"parse: {e}"
            calls.append(call)
            per_judge.setdefault(j, []).append("parse_error")
            continue

        call.winner_label = winner
        call.confidence = conf
        call.reason = reason

        if winner == "TIE":
            call.harness_won = None
            ties += 1
            per_judge.setdefault(j, []).append("tie")
        elif winner == side:
            call.harness_won = True
            wins_h += 1
            per_judge.setdefault(j, []).append("harness")
        elif winner in ("A", "B"):
            call.harness_won = False
            wins_b += 1
            per_judge.setdefault(j, []).append("baseline")
        else:
            per_judge.setdefault(j, []).append("parse_error")

        calls.append(call)

    # Positional consistency: for each judge, do both orderings pick the same side?
    consistency: dict[str, bool] = {}
    for j, labels in per_judge.items():
        clean = [x for x in labels if x in ("harness", "baseline", "tie")]
        consistency[j] = (len(set(clean)) == 1) if clean else False

    # --- Calibrated (weighted) voting ---
    weights = judge_weights or {}
    used: dict[str, float] = {}
    w_harness = 0.0
    w_baseline = 0.0
    for c in calls:
        if c.harness_won is None:
            continue  # ties/parse errors don't cast weighted votes
        w = float(weights.get(c.judge, 1.0))
        # Apply extra penalty if this judge was non-unanimous (positionally inconsistent)
        # on THIS task \u2014 so a biased judge's split vote counts even less.
        if not consistency.get(c.judge, True):
            w *= float(non_unanimous_penalty)
        used[c.judge] = weights.get(c.judge, 1.0)
        if c.harness_won:
            w_harness += w
        else:
            w_baseline += w

    # Unweighted verdict (for record)
    total = wins_h + wins_b + ties
    if total == 0:
        verdict = "unresolved"
    elif w_harness > w_baseline and w_harness >= (w_harness + w_baseline) * 0.5:
        verdict = "harness_win"
    elif w_baseline > w_harness and w_baseline >= (w_harness + w_baseline) * 0.5:
        verdict = "baseline_win"
    else:
        verdict = "tie"

    return PairwiseVerdict(
        task_id=task.task_id, task_file=task_file, task_class=task.task_class,
        calls=calls, harness_wins=wins_h, baseline_wins=wins_b, ties=ties,
        total_calls=len(calls), order_consistent_per_judge=consistency,
        verdict=verdict,
        weighted_harness=round(w_harness, 3),
        weighted_baseline=round(w_baseline, 3),
        judge_weights_used=used,
        baseline_wall_ms=baseline_wall_ms, harness_wall_ms=harness_wall_ms,
    )
