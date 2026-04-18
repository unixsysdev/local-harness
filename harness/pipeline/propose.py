"""Stage 2: proposal generation.

Builds a deterministic generation matrix from (strategies × temperatures) such
that the first strategy gets the lowest temperature, and the total count per
model matches `proposals_per_model`.

Each candidate is produced by an independent, prefix-shared call:
  [system: proposer-role]
  [rubric + task]
  [strategy delta]
Diversity is injected only in the strategy delta and sampling config.
"""
from __future__ import annotations
from itertools import cycle

from ..backends.openai_compat import GenRequest, OpenAICompatBackend
from ..types import Candidate, Rubric, Task
from .jsonutil import parse_json_lenient

SYS = (
    "You generate candidate solutions to a task. "
    "You do not see sibling candidates. "
    "Output JSON only, in the schema provided, no extra prose."
)

STRATEGY_HINTS = {
    "baseline":         "Solve the task directly. Do not pad.",
    "decompose-first":  "First break the task into explicit sub-steps, then solve. Keep steps short.",
    "edge-case-first":  "Start by enumerating likely edge cases, then produce a solution that handles them.",
    "brevity":          "Produce the shortest correct solution. Prefer terse phrasing.",
    "evidence-heavy":   "Ground every claim in concrete details from the task. Cite constraints explicitly.",
    "inverted-assumptions": "Challenge one obvious assumption and solve under the inverted reading.",
}

PROPOSAL_TEMPLATE = """Task:
<<<{task}>>>

HARD CONSTRAINTS (MUST be satisfied exactly; violations count as failure):
{constraints}

Success criteria:
{success}

Rubric:
- Summary: {summary}
- Failure modes to avoid: {failures}

Strategy axis for this candidate: {strategy}
Hint: {hint}

Before writing your solution, confirm to yourself that every hard constraint
above is met. If a constraint specifies a word count, count your words.

Return JSON:
{{
  "solution": "string - the actual deliverable",
  "assumptions": ["string", ...],
  "known_risks": ["string", ...],
  "self_confidence": 0.0
}}
Keep `solution` self-contained.
"""


def build_matrix(strategies: list[str], temperatures: list[float],
                 per_model: int) -> list[tuple[str, float]]:
    """Return (strategy, temperature) pairs of length `per_model`."""
    pairs: list[tuple[str, float]] = []
    strat_cycle = cycle(strategies)
    temp_cycle = cycle(temperatures)
    for _ in range(per_model):
        pairs.append((next(strat_cycle), next(temp_cycle)))
    return pairs


async def generate_proposals(
    task: Task,
    rubric: Rubric,
    backend: OpenAICompatBackend,
    model_keys: list[str],
    strategies: list[str],
    temperatures: list[float],
    per_model: int,
    max_tokens: int,
    top_p: float,
    base_seed: int = 0,
) -> list[Candidate]:
    matrix = build_matrix(strategies, temperatures, per_model)
    requests: list[GenRequest] = []
    meta: list[tuple[str, str, float, int]] = []   # (model_key, strategy, temp, seed)

    for mk in model_keys:
        for i, (strat, temp) in enumerate(matrix):
            seed = base_seed + 1000 * (hash(mk) & 0xFFF) + i
            prompt = PROPOSAL_TEMPLATE.format(
                task=task.raw_input.strip(),
                summary=rubric.task_summary,
                failures="; ".join(rubric.failure_modes) or "(none)",
                success=("\n- " + "\n- ".join(task.success_criteria)) if task.success_criteria else "(none)",
                constraints=("\n- " + "\n- ".join(task.constraints)) if task.constraints else "(none)",
                strategy=strat,
                hint=STRATEGY_HINTS.get(strat, ""),
            )
            requests.append(GenRequest(
                request_id=f"prop:{mk}:{i}",
                model_key=mk,
                prompt=prompt,
                system=SYS,
                temperature=temp,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                response_format={"type": "json_object"},
            ))
            meta.append((mk, strat, temp, seed))

    responses = await backend.generate(requests)

    candidates: list[Candidate] = []
    for (mk, strat, temp, seed), resp in zip(meta, responses):
        if resp.error:
            # Skip failed candidates but record a placeholder so traces show the gap.
            continue
        try:
            data = parse_json_lenient(resp.text)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        solution = str(data.get("solution") or "").strip()
        if not solution:
            continue
        candidates.append(Candidate(
            model=mk,
            strategy=strat,
            temperature=temp,
            seed=seed,
            solution=solution,
            assumptions=[str(x) for x in (data.get("assumptions") or [])],
            known_risks=[str(x) for x in (data.get("known_risks") or [])],
            self_confidence=float(data.get("self_confidence") or 0.5),
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cached_tokens=resp.cached_tokens,
            latency_ms=resp.latency_ms,
        ))
    return candidates
