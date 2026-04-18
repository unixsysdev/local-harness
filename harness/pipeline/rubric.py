"""Stage 1: rubric builder.

Small-model-friendly prompt that returns a compact rubric in JSON.
Weights are normalized to sum to 1.0 post-hoc.
"""
from __future__ import annotations

from ..backends.openai_compat import GenRequest, OpenAICompatBackend
from ..types import Rubric, Task
from .jsonutil import parse_json_lenient

SYS = (
    "You build evaluation rubrics for tasks. "
    "Output JSON only. No prose outside the JSON."
)

TEMPLATE = """Task:
<<<{raw}>>>

Task class: {cls}
Deliverable: {deliv}
Success criteria: {sc}
Constraints: {cons}

Produce a compact rubric as JSON with this schema:
{{
  "task_summary": "<= 25 words",
  "weights": {{
     "correctness": 0.0,
     "completeness": 0.0,
     "constraint_compliance": 0.0,
     "clarity": 0.0,
     "efficiency": 0.0,
     "risk": 0.0
  }},
  "failure_modes": ["string", ...],
  "recommended_strategy_axes": ["string", ...],
  "max_candidate_tokens": 600
}}
Rules: weights sum close to 1.0. Pick at most 4 failure_modes.
"""


async def build_rubric(task: Task, backend: OpenAICompatBackend,
                       model_key: str, max_tokens: int) -> Rubric:
    prompt = TEMPLATE.format(
        raw=task.raw_input.strip(),
        cls=task.task_class,
        deliv=task.deliverable_type,
        sc=", ".join(task.success_criteria) or "(none)",
        cons=", ".join(task.constraints) or "(none)",
    )
    req = GenRequest(
        request_id="rubric",
        model_key=model_key,
        prompt=prompt,
        system=SYS,
        temperature=0.2,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    [resp] = await backend.generate([req])
    if resp.error:
        raise RuntimeError(f"rubric backend error: {resp.error}")
    data = parse_json_lenient(resp.text)
    if not isinstance(data, dict):
        raise RuntimeError(f"rubric returned non-object JSON: {type(data).__name__}")

    weights = data.get("weights") or {}
    total = sum(float(v) for v in weights.values()) or 1.0
    if total and abs(total - 1.0) > 1e-3:
        weights = {k: float(v) / total for k, v in weights.items()}
    return Rubric(
        task_summary=str(data.get("task_summary", task.raw_input[:120])),
        weights={k: float(v) for k, v in weights.items()},
        failure_modes=[str(x) for x in (data.get("failure_modes") or [])][:8],
        recommended_strategy_axes=[str(x) for x in (data.get("recommended_strategy_axes") or [])][:6],
        max_candidate_tokens=int(data.get("max_candidate_tokens") or 600),
    )
