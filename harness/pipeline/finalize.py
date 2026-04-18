"""Stage 7: final synthesis.

Takes the top candidate and produces the user-facing answer.
For verifiable tasks that passed the validator, the solution is returned
verbatim (no silent edits). Otherwise the synthesizer may polish prose,
but must not alter factual content.
"""
from __future__ import annotations

from ..backends.openai_compat import GenRequest, OpenAICompatBackend
from ..types import Candidate, Task

SYS_VERBATIM = (
    "You are a final-answer formatter. The solution has been verified. "
    "Return it verbatim. Do not modify code or data."
)

SYS_POLISH = (
    "You are the final synthesizer. Produce the user-facing answer. "
    "Do not reveal internal scores or candidate comparisons. "
    "Preserve factual content; only improve clarity and flow."
)

POLISH_TEMPLATE = """Task:
<<<{task}>>>

HARD CONSTRAINTS (the final answer MUST satisfy all of these):
{constraints}

Best candidate solution (preserve factual content):
---
{solution}
---

Produce the final answer. Plain text only unless the task requires code.
If the candidate violates any constraint (e.g. word count), fix it while
preserving the facts. Do not add new claims."""


async def synthesize(
    task: Task,
    finalist: Candidate,
    validator_passed: bool | None,
    backend: OpenAICompatBackend,
    model_key: str,
    temperature: float,
    max_tokens: int,
    thinking: bool,
) -> str:
    # Verifiable path with passing candidate: return verbatim, skip the LLM call.
    if validator_passed is True:
        return finalist.solution

    prompt = POLISH_TEMPLATE.format(
        task=task.raw_input.strip(),
        constraints=("\n- " + "\n- ".join(task.constraints)) if task.constraints else "(none)",
        solution=finalist.solution,
    )
    req = GenRequest(
        request_id="synth",
        model_key=model_key,
        prompt=prompt,
        system=SYS_POLISH,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
    )
    [resp] = await backend.generate([req])
    if resp.error:
        # Fall back to the candidate solution rather than failing hard.
        return finalist.solution
    return (resp.text or finalist.solution).strip()
