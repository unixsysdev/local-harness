"""Single-shot baseline. Runs the task with one model call, no search.

Always reported alongside every harness run so we can tell whether the
pipeline earned its compute.
"""
from __future__ import annotations

from .backends.openai_compat import GenRequest, OpenAICompatBackend
from .types import Task


async def run_baseline(
    task: Task, backend: OpenAICompatBackend,
    model_key: str, temperature: float, max_tokens: int, thinking: bool,
) -> dict:
    prompt = task.raw_input.strip()
    if task.success_criteria:
        prompt += "\n\nSuccess criteria: " + "; ".join(task.success_criteria)
    if task.constraints:
        prompt += "\nConstraints: " + "; ".join(task.constraints)

    req = GenRequest(
        request_id="baseline",
        model_key=model_key,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
    )
    [resp] = await backend.generate([req])
    text = resp.text or ""
    # Safety net: if thinking ate the budget and content is empty, fall back to
    # reasoning content so the baseline still produces *something* to judge.
    # Also record truncation so we can spot insufficient-budget cases.
    truncated = resp.finish_reason in ("length", "max_tokens")
    if not text.strip() and resp.reasoning:
        text = resp.reasoning.strip()
    return {
        "text": text,
        "reasoning_len": len(resp.reasoning or ""),
        "finish_reason": resp.finish_reason,
        "truncated": truncated,
        "error": resp.error,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "latency_ms": resp.latency_ms,
    }
