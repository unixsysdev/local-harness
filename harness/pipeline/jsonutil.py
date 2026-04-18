"""Best-effort JSON extraction from model output.

Small models often wrap JSON in prose or fences. We take the first balanced
{...} or [...] block as a pragmatic fallback.
"""
from __future__ import annotations
import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text


def _find_balanced(s: str, open_ch: str, close_ch: str) -> str | None:
    start = s.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_json_lenient(text: str) -> Any:
    """Try straight json.loads; fall back to balanced-brace extraction."""
    if text is None:
        raise ValueError("empty text")
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for op, cl in (("{", "}"), ("[", "]")):
        cand = _find_balanced(cleaned, op, cl)
        if cand:
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from text: {text[:200]!r}")
