"""Minimal JSON-schema validator (no external dep).

Supports a small subset: type, required, properties.type, items.type.
Enough for Phase 1 extraction tasks. Full jsonschema can replace later.
"""
from __future__ import annotations
from typing import Any


def _type_ok(val: Any, t: str) -> bool:
    if t == "string":  return isinstance(val, str)
    if t == "integer": return isinstance(val, int) and not isinstance(val, bool)
    if t == "number":  return isinstance(val, (int, float)) and not isinstance(val, bool)
    if t == "boolean": return isinstance(val, bool)
    if t == "array":   return isinstance(val, list)
    if t == "object":  return isinstance(val, dict)
    if t == "null":    return val is None
    return True  # unknown type -> accept


def validate(instance: Any, schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (ok, errors)."""
    errors: list[str] = []

    def walk(inst: Any, sch: dict[str, Any], path: str):
        t = sch.get("type")
        if t and not _type_ok(inst, t):
            errors.append(f"{path}: expected {t}, got {type(inst).__name__}")
            return
        if t == "object" and isinstance(inst, dict):
            req = sch.get("required") or []
            for r in req:
                if r not in inst:
                    errors.append(f"{path}: missing required '{r}'")
            props = sch.get("properties") or {}
            for k, subschema in props.items():
                if k in inst:
                    walk(inst[k], subschema, f"{path}.{k}")
        elif t == "array" and isinstance(inst, list):
            items = sch.get("items")
            if items:
                for i, it in enumerate(inst):
                    walk(it, items, f"{path}[{i}]")

    walk(instance, schema, "$")
    return len(errors) == 0, errors
