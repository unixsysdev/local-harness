"""Evaluator: compare baseline vs harness output per task, produce verdicts.

Two paths:
- verifiable:     run the task's deterministic validator on BOTH baseline and
                  harness outputs; verdict from pass/fail combination.
- non_verifiable: pairwise LLM preference with both models as judges and
                  order-swap to catch positional bias.
"""
