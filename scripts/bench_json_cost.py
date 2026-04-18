"""A/B test: does response_format: json_object slow decode down?

Fires 6 concurrent requests per mode with a realistic proposal-style prompt.
Runs each mode twice and reports median to wash out variance.
"""
from __future__ import annotations
import asyncio
import statistics
import time

import httpx

EP_A = "http://127.0.0.1:8080/v1/chat/completions"
EP_B = "http://127.0.0.1:8081/v1/chat/completions"
N = 6
MAX_TOK = 400

PROMPT = (
    "Task:\n"
    "<<<Write a Python function is_prime(n: int) -> bool.>>>\n\n"
    "Rubric:\n"
    "- Summary: Implement efficient primality test.\n"
    "- Failure modes: wrong edge cases, too slow.\n\n"
    "Strategy: baseline. Solve directly, do not pad.\n\n"
    "Return JSON:\n"
    "{\n"
    '  "solution": "string",\n'
    '  "assumptions": ["string"],\n'
    '  "known_risks": ["string"],\n'
    '  "self_confidence": 0.0\n'
    "}\n"
)


def body(constrained: bool) -> dict:
    b = {
        "model": "whatever",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7,
        "max_tokens": MAX_TOK,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if constrained:
        b["response_format"] = {"type": "json_object"}
    return b


async def call(client: httpx.AsyncClient, url: str, constrained: bool) -> int:
    r = await client.post(url, json=body(constrained), timeout=600)
    r.raise_for_status()
    return int((r.json().get("usage") or {}).get("completion_tokens", 0) or 0)


async def burst(client, url, n, constrained) -> tuple[float, int]:
    t0 = time.perf_counter()
    outs = await asyncio.gather(*(call(client, url, constrained) for _ in range(n)))
    return time.perf_counter() - t0, sum(outs)


async def dual(client, n, constrained) -> tuple[float, int]:
    t0 = time.perf_counter()
    (wa, oa), (wb, ob) = await asyncio.gather(
        burst(client, EP_A, n, constrained),
        burst(client, EP_B, n, constrained),
    )
    return time.perf_counter() - t0, oa + ob


async def main():
    async with httpx.AsyncClient() as client:
        # warmup
        await burst(client, EP_A, 1, False)
        await burst(client, EP_B, 1, False)

        rows = {}
        for mode_label, constrained in [("constrained", True), ("free", False)]:
            times, outs = [], []
            for _ in range(2):
                w, o = await dual(client, N, constrained)
                times.append(w); outs.append(o)
            rows[mode_label] = {
                "median_wall_s": statistics.median(times),
                "median_out": statistics.median(outs),
                "tps": statistics.median(outs) / statistics.median(times),
            }

        print(f"Per-server concurrency: n={N}  max_tokens={MAX_TOK}  runs=2 each\n")
        for k, v in rows.items():
            print(f"  {k:12}  wall={v['median_wall_s']:.2f}s  "
                  f"out={v['median_out']:.0f}  tps={v['tps']:.1f}")
        c, f = rows["constrained"], rows["free"]
        delta_pct = 100 * (c["median_wall_s"] - f["median_wall_s"]) / f["median_wall_s"]
        print(f"\n  constrained is {delta_pct:+.1f}% vs free "
              f"({c['median_wall_s']:.2f}s vs {f['median_wall_s']:.2f}s)")


if __name__ == "__main__":
    asyncio.run(main())
