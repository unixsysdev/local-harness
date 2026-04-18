"""Micro-benchmark: sweep concurrent-request count against one llama-server,
measure per-stream TPS and aggregate TPS to find the MoE batching knee.

No harness code path involved — raw /v1/chat/completions calls only.
"""
from __future__ import annotations
import argparse
import asyncio
import time

import httpx

PROMPT = (
    "Count from 1 to 200 in a comma-separated list. "
    "Example: 1, 2, 3, 4, 5, ... Start now and do not stop until you reach 200."
)


async def one_call(client: httpx.AsyncClient, url: str, max_tokens: int,
                   thinking: bool) -> dict:
    body = {
        "model": "whatever",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    t0 = time.perf_counter()
    r = await client.post(url, json=body, timeout=600)
    r.raise_for_status()
    data = r.json()
    dt = time.perf_counter() - t0
    usage = data.get("usage") or {}
    return {
        "wall_s": dt,
        "in": int(usage.get("prompt_tokens", 0) or 0),
        "out": int(usage.get("completion_tokens", 0) or 0),
        "tps": (int(usage.get("completion_tokens", 0) or 0) / dt) if dt > 0 else 0,
    }


async def run_level(endpoint: str, n: int, max_tokens: int, thinking: bool) -> dict:
    url = f"{endpoint.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(
            one_call(client, url, max_tokens, thinking) for _ in range(n)
        ))
        wall = time.perf_counter() - t0
    out_sum = sum(r["out"] for r in results)
    per_stream = sorted(r["tps"] for r in results)
    return {
        "n": n,
        "wall_s": round(wall, 2),
        "out_sum": out_sum,
        "aggregate_tps": round(out_sum / wall, 1),
        "per_stream_tps_median": round(per_stream[n // 2], 1),
        "per_stream_tps_min": round(per_stream[0], 1),
        "per_stream_tps_max": round(per_stream[-1], 1),
    }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    p.add_argument("--levels", default="1,2,3,4,6", help="comma-separated concurrency levels")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--thinking", action="store_true")
    args = p.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    print(f"Endpoint: {args.endpoint}")
    print(f"{'n':>3}  {'wall_s':>7}  {'out_sum':>8}  {'agg_tps':>8}  "
          f"{'per_min':>8}  {'per_med':>8}  {'per_max':>8}")
    for n in levels:
        r = await run_level(args.endpoint, n, args.max_tokens, args.thinking)
        print(f"{r['n']:>3}  {r['wall_s']:>7}  {r['out_sum']:>8}  "
              f"{r['aggregate_tps']:>8}  "
              f"{r['per_stream_tps_min']:>8}  "
              f"{r['per_stream_tps_median']:>8}  "
              f"{r['per_stream_tps_max']:>8}")


if __name__ == "__main__":
    asyncio.run(main())
