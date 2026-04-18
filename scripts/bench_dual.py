"""Benchmark both servers simultaneously.

Answers: when Model A and Model B are both decoding at max parallel, do they
saturate the iGPU bandwidth and drag each other down, or do they get their
full individual throughput?
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


async def one_call(client: httpx.AsyncClient, url: str, max_tokens: int) -> dict:
    body = {
        "model": "whatever",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    r = await client.post(url, json=body, timeout=600)
    r.raise_for_status()
    data = r.json()
    dt = time.perf_counter() - t0
    usage = data.get("usage") or {}
    return {
        "wall_s": dt,
        "out": int(usage.get("completion_tokens", 0) or 0),
    }


async def server_load(client: httpx.AsyncClient, endpoint: str, n: int,
                      max_tokens: int) -> dict:
    url = f"{endpoint.rstrip('/')}/chat/completions"
    t0 = time.perf_counter()
    results = await asyncio.gather(*(one_call(client, url, max_tokens) for _ in range(n)))
    wall = time.perf_counter() - t0
    out_sum = sum(r["out"] for r in results)
    return {"endpoint": endpoint, "n": n, "wall_s": wall, "out_sum": out_sum,
            "agg_tps": out_sum / wall if wall else 0}


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ep-a", default="http://127.0.0.1:8080/v1")
    p.add_argument("--ep-b", default="http://127.0.0.1:8081/v1")
    p.add_argument("--n", type=int, default=6, help="concurrency per server")
    p.add_argument("--max-tokens", type=int, default=400)
    args = p.parse_args()

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        a, b = await asyncio.gather(
            server_load(client, args.ep_a, args.n, args.max_tokens),
            server_load(client, args.ep_b, args.n, args.max_tokens),
        )
        wall_total = time.perf_counter() - t0

    print(f"Concurrency per server: n={args.n}  max_tokens={args.max_tokens}")
    print(f"  Model A:  wall={a['wall_s']:.2f}s  out={a['out_sum']}  "
          f"agg_tps={a['agg_tps']:.1f}")
    print(f"  Model B:  wall={b['wall_s']:.2f}s  out={b['out_sum']}  "
          f"agg_tps={b['agg_tps']:.1f}")
    print(f"  Combined: wall={wall_total:.2f}s  "
          f"agg_tps={(a['out_sum']+b['out_sum'])/wall_total:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
