"""Direct comparison: parallel dual (A+B concurrent) vs serial (all A, then all B).

Same total workload in both cases: 6 candidates from each model, 400 max tokens,
same prompt. Measures wall-clock so we can see which scheduling wins.
"""
from __future__ import annotations
import asyncio
import time

import httpx

EP_A = "http://127.0.0.1:8080/v1/chat/completions"
EP_B = "http://127.0.0.1:8081/v1/chat/completions"
N = 6
MAX_TOK = 400

PROMPT = (
    "Count from 1 to 200 in a comma-separated list. "
    "Example: 1, 2, 3, 4, 5, ... Start now and do not stop until you reach 200."
)


def body() -> dict:
    return {
        "model": "whatever",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7,
        "max_tokens": MAX_TOK,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def call(client: httpx.AsyncClient, url: str) -> int:
    r = await client.post(url, json=body(), timeout=600)
    r.raise_for_status()
    return int((r.json().get("usage") or {}).get("completion_tokens", 0) or 0)


async def burst(client: httpx.AsyncClient, url: str, n: int) -> tuple[float, int]:
    t0 = time.perf_counter()
    outs = await asyncio.gather(*(call(client, url) for _ in range(n)))
    return time.perf_counter() - t0, sum(outs)


async def parallel_dual(client: httpx.AsyncClient) -> tuple[float, int]:
    t0 = time.perf_counter()
    (wa, oa), (wb, ob) = await asyncio.gather(burst(client, EP_A, N), burst(client, EP_B, N))
    return time.perf_counter() - t0, oa + ob


async def serial_a_then_b(client: httpx.AsyncClient) -> tuple[float, int]:
    t0 = time.perf_counter()
    wa, oa = await burst(client, EP_A, N)
    wb, ob = await burst(client, EP_B, N)
    return time.perf_counter() - t0, oa + ob


async def main() -> None:
    async with httpx.AsyncClient() as client:
        print(f"Workload: {N} candidates per model × {MAX_TOK} max_tokens = "
              f"{2*N} candidates total\n")

        # Warmup each server with a single tiny call to wake any cache up.
        await burst(client, EP_A, 1)
        await burst(client, EP_B, 1)

        print("PARALLEL (A and B concurrent):")
        pw, po = await parallel_dual(client)
        print(f"  wall={pw:.2f}s  out={po}  agg_tps={po/pw:.1f}\n")

        print("SERIAL (all A, then all B):")
        sw, so = await serial_a_then_b(client)
        print(f"  wall={sw:.2f}s  out={so}  agg_tps={so/sw:.1f}\n")

        faster = "parallel" if pw < sw else "serial"
        pct = 100 * abs(pw - sw) / max(pw, sw)
        print(f"=> {faster} is {pct:.1f}% faster "
              f"({min(pw,sw):.2f}s vs {max(pw,sw):.2f}s)")


if __name__ == "__main__":
    asyncio.run(main())
