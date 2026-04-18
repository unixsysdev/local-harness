"""OpenAI-compatible HTTP adapter (targets llama.cpp --host/--port server).

Async, batched via asyncio.gather. Supports per-request overrides of
temperature, top_p, seed, max_tokens, response_format, and the llama.cpp-
specific `chat_template_kwargs` (e.g. {"enable_thinking": false}).

Streaming mode (default): requests are sent with `stream: true` and SSE chunks
are accumulated token-by-token. Timeout is PER-CHUNK (httpx `read` timeout),
not total-request, so long-running generations don't spuriously fail as long
as the server keeps emitting tokens.
"""
from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass
class GenRequest:
    request_id: str
    model_key: str                       # "model_a" | "model_b" (logical)
    prompt: str | list[dict[str, Any]]   # raw text -> wrapped as user, or messages
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 800
    seed: Optional[int] = None
    stop: list[str] = field(default_factory=list)
    response_format: Optional[dict[str, Any]] = None
    thinking: Optional[bool] = None
    system: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenResponse:
    request_id: str
    model_key: str
    text: str
    reasoning: Optional[str] = None
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0          # prompt_tokens_details.cached_tokens
    latency_ms: int = 0
    error: Optional[str] = None
    raw: Optional[dict[str, Any]] = None


class OpenAICompatBackend:
    """One logical backend binding a model_key -> HTTP endpoint.

    Uses streaming with per-chunk idle timeout so long generations don't
    spuriously fail as long as tokens keep flowing.
    """

    def __init__(
        self,
        endpoints: dict[str, str],        # model_key -> base URL (no /chat/completions)
        served_names: dict[str, str],     # model_key -> model name string sent in payload
        thinking_defaults: dict[str, bool],
        timeout: float = 600.0,           # retained for backward-compat; ignored in stream mode
        concurrency: int = 8,
        stream: bool = True,              # stream by default
        idle_timeout_s: float = 180.0,    # max gap between SSE events before we call it stuck
        connect_timeout_s: float = 30.0,
    ):
        self.endpoints = endpoints
        self.served_names = served_names
        self.thinking_defaults = thinking_defaults
        self.stream = stream
        # In streaming mode: no total timeout, per-chunk read timeout is what catches a stuck server.
        # In non-streaming mode: `timeout` is the total-request timeout (legacy behavior).
        if stream:
            httpx_timeout = httpx.Timeout(
                None,
                connect=connect_timeout_s,
                read=idle_timeout_s,
                write=60.0,
                pool=connect_timeout_s,
            )
        else:
            httpx_timeout = httpx.Timeout(timeout)
        self._client = httpx.AsyncClient(timeout=httpx_timeout)
        self._sem = asyncio.Semaphore(concurrency)
        self._idle_timeout = idle_timeout_s

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self, model_key: str) -> bool:
        base = self.endpoints[model_key]
        # llama.cpp server exposes /health (no /v1 prefix)
        url = base.rstrip("/").removesuffix("/v1") + "/health"
        try:
            r = await self._client.get(url)
            return r.status_code == 200
        except Exception:
            return False

    def _messages(self, req: GenRequest) -> list[dict[str, Any]]:
        if isinstance(req.prompt, list):
            msgs = list(req.prompt)
        else:
            msgs = [{"role": "user", "content": req.prompt}]
        if req.system:
            msgs = [{"role": "system", "content": req.system}] + msgs
        return msgs

    def _build_body(self, req: GenRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.served_names[req.model_key],
            "messages": self._messages(req),
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "stream": self.stream,
        }
        if self.stream:
            # Ask server to include usage in the final chunk
            body["stream_options"] = {"include_usage": True}
        if req.seed is not None:
            body["seed"] = req.seed
        if req.stop:
            body["stop"] = req.stop
        if req.response_format is not None:
            body["response_format"] = req.response_format

        # llama.cpp-specific: chat_template_kwargs controls thinking toggle
        thinking = req.thinking
        if thinking is None:
            thinking = self.thinking_defaults.get(req.model_key, False)
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}

        if req.extra:
            body.update(req.extra)
        return body

    async def _one_nonstream(self, req: GenRequest) -> GenResponse:
        base = self.endpoints[req.model_key].rstrip("/")
        url = f"{base}/chat/completions"
        body = self._build_body(req)
        t0 = time.perf_counter()
        async with self._sem:
            try:
                r = await self._client.post(url, json=body)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPError as e:
                return GenResponse(
                    request_id=req.request_id, model_key=req.model_key,
                    text="", error=str(e),
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content")
        finish = choice.get("finish_reason", "") or ""
        usage = data.get("usage") or {}
        cached = 0
        pdetails = usage.get("prompt_tokens_details") or {}
        if isinstance(pdetails, dict):
            cached = int(pdetails.get("cached_tokens", 0) or 0)
        return GenResponse(
            request_id=req.request_id,
            model_key=req.model_key,
            text=text,
            reasoning=reasoning,
            finish_reason=finish,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            cached_tokens=cached,
            latency_ms=latency_ms,
            raw=data,
        )

    async def _one_stream(self, req: GenRequest) -> GenResponse:
        """Streaming version: accumulate SSE deltas, per-chunk read timeout."""
        base = self.endpoints[req.model_key].rstrip("/")
        url = f"{base}/chat/completions"
        body = self._build_body(req)

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = ""
        usage: dict[str, Any] = {}
        last_raw: dict[str, Any] | None = None
        error: Optional[str] = None

        t0 = time.perf_counter()
        async with self._sem:
            try:
                async with self._client.stream("POST", url, json=body) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        last_raw = chunk
                        # usage-only chunk (stream_options.include_usage)
                        if chunk.get("usage") and not chunk.get("choices"):
                            usage = chunk["usage"]
                            continue
                        choices = chunk.get("choices") or [{}]
                        ch0 = choices[0]
                        if ch0.get("finish_reason"):
                            finish_reason = ch0["finish_reason"]
                        delta = ch0.get("delta") or {}
                        if delta.get("content"):
                            text_parts.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning_parts.append(delta["reasoning_content"])
                        # Some servers include usage on the final choice chunk
                        if chunk.get("usage"):
                            usage = chunk["usage"]
            except httpx.HTTPError as e:
                error = f"{type(e).__name__}: {e}"

        latency_ms = int((time.perf_counter() - t0) * 1000)
        cached = 0
        pdetails = (usage or {}).get("prompt_tokens_details") or {}
        if isinstance(pdetails, dict):
            cached = int(pdetails.get("cached_tokens", 0) or 0)

        return GenResponse(
            request_id=req.request_id,
            model_key=req.model_key,
            text="".join(text_parts),
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
            finish_reason=finish_reason,
            input_tokens=int((usage or {}).get("prompt_tokens", 0) or 0),
            output_tokens=int((usage or {}).get("completion_tokens", 0) or 0),
            cached_tokens=cached,
            latency_ms=latency_ms,
            error=error,
            raw=last_raw,
        )

    async def _one(self, req: GenRequest) -> GenResponse:
        if self.stream:
            return await self._one_stream(req)
        return await self._one_nonstream(req)

    async def generate(self, requests: list[GenRequest]) -> list[GenResponse]:
        return await asyncio.gather(*(self._one(r) for r in requests))
