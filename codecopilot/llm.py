"""Single LLM client wrapper: retries with exponential backoff, timeouts, streaming.
Application code never calls a provider directly — it calls LLMClient."""
from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Iterator

import httpx

from .config import Settings

_RETRYABLE = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        headers = {"Authorization": f"Bearer {cfg.llm_api_key}"} if cfg.llm_api_key else {}
        self._http = httpx.Client(base_url=cfg.llm_base_url.rstrip("/"), timeout=cfg.llm_timeout_s, headers=headers)

    def _request(self, messages: list[dict], stream: bool, temperature: float | None = None,
                 num_ctx: int | None = None) -> tuple[str, dict]:
        c = self.cfg
        temp = c.llm_temperature if temperature is None else temperature
        ctx = c.llm_num_ctx if num_ctx is None else num_ctx
        if c.llm_provider == "ollama":
            # Ollama's default context window (2–4k tokens) silently truncates our ~6k-token prompt
            # from the front, dropping the system prompt and question. Always set num_ctx explicitly.
            return "/api/chat", {"model": c.llm_model, "messages": messages, "stream": stream,
                                 "options": {"temperature": temp, "num_ctx": ctx, "seed": c.llm_seed}}
        return "/chat/completions", {"model": c.llm_model, "messages": messages, "stream": stream,
                                     "temperature": temp, "seed": c.llm_seed}

    def stream_chat(self, messages: list[dict], temperature: float | None = None,
                    num_ctx: int | None = None) -> Iterator[str]:
        path, body = self._request(messages, stream=True, temperature=temperature, num_ctx=num_ctx)
        for attempt in range(self.cfg.llm_max_retries + 1):
            started = False
            try:
                with self._http.stream("POST", path, json=body) as r:
                    if r.status_code in _RETRYABLE:
                        raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                    if r.status_code >= 400:
                        r.read()
                        raise LLMError(f"{r.status_code}: {r.text[:500]}")
                    for line in r.iter_lines():
                        tok = self._parse_stream_line(line)
                        if tok:
                            started = True
                            yield tok
                return
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                # never retry after tokens were emitted — the caller would see duplicated text
                if started or attempt == self.cfg.llm_max_retries:
                    raise LLMError(f"LLM request failed after {attempt + 1} attempt(s): {e}") from e
                time.sleep(min(2 ** attempt, 20) + random.random())

    def chat(self, messages: list[dict], use_cache: bool = True, temperature: float | None = None,
             num_ctx: int | None = None) -> str:
        """Non-streaming call. Cached on disk, keyed by everything that can change the output
        (provider, model, temperature, seed, context size, full messages), so eval reruns cost nothing.
        Callers that need reproducibility (query rewriting) pass temperature=0 and a fixed num_ctx."""
        key = self._cache_key(messages, temperature, num_ctx)
        path = self.cfg.cache_dir / "llm" / key[:2] / f"{key}.json"
        if use_cache and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        text = "".join(self.stream_chat(messages, temperature=temperature, num_ctx=num_ctx))
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": self.cfg.llm_model, "messages": messages, "response": text}),
                            encoding="utf-8")
        return text

    def _cache_key(self, messages: list[dict], temperature: float | None = None, num_ctx: int | None = None) -> str:
        c = self.cfg
        temp = c.llm_temperature if temperature is None else temperature
        ctx = c.llm_num_ctx if num_ctx is None else num_ctx
        blob = json.dumps([c.llm_provider, c.llm_model, temp, c.llm_seed, ctx, messages], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _parse_stream_line(self, line: str) -> str:
        if not line:
            return ""
        if self.cfg.llm_provider == "ollama":
            return json.loads(line).get("message", {}).get("content", "")
        if not line.startswith("data:"):
            return ""
        data = line[5:].strip()
        if data == "[DONE]":
            return ""
        choices = json.loads(data).get("choices") or [{}]
        return (choices[0].get("delta") or {}).get("content") or ""
