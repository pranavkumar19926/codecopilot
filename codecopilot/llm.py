"""Single LLM client wrapper: retries with exponential backoff, timeouts, streaming.
Application code never calls a provider directly — it calls LLMClient."""
from __future__ import annotations

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

    def _request(self, messages: list[dict], stream: bool) -> tuple[str, dict]:
        c = self.cfg
        if c.llm_provider == "ollama":
            # Ollama's default context window (2–4k tokens) silently truncates our ~6k-token prompt
            # from the front, dropping the system prompt and question. Always set num_ctx explicitly.
            return "/api/chat", {"model": c.llm_model, "messages": messages, "stream": stream,
                                 "options": {"temperature": c.llm_temperature, "num_ctx": c.llm_num_ctx}}
        return "/chat/completions", {"model": c.llm_model, "messages": messages, "stream": stream,
                                     "temperature": c.llm_temperature}

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        path, body = self._request(messages, stream=True)
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

    def chat(self, messages: list[dict]) -> str:
        return "".join(self.stream_chat(messages))

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
