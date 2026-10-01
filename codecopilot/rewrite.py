"""Query rewriting (multi-query expansion).

Users ask in their words ("forget the http:// part"); code uses its own ("MissingSchema", "scheme").
An LLM rewrites the question into code-vocabulary queries; each is retrieved separately and the
ranked lists are fused with RRF. The original question keeps double weight so a bad rewrite can
add candidates but can't push out what the original already found."""
from __future__ import annotations

import re

from .llm import LLMClient
from .prompts import load_prompt

_LEAD = re.compile(r"^\s*(?:[-*•]|\d+[.):]|line\s*\d+\s*:)\s*", re.IGNORECASE)


def parse_rewrites(text: str, max_n: int = 3) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        q = _LEAD.sub("", line).strip().strip('"').strip("`").strip()
        if len(q) >= 3 and q.lower() not in {o.lower() for o in out}:
            out.append(q)
    return out[:max_n]


def rewrite_query(llm: LLMClient, question: str, prompt_id: str = "rewrite_v1") -> list[str]:
    prompt = load_prompt(prompt_id)
    return parse_rewrites(llm.chat(prompt.render(question=question)))
