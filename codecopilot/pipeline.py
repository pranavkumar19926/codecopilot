"""Retrieve → assemble context under a token budget → generate → check citations → log the run."""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from .config import Settings
from .embed import Embedder
from .index import Hit, VectorIndex
from .llm import LLMClient
from .prompts import load_prompt

# Accept [path:a-b] (what the prompt asks for) and `path:a-b` (what small models often write instead).
_CITE_RE = re.compile(r"[\[`]([^\[\]`\s]+?\.\w+):(\d+)-(\d+)[\]`]")


def approx_tokens(text: str) -> int:
    return len(text) // 4 + 1  # good enough for budgeting; swap for a real tokenizer later


def assemble_context(hits: list[Hit], budget: int) -> tuple[str, list[Hit]]:
    parts, used, total = [], [], 0
    for h in hits:
        block = f"### {h.chunk.citation}\n```python\n{h.chunk.text}\n```"
        cost = approx_tokens(block)
        if total + cost > budget and used:
            break
        parts.append(block); used.append(h); total += cost
    return "\n\n".join(parts), used


def check_citations(answer: str, used: list[Hit]) -> dict:
    """A citation is valid if its range falls inside a chunk actually shown to the model."""
    cites = [(p, int(a), int(b)) for p, a, b in _CITE_RE.findall(answer)]
    def ok(p, a, b):
        return any(h.chunk.path == p and h.chunk.start_line <= a and b <= h.chunk.end_line for h in used)
    invalid = [f"{p}:{a}-{b}" for p, a, b in cites if not ok(p, a, b)]
    return {"n_citations": len(cites), "invalid": invalid}


@dataclass
class Answer:
    question: str
    hits: list[Hit]
    text: str = ""
    citation_check: dict = field(default_factory=dict)


class Copilot:
    def __init__(self, cfg: Settings, index: VectorIndex, embedder: Embedder, llm: LLMClient | None = None):
        self.cfg, self.index, self.embedder, self.llm = cfg, index, embedder, llm
        self.prompt = load_prompt(cfg.prompt_id)

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.index.search(self.embedder.embed_query(question), k or self.cfg.top_k)

    def ask_stream(self, question: str) -> Iterator[str | Answer]:
        """Yields answer tokens as they arrive, then a final Answer object."""
        if self.llm is None:
            raise RuntimeError("No LLM configured")
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        context, used = assemble_context(hits, self.cfg.context_token_budget)
        t_retrieve = time.perf_counter() - t0
        tokens = []
        for tok in self.llm.stream_chat(self.prompt.render(question=question, context=context)):
            tokens.append(tok)
            yield tok
        ans = Answer(question, used, "".join(tokens))
        ans.citation_check = check_citations(ans.text, used)
        self._log(ans, t_retrieve, time.perf_counter() - t0)
        yield ans

    def _log(self, ans: Answer, t_retrieve: float, t_total: float) -> None:
        # Append-only run log. Becomes the trace store in Project 05 / eval input in Project 03.
        rec = {
            "run_id": uuid.uuid4().hex, "ts": time.time(), "question": ans.question,
            "prompt_id": self.prompt.id, "prompt_hash": self.prompt.version_hash,
            "llm": f"{self.cfg.llm_provider}:{self.cfg.llm_model}", "embed_model": self.index.meta["embed_model"],
            "retrieved": [{"citation": h.chunk.citation, "score": round(h.score, 4)} for h in ans.hits],
            "answer": ans.text, "citation_check": ans.citation_check,
            "latency_s": {"retrieve": round(t_retrieve, 3), "total": round(t_total, 3)},
        }
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.data_dir / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
