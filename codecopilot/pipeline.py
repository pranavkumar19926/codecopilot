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
from .bm25 import BM25Index, identifier_terms, rrf
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
        label = f"  ({h.chunk.kind}: {h.chunk.symbol})" if h.chunk.symbol else ""
        block = f"### {h.chunk.citation}{label}\n```python\n{h.chunk.text}\n```"
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
    def __init__(self, cfg: Settings, index: VectorIndex, embedder: Embedder | None,
                 llm: LLMClient | None = None, bm25: BM25Index | None = None):
        self.cfg, self.index, self.embedder, self.llm, self.bm25 = cfg, index, embedder, llm, bm25
        self.prompt = load_prompt(cfg.prompt_id)

    def retrieve(self, question: str, k: int | None = None, mode: str | None = None) -> list[Hit]:
        """mode: dense (Phase 1) | bm25 | hybrid (dense + BM25 fused with RRF, Phase 2 default)."""
        k, mode = k or self.cfg.top_k, mode or self.cfg.retrieval_mode
        depth = max(k, self.cfg.fusion_depth)
        rankings: dict[str, list[int]] = {}
        if mode in ("dense", "hybrid"):
            if self.embedder is None:
                raise RuntimeError(f"retrieval mode {mode!r} needs an embedder")
            dense = self.index.search(self.embedder.embed_query(question), depth)
            pos = {id(c): i for i, c in enumerate(self.index.chunks)}
            rankings["dense"] = [pos[id(h.chunk)] for h in dense]
        if mode in ("bm25", "hybrid"):
            if self.bm25 is None:
                raise RuntimeError("No BM25 index found. Re-run `codecopilot index <repo>`.")
            rankings["bm25"] = [doc for doc, _ in self.bm25.search(question, depth)]
        if not rankings:
            raise ValueError(f"unknown retrieval mode {mode!r}")

        if len(rankings) == 1:
            (name, ranking), = rankings.items()
            fused = [(doc, 1.0 / (self.cfg.rrf_k + r)) for r, doc in enumerate(ranking, 1)]
        else:
            fused = rrf(list(rankings.values()), self.cfg.rrf_k)
        # Symbol lookup: an identifier in the query that names a definition goes to rank 1, like
        # go-to-definition. Rank fusion alone can demote an exact match that only one retriever found.
        pinned = self._pinned(question) if (mode != "dense" and self.cfg.symbol_pin) else []
        if pinned:
            top = 1.0 / self.cfg.rrf_k + (fused[0][1] if fused else 0.0)
            fused = [(d, top) for d in pinned] + [(d, s) for d, s in fused if d not in set(pinned)]
        rank_of = {name: {doc: r for r, doc in enumerate(rk, 1)} for name, rk in rankings.items()}
        return [Hit(self.index.chunks[doc], score,
                    {**{name: rank_of[name].get(doc) for name in rankings}, **({"symbol": True} if doc in pinned else {})})
                for doc, score in fused[:k]]

    def _pinned(self, question: str) -> list[int]:
        terms = identifier_terms(question)
        if not terms:
            return []
        if not hasattr(self, "_symbols"):
            self._symbols: dict[str, list[int]] = {}
            for i, c in enumerate(self.index.chunks):
                if c.kind in ("window", "module", "class_body") or not c.symbol:
                    continue
                for name in c.symbol.split(", "):
                    for key in {name.lower(), name.split(".")[-1].lower()}:
                        self._symbols.setdefault(key, []).append(i)
        out: list[int] = []
        for t in sorted(terms):
            for d in self._symbols.get(t, [])[: self.cfg.symbol_pin_max]:
                if d not in out:
                    out.append(d)
        return out[: self.cfg.symbol_pin_max]

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
            "retrieval_mode": self.cfg.retrieval_mode, "chunker": self.index.meta.get("chunker", "fixed"),
            "retrieved": [{"citation": h.chunk.citation, "symbol": h.chunk.symbol, "score": round(h.score, 4),
                           "ranks": h.detail} for h in ans.hits],
            "answer": ans.text, "citation_check": ans.citation_check,
            "latency_s": {"retrieve": round(t_retrieve, 3), "total": round(t_total, 3)},
        }
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.data_dir / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
