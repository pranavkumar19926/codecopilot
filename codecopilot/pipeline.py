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
from .citations import check_answer
from .graph import CodeGraph
from .graph import intent as graph_intent
from .index import Hit, VectorIndex
from .llm import LLMClient
from .prompts import load_prompt
from .rerank import Reranker
from .rewrite import rewrite_query

# Accept [path:a-b] (what the prompt asks for) and `path:a-b` (what small models often write instead).
_CITE_RE = re.compile(r"[\[`]([^\[\]`\s]+?\.\w+):(\d+)-(\d+)[\]`]")


def approx_tokens(text: str) -> int:
    return len(text) // 4 + 1  # good enough for budgeting; swap for a real tokenizer later


def assemble_context(hits: list[Hit], budget: int, line_numbers: bool = False) -> tuple[str, list[Hit]]:
    """Format hits as excerpts under a token budget. With line_numbers, each line is prefixed `  216| ` so the
    model can cite exact lines instead of whole chunks (Phase 4 strict citations)."""
    parts, used, total = [], [], 0
    for h in hits:
        label = f"  ({h.chunk.kind}: {h.chunk.symbol})" if h.chunk.symbol else ""
        if h.detail.get("graph"):
            label += f"  [related: {h.detail['graph']}]"
        body = h.chunk.text
        if line_numbers:
            body = "\n".join(f"{n:>5}| {line}" for n, line in enumerate(body.splitlines(), h.chunk.start_line))
        block = f"### {h.chunk.citation}{label}\n```python\n{body}\n```"
        cost = approx_tokens(block)
        if total + cost > budget and used:
            continue   # skip this one but keep trying smaller chunks further down
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
    draft: str = ""                 # first answer, before any citation repair
    draft_check: dict = field(default_factory=dict)
    repaired: bool = False          # True if the repaired answer replaced the draft
    expanded: list[Hit] = field(default_factory=list)


@dataclass
class Repairing:
    """Yielded by ask_stream between the draft and the repaired answer, so the UI can say what's happening."""
    problems: list[str]


class Copilot:
    def __init__(self, cfg: Settings, index: VectorIndex, embedder: Embedder | None,
                 llm: LLMClient | None = None, bm25: BM25Index | None = None, reranker: Reranker | None = None,
                 graph: CodeGraph | None = None):
        self.cfg, self.index, self.embedder, self.llm, self.bm25 = cfg, index, embedder, llm, bm25
        self.reranker, self.graph = reranker, graph
        self._pos = {id(c): i for i, c in enumerate(index.chunks)}
        self.prompt = load_prompt(cfg.prompt_id)
        self.last_rewrites: list[str] = []

    def _rank(self, query: str, mode: str, depth: int) -> tuple[list[tuple[int, float]], dict[str, list[int]]]:
        """First stage for one query string: dense and/or BM25 ranked lists, fused with RRF."""
        rankings: dict[str, list[int]] = {}
        if mode in ("dense", "hybrid"):
            if self.embedder is None:
                raise RuntimeError(f"retrieval mode {mode!r} needs an embedder")
            dense = self.index.search(self.embedder.embed_query(query), depth)
            rankings["dense"] = [self._pos[id(h.chunk)] for h in dense]
        if mode in ("bm25", "hybrid"):
            if self.bm25 is None:
                raise RuntimeError("No BM25 index found. Re-run `codecopilot index <repo>`.")
            rankings["bm25"] = [doc for doc, _ in self.bm25.search(query, depth)]
        if not rankings:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        if len(rankings) == 1:
            (ranking,) = rankings.values()
            return [(doc, 1.0 / (self.cfg.rrf_k + r)) for r, doc in enumerate(ranking, 1)], rankings
        return rrf(list(rankings.values()), self.cfg.rrf_k), rankings

    def retrieve(self, question: str, k: int | None = None, mode: str | None = None,
                 rerank: bool | None = None, rewrite: bool | None = None) -> list[Hit]:
        """Stages: [rewrite → multi-query] → dense/BM25 → RRF → symbol pins → [cross-encoder rerank] → top-k.
        mode: dense (Phase 1) | bm25 | hybrid (Phase 2). rerank / rewrite: Phase 3, default from config."""
        k, mode = k or self.cfg.top_k, mode or self.cfg.retrieval_mode
        rerank = self.cfg.rerank if rerank is None else rerank
        rewrite = self.cfg.query_rewrite if rewrite is None else rewrite
        depth = max(k, self.cfg.fusion_depth, self.cfg.rerank_depth if rerank else 0)

        fused, rankings = self._rank(question, mode, depth)
        self.last_rewrites = []
        if rewrite:
            if self.llm is None:
                raise RuntimeError("query rewriting needs an LLM")
            self.last_rewrites = rewrite_query(self.llm, question, self.cfg.rewrite_prompt_id)
            lists = [[d for d, _ in fused]] * 2  # original question counts twice
            for q in self.last_rewrites:
                lists.append([d for d, _ in self._rank(q, mode, depth)[0]])
            fused = rrf(lists, self.cfg.rrf_k)

        # Symbol lookup: an identifier in the query that names a definition goes to rank 1, like
        # go-to-definition. Rank fusion alone can demote an exact match that only one retriever found.
        pinned = self._pinned(question) if (mode != "dense" and self.cfg.symbol_pin) else []
        rest = [(d, s) for d, s in fused if d not in set(pinned)]

        rerank_rank: dict[int, int] = {}
        if rerank:
            if self.reranker is None:
                raise RuntimeError("reranking needs a reranker")
            n = max(self.cfg.rerank_depth - len(pinned), 0)
            cands, tail = rest[:n], rest[n:]
            scores = self.reranker.score(question, [self.index.chunks[d].embed_text for d, _ in cands])
            cands = sorted(((d, sc) for (d, _), sc in zip(cands, scores)), key=lambda x: -x[1])
            rerank_rank = {d: r for r, (d, _) in enumerate(cands, 1)}
            rest = cands + tail

        top = (rest[0][1] if rest else 0.0) + 1.0
        ordered = [(d, top) for d in pinned] + rest
        self._last_rank = {d: r for r, (d, _) in enumerate(ordered)}   # used by expand() to break ties
        rank_of = {name: {doc: r for r, doc in enumerate(rk, 1)} for name, rk in rankings.items()}
        hits = []
        for doc, score in ordered[:k]:
            detail = {name: rank_of[name].get(doc) for name in rankings}
            if doc in pinned:
                detail["symbol"] = True
            if doc in rerank_rank:
                detail["rerank"] = rerank_rank[doc]
            hits.append(Hit(self.index.chunks[doc], score, detail))
        return hits

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
            # Methods of small classes live inside the class chunk and have no chunk symbol of their own;
            # the call graph knows where every definition is, so it fills those gaps.
            if self.graph is not None:
                for n in self.graph.nodes:
                    if n.kind == "module" or n.chunk < 0:
                        continue
                    for key in {n.qual.lower(), n.name.lower()}:
                        lst = self._symbols.setdefault(key, [])
                        if n.chunk not in lst:
                            lst.append(n.chunk)
        out: list[int] = []
        for t in sorted(terms):
            for d in self._symbols.get(t, [])[: self.cfg.symbol_pin_max]:
                if d not in out:
                    out.append(d)
        return out[: self.cfg.symbol_pin_max]

    # ---- Phase 4: graph expansion ----------------------------------------------------------------
    def _chunk_at(self, path: str, line: int) -> int:
        if not hasattr(self, "_by_path"):
            self._by_path: dict[str, list[int]] = {}
            for i, c in enumerate(self.index.chunks):
                self._by_path.setdefault(c.path, []).append(i)
        for i in self._by_path.get(path, []):
            c = self.index.chunks[i]
            if c.start_line <= line <= c.end_line:
                return i
        return -1

    def expand(self, question: str, hits: list[Hit], max_extra: int | None = None) -> list[Hit]:
        """Add 1-hop graph neighbours of the top hits that retrieval didn't already return.
        Direction follows the question: "what calls X / what breaks" → callers, "what does X call" → callees,
        anything else → both (callees weighted higher: what a function *does* usually explains it)."""
        if self.graph is None or not self.cfg.graph_expand or not hits:
            return []
        direction = graph_intent(question)
        limit = max_extra if max_extra is not None else self.cfg.graph_max_extra * (1 if direction == "related" else 2)
        shown = {self._pos[id(h.chunk)] for h in hits}
        seeds = [h for h in hits if h.detail.get("symbol")] or []
        seeds += [h for h in hits if h not in seeds and h.chunk.kind not in ("module", "module_part", "class_body")]
        seeds = seeds[: self.cfg.graph_seeds]
        g = self.graph
        cand: dict[int, list] = {}   # chunk idx → [score, reason]
        for rank, h in enumerate(seeds):
            seed_w = 1.0 if (rank == 0 or h.detail.get("symbol")) else 0.6
            idx = self._pos[id(h.chunk)]
            for node in g.nodes_in_chunk(idx):
                if node.kind == "module":
                    continue
                if direction in ("callees", "related"):
                    w = 2.0 if direction == "callees" else 1.0
                    for e in g.callees(node.id):
                        tgt = g.nodes[e.dst]
                        self._add(cand, tgt.chunk, shown, seed_w * w, f"{e.kind}ed by {node.qual}"
                                  if e.kind in ("call", "instantiate") else f"{e.kind} of {node.qual}")
                if direction in ("callers", "related"):
                    w = 2.0 if direction == "callers" else 0.75
                    for e in g.callers(node.id):
                        src = g.nodes[e.src]
                        verb = {"call": "calls", "instantiate": "creates", "decorator": "is decorated by",
                                "inherit": "subclasses"}[e.kind]
                        self._add(cand, self._chunk_at(src.path, e.line), shown, seed_w * w,
                                  f"{src.qual} {verb} {node.qual} (line {e.line})")
        # tie-break by how well retrieval itself ranked the neighbour for this question
        rank = getattr(self, "_last_rank", {})
        for i, v in cand.items():
            if i in rank:
                v[0] += 0.5 / (1 + rank[i] / 10)
        best = sorted(cand.items(), key=lambda kv: -kv[1][0])[:limit]
        return [Hit(self.index.chunks[i], sc, {"graph": reason}) for i, (sc, reason) in best]

    @staticmethod
    def _add(cand: dict, chunk_idx: int, shown: set, w: float, reason: str) -> None:
        if chunk_idx < 0 or chunk_idx in shown:
            return
        if chunk_idx in cand:
            cand[chunk_idx][0] += w
        else:
            cand[chunk_idx] = [w, reason]

    # ---- answering --------------------------------------------------------------------------------
    def build_context(self, question: str) -> tuple[list[Hit], list[Hit], str, list[Hit]]:
        hits = self.retrieve(question)
        extra = self.expand(question, hits)
        context, used = assemble_context(hits + extra, self.cfg.context_token_budget, self.cfg.context_line_numbers)
        return hits, extra, context, used

    def _generate(self, question: str, stream: bool) -> Iterator[str | Repairing | Answer]:
        if self.llm is None:
            raise RuntimeError("No LLM configured")
        t0 = time.perf_counter()
        hits, extra, context, used = self.build_context(question)
        t_retrieve = time.perf_counter() - t0
        messages = self.prompt.render(question=question, context=context)

        if stream:
            draft_toks = []
            for tok in self.llm.stream_chat(messages):
                draft_toks.append(tok)
                yield tok
            draft = "".join(draft_toks)
        else:
            draft = self.llm.chat(messages)
        draft_rep = check_answer(draft, used, self.cfg.broad_lines)
        final, final_rep, repaired = draft, draft_rep, False

        if self.cfg.strict_citations and not draft_rep.ok and self.cfg.max_repairs > 0:
            yield Repairing(draft_rep.problems())
            repair = load_prompt(self.cfg.repair_prompt_id)
            msgs2 = messages + [{"role": "assistant", "content": draft},
                                {"role": "user", "content": repair.user.format(
                                    problems="\n".join(f"- {p}" for p in draft_rep.problems()))}]
            if stream:
                toks = []
                for tok in self.llm.stream_chat(msgs2):
                    toks.append(tok)
                    yield tok
                revised = "".join(toks)
            else:
                revised = self.llm.chat(msgs2)
            rev_rep = check_answer(revised, used, self.cfg.broad_lines)
            if rev_rep.penalty < draft_rep.penalty:
                final, final_rep, repaired = revised, rev_rep, True

        ans = Answer(question, used, final, final_rep.as_dict(), draft, draft_rep.as_dict(), repaired, extra)
        self._log(ans, t_retrieve, time.perf_counter() - t0)
        yield ans

    def ask_stream(self, question: str) -> Iterator[str | Repairing | Answer]:
        """Yields draft tokens, then (if strict citations fail) a Repairing marker and the repaired tokens,
        then the final Answer."""
        yield from self._generate(question, stream=True)

    def answer(self, question: str) -> Answer:
        """Non-streaming, LLM-cached version for evaluation."""
        out = None
        for part in self._generate(question, stream=False):
            if isinstance(part, Answer):
                out = part
        return out

    def _log(self, ans: Answer, t_retrieve: float, t_total: float) -> None:
        # Append-only run log. Becomes the trace store in Project 05 / eval input in Project 03.
        rec = {
            "run_id": uuid.uuid4().hex, "ts": time.time(), "question": ans.question,
            "prompt_id": self.prompt.id, "prompt_hash": self.prompt.version_hash,
            "llm": f"{self.cfg.llm_provider}:{self.cfg.llm_model}", "embed_model": self.index.meta["embed_model"],
            "retrieval_mode": self.cfg.retrieval_mode, "chunker": self.index.meta.get("chunker", "fixed"),
            "rerank": self.cfg.rerank and (self.reranker.name if self.reranker else None),
            "rewrites": self.last_rewrites,
            "retrieved": [{"citation": h.chunk.citation, "symbol": h.chunk.symbol, "score": round(h.score, 4),
                           "ranks": h.detail} for h in ans.hits],
            "expanded": [{"citation": h.chunk.citation, "symbol": h.chunk.symbol, "why": h.detail.get("graph")}
                         for h in ans.expanded],
            "answer": ans.text, "citation_check": ans.citation_check,
            "draft": ans.draft if ans.repaired else None, "draft_check": ans.draft_check, "repaired": ans.repaired,
            "latency_s": {"retrieve": round(t_retrieve, 3), "total": round(t_total, 3)},
        }
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.data_dir / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
