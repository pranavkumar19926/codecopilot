"""BM25 keyword index with a code-aware tokenizer, plus reciprocal rank fusion (RRF).

Why BM25 at all: dense embeddings blur exact identifiers. A query for `should_strip_auth` embeds
close to every auth/redirect chunk; BM25 rewards the rare exact token and puts the definition first.

Implemented directly (Okapi BM25, ~40 lines) rather than via SQLite FTS5: no dependency on how
Python's bundled SQLite was compiled, and the scoring is visible for tuning."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .ingest import Chunk

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PARTS = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_STOP = frozenset("""
a an the of to in on at for from by with into onto as is are was were be been being it its this that these those
and or not no do does did how what where when why which who whom whose there here can could should would will
i we you they he she my our your their me us them if then than so such just also very about
""".split())


def _norm(tok: str) -> str:
    """Tiny, consistent stemmer: plural/verb suffixes, then a trailing 'e', so decode/decoded/decoding
    all become 'decod' and raise/raised/raises become 'rais'. Applied identically to documents and queries."""
    if len(tok) > 4 and tok.endswith("ies"):
        tok = tok[:-3] + "y"
    elif len(tok) > 5 and tok.endswith("ing"):
        tok = tok[:-3]
    elif len(tok) > 4 and tok.endswith("ed"):
        tok = tok[:-2]
    elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    if len(tok) > 4 and tok.endswith("e"):
        tok = tok[:-1]
    return tok


_CAMEL = re.compile(r"[a-z0-9][A-Z]")


def identifier_terms(query: str) -> set[str]:
    """Words in a query that look like code identifiers: snake_case, camelCase/PascalCase, `backticked`,
    or a query that is a single bare word. Plain English words are ignored, so 'session' in a sentence
    doesn't pin the `session()` function."""
    words = _IDENT.findall(query)
    ticked = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", query))
    terms = {w for w in words if ("_" in w.strip("_") or _CAMEL.search(w) or w in ticked)}
    terms |= {t.split(".")[-1] for t in ticked}
    if len(words) == 1:
        terms.add(words[0])
    return {t.lower() for t in terms}


def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    """`rebuild_auth` -> [rebuild_auth, rebuild, auth]; `HTTPAdapter` -> [httpadapter, http, adapter].
    The whole identifier is kept as its own token, so an exact symbol query gets a very rare, high-IDF match."""
    out: list[str] = []
    for w in _IDENT.findall(text):
        low = w.lower()
        parts = [p.lower() for p in _PARTS.findall(w)]
        cand = [low] + [p for p in parts if p != low] if len(parts) > 1 or "_" in w else [low]
        for t in cand:
            if len(t) < 2 or (drop_stopwords and t in _STOP):
                continue
            out.append(_norm(t))
    return out


class BM25Index:
    def __init__(self, postings: dict[str, list[list[int]]], doc_len: list[int], k1: float, b: float):
        self.postings, self.doc_len, self.k1, self.b = postings, doc_len, k1, b
        self.n = len(doc_len)
        self.avgdl = (sum(doc_len) / self.n) if self.n else 0.0

    @classmethod
    def build(cls, chunks: list[Chunk], symbol_boost: int = 3, k1: float = 1.2, b: float = 0.75) -> "BM25Index":
        postings: dict[str, list[list[int]]] = defaultdict(list)
        doc_len: list[int] = []
        for i, c in enumerate(chunks):
            toks = tokenize(c.embed_text)
            # BM25F-lite: repeat symbol-name tokens so the defining chunk outranks chunks that merely call it
            toks += tokenize(c.symbol) * symbol_boost
            tf = Counter(toks)
            doc_len.append(len(toks))
            for t, n in tf.items():
                postings[t].append([i, n])
        return cls(dict(postings), doc_len, k1, b)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query, drop_stopwords=True)):
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for doc, tf in plist:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc] / self.avgdl)
                scores[doc] += idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda x: -x[1])[:k]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"k1": self.k1, "b": self.b, "doc_len": self.doc_len, "postings": self.postings}))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        d = json.loads(path.read_text())
        return cls(d["postings"], d["doc_len"], d["k1"], d["b"])


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank). Uses ranks, not raw scores,
    so cosine similarities (~0.7) and BM25 scores (~5-30) can be combined without calibration."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            fused[doc] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])
