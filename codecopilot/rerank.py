"""Second-stage reranking with a cross-encoder.

Bi-encoder (Phase 1/2): question and chunk are embedded *separately*, then compared by cosine.
Fast — chunk vectors are precomputed — but the two texts never attend to each other.

Cross-encoder (here): question and chunk go through the transformer *together* as one sequence
([CLS] question [SEP] chunk), and a head outputs a relevance score. Every question token can attend
to every code token, so it is much more precise, but it costs one forward pass per (question, chunk)
pair and nothing can be precomputed. Hence the standard two-stage design: cheap retrieval narrows
thousands of chunks to ~30, the cross-encoder re-orders those 30."""
from __future__ import annotations

from typing import Protocol

from .bm25 import tokenize
from .config import Settings


class Reranker(Protocol):
    name: str
    def score(self, query: str, docs: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model: str, max_length: int = 512, batch_size: int = 16):
        from sentence_transformers import CrossEncoder  # lazy: heavy import
        self.name = model
        self._m = CrossEncoder(model, max_length=max_length)
        self._bs = batch_size

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        return [float(s) for s in self._m.predict([(query, d) for d in docs], batch_size=self._bs,
                                                  show_progress_bar=False)]


class OverlapReranker:
    """Offline stand-in for tests/CI: fraction of query tokens present in the doc. Not a real reranker."""
    name = "overlap-stub"

    def score(self, query: str, docs: list[str]) -> list[float]:
        q = set(tokenize(query, drop_stopwords=True))
        return [len(q & set(tokenize(d))) / (len(q) or 1) for d in docs]


def get_reranker(cfg: Settings) -> Reranker:
    if cfg.rerank_backend == "stub":
        return OverlapReranker()
    return CrossEncoderReranker(cfg.rerank_model, cfg.rerank_max_length)
