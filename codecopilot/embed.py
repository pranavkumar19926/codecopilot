"""Embedding backends behind one interface, so the model is swappable for A/B tests later."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import Settings


class Embedder(Protocol):
    name: str
    def embed_docs(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    # BGE models are trained with this instruction prefix on queries (not documents)
    _QUERY_PREFIX = {"BAAI/bge": "Represent this sentence for searching relevant passages: "}

    def __init__(self, model: str, batch_size: int):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import
        self.name = model
        self._m = SentenceTransformer(model)
        self._bs = batch_size
        self._qprefix = next((v for k, v in self._QUERY_PREFIX.items() if model.startswith(k)), "")

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._m.encode(texts, batch_size=self._bs, normalize_embeddings=True,
                              show_progress_bar=len(texts) > 200).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._m.encode([self._qprefix + text], normalize_embeddings=True)[0].astype(np.float32)


class HashingEmbedder:
    """Dependency-free bag-of-subtokens embedder. For offline tests/CI only — not semantic."""

    def __init__(self, dim: int = 1024):
        self.name = f"hashing-{dim}"
        self._dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        toks = []
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
            toks.append(w.lower())
            # split snake_case and camelCase so 'rebuild_auth' matches 'auth'
            toks += [p.lower() for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", w) if len(p) > 1]
        return toks

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for t in self._tokens(text):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self._dim] += 1.0 if (h >> 64) & 1 else -1.0
        v = np.sign(v) * np.log1p(np.abs(v))
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self._dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def get_embedder(cfg: Settings) -> Embedder:
    if cfg.embed_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(cfg.embed_model, cfg.embed_batch_size)
