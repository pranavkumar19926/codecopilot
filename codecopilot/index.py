"""Flat in-memory vector index persisted to disk. Brute-force cosine is exact and fast enough
for a mid-size repo (~10k chunks); swap for pgvector/Qdrant once hybrid search lands in Phase 2."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed import Embedder
from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: dict):
        assert len(chunks) == len(vectors)
        self.chunks, self.vectors, self.meta = chunks, vectors, meta

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, repo: Path,
              previous: "VectorIndex | None" = None) -> "VectorIndex":
        # Reuse vectors for unchanged chunks (content-addressed by chunk id) — cheap re-indexing
        cache: dict[str, np.ndarray] = {}
        if previous and previous.meta.get("embed_model") == embedder.name:
            cache = {c.id: v for c, v in zip(previous.chunks, previous.vectors)}
        todo = [i for i, c in enumerate(chunks) if c.id not in cache]
        new_vecs = embedder.embed_docs([chunks[i].text for i in todo])
        for i, v in zip(todo, new_vecs):
            cache[chunks[i].id] = v
        vectors = np.stack([cache[c.id] for c in chunks]) if chunks else np.zeros((0, 1), np.float32)
        meta = {
            "repo": str(repo.resolve()),
            "embed_model": embedder.name,
            "n_chunks": len(chunks),
            "n_embedded": len(todo),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return cls(chunks, vectors, meta)

    def search(self, qvec: np.ndarray, k: int) -> list[Hit]:
        if not self.chunks:
            return []
        scores = self.vectors @ qvec  # vectors are L2-normalised → dot product == cosine
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, d: Path) -> "VectorIndex":
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No index at {d}. Run `codecopilot index <repo>` first.")
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        return cls(chunks, np.load(d / "vectors.npy"), json.loads((d / "meta.json").read_text()))
