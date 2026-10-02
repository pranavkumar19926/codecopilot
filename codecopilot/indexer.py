"""Build everything a repo needs (vector index, BM25, call graph) into one directory.
Shared by `codecopilot index` and the website's "Add repository" flow."""
from __future__ import annotations

import statistics
import time
from pathlib import Path

from .bm25 import BM25Index
from .config import Settings
from .embed import Embedder
from .graph import build_graph
from .index import VectorIndex
from .ingest import ingest_repo


def build_all(repo: Path, data_dir: Path, cfg: Settings, embedder: Embedder, progress=None) -> dict:
    """Returns the saved index meta (plus a 'graph' stats entry and 'seconds')."""
    say = progress or (lambda msg: None)
    t0 = time.perf_counter()
    say("reading and chunking files")
    chunks = ingest_repo(repo, cfg)
    if not chunks:
        raise ValueError("no indexable files found (only .py and .ipynb files are indexed)")
    prev = None
    try:
        prev = VectorIndex.load(data_dir)
    except FileNotFoundError:
        pass
    say(f"embedding {len(chunks)} chunks")
    idx = VectorIndex.build(chunks, embedder, repo, previous=prev)
    sizes = [c.end_line - c.start_line + 1 for c in chunks] or [0]
    idx.meta.update(
        exclude_globs=list(cfg.exclude_globs), chunker=cfg.chunker,
        chunk_lines=cfg.chunk_lines, chunk_overlap=cfg.chunk_overlap,
        ast_max_lines=cfg.ast_max_lines, ast_min_lines=cfg.ast_min_lines,
        mean_chunk_lines=round(statistics.mean(sizes), 1), median_chunk_lines=statistics.median(sizes),
        n_files=len({c.path for c in chunks}),
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    say("building keyword index")
    BM25Index.build(chunks, symbol_boost=cfg.bm25_symbol_boost).save(data_dir / "bm25.json")
    gpath = data_dir / "graph.json"
    if cfg.chunker == "ast":
        say("building call graph")
        g = build_graph(repo, chunks)
        g.save(gpath)
        idx.meta["graph"] = g.stats
    elif gpath.exists():
        gpath.unlink()   # a graph from an older AST index would point at the wrong chunks
    idx.save(data_dir)   # meta last: its presence marks a complete index
    return {**idx.meta, "seconds": round(time.perf_counter() - t0, 1)}
