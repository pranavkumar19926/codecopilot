"""Retrieval eval. Gold items name a (path, line) that MUST appear in a retrieved chunk.
Line-based targets stay valid when chunking changes (fixed → AST), so Phase 1 vs Phase 2 is a fair comparison.

Caveat worth stating in any write-up: bigger chunks make line-containment easier. Report mean chunk size
next to recall (the CLI does) so a recall gain can't quietly come from just making chunks larger."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .pipeline import Copilot


def load_gold(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip() and not l.lstrip().startswith("//")]


def validate_gold(gold: list[dict], repo: Path) -> list[str]:
    """Catch a repo checked out at the wrong commit: gold lines must exist and be non-blank."""
    problems = []
    for g in gold:
        f = repo / g["path"]
        if not f.exists():
            problems.append(f"missing file {g['path']}")
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        if g["line"] > len(lines) or not lines[g["line"] - 1].strip():
            problems.append(f"{g['path']}:{g['line']} is out of range or blank")
    return problems


def evaluate_retrieval(cp: Copilot, gold: list[dict], k: int, mode: str | None = None,
                       rerank: bool | None = None, rewrite: bool | None = None, progress=None) -> dict:
    per_q, latencies = [], []
    for i, g in enumerate(gold, 1):
        t0 = time.perf_counter()
        hits = cp.retrieve(g["question"], k, mode, rerank=rerank, rewrite=rewrite)
        latencies.append(time.perf_counter() - t0)
        rank = next((i for i, h in enumerate(hits, 1)
                     if h.chunk.path == g["path"] and h.chunk.start_line <= g["line"] <= h.chunk.end_line), None)
        per_q.append({"question": g["question"], "type": g.get("type", "semantic"), "rank": rank,
                      "rewrites": list(cp.last_rewrites)})
        if progress:
            progress(i, len(gold))

    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "n": n,
            "recall": sum(r["rank"] is not None for r in rows) / n if n else 0.0,
            "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n if n else 0.0,
            "hit1": sum(r["rank"] == 1 for r in rows) / n if n else 0.0,
        }

    out = summarize(per_q)
    out["by_type"] = {t: summarize([r for r in per_q if r["type"] == t]) for t in sorted({r["type"] for r in per_q})}
    out["per_question"] = per_q
    lat = sorted(latencies)
    out["latency_ms"] = {"p50": round(1000 * lat[len(lat) // 2], 1),
                         "p95": round(1000 * lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1)} if lat else {}
    return out
