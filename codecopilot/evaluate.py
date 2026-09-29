"""Retrieval eval. Gold items name a (path, line) that MUST appear in a retrieved chunk.
Line-based targets stay valid when chunking changes (fixed → AST), so Phase 1 vs Phase 2 is a fair comparison."""
from __future__ import annotations

import json
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


def evaluate_retrieval(cp: Copilot, gold: list[dict], k: int) -> dict:
    per_q = []
    for g in gold:
        hits = cp.retrieve(g["question"], k)
        rank = next((i for i, h in enumerate(hits, 1)
                     if h.chunk.path == g["path"] and h.chunk.start_line <= g["line"] <= h.chunk.end_line), None)
        per_q.append({"question": g["question"], "type": g.get("type", "semantic"), "rank": rank})
    n = len(per_q)
    by_type: dict[str, dict] = {}
    for t in sorted({r["type"] for r in per_q}):
        rs = [r for r in per_q if r["type"] == t]
        by_type[t] = {"n": len(rs), "recall": sum(r["rank"] is not None for r in rs) / len(rs)}
    return {
        "n": n,
        "recall": sum(r["rank"] is not None for r in per_q) / n if n else 0.0,
        "mrr": sum(1 / r["rank"] for r in per_q if r["rank"]) / n if n else 0.0,
        "by_type": by_type,
        "per_question": per_q,
    }
