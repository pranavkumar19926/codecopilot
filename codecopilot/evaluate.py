"""Retrieval eval. Gold items name a (path, line) that MUST appear in a retrieved chunk.
Line-based targets stay valid when chunking changes (fixed → AST), so Phase 1 vs Phase 2 is a fair comparison.

Caveat worth stating in any write-up: bigger chunks make line-containment easier. Report mean chunk size
next to recall (the CLI does) so a recall gain can't quietly come from just making chunks larger."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .citations import parse_citations
from .pipeline import Copilot


def load_gold(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip() and not l.lstrip().startswith("//")]


def validate_gold(gold: list[dict], repo: Path) -> list[str]:
    """Catch a repo checked out at the wrong commit: gold lines must exist and be non-blank."""
    problems = []
    for g in gold:
        for t in g.get("targets") or [g]:
            f = repo / t["path"]
            if not f.exists():
                problems.append(f"missing file {t['path']}")
                continue
            lines = f.read_text(encoding="utf-8").splitlines()
            if t["line"] > len(lines) or not lines[t["line"] - 1].strip():
                problems.append(f"{t['path']}:{t['line']} is out of range or blank")
    return problems


def _covers(hits, t) -> bool:
    return any(h.chunk.path == t["path"] and h.chunk.start_line <= t["line"] <= h.chunk.end_line for h in hits)


def evaluate_trace(cp: Copilot, gold: list[dict], k: int, progress=None) -> dict:
    """Trace questions have several required locations (all callers of X, or what X calls).
    For each question we compare, at the SAME number of chunks in context:
      expand    top-k retrieval + graph neighbours (Phase 4)
      baseline  top-(k + n_added) retrieval, no graph
    so any gain comes from *which* chunks the graph picks, not from simply showing more chunks."""
    rows = []
    for i, g in enumerate(gold, 1):
        hits = cp.retrieve(g["question"], k)
        extra = cp.expand(g["question"], hits)
        base = cp.retrieve(g["question"], k + len(extra))
        tg = g["targets"]
        rows.append({"question": g["question"], "type": g.get("type", "trace"), "n_targets": len(tg),
                     "expand": sum(_covers(hits + extra, t) for t in tg) / len(tg),
                     "baseline": sum(_covers(base, t) for t in tg) / len(tg),
                     "topk": sum(_covers(hits, t) for t in tg) / len(tg),
                     "n_added": len(extra), "added": [h.detail.get("graph") for h in extra]})
        if progress:
            progress(i, len(gold))

    def agg(rs, key):
        n = len(rs)
        return {"target_recall": sum(r[key] for r in rs) / n if n else 0.0,
                "all_found": sum(r[key] == 1.0 for r in rs) / n if n else 0.0}

    out = {"n": len(rows), "per_question": rows,
           "mean_added": sum(r["n_added"] for r in rows) / len(rows) if rows else 0.0}
    for key in ("topk", "baseline", "expand"):
        out[key] = agg(rows, key)
        out[key]["by_type"] = {t: agg([r for r in rows if r["type"] == t], key)
                               for t in sorted({r["type"] for r in rows})}
    return out


def evaluate_answers(cp: Copilot, gold: list[dict], progress=None) -> dict:
    """Answer-level eval (needs the LLM). Script-checkable metrics only, no LLM judge:
      gold_cited      a valid citation in the final answer contains the gold line
      valid_rate      share of citations that point inside the shown excerpts
      claims_cited    share of sentences naming code that carry a citation
      supported       share of named identifiers that occur in the lines cited for them
      mean_width      average cited range width in lines (smaller = more pinpoint)"""
    rows = []
    for i, g in enumerate(gold, 1):
        ans = cp.answer(g["question"])
        cc = ans.citation_check
        valid = [(p, a, b) for p, a, b in parse_citations(ans.text) if f"{p}:{a}-{b}" not in cc["invalid"]]
        targets = g.get("targets") or [g]
        gold_cited = any(p == t["path"] and a <= t["line"] <= b for p, a, b in valid for t in targets)
        rows.append({"question": g["question"], "gold_cited": gold_cited, "n_citations": cc["n_citations"],
                     "invalid": len(cc["invalid"]), "claims": cc["n_claims"], "cited_claims": cc["cited_claims"],
                     "unsupported": len(cc["unsupported"]), "mean_width": cc["mean_width"], "repaired": ans.repaired,
                     "answer": ans.text})
        if progress:
            progress(i, len(gold))
    n = len(rows)
    tot_c = sum(r["n_citations"] for r in rows)
    tot_claims = sum(r["claims"] for r in rows)
    widths = [r["mean_width"] for r in rows if r["mean_width"] is not None]
    return {
        "n": n, "per_question": rows,
        "gold_cited": sum(r["gold_cited"] for r in rows) / n if n else 0.0,
        "valid_rate": (tot_c - sum(r["invalid"] for r in rows)) / tot_c if tot_c else 0.0,
        "claims_cited": sum(r["cited_claims"] for r in rows) / tot_claims if tot_claims else 0.0,
        "unsupported_per_answer": sum(r["unsupported"] for r in rows) / n if n else 0.0,
        "mean_width": round(sum(widths) / len(widths), 1) if widths else None,
        "repaired": sum(r["repaired"] for r in rows),
    }


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
