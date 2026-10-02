"""CLI: index / search / ask / eval / report."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .bm25 import BM25Index
from .graph import CodeGraph, build_graph
from .config import settings
from .embed import get_embedder
from .index import VectorIndex
from .ingest import ingest_repo
from .llm import LLMClient, LLMError
from .pipeline import Answer, Copilot, Repairing, Retrieved

app = typer.Typer(add_completion=False, help="Codebase Intelligence Copilot — hybrid search, query rewriting, call graph, strict citations")
con = Console()
RESULTS = Path("eval/results.jsonl")
MODES = ("dense", "bm25", "hybrid")
_ModeOpt = typer.Option(None, "--mode", "-m", help="dense | bm25 | hybrid (default from config: hybrid)")


def get_embedder_name() -> str:
    return "hashing-1024" if settings.embed_backend == "hashing" else settings.embed_model


def _copilot(modes: tuple[str, ...], with_llm: bool = False) -> Copilot:
    idx = VectorIndex.load(settings.data_dir)
    bm25_path = settings.data_dir / "bm25.json"
    bm25 = BM25Index.load(bm25_path) if bm25_path.exists() else None
    needs_dense = any(m in ("dense", "hybrid") for m in modes)
    if needs_dense and idx.meta["embed_model"] != get_embedder_name():
        con.print(f"[yellow]Index built with {idx.meta['embed_model']}, config says {get_embedder_name()}. Re-index.[/]")
        raise typer.Exit(1)
    if bm25 is None and any(m in ("bm25", "hybrid") for m in modes):
        con.print("[yellow]This index has no BM25 part (built by Phase 1). Re-run `codecopilot index <repo>`.[/]")
        raise typer.Exit(1)
    embedder = get_embedder(settings) if needs_dense else None  # bm25-only skips loading the model
    llm = LLMClient(settings) if (with_llm or settings.query_rewrite) else None
    reranker = None
    if settings.rerank:
        from .rerank import get_reranker
        reranker = get_reranker(settings)
    graph_path = settings.data_dir / "graph.json"
    graph = CodeGraph.load(graph_path) if graph_path.exists() else None
    return Copilot(settings, idx, embedder, llm, bm25, reranker, graph)


def _apply_flags(rerank: bool | None, rewrite: bool | None) -> None:
    if rerank is not None:
        settings.rerank = rerank
    if rewrite is not None:
        settings.query_rewrite = rewrite


_RerankOpt = typer.Option(None, "--rerank/--no-rerank", help="cross-encoder second stage (Phase 3)")
_RewriteOpt = typer.Option(None, "--rewrite/--no-rewrite", help="LLM query rewriting, needs Ollama (Phase 3)")


def _stages() -> str:
    return "+".join(["rerank"] * settings.rerank + ["rewrite"] * settings.query_rewrite) or "none"


@app.command()
def index(repo: Path = typer.Argument(..., exists=True, file_okay=False),
          exclude: list[str] = typer.Option(None, "--exclude", "-x",
                                            help="Glob on repo-relative paths, repeatable. e.g. -x 'tests/*'"),
          chunker: str = typer.Option(None, "--chunker", "-c", help="ast (default) | fixed (Phase 1 baseline)")):
    """Chunk a repository, embed the chunks, and build the BM25 index."""
    update = {}
    if exclude:
        update["exclude_globs"] = tuple(exclude)
    if chunker:
        if chunker not in ("ast", "fixed"):
            raise typer.BadParameter("chunker must be 'ast' or 'fixed'")
        update["chunker"] = chunker
    cfg = settings.model_copy(update=update) if update else settings

    from .indexer import build_all
    try:
        m = build_all(repo, settings.data_dir, cfg, get_embedder(settings))
    except ValueError as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    g = m.get("graph")
    graph_note = f" + call graph ({g['nodes']} definitions, {g['edges']} edges)" if g else ""
    con.print(f"[green]Indexed[/] {m['n_files']} files → {m['n_chunks']} {m['chunker']} chunks "
              f"(mean {m['mean_chunk_lines']} lines; {m['n_embedded']} newly embedded) "
              f"with {m['embed_model']} + BM25{graph_note} in {m['seconds']:.1f}s"
              + (f", excluding {m['exclude_globs']}" if m["exclude_globs"] else ""))


@app.command()
def search(query: str, k: int = typer.Option(settings.top_k, "-k"), mode: str = _ModeOpt,
           rerank: bool = _RerankOpt, rewrite: bool = _RewriteOpt):
    """Show top-k retrieved chunks (no answer generation) — use this to debug retrieval."""
    _apply_flags(rerank, rewrite)
    mode = mode or settings.retrieval_mode
    cp = _copilot((mode,))
    hits = cp.retrieve(query, k, mode)
    if cp.last_rewrites:
        con.print("[dim]rewrites:[/]")
        for q in cp.last_rewrites:
            con.print(f"  [dim]- {q}[/]", highlight=False)
    t = Table("rank", "location", "symbol", "dense", "bm25", "rerank", title=f"mode={mode}, stages={_stages()}")
    for i, h in enumerate(hits, 1):
        d = h.detail
        t.add_row(str(i), h.chunk.citation, ("* " if d.get("symbol") else "") + (h.chunk.symbol[:45] or "-"),
                  str(d.get("dense") or "-") if "dense" in d else "", str(d.get("bm25") or "-") if "bm25" in d else "",
                  str(d.get("rerank", "")))
    con.print(t)


_StrictOpt = typer.Option(None, "--strict/--no-strict",
                          help="Phase 4 strict citations (line-numbered context, claim checks, one repair round)")
_ExpandOpt = typer.Option(None, "--expand/--no-expand", help="Phase 4 call-graph expansion of the context")


def _apply_phase4(strict: bool | None, expand: bool | None) -> None:
    if strict is not None:
        settings.strict_citations = strict
        settings.context_line_numbers = strict
        settings.prompt_id = "answer_v4" if strict else "answer_v2"
    if expand is not None:
        settings.graph_expand = expand


@app.command()
def ask(question: str, show_sources: bool = typer.Option(True, "--sources/--no-sources"), mode: str = _ModeOpt,
        rerank: bool = _RerankOpt, rewrite: bool = _RewriteOpt, strict: bool = _StrictOpt, expand: bool = _ExpandOpt):
    """Answer a question with path:line citations."""
    _apply_flags(rerank, rewrite)
    _apply_phase4(strict, expand)
    if mode:
        settings.retrieval_mode = mode
    cp = _copilot((settings.retrieval_mode,), with_llm=True)
    final: Answer | None = None
    try:
        for part in cp.ask_stream(question):
            if isinstance(part, Answer):
                final = part
            elif isinstance(part, Retrieved):
                continue
            elif isinstance(part, Repairing):
                con.print()
                con.rule("[yellow]citation check failed — revising[/]")
                for p in part.problems[:6]:
                    con.print(f"  [dim]- {p}[/]", highlight=False)
                con.rule("revised answer")
            else:
                con.print(part, end="", markup=False, highlight=False)
    except LLMError as e:
        con.print(f"\n[red]LLM error:[/] {e}\nIs Ollama running (`ollama serve`) and the model pulled?")
        raise typer.Exit(1)
    con.print()
    if final and final.draft and final.draft != final.text and not final.repaired:
        con.print("[dim](revision scored worse on the citation check; kept the first answer)[/]")
    if final and show_sources:
        if cp.last_rewrites:
            con.rule("rewrites")
            for q in cp.last_rewrites:
                con.print(f"  {q}", highlight=False)
        con.rule(f"retrieved ({settings.retrieval_mode}, stages={_stages()})")
        shown = {id(h.chunk) for h in final.hits}
        for h in final.hits:
            if "graph" not in h.detail:
                con.print(f"  {h.chunk.citation:<42} {h.chunk.symbol}", highlight=False)
        related = [h for h in final.hits if "graph" in h.detail]
        if related:
            con.rule("added from the call graph")
            for h in related:
                con.print(f"  {h.chunk.citation:<42} {h.detail['graph']}", highlight=False)
        cc = final.citation_check
        good = cc["n_citations"] and not cc["invalid"] and not cc.get("unsupported") and not cc.get("uncited")
        style = "green" if good else "yellow"
        line = f"citations: {cc['n_citations']}, invalid: {cc['invalid'] or 'none'}"
        if "n_claims" in cc:
            line += (f", claims cited: {cc['cited_claims']}/{cc['n_claims']}, unsupported: {len(cc['unsupported'])}"
                     f", mean width: {cc['mean_width']} lines")
        con.print(f"[{style}]{line}[/]" + ("  [dim](after repair)[/]" if final.repaired else ""))


@app.command("eval")
def eval_(gold: Path = typer.Argument(..., exists=True), k: int = typer.Option(10, "-k"),
          mode: str = typer.Option(None, "--mode", "-m", help="dense | bm25 | hybrid | all"),
          rerank: bool = _RerankOpt, rewrite: bool = _RewriteOpt,
          verbose: bool = typer.Option(False, "-v"),
          save: str = typer.Option("", "--save", help="Label; appends each result to eval/results.jsonl")):
    """Retrieval recall@k, MRR, hit@1 and latency against a gold set. `--mode all` runs dense, bm25, hybrid."""
    from .evaluate import evaluate_retrieval, load_gold, validate_gold
    _apply_flags(rerank, rewrite)
    items = load_gold(gold)
    if items and "targets" in items[0]:
        return _eval_trace(gold, items, k if k != 10 else settings.top_k, verbose, save)
    modes = MODES if mode == "all" else (mode or settings.retrieval_mode,)
    cp = _copilot(modes)
    problems = validate_gold(items, Path(cp.index.meta["repo"]))
    if problems:
        con.print(f"[red]{len(problems)} gold item(s) don't match the indexed repo — wrong commit?[/] "
                  f"e.g. {problems[0]}\nFor requests: git -C requests checkout 611c616, then re-index.")
        raise typer.Exit(1)

    m = cp.index.meta
    chunker = m.get("chunker", "fixed")
    con.print(f"[dim]{gold.name}: {chunker} chunks, n={m['n_chunks']}, mean {m.get('mean_chunk_lines', '?')} lines, "
              f"embed={m['embed_model']}, stages={_stages()}"
              + (f", reranker={cp.reranker.name}" if cp.reranker else "")
              + (f", rewriter={settings.llm_model}" if settings.query_rewrite else "") + "[/]")
    if settings.query_rewrite:
        con.print("[dim]query rewriting calls the LLM once per question; first run is slow, reruns use the cache[/]")
    for md in modes:
        with con.status(f"evaluating {md}...") as status:
            res = evaluate_retrieval(cp, items, k, md,
                                     progress=lambda i, n: status.update(f"evaluating {md}: {i}/{n}"))
        if verbose:
            for r in res["per_question"]:
                mark = "[green]✓[/]" if r["rank"] else "[red]✗[/]"
                con.print(f"{mark} rank={r['rank'] or '-':<3} [dim]{r['type']:<10}[/] {r['question']}")
                for q in r["rewrites"]:
                    con.print(f"      [dim]↳ {q}[/]", highlight=False)
        by_type = "   ".join(f"{t}: {v['recall']:.2f} (hit@1 {v['hit1']:.2f}, n={v['n']})"
                             for t, v in res["by_type"].items())
        lat = res["latency_ms"]
        con.print(f"[bold]{chunker}+{md}+{_stages()}[/]  recall@{k} = [bold]{res['recall']:.3f}[/]   "
                  f"MRR = [bold]{res['mrr']:.3f}[/]   hit@1 = {res['hit1']:.3f}   "
                  f"latency p50 {lat['p50']:.0f} ms, p95 {lat['p95']:.0f} ms   (n={res['n']})")
        con.print(f"   by type → {by_type}")
        if save:
            label = f"{save}-{md}" if len(modes) > 1 else save
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "gold": gold.name, "k": k,
                   "chunker": chunker, "mode": md, "symbol_pin": settings.symbol_pin and md != "dense",
                   "rerank": cp.reranker.name if (settings.rerank and cp.reranker) else None,
                   "rewrite": f"{settings.rewrite_prompt_id}@{settings.llm_model}" if settings.query_rewrite else None,
                   "recall": round(res["recall"], 4), "mrr": round(res["mrr"], 4), "hit1": round(res["hit1"], 4),
                   "n": res["n"], "latency_ms": lat,
                   "by_type": {t: {"recall": round(v["recall"], 4), "hit1": round(v["hit1"], 4)}
                               for t, v in res["by_type"].items()},
                   "index": {key: m.get(key) for key in ("embed_model", "chunker", "n_chunks", "mean_chunk_lines",
                                                         "chunk_lines", "chunk_overlap", "ast_max_lines",
                                                         "exclude_globs")},
                   "misses": [r["question"] for r in res["per_question"] if not r["rank"]]}
            RESULTS.parent.mkdir(exist_ok=True)
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            con.print(f"   [green]saved[/] as {label!r}")


def _check_gold(cp: Copilot, items: list[dict]) -> None:
    from .evaluate import validate_gold
    problems = validate_gold(items, Path(cp.index.meta["repo"]))
    if problems:
        con.print(f"[red]{len(problems)} gold item(s) don't match the indexed repo — wrong commit?[/] "
                  f"e.g. {problems[0]}\nFor requests: git -C requests checkout 611c616, then re-index.")
        raise typer.Exit(1)


def _save(rec: dict) -> None:
    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    con.print(f"   [green]saved[/] as {rec['label']!r}")


def _eval_trace(gold: Path, items: list[dict], k: int, verbose: bool, save: str) -> None:
    from .evaluate import evaluate_trace
    cp = _copilot((settings.retrieval_mode,))
    if cp.graph is None:
        con.print("[yellow]No call graph in this index. Re-run `codecopilot index <repo>` (AST chunker).[/]")
        raise typer.Exit(1)
    _check_gold(cp, items)
    con.print(f"[dim]{gold.name}: trace questions, k={k}, graph: {cp.graph.stats['edges']} edges, "
              f"stages={_stages()}[/]")
    with con.status("evaluating trace questions...") as status:
        res = evaluate_trace(cp, items, k, progress=lambda i, n: status.update(f"trace questions: {i}/{n}"))
    if verbose:
        for r in res["per_question"]:
            con.print(f"  top-k {r['topk']:.2f}  baseline {r['baseline']:.2f}  expand {r['expand']:.2f}  "
                      f"[dim]{r['type']:<8}[/] {r['question']}", highlight=False)
    for key, name in (("topk", f"top-{k} only"), ("baseline", f"top-{k}+N, no graph"), ("expand", f"top-{k} + graph")):
        v = res[key]
        con.print(f"[bold]{name:<22}[/] target recall = [bold]{v['target_recall']:.3f}[/]   "
                  f"all targets found = {v['all_found']:.3f}   "
                  + "  ".join(f"{t}: {b['target_recall']:.2f}" for t, b in v["by_type"].items()))
    con.print(f"   graph added {res['mean_added']:.1f} chunks per question on average (baseline got the same number)")
    if save:
        for key in ("baseline", "expand"):
            v = res[key]
            _save({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "trace", "label": f"{save}-{key}",
                   "gold": gold.name, "k": k, "variant": key, "n": res["n"], "mean_added": round(res["mean_added"], 2),
                   "target_recall": round(v["target_recall"], 4), "all_found": round(v["all_found"], 4),
                   "by_type": {t: round(b["target_recall"], 4) for t, b in v["by_type"].items()},
                   "rewrite": settings.query_rewrite})


@app.command("eval-answers")
def eval_answers(gold: Path = typer.Argument(..., exists=True),
                 n: int = typer.Option(8, "-n", help="How many questions (each needs 1-2 LLM calls)"),
                 strict: bool = _StrictOpt, expand: bool = _ExpandOpt,
                 verbose: bool = typer.Option(False, "-v"), save: str = typer.Option("", "--save")):
    """Answer-level eval with script-checkable citation metrics (needs Ollama; responses are cached)."""
    from .evaluate import evaluate_answers, load_gold
    _apply_phase4(strict, expand)
    items = load_gold(gold)[:n]
    cp = _copilot((settings.retrieval_mode,), with_llm=True)
    _check_gold(cp, items)
    mode = "strict" if settings.strict_citations else "plain"
    con.print(f"[dim]{gold.name}: {len(items)} questions, citations={mode}, prompt={settings.prompt_id}, "
              f"graph={'on' if settings.graph_expand and cp.graph else 'off'}, llm={settings.llm_model}[/]")
    con.print("[dim]this makes 1-2 LLM calls per question; first run is slow, reruns use the cache[/]")
    try:
        with con.status("answering...") as status:
            res = evaluate_answers(cp, items, progress=lambda i, t: status.update(f"answering {i}/{t}"))
    except LLMError as e:
        con.print(f"[red]LLM error:[/] {e}")
        raise typer.Exit(1)
    if verbose:
        for r in res["per_question"]:
            mark = "[green]✓[/]" if r["gold_cited"] else "[red]✗[/]"
            con.print(f"{mark} cites={r['n_citations']} invalid={r['invalid']} unsupported={r['unsupported']} "
                      f"width={r['mean_width']} {'(repaired) ' if r['repaired'] else ''}{r['question']}", highlight=False)
    con.print(f"[bold]{mode}[/]  gold line cited = [bold]{res['gold_cited']:.3f}[/]   "
              f"valid citations = {res['valid_rate']:.3f}   claims cited = {res['claims_cited']:.3f}   "
              f"unsupported/answer = {res['unsupported_per_answer']:.2f}   mean width = {res['mean_width']} lines   "
              f"repaired = {res['repaired']}/{res['n']}")
    if save:
        _save({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "answers", "label": save, "gold": gold.name,
               "n": res["n"], "citations": mode, "prompt": settings.prompt_id, "graph": settings.graph_expand,
               "llm": settings.llm_model, **{key: (round(res[key], 4) if isinstance(res[key], float) else res[key])
                                             for key in ("gold_cited", "valid_rate", "claims_cited",
                                                         "unsupported_per_answer", "mean_width", "repaired")}})


# ---- call graph commands -----------------------------------------------------------------------
def _graph_node(symbol: str):
    cp_graph_path = settings.data_dir / "graph.json"
    if not cp_graph_path.exists():
        con.print("[yellow]No call graph for this index. Re-run `codecopilot index <repo>` (AST chunker).[/]")
        raise typer.Exit(1)
    g = CodeGraph.load(cp_graph_path)
    found = g.find(symbol)
    if not found:
        con.print(f"[yellow]No definition named {symbol!r} in this index.[/]")
        raise typer.Exit(1)
    if len(found) > 1:
        con.print(f"[dim]{len(found)} definitions match {symbol!r}; showing all. Use Class.method to pick one.[/]")
    return g, found


_VERB = {"call": "calls", "instantiate": "creates", "decorator": "decorated by", "inherit": "subclasses"}


@app.command()
def callers(symbol: str):
    """Who calls / uses / is decorated by / subclasses SYMBOL (one hop)."""
    g, found = _graph_node(symbol)
    for n in found:
        t = Table("caller", "how", "where", title=f"callers of {n.label}")
        for e in g.callers(n.id):
            src = g.nodes[e.src]
            t.add_row(src.qual, _VERB[e.kind], f"{src.path}:{e.line}")
        con.print(t if t.row_count else f"[dim]no resolved callers of {n.label}[/]")


@app.command()
def callees(symbol: str):
    """What SYMBOL calls / creates / is decorated with / inherits from (one hop, resolved definitions only)."""
    g, found = _graph_node(symbol)
    for n in found:
        t = Table("callee", "how", "called at", "defined at", title=f"callees of {n.label}")
        for e in g.callees(n.id):
            dst = g.nodes[e.dst]
            t.add_row(dst.qual, e.kind, f"{n.path}:{e.line}", f"{dst.path}:{dst.line}")
        con.print(t if t.row_count else f"[dim]no resolved callees of {n.label}[/]")


@app.command()
def impact(symbol: str, depth: int = typer.Option(None, "--depth", "-d")):
    """Blast radius: everything that transitively calls SYMBOL, i.e. what could break if it changes."""
    g, found = _graph_node(symbol)
    depth = depth or settings.impact_depth
    for n in found:
        rows = g.impact(n.id, depth)
        t = Table("hop", "affected", "via", "where", title=f"impact of changing {n.label} (depth ≤ {depth})")
        for nid, d, e in sorted(rows, key=lambda r: (r[1], g.nodes[r[0]].path)):
            via = g.nodes[e.dst]
            t.add_row(str(d), g.nodes[nid].qual, f"{_VERB[e.kind]} {via.qual}", f"{g.nodes[nid].path}:{e.line}")
        con.print(t if t.row_count else f"[dim]nothing in this repo calls {n.label}[/]")
        if rows:
            files = {g.nodes[nid].path for nid, _, _ in rows}
            con.print(f"[bold]{len(rows)}[/] definitions in [bold]{len(files)}[/] files could be affected. "
                      "[dim](static analysis: dynamic dispatch like obj.method() on unknown types is not tracked)[/]")


@app.command()
def report(path: Path = typer.Argument(RESULTS),
           gold: str = typer.Option("", "--gold", "-g", help="Only rows whose gold file name contains this")):
    """Print every saved eval run as tables — your before/after evidence."""
    if not path.exists():
        con.print(f"[yellow]No results yet at {path}. Run `codecopilot eval ... --save <label>` first.[/]")
        raise typer.Exit(1)
    t = Table("label", "gold set", "chunks", "mode", "rerank", "rewrite", "recall@k", "MRR", "hit@1", "p50 ms",
              "recall by type", title="retrieval")
    tt = Table("label", "gold set", "variant", "target recall", "all targets found", "chunks added", "by type",
               title="trace questions (call graph)")
    ta = Table("label", "gold set", "citations", "n", "gold line cited", "valid", "claims cited", "unsupported/ans",
               "mean width", "repaired", title="answers (citation quality)")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gname = Path(str(r.get("gold", "?")).replace("\\", "/")).name.replace("requests_gold", "").replace(".jsonl", "") or "standard"
        gname = gname.strip("_") or "standard"
        if gold and gold not in gname:
            continue
        kind = r.get("kind", "retrieval")
        if kind == "trace":
            tt.add_row(r["label"], gname, r["variant"], f"{r['target_recall']:.3f}", f"{r['all_found']:.3f}",
                       f"{r['mean_added']:.1f}", "  ".join(f"{k} {v:.2f}" for k, v in r.get("by_type", {}).items()))
            continue
        if kind == "answers":
            ta.add_row(r["label"], gname, r["citations"], str(r["n"]), f"{r['gold_cited']:.3f}",
                       f"{r['valid_rate']:.3f}", f"{r['claims_cited']:.3f}", f"{r['unsupported_per_answer']:.2f}",
                       str(r["mean_width"]), f"{r['repaired']}/{r['n']}")
            continue
        bt = r.get("by_type", {})
        types = "  ".join(f"{typ[:5]} {(v['recall'] if isinstance(v, dict) else v):.2f}" for typ, v in bt.items())
        lat = r.get("latency_ms", {}).get("p50")
        t.add_row(r["label"], gname, r.get("chunker") or "fixed", r.get("mode") or "dense",
                  "yes" if r.get("rerank") else "-", "yes" if r.get("rewrite") else "-",
                  f"{r['recall']:.3f}", f"{r['mrr']:.3f}", f"{r['hit1']:.3f}" if "hit1" in r else "-",
                  f"{lat:.0f}" if lat is not None else "-", types)
    for table in (t, tt, ta):
        if table.row_count:
            con.print(table)


@app.command()
def serve(host: str = typer.Option("127.0.0.1", "--host"), port: int = typer.Option(8000, "--port", "-p"),
          llm: str = typer.Option("ollama", "--llm", help="ollama (local) | groq (needs GROQ_API_KEY)")):
    """Start the website: http://127.0.0.1:8000"""
    from .web import serve as run
    if llm not in ("ollama", "groq"):
        raise typer.BadParameter("--llm must be ollama or groq")
    con.print(f"[green]Codebase Copilot[/] on http://{host}:{port}  (answers: {llm}; Ctrl+C to stop)")
    run(host, port, llm)


def main() -> None:
    # Click expands wildcards in argv on Windows, which turns -x "tests/*" into local file paths.
    app(windows_expand_args=False)


if __name__ == "__main__":
    main()
