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
from .config import settings
from .embed import get_embedder
from .index import VectorIndex
from .ingest import ingest_repo
from .llm import LLMClient, LLMError
from .pipeline import Answer, Copilot

app = typer.Typer(add_completion=False, help="Codebase Intelligence Copilot — Phase 2 (AST chunks + hybrid search)")
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
    return Copilot(settings, idx, embedder, LLMClient(settings) if with_llm else None, bm25)


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

    t0 = time.perf_counter()
    chunks = ingest_repo(repo, cfg)
    prev = None
    try:
        prev = VectorIndex.load(settings.data_dir)
    except FileNotFoundError:
        pass
    idx = VectorIndex.build(chunks, get_embedder(settings), repo, previous=prev)
    sizes = [c.end_line - c.start_line + 1 for c in chunks] or [0]
    idx.meta.update(
        exclude_globs=list(cfg.exclude_globs), chunker=cfg.chunker,
        chunk_lines=cfg.chunk_lines, chunk_overlap=cfg.chunk_overlap,
        ast_max_lines=cfg.ast_max_lines, ast_min_lines=cfg.ast_min_lines,
        mean_chunk_lines=round(statistics.mean(sizes), 1), median_chunk_lines=statistics.median(sizes),
        n_files=len({c.path for c in chunks}),
    )
    idx.save(settings.data_dir)
    BM25Index.build(chunks, symbol_boost=cfg.bm25_symbol_boost).save(settings.data_dir / "bm25.json")
    m = idx.meta
    con.print(f"[green]Indexed[/] {m['n_files']} files → {m['n_chunks']} {m['chunker']} chunks "
              f"(mean {m['mean_chunk_lines']} lines; {m['n_embedded']} newly embedded) "
              f"with {m['embed_model']} + BM25 in {time.perf_counter() - t0:.1f}s"
              + (f", excluding {m['exclude_globs']}" if m["exclude_globs"] else ""))


@app.command()
def search(query: str, k: int = typer.Option(settings.top_k, "-k"), mode: str = _ModeOpt):
    """Show top-k retrieved chunks (no LLM) — use this to debug retrieval."""
    mode = mode or settings.retrieval_mode
    cp = _copilot((mode,))
    t = Table("rank", "location", "symbol", "dense", "bm25", title=f"mode={mode}")
    for i, h in enumerate(cp.retrieve(query, k, mode), 1):
        d = h.detail
        t.add_row(str(i), h.chunk.citation, ("* " if d.get("symbol") else "") + (h.chunk.symbol[:45] or "-"),
                  str(d.get("dense") or "-") if "dense" in d else "", str(d.get("bm25") or "-") if "bm25" in d else "")
    con.print(t)


@app.command()
def ask(question: str, show_sources: bool = typer.Option(True, "--sources/--no-sources"), mode: str = _ModeOpt):
    """Answer a question with path:line citations."""
    if mode:
        settings.retrieval_mode = mode
    cp = _copilot((settings.retrieval_mode,), with_llm=True)
    final: Answer | None = None
    try:
        for part in cp.ask_stream(question):
            if isinstance(part, Answer):
                final = part
            else:
                con.print(part, end="", markup=False, highlight=False)
    except LLMError as e:
        con.print(f"\n[red]LLM error:[/] {e}\nIs Ollama running (`ollama serve`) and the model pulled?")
        raise typer.Exit(1)
    con.print()
    if final and show_sources:
        con.rule(f"retrieved ({settings.retrieval_mode})")
        for h in final.hits:
            con.print(f"  {h.chunk.citation:<42} {h.chunk.symbol}", highlight=False)
        cc = final.citation_check
        style = "green" if cc["n_citations"] and not cc["invalid"] else "yellow"
        con.print(f"[{style}]citations: {cc['n_citations']}, invalid: {cc['invalid'] or 'none'}[/]")


@app.command("eval")
def eval_(gold: Path = typer.Argument(..., exists=True), k: int = typer.Option(10, "-k"),
          mode: str = typer.Option(None, "--mode", "-m", help="dense | bm25 | hybrid | all"),
          verbose: bool = typer.Option(False, "-v"),
          save: str = typer.Option("", "--save", help="Label; appends each result to eval/results.jsonl")):
    """Retrieval recall@k and MRR against a gold set. `--mode all` runs dense, bm25 and hybrid in one go."""
    from .evaluate import evaluate_retrieval, load_gold, validate_gold
    modes = MODES if mode == "all" else (mode or settings.retrieval_mode,)
    cp = _copilot(modes)
    items = load_gold(gold)
    problems = validate_gold(items, Path(cp.index.meta["repo"]))
    if problems:
        con.print(f"[red]{len(problems)} gold item(s) don't match the indexed repo — wrong commit?[/] "
                  f"e.g. {problems[0]}\nFor requests: git -C requests checkout 611c616, then re-index.")
        raise typer.Exit(1)

    m = cp.index.meta
    chunker = m.get("chunker", "fixed")
    con.print(f"[dim]index: {chunker} chunks, n={m['n_chunks']}, mean {m.get('mean_chunk_lines', '?')} lines, "
              f"embed={m['embed_model']}, exclude={m.get('exclude_globs') or 'none'}[/]")
    for md in modes:
        res = evaluate_retrieval(cp, items, k, md)
        if verbose:
            for r in res["per_question"]:
                mark = "[green]✓[/]" if r["rank"] else "[red]✗[/]"
                con.print(f"{mark} rank={r['rank'] or '-':<3} [dim]{r['type']:<10}[/] {r['question']}")
        by_type = "   ".join(f"{t}: {v['recall']:.2f} (hit@1 {v['hit1']:.2f}, n={v['n']})"
                             for t, v in res["by_type"].items())
        con.print(f"[bold]{chunker}+{md:<6}[/] recall@{k} = [bold]{res['recall']:.3f}[/]   "
                  f"MRR = [bold]{res['mrr']:.3f}[/]   hit@1 = {res['hit1']:.3f}   (n={res['n']})")
        con.print(f"   by type → {by_type}")
        if save:
            label = f"{save}-{md}" if len(modes) > 1 else save
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "gold": str(gold), "k": k,
                   "chunker": chunker, "mode": md, "symbol_pin": settings.symbol_pin and md != "dense",
                   "recall": round(res["recall"], 4), "mrr": round(res["mrr"], 4), "hit1": round(res["hit1"], 4),
                   "n": res["n"],
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


@app.command()
def report(path: Path = typer.Argument(RESULTS)):
    """Print every saved eval run as one table — your before/after evidence."""
    if not path.exists():
        con.print(f"[yellow]No results yet at {path}. Run `codecopilot eval ... --save <label>` first.[/]")
        raise typer.Exit(1)
    t = Table("label", "chunks", "mode", "recall@k", "MRR", "hit@1", "semantic", "why", "identifier", "ident hit@1")
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        bt = r.get("by_type", {})
        rec = lambda typ: (f"{bt[typ]['recall']:.2f}" if isinstance(bt.get(typ), dict)
                           else f"{bt[typ]:.2f}" if typ in bt else "-")
        h1 = bt.get("identifier", {}).get("hit1") if isinstance(bt.get("identifier"), dict) else None
        t.add_row(r["label"], r.get("chunker") or "fixed", r.get("mode") or "dense", f"{r['recall']:.3f}",
                  f"{r['mrr']:.3f}", f"{r['hit1']:.3f}" if "hit1" in r else "-",
                  rec("semantic"), rec("why"), rec("identifier"), f"{h1:.2f}" if h1 is not None else "-")
    con.print(t)


def main() -> None:
    # Click expands wildcards in argv on Windows, which turns -x "tests/*" into local file paths.
    app(windows_expand_args=False)


if __name__ == "__main__":
    main()
