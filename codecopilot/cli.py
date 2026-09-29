"""CLI: index / search / ask / eval."""
from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .config import settings
from .embed import get_embedder
from .index import VectorIndex
from .ingest import ingest_repo
from .llm import LLMClient, LLMError
from .pipeline import Answer, Copilot

app = typer.Typer(add_completion=False, help="Codebase Intelligence Copilot — Phase 1")
con = Console()


def _load() -> Copilot:
    idx = VectorIndex.load(settings.data_dir)
    if idx.meta["embed_model"] != get_embedder_name():
        con.print(f"[yellow]Index built with {idx.meta['embed_model']}, config says {get_embedder_name()}. Re-index.[/]")
        raise typer.Exit(1)
    return Copilot(settings, idx, get_embedder(settings), LLMClient(settings))


def get_embedder_name() -> str:
    return "hashing-1024" if settings.embed_backend == "hashing" else settings.embed_model


@app.command()
def index(repo: Path = typer.Argument(..., exists=True, file_okay=False),
          exclude: list[str] = typer.Option(None, "--exclude", "-x",
                                            help="Glob on repo-relative paths, repeatable. e.g. -x 'tests/*'")):
    """Chunk and embed a repository."""
    cfg = settings.model_copy(update={"exclude_globs": tuple(exclude)}) if exclude else settings
    chunks = ingest_repo(repo, cfg)
    prev = None
    try:
        prev = VectorIndex.load(settings.data_dir)
    except FileNotFoundError:
        pass
    idx = VectorIndex.build(chunks, get_embedder(settings), repo, previous=prev)
    idx.meta.update(exclude_globs=list(cfg.exclude_globs), chunk_lines=cfg.chunk_lines,
                    chunk_overlap=cfg.chunk_overlap)
    idx.save(settings.data_dir)
    m = idx.meta
    con.print(f"[green]Indexed[/] {len({c.path for c in chunks})} files → {m['n_chunks']} chunks "
              f"({m['n_embedded']} newly embedded) with {m['embed_model']}"
              + (f", excluding {m['exclude_globs']}" if m["exclude_globs"] else ""))


@app.command()
def search(query: str, k: int = typer.Option(settings.top_k, "-k")):
    """Show top-k retrieved chunks (no LLM) — use this to debug retrieval."""
    cp = Copilot(settings, VectorIndex.load(settings.data_dir), get_embedder(settings))
    t = Table("rank", "score", "location", "first line")
    for i, h in enumerate(cp.retrieve(query, k), 1):
        first = next((l.strip() for l in h.chunk.text.splitlines() if l.strip()), "")
        t.add_row(str(i), f"{h.score:.3f}", h.chunk.citation, first[:70])
    con.print(t)


@app.command()
def ask(question: str, show_sources: bool = typer.Option(True, "--sources/--no-sources")):
    """Answer a question with path:line citations."""
    cp = _load()
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
        con.rule("retrieved")
        for h in final.hits:
            con.print(f"  {h.score:.3f}  {h.chunk.citation}")
        cc = final.citation_check
        style = "green" if cc["n_citations"] and not cc["invalid"] else "yellow"
        con.print(f"[{style}]citations: {cc['n_citations']}, invalid: {cc['invalid'] or 'none'}[/]")


@app.command("eval")
def eval_(gold: Path = typer.Argument(..., exists=True), k: int = typer.Option(10, "-k"),
          verbose: bool = typer.Option(False, "-v"),
          save: str = typer.Option("", "--save", help="Label; appends the result to eval/results.jsonl")):
    """Retrieval recall@k and MRR against a gold set (JSONL: question + expected path/line)."""
    from .evaluate import evaluate_retrieval, load_gold, validate_gold
    cp = Copilot(settings, VectorIndex.load(settings.data_dir), get_embedder(settings))
    items = load_gold(gold)
    problems = validate_gold(items, Path(cp.index.meta["repo"]))
    if problems:
        con.print(f"[red]{len(problems)} gold item(s) don't match the indexed repo — wrong commit?[/] "
                  f"e.g. {problems[0]}\nFor requests: git -C requests checkout 611c616, then re-index.")
        raise typer.Exit(1)
    res = evaluate_retrieval(cp, items, k)
    m = cp.index.meta
    if verbose:
        for r in res["per_question"]:
            mark = "[green]✓[/]" if r["rank"] else "[red]✗[/]"
            con.print(f"{mark} rank={r['rank'] or '-':<3} [dim]{r['type']:<10}[/] {r['question']}")
    by_type = "   ".join(f"{t}: {v['recall']:.2f} (n={v['n']})" for t, v in res["by_type"].items())
    con.print(f"[bold]recall@{k}[/] = {res['recall']:.3f}   [bold]MRR[/] = {res['mrr']:.3f}   (n={res['n']})")
    con.print(f"  by type → {by_type}")
    con.print(f"  config  → chunking={m.get('chunk_lines')}/{m.get('chunk_overlap')}, "
              f"embed={m['embed_model']}, exclude={m.get('exclude_globs') or 'none'}")
    if save:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "gold": str(gold), "k": k, "label": save,
               "recall": round(res["recall"], 4), "mrr": round(res["mrr"], 4), "n": res["n"],
               "by_type": {t: round(v["recall"], 4) for t, v in res["by_type"].items()},
               "index": {key: m.get(key) for key in ("embed_model", "chunk_lines", "chunk_overlap", "exclude_globs")},
               "misses": [r["question"] for r in res["per_question"] if not r["rank"]]}
        with open("eval/results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        con.print(f"[green]saved[/] to eval/results.jsonl as {save!r}")


def main() -> None:
    # Click expands wildcards in argv on Windows, which turns -x "tests/*" into local file paths.
    app(windows_expand_args=False)


if __name__ == "__main__":
    main()
