"""Repo walking + chunk model. Two chunkers: `fixed` (Phase 1 baseline, line windows) and `ast`
(Phase 2, syntactic boundaries via tree-sitter). Every chunk carries path + line range so answers can cite path:line."""
from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from .config import Settings


@dataclass
class Chunk:
    id: str           # sha1(path:start-end:embed_text) — stable across reindexing of unchanged code
    path: str         # repo-relative, posix
    start_line: int   # 1-indexed, inclusive
    end_line: int     # inclusive
    text: str         # exact source lines start..end (what the LLM sees and what citations point into)
    symbol: str = ""  # qualified name, e.g. "SessionRedirectMixin.rebuild_auth"; "" for fixed windows
    kind: str = "window"  # window | function | class | method | class_header | module | group
    parent: str = ""  # enclosing class for methods / class_header, else ""
    context: str = "" # extra header line(s) used only for embedding/BM25, e.g. a class's method list

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def embed_text(self) -> str:
        """Text that gets embedded. AST chunks get a header so a method body that never mentions its
        class or file still matches questions about them. Fixed windows are embedded raw (Phase 1 baseline)."""
        if not self.symbol:
            return self.text
        head = f"# file: {self.path}\n# {self.kind}: {self.symbol}\n"
        if self.context:
            head += self.context + "\n"
        return head + self.text

    def to_dict(self) -> dict:
        return asdict(self)


def make_chunk(path: str, start: int, end: int, text: str, symbol: str = "", kind: str = "window",
               parent: str = "", context: str = "") -> Chunk:
    c = Chunk("", path, start, end, text, symbol, kind, parent, context)
    c.id = hashlib.sha1(f"{path}:{start}-{end}:{c.embed_text}".encode()).hexdigest()[:16]
    return c


def iter_source_files(root: Path, cfg: Settings) -> Iterator[Path]:
    excluded = set(cfg.exclude_dirs)
    for dirpath, dirnames, filenames in os.walk(root):
        # prune in place so os.walk never descends into excluded dirs
        dirnames[:] = sorted(d for d in dirnames if d not in excluded and not d.startswith("."))
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.suffix in cfg.include_ext and p.stat().st_size <= cfg.max_file_bytes:
                yield p


def chunk_file(text: str, rel_path: str, size: int, overlap: int) -> list[Chunk]:
    """Phase 1 baseline: fixed line windows with overlap."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_lines")
    lines = text.splitlines()
    chunks: list[Chunk] = []
    step = size - overlap
    for start in range(0, max(len(lines), 1), step):
        window = lines[start : start + size]
        body = "\n".join(window)
        if body.strip():
            chunks.append(make_chunk(rel_path, start + 1, start + len(window), body))
        if start + size >= len(lines):
            break
    return chunks


def is_excluded(rel_path: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(rel_path, g) for g in globs)


def ingest_repo(root: Path, cfg: Settings) -> list[Chunk]:
    root = root.resolve()
    out: list[Chunk] = []
    for f in iter_source_files(root, cfg):
        rel = f.relative_to(root).as_posix()
        if is_excluded(rel, cfg.exclude_globs):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if cfg.chunker == "ast" and f.suffix == ".py":
            from .chunking import chunk_python  # lazy: tree-sitter only needed for the AST chunker
            out.extend(chunk_python(text, rel, cfg))
        else:
            out.extend(chunk_file(text, rel, cfg.chunk_lines, cfg.chunk_overlap))
    return out
