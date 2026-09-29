"""Repo walking + Phase 1 fixed-size chunking. Each chunk carries path and line range so answers can cite path:line."""
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
    id: str           # sha1(path:start-end:content) — stable across reindexing of unchanged code
    path: str         # repo-relative, posix
    start_line: int   # 1-indexed, inclusive
    end_line: int     # inclusive
    text: str

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict:
        return asdict(self)


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
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_lines")
    lines = text.splitlines()
    chunks: list[Chunk] = []
    step = size - overlap
    for start in range(0, max(len(lines), 1), step):
        window = lines[start : start + size]
        body = "\n".join(window)
        if body.strip():
            end = start + len(window)
            cid = hashlib.sha1(f"{rel_path}:{start+1}-{end}:{body}".encode()).hexdigest()[:16]
            chunks.append(Chunk(cid, rel_path, start + 1, end, body))
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
        out.extend(chunk_file(text, rel, cfg.chunk_lines, cfg.chunk_overlap))
    return out
