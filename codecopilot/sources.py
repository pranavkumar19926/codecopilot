"""Reading source files. `.py` files are read as they are; Jupyter notebooks (`.ipynb`) are turned into a
Python script view first, so chunking, the call graph, citations and the code viewer all work on them.

Notebook view (line numbers in citations refer to this view, and the website shows the same view):

    # %% [markdown] cell 1
    # # Titanic survival model
    # Loads the data and trains a classifier.

    # %% cell 2
    import pandas as pd
    # !pip install xgboost          <- shell / magic lines are commented out so the code still parses
    df = pd.read_csv("train.csv")

Outputs (plots, tables, tracebacks) are dropped: they are often megabytes and are not code."""
from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOK_EXT = ".ipynb"
CODE_EXT = (".py", NOTEBOOK_EXT)
_MAGIC = re.compile(r"^\s*[%!]")


def _lines(src) -> list[str]:
    text = "".join(src) if isinstance(src, list) else (src or "")
    return text.splitlines()


def notebook_to_text(raw: str) -> str:
    nb = json.loads(raw)
    cells = nb.get("cells")
    if cells is None:                      # very old nbformat 3: cells live inside worksheets
        cells = [c for ws in nb.get("worksheets", []) for c in ws.get("cells", [])]
    out: list[str] = []
    for i, cell in enumerate(cells, 1):
        kind = cell.get("cell_type", "code")
        body = _lines(cell.get("source", cell.get("input", "")))
        if not any(l.strip() for l in body):
            continue
        if out:
            out.append("")
        if kind == "code":
            out.append(f"# %% cell {i}")
            out.extend(f"# {l}" if _MAGIC.match(l) else l for l in body)
        else:
            out.append(f"# %% [{kind}] cell {i}")
            out.extend(f"# {l}".rstrip() for l in body)
    return "\n".join(out) + "\n"


def read_source(path: Path) -> str:
    """Text of a source file as the rest of the system sees it. Raises UnicodeDecodeError / ValueError
    for files that can't be read (binary, broken notebook JSON)."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix == NOTEBOOK_EXT:
        try:
            return notebook_to_text(raw)
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            raise ValueError(f"not a valid notebook: {path.name}") from e
    return raw


def is_python_like(path: str) -> bool:
    return path.endswith(CODE_EXT)


# ---- file names Windows can't create --------------------------------------------------------------
# git checkout of a repo with e.g. "NGC 6744: Close-Up.mp3" fails on Windows. We only check out code files
# whose names are valid there, so one odd media file elsewhere in the repo no longer breaks the whole clone.
_BAD_CHARS = re.compile(r'[<>:"|?*\\\x00-\x1f]')
_RESERVED = re.compile(r"^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$", re.IGNORECASE)


def windows_safe(rel_path: str) -> bool:
    for part in rel_path.split("/"):
        if not part or _BAD_CHARS.search(part) or _RESERVED.match(part) or part.endswith((" ", ".")):
            return False
    return len(rel_path) < 240
