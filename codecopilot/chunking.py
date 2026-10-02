"""Phase 2: structure-aware chunking for Python via tree-sitter.

One chunk per function / method / small class, so a retrieved chunk is a complete unit of meaning
instead of an arbitrary 60-line window that cuts a function in half. Rules, in order:

  1. Definitions keep their decorators and the comment block directly above them
     (that comment is often the "why", which the `why` gold questions depend on).
  2. `@overload` stubs are merged into the implementation they describe (no near-duplicate chunks).
  3. A class that fits in `ast_max_lines` is one chunk. A bigger class becomes a class_header chunk
     (signature, docstring, class attributes + a list of its methods) plus one chunk per method,
     each linked back via `parent` — the guide's "hierarchical chunks with parent links".
  4. A function longer than `ast_max_lines` is split between top-level statements, never mid-statement.
  5. Runs of tiny siblings (3-line getters, one-line helpers) are packed into one `group` chunk,
     so they don't flood the top-k with fragments.
  6. Module-level code between definitions (imports, constants, `__all__`) becomes `module` chunks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from .config import Settings
from .ingest import Chunk, chunk_file, make_chunk

_DEF_TYPES = {"function_definition", "class_definition"}


@lru_cache(maxsize=1)
def _parser():
    import tree_sitter as ts
    import tree_sitter_python as tsp
    return ts.Parser(ts.Language(tsp.language()))


@dataclass
class _Unit:
    start: int            # 1-indexed inclusive
    end: int
    symbol: str
    kind: str
    parent: str = ""
    body: object = None   # tree-sitter block node, used to split long functions
    context: str = ""     # extra line(s) for the embedding header only
    names: list[str] = field(default_factory=list)  # member names when packed into a group

    @property
    def n_lines(self) -> int:
        return self.end - self.start + 1


def module_name(rel_path: str) -> str:
    p = rel_path.removesuffix(".py").removesuffix(".ipynb")
    if p.startswith("src/"):
        p = p[4:]
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    return p.replace("/", ".")


def _definition(node):
    """Return the function/class node for a (possibly decorated) definition, else None."""
    if node.type in _DEF_TYPES:
        return node
    if node.type == "decorated_definition":
        d = node.child_by_field_name("definition")
        return d if d is not None and d.type in _DEF_TYPES else None
    return None


def _name(defn, src: bytes) -> str:
    n = defn.child_by_field_name("name")
    return src[n.start_byte:n.end_byte].decode("utf-8", "replace") if n is not None else "<anonymous>"


def _is_overload(node, src: bytes) -> bool:
    if node.type != "decorated_definition":
        return False
    return any(c.type == "decorator" and b"overload" in src[c.start_byte:c.end_byte] for c in node.children)


def _leading_comment_start(start: int, lines: list[str], floor: int) -> int:
    """Walk upward from a definition over a contiguous comment block (no blank line in between)."""
    s = start
    while s - 1 > floor and lines[s - 2].strip().startswith("#"):
        s -= 1
    return s


def _loose_segments(lo: int, hi: int, lines: list[str]) -> list[tuple[int, int]]:
    """Non-blank span inside [lo, hi] that contains real code (not only comments)."""
    while lo <= hi and not lines[lo - 1].strip():
        lo += 1
    while hi >= lo and not lines[hi - 1].strip():
        hi -= 1
    if lo > hi:
        return []
    if all(not l.strip() or l.strip().startswith("#") for l in lines[lo - 1:hi]):
        return []
    return [(lo, hi)]


def _scope_units(children, lines: list[str], src: bytes, prefix: str, parent: str,
                 lo: int, hi: int, cfg: Settings, loose_symbol: str, loose_kind: str) -> list[_Unit]:
    """Turn the children of a module / class body into ordered units covering [lo, hi]."""
    units: list[_Unit] = []
    cursor = lo  # first line not yet claimed
    pending_overload_start: int | None = None

    for node in children:
        defn = _definition(node)
        if defn is None:
            continue
        start = _leading_comment_start(node.start_point[0] + 1, lines, cursor - 1)
        end = node.end_point[0] + 1
        for a, b in _loose_segments(cursor, start - 1, lines):
            units.append(_Unit(a, b, loose_symbol, loose_kind, parent))
        cursor = end + 1

        name = _name(defn, src)
        qual = f"{prefix}{name}"
        if _is_overload(node, src):
            # stub: remember where it starts; the real implementation will absorb it
            pending_overload_start = pending_overload_start or start
            continue
        if pending_overload_start is not None:
            start, pending_overload_start = pending_overload_start, None

        if defn.type == "class_definition":
            units.extend(_class_units(defn, start, end, qual, parent, lines, src, cfg))
        else:
            kind = "method" if parent else "function"
            units.append(_Unit(start, end, qual, kind, parent, body=defn.child_by_field_name("body")))

    for a, b in _loose_segments(cursor, hi, lines):
        units.append(_Unit(a, b, loose_symbol, loose_kind, parent))
    return units


def _class_units(defn, start: int, end: int, qual: str, parent: str, lines: list[str], src: bytes,
                 cfg: Settings) -> list[_Unit]:
    if end - start + 1 <= cfg.ast_max_lines:
        return [_Unit(start, end, qual, "class", parent)]

    body = defn.child_by_field_name("body")
    members = [c for c in body.named_children if _definition(c) is not None]
    first_member_line = (_leading_comment_start(members[0].start_point[0] + 1, lines, start)
                         if members else end + 1)
    method_names = [_name(_definition(m), src) for m in members]

    header_end = first_member_line - 1
    while header_end > start and not lines[header_end - 1].strip():
        header_end -= 1
    header = _Unit(start, header_end, qual, "class_header", parent,
                   context=f"# methods: {', '.join(method_names)}" if method_names else "")
    inner = _scope_units(body.named_children, lines, src, prefix=f"{qual}.", parent=qual,
                         lo=first_member_line, hi=end, cfg=cfg,
                         loose_symbol=qual, loose_kind="class_body")
    return [header, *inner]


def _pack_small(units: list[_Unit], min_lines: int, max_lines: int) -> list[_Unit]:
    """Merge runs of consecutive tiny function/method units into a single `group` unit."""
    out: list[_Unit] = []
    run: list[_Unit] = []

    def flush():
        if len(run) == 1:
            out.append(run[0])
        elif run:
            names = [u.symbol for u in run]
            out.append(_Unit(run[0].start, run[-1].end, ", ".join(names), "group", run[0].parent, names=names))
        run.clear()

    for u in units:
        small = u.kind in ("function", "method") and u.n_lines < min_lines
        if small and (not run or (run[-1].parent == u.parent and u.end - run[0].start + 1 <= max_lines)):
            run.append(u)
            continue
        flush()
        if small:
            run.append(u)
        else:
            out.append(u)
    flush()
    return out


def _split_long(u: _Unit, cfg: Settings) -> list[tuple[int, int]]:
    """Split a long function between its top-level statements. First part keeps the signature."""
    if u.n_lines <= cfg.ast_max_lines:
        return [(u.start, u.end)]
    if u.body is None or not u.body.named_children:
        return [(a, min(a + cfg.ast_max_lines - 1, u.end)) for a in range(u.start, u.end + 1, cfg.ast_max_lines)]
    parts: list[tuple[int, int]] = []
    part_start, part_end = u.start, None
    for stmt in u.body.named_children:
        s, e = stmt.start_point[0] + 1, stmt.end_point[0] + 1
        if part_end is not None and e - part_start + 1 > cfg.ast_max_lines:
            parts.append((part_start, part_end))
            part_start = s
        part_end = e
    parts.append((part_start, u.end))
    return parts


def chunk_python(text: str, rel_path: str, cfg: Settings) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []
    src = text.encode("utf-8")
    try:
        tree = _parser().parse(src)
    except Exception:  # never let one odd file kill indexing
        return chunk_file(text, rel_path, cfg.chunk_lines, cfg.chunk_overlap)

    mod = module_name(rel_path)
    units = _scope_units(tree.root_node.named_children, lines, src, prefix="", parent="",
                         lo=1, hi=len(lines), cfg=cfg, loose_symbol=mod, loose_kind="module")
    units = _pack_small(units, cfg.ast_min_lines, cfg.ast_max_lines)

    chunks: list[Chunk] = []
    for u in units:
        if u.kind in ("module", "class_body") and u.n_lines > cfg.ast_max_lines:
            spans = [(a, min(a + cfg.ast_max_lines - 1, u.end)) for a in range(u.start, u.end + 1, cfg.ast_max_lines)]
        else:
            spans = _split_long(u, cfg)
        for i, (a, b) in enumerate(spans):
            kind = u.kind if len(spans) == 1 else f"{u.kind}_part"
            c = make_chunk(rel_path, a, b, "\n".join(lines[a - 1:b]), u.symbol, kind, u.parent,
                           context=u.context if i == 0 else f"# part {i + 1} of {len(spans)}")
            chunks.append(c)
    return chunks
