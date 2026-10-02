"""Phase 4: a static call graph for Python, built with tree-sitter.

Retrieval finds code by *content*. Questions like "what calls X?" or "what breaks if I change X?" are about
*relationships*: the callers of X may never mention anything that looks like the question. So at index time
we extract, per file:

  nodes  every function / method / class definition (+ one pseudo-node per module for top-level code)
  edges  caller → callee, of kind
           call         foo(), self.foo(), obj.foo()
           decorator    @admin_required on dashboard → dashboard depends on admin_required
           instantiate  Session() → the Session class
           inherit      class Session(SessionRedirectMixin) → the base class

Python is dynamically typed, so `obj.send()` can't be resolved without type inference. Resolution is
deliberately conservative: an edge is kept only when the target is unambiguous, which means some edges are
missed rather than invented.
  1. self.foo / cls.foo      → a method of the enclosing class, else of a base class (transitively)
  2. bare foo()              → same-file definition, else a name imported into this file, else a unique top-level def
  3. Name.foo                → a method of class `Name`
  4. anything else           → only if exactly one definition has that name and it isn't a common name
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunking import _parser, module_name
from .ingest import Chunk

# Method names too generic to resolve by name alone (dicts, files, sockets, strings all have them).
COMMON = frozenset("""
get set send close update items keys values append pop copy read write join split format encode decode strip
lower upper startswith endswith replace add remove clear extend insert open run setdefault sort index count
__init__ __call__ __enter__ __exit__ __iter__ __getitem__ __setitem__ __contains__ __len__ __repr__ __eq__
info debug warning error exception commit flush seek tell readline lstrip rstrip
""".split())


@dataclass
class Node:
    id: int
    name: str        # short name, e.g. rebuild_auth
    qual: str        # qualified, e.g. SessionRedirectMixin.rebuild_auth
    path: str
    line: int        # definition line (1-indexed)
    end_line: int
    kind: str        # function | method | class | module
    cls: str = ""    # enclosing class for methods
    chunk: int = -1  # index of the chunk that contains the definition line

    @property
    def label(self) -> str:
        return f"{self.qual}  ({self.path}:{self.line})"


@dataclass
class Edge:
    src: int
    dst: int
    kind: str        # call | decorator | instantiate | inherit
    line: int        # where the reference happens (inside src)
    how: str         # resolution rule used: self | base | local | import | class | unique


class CodeGraph:
    def __init__(self, nodes: list[Node], edges: list[Edge], stats: dict | None = None):
        self.nodes, self.edges, self.stats = nodes, edges, stats or {}
        self._out: dict[int, list[Edge]] = defaultdict(list)
        self._in: dict[int, list[Edge]] = defaultdict(list)
        for e in edges:
            self._out[e.src].append(e)
            self._in[e.dst].append(e)

    # ---- queries -------------------------------------------------------------------------------
    def find(self, symbol: str) -> list[Node]:
        """`rebuild_auth`, `SessionRedirectMixin.rebuild_auth` or `sessions.rebuild_auth` style lookups."""
        s = symbol.strip().strip("`").removesuffix("()")
        exact = [n for n in self.nodes if n.kind != "module" and n.qual == s]
        if exact:
            return exact
        return [n for n in self.nodes if n.kind != "module" and (n.qual.endswith("." + s) or n.name == s)]

    def callers(self, node_id: int) -> list[Edge]:
        return sorted(self._in.get(node_id, []), key=lambda e: (self.nodes[e.src].path, e.line))

    def callees(self, node_id: int) -> list[Edge]:
        return sorted(self._out.get(node_id, []), key=lambda e: e.line)

    def impact(self, node_id: int, depth: int = 3) -> list[tuple[int, int, Edge]]:
        """Transitive callers (BFS): everything that could break if `node_id` changes. Returns (node, depth, edge)."""
        seen, out, q = {node_id}, [], deque([(node_id, 0)])
        while q:
            cur, d = q.popleft()
            if d == depth:
                continue
            for e in self.callers(cur):
                if e.src not in seen:
                    seen.add(e.src)
                    out.append((e.src, d + 1, e))
                    q.append((e.src, d + 1))
        return out

    def nodes_in_chunk(self, chunk_idx: int) -> list[Node]:
        return [n for n in self.nodes if n.chunk == chunk_idx]

    # ---- persistence ---------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"stats": self.stats, "nodes": [asdict(n) for n in self.nodes],
                                    "edges": [asdict(e) for e in self.edges]}))

    @classmethod
    def load(cls, path: Path) -> "CodeGraph":
        d = json.loads(path.read_text())
        return cls([Node(**n) for n in d["nodes"]], [Edge(**e) for e in d["edges"]], d.get("stats"))


# ---- construction ------------------------------------------------------------------------------

def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _callee(expr, src: bytes) -> tuple[str, str | None] | None:
    """(name, receiver) for the expression being called / used as decorator / base class."""
    if expr is None:
        return None
    if expr.type == "call":                      # @route(...) → resolve `route`
        return _callee(expr.child_by_field_name("function"), src)
    if expr.type == "identifier":
        return _text(expr, src), None
    if expr.type == "attribute":
        attr = expr.child_by_field_name("attribute")
        obj = expr.child_by_field_name("object")
        if attr is None:
            return None
        return _text(attr, src), (_text(obj, src) if obj is not None else None)
    return None


_IMPORT = re.compile(r"^\s*(?:from\s+\S+\s+)?import\s+(.+)$", re.MULTILINE)


def _imported_names(text: str) -> set[str]:
    names = set()
    for m in _IMPORT.finditer(text):
        for part in m.group(1).replace("(", " ").replace(")", " ").split(","):
            bits = part.split()
            if bits:
                names.add(bits[-1].split(".")[-1])   # handles `a as b` (→ b) and `pkg.mod` (→ mod)
    return names


class _FileScan:
    """Walks one file: collects definitions and raw references (resolved later, once all files are known)."""

    def __init__(self, path: str, text: str):
        self.path, self.text = path, text
        self.src = text.encode("utf-8")
        self.defs: list[dict] = []
        self.refs: list[dict] = []   # {owner (index into defs), name, receiver, kind, line}
        self.imports = _imported_names(text)
        tree = _parser().parse(self.src)
        mod = {"name": module_name(path), "qual": module_name(path), "kind": "module", "cls": "",
               "line": 1, "end_line": text.count("\n") + 1, "bases": []}
        self.defs.append(mod)
        self._walk(tree.root_node, owner=0, cls="", prefix="")

    def _walk(self, node, owner: int, cls: str, prefix: str):
        for child in node.named_children:
            t = child.type
            if t == "decorated_definition":
                inner = child.child_by_field_name("definition")
                idx = self._define(inner, cls, prefix) if inner is not None else None
                for dec in child.children:
                    if dec.type == "decorator" and idx is not None:
                        expr = next((c for c in dec.named_children), None)
                        self._ref(idx, _callee(expr, self.src), "decorator", dec.start_point[0] + 1)
                if inner is not None and idx is not None:
                    self._descend(inner, idx, cls, prefix)
                continue
            if t in ("function_definition", "class_definition"):
                idx = self._define(child, cls, prefix)
                self._descend(child, idx, cls, prefix)
                continue
            if t == "call":
                self._ref(owner, _callee(child.child_by_field_name("function"), self.src), "call",
                          child.start_point[0] + 1)
            self._walk(child, owner, cls, prefix)

    def _define(self, node, cls: str, prefix: str) -> int:
        name_node = node.child_by_field_name("name")
        name = _text(name_node, self.src) if name_node is not None else "<anonymous>"
        is_class = node.type == "class_definition"
        bases = []
        if is_class:
            sup = node.child_by_field_name("superclasses")
            if sup is not None:
                bases = [b for b in (_callee(a, self.src) for a in sup.named_children) if b]
        self.defs.append({
            "name": name, "qual": f"{prefix}{name}",
            "kind": "class" if is_class else ("method" if cls and prefix == cls + "." else "function"),
            "cls": cls if not is_class else "", "line": node.start_point[0] + 1, "end_line": node.end_point[0] + 1,
            "bases": bases,
        })
        idx = len(self.defs) - 1
        for b in bases:
            self._ref(idx, b, "inherit", node.start_point[0] + 1)
        return idx

    def _descend(self, node, idx: int, cls: str, prefix: str):
        d = self.defs[idx]
        body = node.child_by_field_name("body")
        if body is None:
            return
        if d["kind"] == "class":
            self._walk(body, owner=idx, cls=d["qual"], prefix=d["qual"] + ".")
        else:   # nested defs inside a function get qualified under it, but keep the enclosing class for self.*
            self._walk(body, owner=idx, cls=cls, prefix=d["qual"] + ".")

    def _ref(self, owner: int, target: tuple[str, str | None] | None, kind: str, line: int):
        if target:
            self.refs.append({"owner": owner, "name": target[0], "receiver": target[1], "kind": kind, "line": line})


def build_graph(repo: Path, chunks: list[Chunk]) -> CodeGraph:
    repo = repo.resolve()
    by_path: dict[str, list[tuple[int, Chunk]]] = defaultdict(list)
    for i, c in enumerate(chunks):
        by_path[c.path].append((i, c))

    def chunk_at(path: str, line: int) -> int:
        for i, c in by_path.get(path, []):
            if c.start_line <= line <= c.end_line:
                return i
        return -1

    scans: list[_FileScan] = []
    for path in sorted(p for p in by_path if p.endswith(".py")):
        try:
            scans.append(_FileScan(path, (repo / path).read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue

    nodes: list[Node] = []
    local_ids: list[list[int]] = []
    for s in scans:
        ids = []
        for d in s.defs:
            n = Node(len(nodes), d["name"], d["qual"], s.path, d["line"], d["end_line"], d["kind"], d["cls"],
                     chunk_at(s.path, d["line"]))
            nodes.append(n)
            ids.append(n.id)
        local_ids.append(ids)

    by_name: dict[str, list[Node]] = defaultdict(list)
    classes: dict[str, list[Node]] = defaultdict(list)
    for n in nodes:
        if n.kind == "module":
            continue
        by_name[n.name].append(n)
        if n.kind == "class":
            classes[n.qual].append(n)
            classes[n.name].append(n)
    bases: dict[str, list[str]] = {}
    for s, ids in zip(scans, local_ids):
        for d, nid in zip(s.defs, ids):
            if d["kind"] == "class":
                bases[nodes[nid].qual] = [b[0] for b in d["bases"]]

    def ancestry(cls_qual: str) -> list[str]:
        out, todo, seen = [], list(bases.get(cls_qual, [])), set()
        while todo:
            b = todo.pop(0)
            if b in seen:
                continue
            seen.add(b)
            out.append(b)
            for cand in classes.get(b, []):
                todo.extend(bases.get(cand.qual, []))
        return out

    edges: list[Edge] = []
    unresolved = ambiguous = 0
    for s, ids in zip(scans, local_ids):
        for r in s.refs:
            src = nodes[ids[r["owner"]]]
            name, recv = r["name"], r["receiver"]
            cands = by_name.get(name, [])
            if not cands:
                unresolved += 1
                continue
            pick, how = [], ""
            owner_cls = src.cls if src.kind != "class" else src.qual
            if recv in ("self", "cls") or (recv or "").startswith("super("):
                if owner_cls and not (recv or "").startswith("super("):
                    pick, how = [c for c in cands if c.cls == owner_cls and c.kind == "method"], "self"
                if not pick and owner_cls:
                    for b in ancestry(owner_cls):
                        pick = [c for c in cands if c.kind == "method" and (c.cls == b or c.cls.endswith("." + b))]
                        if pick:
                            how = "base"
                            break
            elif recv is None:
                same = [c for c in cands if c.path == s.path and c.kind != "method"]
                if same:
                    pick, how = same, "local"
                elif name in s.imports:
                    top = [c for c in cands if c.kind != "method"]
                    pick, how = (top, "import") if len(top) == 1 else ([], "")
                else:
                    top = [c for c in cands if c.kind != "method"]
                    if len(top) == 1 and name not in COMMON:
                        pick, how = top, "unique"
            else:
                cls_match = [c for c in cands if c.cls and (c.cls == recv or c.cls.endswith("." + recv))]
                if cls_match:
                    pick, how = cls_match, "class"
                elif len(cands) == 1 and name not in COMMON:
                    pick, how = cands, "unique"
            if len(pick) != 1:
                ambiguous += 1
                continue
            dst = pick[0]
            if dst.id == src.id:
                continue
            kind = r["kind"]
            if kind == "call" and dst.kind == "class":
                kind = "instantiate"
            edges.append(Edge(src.id, dst.id, kind, r["line"], how))

    # one edge per (src, dst, kind): keep the first reference line
    dedup: dict[tuple[int, int, str], Edge] = {}
    for e in sorted(edges, key=lambda e: e.line):
        dedup.setdefault((e.src, e.dst, e.kind), e)
    edges = list(dedup.values())
    stats = {"nodes": sum(n.kind != "module" for n in nodes), "edges": len(edges),
             "refs_unresolved_external": unresolved, "refs_ambiguous_dropped": ambiguous,
             "by_kind": {k: sum(e.kind == k for e in edges) for k in ("call", "decorator", "instantiate", "inherit")}}
    return CodeGraph(nodes, edges, stats)


# ---- question intent ---------------------------------------------------------------------------

_CALLEE_INTENT = re.compile(r"\bdoes\b.*\b(call|calls|use|uses|invoke|depend|depends|rely)\b", re.IGNORECASE)
_CALLER_INTENT = re.compile(
    r"\b(calls?|called|callers?|uses?|used|usages?|invok\w*|depend\w*|breaks?|impact\w*|affect\w*|referenc\w*|"
    r"protected|wrapped|decorated|subclass\w*|inherit\w*)\b", re.IGNORECASE)


def intent(question: str) -> str:
    """'callees' (what does X call), 'callers' (who calls / uses / breaks), or 'related' (anything else)."""
    if _CALLEE_INTENT.search(question):
        return "callees"
    if _CALLER_INTENT.search(question):
        return "callers"
    return "related"
