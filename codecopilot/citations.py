"""Strict citation checking (Phase 4).

Phase 1 only checked that a citation pointed somewhere inside a chunk the model was shown. That catches
invented locations, but not the subtler failure: citing a real range that doesn't contain what the sentence
claims (e.g. citing a 100-line function for a claim about one line in it, or citing the wrong function).

Here every sentence is checked on its own:
  * invalid      the cited range is not inside any chunk shown to the model
  * unsupported  the sentence names a code identifier (in `backticks`) that does not occur in the lines it cites
  * uncited      the sentence names code identifiers but cites nothing
  * broad        the cited range is wider than `broad_lines` (technically valid, but not a pinpoint citation)
The identifier check is a cheap, script-checkable proxy for "the cited lines support the claim"."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index import Hit

CITE_RE = re.compile(r"[\[`]([^\[\]`\s]+?\.\w+):(\d+)(?:-(\d+))?[\]`]")
_TICK = re.compile(r"`([^`\n]+)`")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[`*(-])|\n+")
_SKIP = frozenset({"None", "True", "False", "self", "cls", "str", "int", "dict", "list", "bool", "bytes", "def",
                   "class", "return", "raise", "if", "else", "for", "in", "and", "or", "not"})


def parse_citations(text: str) -> list[tuple[str, int, int]]:
    return [(p, int(a), int(b or a)) for p, a, b in CITE_RE.findall(text)]


def _identifiers(sentence: str) -> list[str]:
    """Backticked code names in a sentence, reduced to the name the code would contain: `Session.send()` → send."""
    out = []
    for span in _TICK.findall(sentence):
        if CITE_RE.fullmatch(f"`{span}`"):
            continue                                   # it's a citation, not a name
        name = span.split("(")[0].strip().split(".")[-1]
        if _IDENT.match(name) and len(name) >= 3 and name not in _SKIP and not span.endswith((".py", ".md")):
            out.append(name)
    return out


@dataclass
class CitationReport:
    n_citations: int = 0
    invalid: list[str] = field(default_factory=list)
    n_claims: int = 0                 # sentences that name code identifiers
    cited_claims: int = 0
    unsupported: list[str] = field(default_factory=list)   # "name @ path:a-b"
    uncited: list[str] = field(default_factory=list)       # sentence snippets
    broad: list[str] = field(default_factory=list)
    widths: list[int] = field(default_factory=list)

    @property
    def penalty(self) -> int:
        return 3 * len(self.invalid) + 2 * len(self.unsupported) + len(self.uncited) + (self.n_citations == 0) * 3

    @property
    def ok(self) -> bool:
        return self.penalty == 0

    def as_dict(self) -> dict:
        return {"n_citations": self.n_citations, "invalid": self.invalid, "n_claims": self.n_claims,
                "cited_claims": self.cited_claims, "unsupported": self.unsupported, "uncited": self.uncited,
                "broad": self.broad,
                "mean_width": round(sum(self.widths) / len(self.widths), 1) if self.widths else None}

    def problems(self) -> list[str]:
        out = [f"{c} is not inside any of the code excerpts you were shown" for c in self.invalid]
        out += [f"`{u.split(' @ ')[0]}` does not appear in the lines you cited ({u.split(' @ ')[1]})"
                for u in self.unsupported]
        out += [f'this claim has no citation: "{s}"' for s in self.uncited]
        if self.n_citations == 0:
            out.append("the answer has no [path:start-end] citations at all")
        return out


def check_answer(answer: str, shown: list[Hit], broad_lines: int = 40) -> CitationReport:
    rep = CitationReport()

    def lines_for(path: str, a: int, b: int) -> str | None:
        for h in shown:
            c = h.chunk
            if c.path == path and c.start_line <= a and b <= c.end_line:
                rows = c.text.splitlines()
                return "\n".join(rows[a - c.start_line: b - c.start_line + 1])
        return None

    seen_invalid = set()
    for sentence in (s.strip() for s in _SENT_SPLIT.split(answer)):
        if not sentence:
            continue
        cites = parse_citations(sentence)
        names = _identifiers(sentence)
        cited_text = []
        for p, a, b in cites:
            rep.n_citations += 1
            rep.widths.append(b - a + 1)
            body = lines_for(p, a, b)
            if body is None:
                key = f"{p}:{a}-{b}"
                if key not in seen_invalid:
                    rep.invalid.append(key)
                    seen_invalid.add(key)
                continue
            cited_text.append(body)
            if b - a + 1 > broad_lines:
                rep.broad.append(f"{p}:{a}-{b}")
        if names:
            rep.n_claims += 1
            if cites:
                rep.cited_claims += 1
                joined = "\n".join(cited_text)
                for n in names:
                    if cited_text and n not in joined:
                        rep.unsupported.append(f"{n} @ " + ", ".join(f"{p}:{a}-{b}" for p, a, b in cites))
            else:
                rep.uncited.append(sentence[:90] + ("…" if len(sentence) > 90 else ""))
    return rep
