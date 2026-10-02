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

GAP = 3   # blank lines allowed between two shown excerpts that one citation spans
CITE_RE =re.compile(r"[\[`]([^\[\]`\s]+?\.\w+):(\d+)(?:-(\d+))?[\]`]")
_TICK = re.compile(r"`([^`\n]+)`")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[`*(-])|\n+")
# The prompts tell the model to say this instead of guessing. An honest "not found" needs no citations.
_REFUSAL = re.compile(r"couldn.?t find (this|that|it|any)|could not find (this|that|it|any)|"
                      r"not (present|found|shown) in the (retrieved|provided) (code|excerpts)|"
                      r"(excerpts|retrieved code) (do|does) not (contain|show|include)", re.IGNORECASE)
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
    refusal: bool = False             # the answer says the code doesn't contain it, and claims nothing

    @property
    def penalty(self) -> int:
        if self.refusal:
            return 0
        return 3 * len(self.invalid) + 2 * len(self.unsupported) + len(self.uncited) + (self.n_citations == 0) * 3

    @property
    def ok(self) -> bool:
        return self.penalty == 0

    def as_dict(self) -> dict:
        return {"n_citations": self.n_citations, "invalid": self.invalid, "n_claims": self.n_claims,
                "cited_claims": self.cited_claims, "unsupported": self.unsupported, "uncited": self.uncited,
                "broad": self.broad, "refusal": self.refusal,
                "mean_width": round(sum(self.widths) / len(self.widths), 1) if self.widths else None}

    def problems(self) -> list[str]:
        out = [f"{c} is not inside any of the code excerpts you were shown" for c in self.invalid]
        out += [f"`{u.split(' @ ')[0]}` does not appear in the lines you cited ({u.split(' @ ')[1]})"
                for u in self.unsupported]
        out += [f'this claim has no citation: "{s}"' for s in self.uncited]
        if self.n_citations == 0 and not self.refusal:
            out.append("the answer has no [path:start-end] citations at all")
        return out


def check_answer(answer: str, shown: list[Hit], broad_lines: int = 40) -> CitationReport:
    rep = CitationReport()

    def lines_for(path: str, a: int, b: int) -> tuple[str, str] | None:
        """(cited lines, names of the definition(s) those lines sit inside)."""
        for h in shown:
            c = h.chunk
            if c.path == path and c.start_line <= a and b <= c.end_line:
                rows = c.text.splitlines()
                return "\n".join(rows[a - c.start_line: b - c.start_line + 1]), c.symbol
        # A range spanning several shown excerpts of the same file is fine too ("what is this project about?"
        # → [jarvis.py:1-201] when every chunk of jarvis.py was shown). AST chunks leave the blank lines
        # between definitions out, so gaps of up to GAP lines between excerpts are allowed.
        parts = sorted((h.chunk for h in shown if h.chunk.path == path and h.chunk.end_line >= a
                        and h.chunk.start_line <= b), key=lambda c: c.start_line)
        cur, bodies, owners = a, [], []
        for c in parts:
            if c.start_line > cur + GAP:
                return None
            rows = c.text.splitlines()
            lo, hi = max(a, c.start_line), min(b, c.end_line)
            bodies.append("\n".join(rows[lo - c.start_line: hi - c.start_line + 1]))
            if c.symbol:
                owners.append(c.symbol)
            cur = max(cur, c.end_line + 1)
        if parts and cur + GAP > b:
            return "\n".join(bodies), ", ".join(owners)
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
            found = lines_for(p, a, b)
            if found is None:
                key = f"{p}:{a}-{b}"
                if key not in seen_invalid:
                    rep.invalid.append(key)
                    seen_invalid.add(key)
                continue
            body, owner = found
            # a name is supported if it occurs in the cited lines, OR the cited lines are inside that
            # definition: "`resolve_redirects` calls `rebuild_auth` [sessions.py:273]" cites the call site,
            # which is inside resolve_redirects even though that name is on line 186.
            owners = " ".join(part.split(".")[-1] for part in owner.split(", ")) if owner else ""
            cited_text.append(body + "\n" + owners)
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
    rep.refusal = rep.n_citations == 0 and rep.n_claims == 0 and bool(_REFUSAL.search(answer))
    return rep
