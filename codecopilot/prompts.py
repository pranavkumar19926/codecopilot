"""Prompt registry: prompts are versioned files with IDs, not string literals. Every answer records
the prompt id + content hash that produced it — this is what makes evals and tracing possible later."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    id: str
    version_hash: str
    system: str
    user: str

    def render(self, **kw: str) -> list[dict]:
        return [{"role": "system", "content": self.system.format(**kw)},
                {"role": "user", "content": self.user.format(**kw)}]


def load_prompt(prompt_id: str) -> Prompt:
    path = _DIR / f"{prompt_id}.md"
    if not path.exists():
        raise KeyError(f"Unknown prompt id {prompt_id!r}; available: {[p.stem for p in _DIR.glob('*.md')]}")
    raw = path.read_text(encoding="utf-8")
    body = raw.split("---", 2)[2] if raw.startswith("---") else raw
    system, user = body.split("[user]", 1)
    system = system.split("[system]", 1)[1]
    return Prompt(prompt_id, hashlib.sha1(raw.encode()).hexdigest()[:10], system.strip(), user.strip())
