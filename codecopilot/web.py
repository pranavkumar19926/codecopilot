"""Web app: FastAPI backend + a single static page (codecopilot/static).

Routes
  GET  /                         the page
  GET  /api/config               which LLM, which stages are on
  GET  /api/repos                indexed repositories (website workspace + any .cc_* CLI indexes in the cwd)
  POST /api/repos                {"url": "https://github.com/o/r"} or {"path": "../requests"}: clone/index in background
  GET  /api/repos/{id}           status of one repo / indexing job
  DELETE /api/repos/{id}         remove a website-added repo
  GET  /api/search?repo&q&k      retrieval only (fast)
  POST /api/ask                  {"repo","question"} → NDJSON stream: sources, token…, repair?, token…, done | error
  GET  /api/file?repo&path       a source file, for the code viewer
  GET  /api/graph?repo&symbol    callers / callees / impact of a definition
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .bm25 import BM25Index
from .config import Settings, settings
from .embed import get_embedder
from .graph import CodeGraph
from .index import Hit, VectorIndex
from .llm import LLMClient, LLMError
from .pipeline import Answer, Copilot, Repairing, Retrieved
from .sources import read_source, windows_safe

STATIC = Path(__file__).parent / "static"
_GITHUB = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")
_ID = re.compile(r"[^a-z0-9_-]+")


def _slug(text: str) -> str:
    return _ID.sub("-", text.lower()).strip("-")[:60] or "repo"


@dataclass
class RepoEntry:
    id: str
    name: str
    data_dir: Path
    source: str                # GitHub URL, local path, or "cli"
    removable: bool = True
    status: str = "ready"      # queued | cloning | indexing | ready | error
    message: str = ""
    meta: dict = field(default_factory=dict)

    def public(self) -> dict:
        m = self.meta
        return {"id": self.id, "name": self.name, "source": self.source, "status": self.status,
                "message": self.message, "removable": self.removable,
                "files": m.get("n_files"), "chunks": m.get("n_chunks"), "built_at": m.get("built_at"),
                "graph_edges": (m.get("graph") or {}).get("edges"), "chunker": m.get("chunker")}


class AskBody(BaseModel):
    repo: str
    question: str


class AddBody(BaseModel):
    url: str | None = None
    path: str | None = None
    exclude_tests: bool = True


class Workspace:
    def __init__(self, cfg: Settings, root: Path, scan_dir: Path | None = None):
        self.cfg, self.root = cfg, root
        self.repos: dict[str, RepoEntry] = {}
        self._copilots: dict[str, Copilot] = {}
        self._embedder = None
        self._llm = None
        self.lock = threading.Lock()        # one generation at a time: the LLM is the bottleneck anyway
        self.index_lock = threading.Lock()  # one indexing job at a time (CPU + memory)
        (root / "indexes").mkdir(parents=True, exist_ok=True)
        (root / "repos").mkdir(parents=True, exist_ok=True)
        self._load(scan_dir)

    # ---- registry ------------------------------------------------------------------------------
    def _load(self, scan_dir: Path | None) -> None:
        for d in sorted((self.root / "indexes").iterdir()):
            info = d / "web.json"
            if (d / "meta.json").exists() and info.exists():
                w = json.loads(info.read_text())
                self.repos[d.name] = RepoEntry(d.name, w["name"], d, w["source"],
                                               meta=json.loads((d / "meta.json").read_text()))
        if scan_dir:   # indexes made with the CLI (.cc_ast, .cc_store, …) show up read-only
            for d in sorted(scan_dir.glob(".cc_*")):
                if (d / "meta.json").exists() and (d / "bm25.json").exists():
                    meta = json.loads((d / "meta.json").read_text())
                    if "graph" not in meta and (d / "graph.json").exists():   # indexed before the website existed
                        meta["graph"] = json.loads((d / "graph.json").read_text()).get("stats", {})
                    rid = "cli-" + _slug(d.name.removeprefix(".cc_"))
                    name = Path(meta["repo"]).name
                    if any(e.name == name for e in self.repos.values()):
                        name = f"{name} ({d.name})"
                    self.repos[rid] = RepoEntry(rid, name, d, "cli", removable=False, meta=meta)

    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.cfg)
        return self._embedder

    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(self.cfg)
        return self._llm

    def copilot(self, rid: str) -> Copilot:
        e = self.repos.get(rid)
        if e is None:
            raise HTTPException(404, f"No repository with id {rid!r}.")
        if e.status != "ready":
            raise HTTPException(409, f"{e.name} is still {e.status}.")
        if rid not in self._copilots:
            cfg = self.cfg.model_copy(update={"data_dir": e.data_dir})
            idx = VectorIndex.load(e.data_dir)
            emb_name = "hashing-1024" if cfg.embed_backend == "hashing" else cfg.embed_model
            if idx.meta["embed_model"] != emb_name:
                raise HTTPException(409, f"{e.name} was indexed with {idx.meta['embed_model']}; "
                                         f"the server uses {emb_name}. Re-index it.")
            gpath = e.data_dir / "graph.json"
            self._copilots[rid] = Copilot(cfg, idx, self.embedder(), self.llm(),
                                          BM25Index.load(e.data_dir / "bm25.json"),
                                          graph=CodeGraph.load(gpath) if gpath.exists() else None)
        return self._copilots[rid]

    # ---- adding repositories -------------------------------------------------------------------
    def add(self, body: AddBody) -> RepoEntry:
        if bool(body.url) == bool(body.path):
            raise HTTPException(400, "Give either a GitHub URL or a local folder path.")
        if body.url:
            m = _GITHUB.match(body.url.strip())
            if not m:
                raise HTTPException(400, "Use a public GitHub URL like https://github.com/psf/requests")
            name = f"{m.group(1)}/{m.group(2)}"
            source = f"https://github.com/{name}"
        else:
            if not self.cfg.web_allow_local:
                raise HTTPException(403, "Adding local folders is disabled on this server.")
            p = Path(body.path).expanduser().resolve()
            if not p.is_dir():
                raise HTTPException(400, f"Folder not found: {p}")
            name, source = p.name, str(p)
        rid = _slug(name)
        if rid in self.repos and self.repos[rid].status not in ("error",):
            raise HTTPException(409, f"{name} is already added.")
        e = RepoEntry(rid, name, self.root / "indexes" / rid, source, status="queued",
                      message="waiting for another indexing job" if self.index_lock.locked() else "")
        self.repos[rid] = e
        threading.Thread(target=self._index_job, args=(e, body.exclude_tests), daemon=True).start()
        return e

    def _index_job(self, e: RepoEntry, exclude_tests: bool) -> None:
        from .indexer import build_all
        with self.index_lock:
            try:
                skipped = 0
                if e.source.startswith("https://"):
                    e.status, e.message = "cloning", "downloading the repository"
                    dest = self.root / "repos" / e.id
                    shutil.rmtree(dest, ignore_errors=True)
                    repo, skipped = self._clone(e.source, dest)
                else:
                    repo = Path(e.source)
                e.status = "indexing"
                globs = ("tests/*", "test/*", "docs/*", "*/tests/*", "*/test/*") if exclude_tests else ()
                cfg = self.cfg.model_copy(update={"exclude_globs": globs, "chunker": "ast"})

                def say(msg):
                    e.message = msg
                e.meta = build_all(repo, e.data_dir, cfg, self.embedder(), progress=say)
                (e.data_dir / "web.json").write_text(json.dumps({"name": e.name, "source": e.source}))
                self._copilots.pop(e.id, None)
                e.status, e.message = "ready", f"indexed in {e.meta['seconds']}s" + (
                    f"; skipped {skipped} file(s) with names Windows can't use" if skipped else "")
            except subprocess.TimeoutExpired:
                e.status, e.message = "error", "cloning took too long"
            except Exception as ex:   # surfaced to the UI; the server keeps running
                e.status, e.message = "error", str(ex) or ex.__class__.__name__
                traceback.print_exc()

    def _git(self, *args: str, cwd: Path | None = None) -> str:
        r = subprocess.run(["git", "-c", "core.longpaths=true", *args], cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=self.cfg.web_clone_timeout_s)
        if r.returncode != 0:
            lines = [l.strip() for l in r.stderr.splitlines() if l.strip()]
            # git ends with hints ("retry with 'git restore ...'"); the real reason is the fatal:/error: line
            reason = next((l for l in lines if l.startswith(("fatal:", "error:"))), lines[-1] if lines else "unknown")
            if "not found" in reason.lower() or "could not read username" in reason.lower():
                reason = "repository not found (is it public?)"
            raise ValueError("git failed: " + reason.removeprefix("fatal: ").removeprefix("error: "))
        return r.stdout

    def _clone(self, url: str, dest: Path) -> tuple[Path, int]:
        """Download only the code files we index. A full checkout breaks on Windows when the repo has *any*
        file with a name Windows can't create (e.g. "NGC 6744: Close-Up.mp3"), and it wastes time and disk on
        images, datasets and models. Partial clone (--filter=blob:none) fetches file contents on demand,
        so only the checked-out files are downloaded."""
        self._git("clone", "--depth", "1", "--single-branch", "--filter=blob:none", "--no-checkout", url, str(dest))
        names = self._git("-c", "core.quotepath=off", "ls-tree", "-r", "-z", "--name-only", "HEAD", cwd=dest)
        wanted, skipped = [], 0
        for path in names.split("\0"):
            if not path or not path.endswith(tuple(self.cfg.include_ext)):
                continue
            if any(part in self.cfg.exclude_dirs or part.startswith(".") for part in path.split("/")[:-1]):
                continue
            if windows_safe(path):
                wanted.append(path)
            else:
                skipped += 1
        if not wanted:
            raise ValueError("no indexable files found (only .py and .ipynb files are indexed)")
        for i in range(0, len(wanted), 100):     # batches keep the command line short on Windows
            self._git("--literal-pathspecs", "checkout", "HEAD", "--", *wanted[i:i + 100], cwd=dest)
        size = sum((dest / p).stat().st_size for p in wanted if (dest / p).is_file())
        if size > self.cfg.web_max_repo_mb * 1_000_000:
            shutil.rmtree(dest, ignore_errors=True)
            raise ValueError(f"the repository's code files are {size / 1e6:.0f} MB; this server's limit is "
                             f"{self.cfg.web_max_repo_mb} MB")
        return dest, skipped

    def remove(self, rid: str) -> None:
        e = self.repos.get(rid)
        if e is None:
            raise HTTPException(404, "No such repository.")
        if not e.removable:
            raise HTTPException(400, "Indexes made with the command line can't be removed from the website.")
        if e.status in ("queued", "cloning", "indexing"):
            raise HTTPException(409, "Wait for indexing to finish first.")
        self.repos.pop(rid)
        self._copilots.pop(rid, None)
        shutil.rmtree(e.data_dir, ignore_errors=True)
        shutil.rmtree(self.root / "repos" / rid, ignore_errors=True)


def _hit(h: Hit) -> dict:
    c = h.chunk
    return {"path": c.path, "start": c.start_line, "end": c.end_line, "symbol": c.symbol, "kind": c.kind,
            "why": h.detail.get("graph"), "pinned": bool(h.detail.get("symbol"))}


def create_app(cfg: Settings | None = None, root: Path | None = None, scan_dir: Path | None = None) -> FastAPI:
    cfg = cfg or settings
    ws = Workspace(cfg, root or cfg.web_dir, scan_dir if scan_dir is not None else Path.cwd())
    app = FastAPI(title="Codebase Copilot", docs_url="/api/docs", redoc_url=None)
    app.state.ws = ws

    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/config")
    def config():
        return {"llm": cfg.llm_model, "provider": "Groq" if "groq.com" in cfg.llm_base_url else
                ("Ollama" if cfg.llm_provider == "ollama" else cfg.llm_base_url),
                "rewrite": cfg.query_rewrite, "strict_citations": cfg.strict_citations,
                "graph": cfg.graph_expand, "allow_local": cfg.web_allow_local}

    @app.get("/api/repos")
    def repos():
        return [e.public() for e in ws.repos.values()]

    @app.post("/api/repos", status_code=202)
    def add(body: AddBody):
        return ws.add(body).public()

    @app.get("/api/repos/{rid}")
    def repo(rid: str):
        if rid not in ws.repos:
            raise HTTPException(404, "No such repository.")
        return ws.repos[rid].public()

    @app.delete("/api/repos/{rid}", status_code=204)
    def delete(rid: str):
        ws.remove(rid)

    @app.get("/api/repos/{rid}/suggestions")
    def suggestions(rid: str):
        """Starter questions built from the repo itself: its most-called definitions."""
        cp = ws.copilot(rid)
        g = cp.graph
        if g is None:
            return {"questions": ["Where is the main entry point?", "How are errors handled?"]}
        ranked = sorted((n for n in g.nodes if n.kind in ("function", "method") and not n.name.startswith("_")),
                        key=lambda n: -len(g.callers(n.id)))
        top = [n for n in ranked if len(g.callers(n.id)) > 0][:2]
        qs = [f"What does {top[0].qual} do?"] if top else []
        if len(top) > 1:
            qs.append(f"What calls {top[1].name}, and what would break if I changed it?")
        qs.append("How are errors handled?")
        return {"questions": qs}

    @app.get("/api/search")
    def search(repo: str, q: str, k: int = 8):
        cp = ws.copilot(repo)
        with ws.lock:
            try:
                hits = cp.retrieve(q, max(1, min(k, 20)))
            except LLMError as e:
                raise HTTPException(503, f"The language model isn't reachable: {e}")
            return {"rewrites": cp.last_rewrites, "hits": [_hit(h) for h in hits]}

    @app.post("/api/ask")
    def ask(body: AskBody):
        q = body.question.strip()
        if not q:
            raise HTTPException(400, "Type a question first.")
        if len(q) > 1000:
            raise HTTPException(400, "Questions are limited to 1000 characters.")
        cp = ws.copilot(body.repo)

        def events():
            def line(obj):
                return json.dumps(obj) + "\n"
            with ws.lock:
                t0 = time.perf_counter()
                try:
                    for part in cp.ask_stream(q):
                        if isinstance(part, Retrieved):
                            related = {id(h) for h in part.extra}
                            yield line({"type": "sources", "rewrites": part.rewrites,
                                        "hits": [_hit(h) for h in part.used if id(h) not in related],
                                        "related": [_hit(h) for h in part.used if id(h) in related]})
                        elif isinstance(part, Repairing):
                            yield line({"type": "repair", "problems": part.problems})
                        elif isinstance(part, Answer):
                            yield line({"type": "done", "text": part.text, "repaired": part.repaired,
                                        "check": part.citation_check, "seconds": round(time.perf_counter() - t0, 1)})
                        else:
                            yield line({"type": "token", "text": part})
                except LLMError as e:
                    yield line({"type": "error", "message": f"The language model isn't reachable. {e}"})
                except Exception as e:
                    traceback.print_exc()
                    yield line({"type": "error", "message": f"{e.__class__.__name__}: {e}"})

        return StreamingResponse(events(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/file")
    def file(repo: str, path: str):
        cp = ws.copilot(repo)
        root = Path(cp.index.meta["repo"]).resolve()
        target = (root / path).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(404, "File not found in this repository.")
        try:
            text = read_source(target)      # notebooks: the same script view the citations point into
        except (UnicodeDecodeError, ValueError):
            text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) > 20000:
            raise HTTPException(413, "File is too large to display.")
        return {"path": path, "lines": lines}

    @app.get("/api/graph")
    def graph(repo: str, symbol: str):
        cp = ws.copilot(repo)
        g = cp.graph
        if g is None:
            raise HTTPException(404, "This repository has no call graph. Re-index it.")
        verbs = {"call": "calls", "instantiate": "creates", "decorator": "is decorated by", "inherit": "subclasses"}
        out = []
        for n in g.find(symbol)[:5]:
            out.append({
                "qual": n.qual, "path": n.path, "line": n.line, "end_line": n.end_line, "kind": n.kind,
                "callers": [{"qual": g.nodes[e.src].qual, "path": g.nodes[e.src].path, "line": e.line,
                             "how": verbs[e.kind]} for e in g.callers(n.id)],
                "callees": [{"qual": g.nodes[e.dst].qual, "path": g.nodes[e.dst].path, "line": g.nodes[e.dst].line,
                             "how": e.kind, "at": e.line} for e in g.callees(n.id)],
                "impact": len(g.impact(n.id, cfg.impact_depth)),
            })
        similar = [] if out else g.similar(symbol)
        return {"symbol": symbol, "matches": out, "similar": similar}

    return app


def serve(host: str, port: int, llm: str) -> None:
    import os

    import uvicorn
    if llm == "groq":
        key = (os.environ.get("GROQ_API_KEY") or settings.llm_api_key).strip().strip('"').strip("'")
        if not key:
            raise SystemExit("Set GROQ_API_KEY first (free key: https://console.groq.com/keys).")
        settings.llm_provider = "openai_compat"
        settings.llm_base_url = "https://api.groq.com/openai/v1"
        settings.llm_api_key = key
        if settings.llm_model == "qwen2.5-coder:7b":
            settings.llm_model = settings.groq_model
        settings.context_token_budget = min(settings.context_token_budget, settings.groq_context_budget)
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
