import json
import textwrap

from fastapi.testclient import TestClient

from codecopilot.config import Settings
from codecopilot.embed import HashingEmbedder
from codecopilot.indexer import build_all
from codecopilot.web import create_app

SRC = textwrap.dedent('''\
    def should_strip(url):
        return url.startswith("https")


    def rebuild_auth(req):
        if should_strip(req.url):
            del req.headers["Authorization"]
        return req


    def resolve(resp):
        return rebuild_auth(resp.request)
    ''')


class ScriptedLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, use_cache=True, **kw):
        return "".join(self.stream_chat(messages))

    def stream_chat(self, messages, **kw):
        self.calls += 1
        user = messages[1]["content"]
        if len(messages) > 2:   # repair turn
            yield "`rebuild_auth` deletes it [pkg/auth.py:5-7]."
        elif "Question:" in user and "excerpts" in user:
            for tok in ["`rebuild_auth` ", "deletes it ", "[nowhere.py:1-2]."]:
                yield tok
        else:
            yield "rebuild_auth\nAuthorization header\nremoves credentials"


def _client(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "auth.py").write_text(SRC)
    cfg = Settings(embed_backend="hashing", cache_dir=tmp_path / "cache", web_dir=tmp_path / "web", ast_min_lines=1)
    build_all(repo, tmp_path / "scan" / ".cc_demo", cfg, HashingEmbedder())
    app = create_app(cfg, root=tmp_path / "web", scan_dir=tmp_path / "scan")
    app.state.ws._llm = ScriptedLLM()
    return TestClient(app), app


def test_repos_listing_and_config(tmp_path):
    c, _ = _client(tmp_path)
    repos = c.get("/api/repos").json()
    assert [r["id"] for r in repos] == ["cli-demo"] and repos[0]["name"] == "repo" and repos[0]["status"] == "ready"
    assert c.get("/api/config").json()["strict_citations"] is True
    assert c.get("/").status_code == 200 and "Codebase Copilot" in c.get("/").text


def test_ask_streams_sources_tokens_repair_done(tmp_path):
    c, _ = _client(tmp_path)
    with c.stream("POST", "/api/ask", json={"repo": "cli-demo", "question": "what calls should_strip"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    types = [e["type"] for e in events]
    assert types[0] == "sources" and "token" in types and "repair" in types and types[-1] == "done"
    assert types.index("repair") < types.index("done")
    done = events[-1]
    assert done["repaired"] and done["check"]["invalid"] == [] and "pkg/auth.py:5-7" in done["text"]
    assert any(h["symbol"] == "should_strip" for h in events[0]["hits"])


def test_file_endpoint_blocks_path_traversal(tmp_path):
    c, _ = _client(tmp_path)
    assert c.get("/api/file", params={"repo": "cli-demo", "path": "pkg/auth.py"}).json()["lines"][0].startswith("def")
    for bad in ["../scan/.cc_demo/meta.json", "/etc/passwd", "pkg/../../web"]:
        assert c.get("/api/file", params={"repo": "cli-demo", "path": bad}).status_code == 404


def test_graph_search_and_suggestions(tmp_path):
    c, _ = _client(tmp_path)
    g = c.get("/api/graph", params={"repo": "cli-demo", "symbol": "rebuild_auth"}).json()["matches"][0]
    assert [x["qual"] for x in g["callers"]] == ["resolve"] and [x["qual"] for x in g["callees"]] == ["should_strip"]
    hits = c.get("/api/search", params={"repo": "cli-demo", "q": "should_strip", "k": 2}).json()["hits"]
    assert hits[0]["pinned"]
    assert c.get("/api/repos/cli-demo/suggestions").json()["questions"]


def test_add_repo_validation(tmp_path):
    c, app = _client(tmp_path)
    assert c.post("/api/repos", json={"url": "https://example.com/x/y"}).status_code == 400
    assert c.post("/api/repos", json={}).status_code == 400
    app.state.ws.cfg.web_allow_local = False
    assert c.post("/api/repos", json={"path": str(tmp_path)}).status_code == 403
    assert c.delete("/api/repos/cli-demo").status_code == 400      # CLI indexes are read-only
    assert c.get("/api/repos/nope").status_code == 404


def test_add_local_folder_indexes_in_background(tmp_path):
    c, _ = _client(tmp_path)
    r = c.post("/api/repos", json={"path": str(tmp_path / "repo"), "exclude_tests": True})
    assert r.status_code == 202
    rid = r.json()["id"]
    import time
    for _ in range(50):
        st = c.get(f"/api/repos/{rid}").json()
        if st["status"] in ("ready", "error"):
            break
        time.sleep(0.1)
    assert st["status"] == "ready", st
    assert c.delete(f"/api/repos/{rid}").status_code == 204
