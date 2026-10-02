"""Update 4: Jupyter notebooks, Windows-safe cloning, citations spanning excerpts, "did you mean" lookups."""
import json
import subprocess

from fastapi.testclient import TestClient

from codecopilot.citations import check_answer
from codecopilot.config import Settings
from codecopilot.embed import HashingEmbedder
from codecopilot.graph import build_graph
from codecopilot.index import Hit
from codecopilot.indexer import build_all
from codecopilot.ingest import ingest_repo, make_chunk
from codecopilot.sources import notebook_to_text, windows_safe
from codecopilot.web import Workspace, create_app

NB = {
    "nbformat": 4, "nbformat_minor": 5, "metadata": {},
    "cells": [
        {"cell_type": "markdown", "source": ["# Titanic model\n", "Trains a classifier."]},
        {"cell_type": "code", "source": ["%matplotlib inline\n", "!pip install xgboost\n", "import pandas as pd"],
         "outputs": [{"output_type": "stream", "text": ["lots of output\n"] * 500}]},
        {"cell_type": "code", "source": "def load_data(path):\n    return pd.read_csv(path)\n"},
        {"cell_type": "code", "source": []},
        {"cell_type": "code", "source": ["def train(path):\n", "    df = load_data(path)\n", "    return df"]},
    ],
}


def _nb_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "model.ipynb").write_text(json.dumps(NB))
    (repo / ".ipynb_checkpoints").mkdir()
    (repo / ".ipynb_checkpoints" / "model-checkpoint.ipynb").write_text(json.dumps(NB))
    return repo


def test_notebook_view_keeps_code_comments_magics_and_drops_outputs():
    text = notebook_to_text(json.dumps(NB))
    lines = text.splitlines()
    assert lines[0] == "# %% [markdown] cell 1" and lines[1] == "# # Titanic model"
    assert "# %matplotlib inline" in lines and "# !pip install xgboost" in lines and "import pandas as pd" in lines
    assert "lots of output" not in text and "cell 4" not in text      # outputs and empty cells are dropped
    compile(text, "model.ipynb", "exec")                                # the view is valid Python


def test_notebooks_are_chunked_graphed_and_citable(tmp_path):
    repo = _nb_repo(tmp_path)
    chunks = ingest_repo(repo, Settings(ast_min_lines=1))
    assert {c.path for c in chunks} == {"model.ipynb"}                  # checkpoint copies are skipped
    assert {"load_data", "train"} <= {c.symbol for c in chunks}
    g = build_graph(repo, chunks)
    (n,) = g.find("load_data")
    assert [g.nodes[e.src].qual for e in g.callers(n.id)] == ["train"]
    c = next(c for c in chunks if c.symbol == "train")
    view = notebook_to_text(json.dumps(NB)).splitlines()
    assert view[c.start_line - 1: c.end_line] == c.text.splitlines()   # line numbers match the viewer
    assert c.text.startswith("# %% cell 5\ndef train")                  # the cell marker travels with the code


def test_citation_may_span_several_shown_excerpts_of_one_file():
    shown = [Hit(make_chunk("j.py", 1, 10, "\n".join(f"x{i} = {i}" for i in range(1, 11)), "a", "function"), 1),
             Hit(make_chunk("j.py", 12, 20, "def speak(audio):\n" + "\n".join(["    pass"] * 8), "speak", "function"), 1)]
    rep = check_answer("The project defines `speak` [j.py:1-20].", shown)
    assert rep.invalid == [] and rep.unsupported == []
    assert check_answer("It does this [j.py:1-40].", shown).invalid == ["j.py:1-40"]      # past the shown code
    far = [shown[0], Hit(make_chunk("j.py", 30, 35, "y = 1\n" * 6, "b", "function"), 1)]
    assert check_answer("It does this [j.py:1-35].", far).invalid == ["j.py:1-35"]       # 19-line hole


def test_windows_safe_names():
    assert windows_safe("pkg/mod.py") and windows_safe("Number Guessing/main.py")
    for bad in ["Astro/NGC 6744: Close-Up.py", "a/con.py", "a/b?.py", "dir./x.py", "a\\b.py"]:
        assert not windows_safe(bad), bad


def test_clone_checks_out_only_safe_code_files(tmp_path):
    src = tmp_path / "src"
    (src / "app").mkdir(parents=True)
    (src / "app" / "main.py").write_text("def main():\n    return 1\n")
    (src / "nb.ipynb").write_text(json.dumps(NB))
    (src / "big.csv").write_text("a,b\n" * 1000)
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(src)], check=True)
    subprocess.run([*git, "-C", str(src), "add", "-A"], check=True)
    # Windows can't create "Bad: name.py" on disk, so put it straight into git, as it sits in the GitHub repo
    blob = subprocess.run(["git", "-C", str(src), "hash-object", "-w", "--stdin"], input="x = 1\n",
                          capture_output=True, text=True, check=True).stdout.strip()
    # core.protectNTFS=false: Git for Windows otherwise refuses to record the name (only in this throwaway repo)
    subprocess.run(["git", "-c", "core.protectNTFS=false", "-C", str(src), "update-index", "--add", "--cacheinfo",
                    f"100644,{blob},Bad: name.py"], check=True)
    subprocess.run([*git, "-c", "core.protectNTFS=false", "-C", str(src), "commit", "-qm", "init"], check=True)
    ws = Workspace(Settings(web_dir=tmp_path / "web"), tmp_path / "web")
    dest, skipped = ws._clone(src.as_uri(), tmp_path / "out")
    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file() and ".git" not in p.parts)
    assert files == ["app/main.py", "nb.ipynb"] and skipped == 1


def test_graph_lookup_is_case_insensitive_and_suggests_names(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "svc.py").write_text("class PersonServer:\n    def people(self, ids):\n        return ids\n\n\n"
                                 "class UserService:\n    def add_user(self):\n        pass\n")
    cfg = Settings(embed_backend="hashing", cache_dir=tmp_path / "c", web_dir=tmp_path / "web", ast_min_lines=1)
    build_all(repo, tmp_path / "scan" / ".cc_demo", cfg, HashingEmbedder())
    c = TestClient(create_app(cfg, root=tmp_path / "web", scan_dir=tmp_path / "scan"))
    assert c.get("/api/graph", params={"repo": "cli-demo", "symbol": "personserver"}).json()["matches"][0]["qual"] \
        == "PersonServer"
    res = c.get("/api/graph", params={"repo": "cli-demo", "symbol": "PersonService"}).json()
    assert res["matches"] == [] and res["similar"][0] == "PersonServer"


def test_file_endpoint_shows_notebook_view(tmp_path):
    repo = _nb_repo(tmp_path)
    cfg = Settings(embed_backend="hashing", cache_dir=tmp_path / "c", web_dir=tmp_path / "web", ast_min_lines=1)
    build_all(repo, tmp_path / "scan" / ".cc_nb", cfg, HashingEmbedder())
    c = TestClient(create_app(cfg, root=tmp_path / "web", scan_dir=tmp_path / "scan"))
    lines = c.get("/api/file", params={"repo": "cli-nb", "path": "model.ipynb"}).json()["lines"]
    assert lines == notebook_to_text(json.dumps(NB)).splitlines()
