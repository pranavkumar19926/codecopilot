from pathlib import Path

import numpy as np
import pytest

from codecopilot.config import Settings
from codecopilot.embed import HashingEmbedder
from codecopilot.index import VectorIndex
from codecopilot.ingest import chunk_file, ingest_repo
from codecopilot.pipeline import Copilot, assemble_context, check_citations


def test_chunks_cover_every_line_with_overlap():
    text = "\n".join(f"line{i}" for i in range(1, 131))
    chunks = chunk_file(text, "a.py", size=60, overlap=15)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (46, 105), (91, 130)]
    covered = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line + 1))
        assert c.text.splitlines()[0] == f"line{c.start_line}"
    assert covered == set(range(1, 131))


def test_short_and_empty_files():
    assert [(c.start_line, c.end_line) for c in chunk_file("x = 1", "a.py", 60, 15)] == [(1, 1)]
    assert chunk_file("", "a.py", 60, 15) == []
    with pytest.raises(ValueError):
        chunk_file("x", "a.py", 10, 10)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "auth.py").write_text("def refresh_token(session):\n    return session.renew()\n")
    (tmp_path / "pkg" / "db.py").write_text("def connect_database(url):\n    return open_pool(url)\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk.py").write_text("def refresh_token(): pass\n")
    return tmp_path


def test_ingest_skips_excluded_dirs(tmp_path):
    chunks = ingest_repo(_repo(tmp_path), Settings())
    assert sorted(c.path for c in chunks) == ["pkg/auth.py", "pkg/db.py"]


def test_retrieval_and_incremental_reindex(tmp_path):
    repo, cfg = _repo(tmp_path), Settings(data_dir=tmp_path / "idx")
    emb = HashingEmbedder()
    idx = VectorIndex.build(ingest_repo(repo, cfg), emb, repo)
    idx.save(cfg.data_dir)
    idx = VectorIndex.load(cfg.data_dir)
    hits = Copilot(cfg, idx, emb).retrieve("where is the token refreshed", k=1, mode="dense")
    assert hits[0].chunk.path == "pkg/auth.py"

    (repo / "pkg" / "db.py").write_text("def connect_database(url, timeout):\n    return open_pool(url)\n")
    idx2 = VectorIndex.build(ingest_repo(repo, cfg), emb, repo, previous=idx)
    assert idx2.meta["n_embedded"] == 1  # only the changed file is re-embedded
    assert np.allclose(idx2.vectors[0], idx.vectors[0])


def test_citation_check(tmp_path):
    repo, cfg = _repo(tmp_path), Settings()
    idx = VectorIndex.build(ingest_repo(repo, cfg), HashingEmbedder(), repo)
    _, used = assemble_context([h for h in idx.search(HashingEmbedder().embed_query("token"), 2)], 10_000)
    res = check_citations("Refreshed in [pkg/auth.py:1-2] not [pkg/auth.py:1-9] or [other.py:1-2].", used)
    assert res == {"n_citations": 3, "invalid": ["pkg/auth.py:1-9", "other.py:1-2"]}


def test_exclude_globs(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_auth.py").write_text("def test_refresh():\n    assert True\n")
    cfg = Settings(exclude_globs=("tests/*",))
    assert "tests/test_auth.py" not in {c.path for c in ingest_repo(repo, cfg)}
    assert "tests/test_auth.py" in {c.path for c in ingest_repo(repo, Settings())}


def test_citation_check_accepts_backticks(tmp_path):
    repo, cfg = _repo(tmp_path), Settings()
    idx = VectorIndex.build(ingest_repo(repo, cfg), HashingEmbedder(), repo)
    _, used = assemble_context(idx.search(HashingEmbedder().embed_query("token"), 2), 10_000)
    res = check_citations("It is in `pkg/auth.py:1-2`, see also [pkg/db.py:1-2].", used)
    assert res == {"n_citations": 2, "invalid": []}
