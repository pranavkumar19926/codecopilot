import textwrap

from codecopilot.bm25 import BM25Index, identifier_terms, rrf, tokenize
from codecopilot.chunking import chunk_python, module_name
from codecopilot.config import Settings
from codecopilot.embed import HashingEmbedder
from codecopilot.index import VectorIndex
from codecopilot.ingest import make_chunk
from codecopilot.pipeline import Copilot

SRC = textwrap.dedent('''\
    import os
    from typing import overload

    TIMEOUT = 30


    # Retry exists because the upstream API drops ~1% of connections.
    @cached
    def fetch(url):
        return os.get(url)


    @overload
    def parse(x: int) -> int: ...
    @overload
    def parse(x: str) -> str: ...
    def parse(x):
        """Parse things."""
        value = x
        return value


    def a(): return 1
    def b(): return 2
    def c(): return 3


    class Small:
        def run(self):
            return 1
    ''')


def _cfg(**kw):
    return Settings(**kw)


def _by_symbol(chunks):
    return {c.symbol: c for c in chunks}


def test_definitions_keep_decorators_and_the_comment_above():
    ch = _by_symbol(chunk_python(SRC, "pkg/mod.py", _cfg()))
    fetch = ch["fetch"]
    assert fetch.text.splitlines()[0].startswith("# Retry exists")
    assert "@cached" in fetch.text and fetch.kind == "function"


def test_overload_stubs_merge_into_implementation():
    chunks = chunk_python(SRC, "pkg/mod.py", _cfg())
    parse = [c for c in chunks if c.symbol == "parse"]
    assert len(parse) == 1
    assert parse[0].text.count("def parse") == 3


def test_tiny_siblings_are_packed_and_module_code_is_kept():
    ch = _by_symbol(chunk_python(SRC, "pkg/mod.py", _cfg()))
    assert ch["a, b, c"].kind == "group"
    mod = ch["pkg.mod"]
    assert mod.kind == "module" and "import os" in mod.text and "TIMEOUT = 30" in mod.text
    assert ch["Small"].kind == "class"


def test_big_class_splits_into_header_and_linked_methods():
    methods = "\n".join(f"    def m{i}(self):\n" + "        x = 1\n" * 8 + "        return x\n" for i in range(12))
    src = f'class Big:\n    """Doc."""\n    limit = 5\n\n{methods}'
    chunks = chunk_python(src, "big.py", _cfg(ast_max_lines=40))
    header = next(c for c in chunks if c.kind == "class_header")
    assert header.symbol == "Big" and "limit = 5" in header.text and "def m0" not in header.text
    assert "# methods: m0, m1" in header.embed_text
    m3 = next(c for c in chunks if c.symbol == "Big.m3")
    assert m3.kind == "method" and m3.parent == "Big"


def test_long_function_splits_between_statements_and_covers_every_line():
    body = "".join(f"    x{i} = {i}\n" for i in range(50))
    src = f"def long():\n{body}    return x0\n"
    chunks = chunk_python(src, "l.py", _cfg(ast_max_lines=20))
    assert len(chunks) >= 3 and all(c.kind == "function_part" for c in chunks)
    covered = sorted(l for c in chunks for l in range(c.start_line, c.end_line + 1))
    assert covered == list(range(1, len(src.splitlines()) + 1))
    assert chunks[0].text.startswith("def long")


def test_every_code_line_is_covered():
    chunks = chunk_python(SRC, "pkg/mod.py", _cfg())
    covered = {l for c in chunks for l in range(c.start_line, c.end_line + 1)}
    code = {i for i, l in enumerate(SRC.splitlines(), 1) if l.strip() and not l.strip().startswith("#")}
    assert code <= covered


def test_module_name():
    assert module_name("src/requests/sessions.py") == "requests.sessions"
    assert module_name("pkg/__init__.py") == "pkg"


def test_tokenizer_splits_identifiers_and_stems_consistently():
    assert tokenize("rebuild_auth")[:3] == ["rebuild_auth", "rebuild", "auth"]
    assert set(tokenize("HTTPAdapter")) == {"httpadapter", "http", "adapter"}
    assert tokenize("decoded") == tokenize("decode") == tokenize("decoding")
    assert tokenize("raised") == tokenize("raise")
    assert "where" not in tokenize("where is it", drop_stopwords=True)


def test_identifier_terms_ignore_plain_english():
    assert identifier_terms("How does a session merge settings?") == set()
    assert identifier_terms("what calls prepare_hooks") == {"prepare_hooks"}
    assert identifier_terms("LookupDict") == {"lookupdict"}
    assert identifier_terms("who uses `Session.send`?") == {"session.send"}
    assert identifier_terms("What does Session.send call?") == {"session.send"}


def test_rrf_rewards_agreement():
    fused = dict(rrf([[1, 2, 3], [3, 1, 2]], k=60))
    assert max(fused, key=fused.get) == 1
    assert abs(fused[1] - (1 / 61 + 1 / 62)) < 1e-12


def _chunks():
    return [
        make_chunk("a.py", 1, 5, "def rebuild_auth(req):\n    del req.headers['Authorization']", "rebuild_auth", "function"),
        make_chunk("b.py", 1, 5, "def send(r):\n    rebuild_auth(r)\n    rebuild_auth(r)\n    return r", "send", "function"),
        make_chunk("c.py", 1, 5, "def auth_header():\n    return 'Basic'", "auth_header", "function"),
    ]


def test_bm25_ranks_definition_above_callers():
    chunks = _chunks()
    top = BM25Index.build(chunks).search("rebuild_auth", 3)
    assert chunks[top[0][0]].symbol == "rebuild_auth"


def test_hybrid_retrieval_pins_exact_symbol(tmp_path):
    chunks = _chunks()
    emb = HashingEmbedder()
    cfg = Settings(data_dir=tmp_path)
    cp = Copilot(cfg, VectorIndex.build(chunks, emb, tmp_path), emb, bm25=BM25Index.build(chunks))
    hits = cp.retrieve("where is rebuild_auth called", k=3, mode="hybrid")
    assert hits[0].chunk.symbol == "rebuild_auth" and hits[0].detail.get("symbol")
    dense_only = cp.retrieve("rebuild_auth", k=3, mode="dense")
    assert all("symbol" not in h.detail for h in dense_only)


def test_bm25_roundtrip(tmp_path):
    idx = BM25Index.build(_chunks())
    idx.save(tmp_path / "bm25.json")
    again = BM25Index.load(tmp_path / "bm25.json")
    assert again.search("auth header", 3) == idx.search("auth header", 3)
