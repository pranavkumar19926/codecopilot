from codecopilot.bm25 import BM25Index
from codecopilot.config import Settings
from codecopilot.embed import HashingEmbedder
from codecopilot.index import VectorIndex
from codecopilot.ingest import make_chunk
from codecopilot.llm import LLMClient
from codecopilot.pipeline import Copilot
from codecopilot.rerank import OverlapReranker
from codecopilot.rewrite import parse_rewrites


def test_parse_rewrites_strips_numbering_bullets_and_dupes():
    text = '1. missing scheme error\n- MissingSchema prepare_url\n3) "validates the URL scheme"\n\n- missing scheme error\nextra'
    assert parse_rewrites(text) == ["missing scheme error", "MissingSchema prepare_url", "validates the URL scheme"]


def test_llm_cache_hits_skip_the_network(tmp_path, monkeypatch):
    llm = LLMClient(Settings(cache_dir=tmp_path))
    calls = []
    monkeypatch.setattr(llm, "stream_chat", lambda m: (calls.append(1), iter(["a", "b"]))[1])
    msgs = [{"role": "user", "content": "q"}]
    assert llm.chat(msgs) == "ab" and llm.chat(msgs) == "ab"
    assert len(calls) == 1
    llm.cfg = Settings(cache_dir=tmp_path, llm_model="other-model")   # different model → different key
    llm.chat(msgs)
    assert len(calls) == 2


def _chunks():
    return [
        make_chunk("a.py", 1, 3, "def prepare_url(url):\n    raise MissingSchema('no scheme')", "prepare_url", "function"),
        make_chunk("b.py", 1, 3, "def send(r):\n    return adapter.send(r)", "send", "function"),
        make_chunk("c.py", 1, 3, "def get_redirect_target(resp):\n    return resp.headers['location']",
                   "get_redirect_target", "function"),
        make_chunk("d.py", 1, 3, "def rebuild_auth(req):\n    del req.headers['Authorization']", "rebuild_auth", "function"),
    ]


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def chat(self, messages, use_cache=True):
        self.calls += 1
        return self.reply


def _copilot(tmp_path, llm=None, reranker=None, **cfg):
    chunks, emb = _chunks(), HashingEmbedder()
    return Copilot(Settings(data_dir=tmp_path, **cfg), VectorIndex.build(chunks, emb, tmp_path), emb,
                   llm=llm, bm25=BM25Index.build(chunks), reranker=reranker)


def test_rewrite_adds_code_vocabulary_candidates(tmp_path):
    q = "what happens if I forget the http part"
    plain = _copilot(tmp_path).retrieve(q, k=1, mode="bm25")
    assert plain == [] or plain[0].chunk.symbol != "prepare_url"
    llm = FakeLLM("MissingSchema error\nprepare_url scheme\nraise when the URL has no scheme")
    cp = _copilot(tmp_path, llm=llm)
    hits = cp.retrieve(q, k=1, mode="bm25", rewrite=True)
    assert hits[0].chunk.symbol == "prepare_url"
    assert llm.calls == 1 and len(cp.last_rewrites) == 3


def test_rerank_reorders_but_keeps_symbol_pins_first(tmp_path):
    cp = _copilot(tmp_path, reranker=OverlapReranker())
    hits = cp.retrieve("where is the Authorization header deleted from req", k=4, mode="hybrid", rerank=True)
    assert hits[0].chunk.symbol == "rebuild_auth" and hits[0].detail.get("rerank") == 1
    pinned = cp.retrieve("where is rebuild_auth vs get_redirect_target", k=4, mode="hybrid", rerank=True)
    assert {h.chunk.symbol for h in pinned[:2]} == {"rebuild_auth", "get_redirect_target"}
    assert all(h.detail.get("symbol") for h in pinned[:2])


def test_stages_off_by_default_and_need_their_components(tmp_path):
    cp = _copilot(tmp_path)
    assert cp.retrieve("redirect", k=2) and cp.last_rewrites == []
    for kw in ({"rerank": True}, {"rewrite": True}):
        try:
            cp.retrieve("redirect", k=2, **kw)
        except RuntimeError:
            continue
        raise AssertionError(f"{kw} should require its component")
