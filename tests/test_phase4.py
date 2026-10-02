import textwrap

from codecopilot.bm25 import BM25Index
from codecopilot.citations import check_answer
from codecopilot.config import Settings
from codecopilot.embed import HashingEmbedder
from codecopilot.graph import CodeGraph, build_graph, intent
from codecopilot.index import Hit, VectorIndex
from codecopilot.ingest import ingest_repo, make_chunk
from codecopilot.pipeline import Answer, Copilot, Repairing, assemble_context

AUTH = textwrap.dedent('''\
    from .utils import get_netrc_auth


    def admin_required(f):
        def wrapper(*a):
            return f(*a)
        return wrapper


    class Base:
        def strip(self, url):
            return url


    class Session(Base):
        def rebuild_auth(self, req):
            if self.should_strip(req.url):
                del req.headers["Authorization"]
            return get_netrc_auth(req.url)

        def should_strip(self, url):
            return self.strip(url) != url

        def resolve(self, resp):
            return self.rebuild_auth(resp.request)


    @admin_required
    def dashboard():
        s = Session()
        return s
    ''')
UTILS = textwrap.dedent('''\
    def get_netrc_auth(url):
        return None
    ''')


def _repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "auth.py").write_text(AUTH)
    (tmp_path / "pkg" / "utils.py").write_text(UTILS)
    return tmp_path


def _graph(tmp_path):
    repo = _repo(tmp_path)
    chunks = ingest_repo(repo, Settings(chunker="ast", ast_min_lines=1, ast_max_lines=8))
    return repo, chunks, build_graph(repo, chunks)


def _edges(g: CodeGraph, symbol):
    (n,) = g.find(symbol)
    return {(g.nodes[e.src].qual, e.kind, e.how) for e in g.callers(n.id)}


def test_graph_resolves_self_base_import_decorator_and_instantiation(tmp_path):
    _, _, g = _graph(tmp_path)
    assert _edges(g, "should_strip") == {("Session.rebuild_auth", "call", "self")}
    assert _edges(g, "Base.strip") == {("Session.should_strip", "call", "base")}       # inherited method
    assert _edges(g, "get_netrc_auth") == {("Session.rebuild_auth", "call", "import")}  # cross-file via import
    assert _edges(g, "admin_required") == {("dashboard", "decorator", "local")}
    assert _edges(g, "Session") == {("dashboard", "instantiate", "local")}
    assert ("Session", "inherit", "local") in _edges(g, "Base")


def test_impact_is_transitive_and_dedups(tmp_path):
    _, _, g = _graph(tmp_path)
    (n,) = g.find("Session.should_strip")
    hits = {(g.nodes[nid].qual, d) for nid, d, _ in g.impact(n.id, depth=3)}
    assert hits == {("Session.rebuild_auth", 1), ("Session.resolve", 2)}


def test_graph_roundtrip(tmp_path):
    _, _, g = _graph(tmp_path)
    g.save(tmp_path / "g.json")
    g2 = CodeGraph.load(tmp_path / "g.json")
    assert len(g2.edges) == len(g.edges) and g2.find("dashboard")[0].qual == "dashboard"


def test_intent():
    assert intent("What does Session.send call?") == "callees"
    assert intent("What calls rebuild_auth?") == "callers"
    assert intent("What would break if I changed get_netrc_auth?") == "callers"
    assert intent("How are passwords hashed?") == "related"


def test_expand_adds_callers_for_caller_questions(tmp_path):
    repo, chunks, g = _graph(tmp_path)
    emb = HashingEmbedder()
    cfg = Settings(data_dir=tmp_path / "idx", query_rewrite=False)
    cp = Copilot(cfg, VectorIndex.build(chunks, emb, repo), emb, bm25=BM25Index.build(chunks), graph=g)
    hits = cp.retrieve("what calls should_strip", k=1)
    assert hits[0].chunk.symbol == "Session.should_strip"
    extra = cp.expand("what calls should_strip", hits)
    assert extra and extra[0].chunk.symbol == "Session.rebuild_auth"
    assert "calls Session.should_strip" in extra[0].detail["graph"]
    cfg.graph_expand = False
    assert cp.expand("what calls should_strip", hits) == []


def _shown():
    text = "def rebuild_auth(req):\n    if should_strip(req):\n        del req.headers['Authorization']"
    return [Hit(make_chunk("a.py", 10, 12, text, "rebuild_auth", "function"), 1.0)]


def test_strict_checker_flags_unsupported_uncited_invalid_and_broad():
    shown = _shown()
    good = "`rebuild_auth` deletes the header [a.py:10-12]. It first calls `should_strip` [a.py:11-11]."
    rep = check_answer(good, shown)
    assert rep.ok and rep.n_claims == 2 and rep.widths == [3, 1]
    bad = ("`get_netrc_auth` deletes the header [a.py:12-12]. "   # unsupported: not on line 12, not the owner
           "It uses `should_strip` to decide. "                     # uncited
           "See [b.py:1-3].")                                       # invalid
    rep = check_answer(bad, shown)
    assert rep.unsupported == ["get_netrc_auth @ a.py:12-12"]
    assert len(rep.uncited) == 1 and rep.invalid == ["b.py:1-3"]
    assert not rep.ok and len(rep.problems()) == 3
    assert check_answer("`rebuild_auth` [a.py:10-12]", shown, broad_lines=2).broad == ["a.py:10-12"]


def test_line_numbers_in_context():
    ctx, used = assemble_context(_shown(), 10_000, line_numbers=True)
    assert "   11|     if should_strip(req):" in ctx and len(used) == 1


class ScriptedLLM:
    """Returns a bad first draft, then a corrected one."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, use_cache=True, **kw):
        self.calls.append(messages)
        return self.replies.pop(0)

    def stream_chat(self, messages, **kw):
        yield self.chat(messages)


def _answer_copilot(tmp_path, llm):
    repo, chunks, g = _graph(tmp_path)
    emb = HashingEmbedder()
    cfg = Settings(data_dir=tmp_path / "idx", query_rewrite=False)
    return Copilot(cfg, VectorIndex.build(chunks, emb, repo), emb, llm=llm, bm25=BM25Index.build(chunks), graph=g)


def test_repair_round_replaces_a_draft_with_bad_citations(tmp_path):
    (tmp_path / "x").mkdir()
    loc = None
    llm = ScriptedLLM(["`should_strip` is defined in [pkg/nowhere.py:1-2].", ""])
    cp = _answer_copilot(tmp_path / "x", llm)
    hits = cp.retrieve("what calls should_strip", k=1)
    c = hits[0].chunk
    loc = f"{c.path}:{c.start_line}-{c.start_line}"
    llm.replies[1] = f"`should_strip` is defined at [{loc}]."
    parts = list(cp.ask_stream("what calls should_strip"))
    assert any(isinstance(p, Repairing) for p in parts)
    ans = parts[-1]
    assert isinstance(ans, Answer) and ans.repaired and ans.citation_check["invalid"] == []
    assert ans.draft_check["invalid"] == ["pkg/nowhere.py:1-2"]
    assert llm.calls[1][-1]["role"] == "user" and "not inside any" in llm.calls[1][-1]["content"]


def test_no_repair_when_strict_off_or_draft_is_clean(tmp_path):
    llm = ScriptedLLM(["`x` [pkg/nowhere.py:1-2]."])
    cp = _answer_copilot(tmp_path, llm)
    cp.cfg.strict_citations = False
    ans = cp.answer("what calls should_strip")
    assert not ans.repaired and len(llm.calls) == 1


def test_small_class_methods_are_pinnable_through_the_graph(tmp_path):
    repo = _repo(tmp_path)
    chunks = ingest_repo(repo, Settings(chunker="ast", ast_min_lines=1))   # Session stays one chunk
    g = build_graph(repo, chunks)
    emb = HashingEmbedder()
    cp = Copilot(Settings(data_dir=tmp_path / "i", query_rewrite=False), VectorIndex.build(chunks, emb, repo), emb,
                 bm25=BM25Index.build(chunks), graph=g)
    hits = cp.retrieve("where is should_strip", k=1)
    assert hits[0].chunk.symbol == "Session" and hits[0].detail.get("symbol")


def test_honest_not_found_answer_passes_without_citations():
    shown = _shown()
    rep = check_answer("I couldn't find this in the retrieved code. I would search for login or token handling.", shown)
    assert rep.refusal and rep.ok and rep.problems() == []
    rep = check_answer("Authentication happens somewhere in the app.", shown)   # vague, uncited, not a refusal
    assert not rep.refusal and not rep.ok
    rep = check_answer("I couldn't find all of it, but `rebuild_auth` handles it.", shown)  # claims a name: not a refusal
    assert not rep.refusal


def test_citing_a_call_site_supports_the_enclosing_function_name():
    body = "def resolve(resp):\n    x = 1\n    return rebuild_auth(resp.request)"
    shown = [Hit(make_chunk("s.py", 186, 188, body, "Mixin.resolve", "method"), 1.0)]
    rep = check_answer("`Mixin.resolve` calls `rebuild_auth` [s.py:188-188].", shown)
    assert rep.ok and rep.unsupported == []
    rep = check_answer("`other_function` calls `rebuild_auth` [s.py:188-188].", shown)
    assert rep.unsupported == ["other_function @ s.py:188-188"]


def test_repair_never_replaces_a_cited_draft_with_a_refusal(tmp_path):
    llm = ScriptedLLM(["", "I couldn't find this in the retrieved code."])
    cp = _answer_copilot(tmp_path, llm)
    c = cp.retrieve("what calls should_strip", k=1)[0].chunk
    # valid citation + one unsupported name → repair is attempted, revision gives up → draft must be kept
    llm.replies[0] = f"`should_strip` is used here [{c.path}:{c.start_line}-{c.start_line}], see `made_up_name` [{c.path}:{c.start_line}-{c.start_line}]."
    ans = cp.answer("what calls should_strip")
    assert len(llm.calls) == 2 and not ans.repaired
    assert ans.text.startswith("`should_strip` is used here")


def test_prompt_example_citation_uses_a_real_shown_file(tmp_path):
    llm = ScriptedLLM(["`should_strip` [pkg/auth.py:1-1].", "I couldn't find this in the retrieved code."])
    cp = _answer_copilot(tmp_path, llm)
    cp.answer("what calls should_strip")
    system = llm.calls[0][0]["content"]
    assert "src/pkg/auth.py" not in system and "e.g. [pkg/" in system


def test_repair_message_lists_citable_files(tmp_path):
    llm = ScriptedLLM(["`should_strip` lives in [src/pkg/auth.py:18-20].", "I couldn't find this in the retrieved code."])
    cp = _answer_copilot(tmp_path, llm)
    cp.answer("what calls should_strip")
    repair_msg = llm.calls[1][-1]["content"]
    assert "The only files you can cite are: pkg/" in repair_msg
