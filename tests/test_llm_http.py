"""Exercises the real LLMClient HTTP + stream-parsing path against a local fake server (no mocks of LLMClient)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from codecopilot.config import Settings
from codecopilot.llm import LLMClient


def _serve(lines, seen):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            for line in lines:
                self.wfile.write((line + "\n").encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_ollama_stream_parsing_and_options(tmp_path):
    seen = []
    srv = _serve([json.dumps({"message": {"content": "hel"}}), json.dumps({"message": {"content": "lo"}}),
                  json.dumps({"done": True})], seen)
    try:
        llm = LLMClient(Settings(llm_base_url=f"http://127.0.0.1:{srv.server_port}", cache_dir=tmp_path))
        assert llm.chat([{"role": "user", "content": "q"}], temperature=0.0, num_ctx=4096) == "hello"
        opts = seen[0]["options"]
        assert opts["temperature"] == 0.0 and opts["num_ctx"] == 4096 and "seed" in opts
        assert llm.chat([{"role": "user", "content": "q"}], temperature=0.0, num_ctx=4096) == "hello"
        assert len(seen) == 1   # second call served from cache
    finally:
        srv.shutdown()


def test_openai_compatible_stream_parsing(tmp_path):
    seen = []
    chunk = lambda t: "data: " + json.dumps({"choices": [{"delta": {"content": t}}]})
    srv = _serve([chunk("a"), chunk("b"), "data: [DONE]"], seen)
    try:
        cfg = Settings(llm_provider="openai_compat", llm_base_url=f"http://127.0.0.1:{srv.server_port}",
                       cache_dir=tmp_path)
        assert "".join(LLMClient(cfg).stream_chat([{"role": "user", "content": "q"}])) == "ab"
        assert seen[0]["temperature"] == cfg.llm_temperature and "seed" in seen[0]
    finally:
        srv.shutdown()


def test_rate_limit_waits_for_retry_after_then_succeeds(tmp_path):
    seen, statuses = [], [429, 200]

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            code = statuses.pop(0)
            self.send_response(code)
            if code == 429:
                self.send_header("retry-after", "0.01")
            self.end_headers()
            if code == 200:
                self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}) + "\n").encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = Settings(llm_provider="openai_compat", llm_base_url=f"http://127.0.0.1:{srv.server_port}",
                       cache_dir=tmp_path, llm_model="openai/gpt-oss-120b")
        assert "".join(LLMClient(cfg).stream_chat([{"role": "user", "content": "q"}])) == "ok"
        assert len(seen) == 2 and seen[0]["reasoning_effort"] == "low"
    finally:
        srv.shutdown()


def test_rate_limit_gives_a_clear_message(tmp_path):
    import pytest
    from codecopilot.llm import LLMError

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(429)
            self.send_header("retry-after", "0.01")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = Settings(llm_provider="openai_compat", llm_base_url=f"http://127.0.0.1:{srv.server_port}",
                       cache_dir=tmp_path, llm_max_retries=1)
        with pytest.raises(LLMError, match="rate limit"):
            list(LLMClient(cfg).stream_chat([{"role": "user", "content": "q"}]))
    finally:
        srv.shutdown()


def test_key_is_trimmed_and_never_shown_in_errors(tmp_path):
    import pytest
    from codecopilot.llm import LLMError
    cfg = Settings(llm_provider="openai_compat", llm_base_url="http://127.0.0.1:9", cache_dir=tmp_path,
                   llm_api_key="gsk_SECRETabc123XYZ ", llm_max_retries=0)
    llm = LLMClient(cfg)
    assert llm._http.headers["Authorization"] == "Bearer gsk_SECRETabc123XYZ"
    with pytest.raises(LLMError) as e:
        list(llm.stream_chat([{"role": "user", "content": "q"}]))
    assert "SECRET" not in str(e.value)
    assert "SECRET" not in llm._scrub("Illegal header value b'Bearer gsk_SECRETabc123XYZ '")
