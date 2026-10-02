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
