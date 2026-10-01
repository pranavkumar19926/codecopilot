import os

# Tests run offline: query rewriting (on by default in the app) needs Ollama, so tests opt in explicitly
# with rewrite=True and a fake LLM.
os.environ.setdefault("CC_QUERY_REWRITE", "false")
