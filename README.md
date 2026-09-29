# Codebase Intelligence Copilot — Phase 1

Ask questions about a repository; get answers with `path:start-end` citations to the actual code.

**Phase 1 scope (baseline):** fixed-size line chunks → dense embeddings → cosine top-k → local LLM answer with citations.
This is deliberately naive. It exists so Phase 2 (AST chunking + BM25) has a number to beat.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Local LLM (free): https://ollama.com
ollama pull qwen2.5-coder:7b        # ~4.7 GB; use qwen2.5-coder:3b on 8 GB RAM

# Test repo, pinned to the commit the gold set was labelled against
git clone https://github.com/psf/requests && git -C requests checkout 611c616
```

The embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) downloads automatically on first `index`.

## Usage

```bash
codecopilot index requests                        # chunk + embed (re-runs only embed changed chunks)
codecopilot search "where is auth stripped on redirect" -k 5    # retrieval only, no LLM — debug here first
codecopilot ask "Where is the Authorization header removed on a cross-host redirect?"
codecopilot eval eval/requests_gold.jsonl -k 10 -v              # recall@10 + MRR
pytest -q
```

`ask` streams the answer, lists retrieved chunks, and flags any citation that doesn't point inside a chunk the model was shown.
Every `ask` is appended to `.codecopilot/runs.jsonl` with prompt id + hash, models, retrieved chunks, and latency.

## Configuration

All settings live in `codecopilot/config.py` and can be overridden with `CC_*` env vars or a `.env` file:

| Variable | Default | Notes |
|---|---|---|
| `CC_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Try `jinaai/jina-embeddings-v2-base-code` for a code-tuned A/B |
| `CC_CHUNK_LINES` / `CC_CHUNK_OVERLAP` | `60` / `15` | Phase 1 baseline window |
| `CC_TOP_K` | `8` | Chunks sent to the LLM |
| `CC_LLM_MODEL` | `qwen2.5-coder:7b` | Any Ollama model |
| `CC_LLM_PROVIDER` | `ollama` | `openai_compat` for Groq etc. (see below) |
| `CC_EMBED_BACKEND` | `sentence-transformers` | `hashing` = offline stub for tests, **not** semantic |

Groq free tier instead of Ollama:
```bash
export CC_LLM_PROVIDER=openai_compat CC_LLM_BASE_URL=https://api.groq.com/openai/v1 \
       CC_LLM_MODEL=llama-3.3-70b-versatile CC_LLM_API_KEY=gsk_...
```

## Layout

```
codecopilot/
  config.py     typed settings (one place for model names, budgets, thresholds)
  ingest.py     repo walk + fixed-size chunking with path/line metadata
  embed.py      Embedder interface: sentence-transformers | hashing stub
  index.py      flat cosine index on disk; content-addressed reuse of unchanged chunks
  llm.py        LLM client wrapper: retries + backoff, timeouts, streaming (Ollama / OpenAI-compatible)
  prompts/      versioned prompt files; prompts.py loads them by id and hashes them
  pipeline.py   retrieve → context budget → generate → citation check → run log
  evaluate.py   recall@k and MRR against line-anchored gold items
  cli.py
eval/requests_gold.jsonl   50 labelled questions (35 semantic, 5 why, 10 identifier)
```

## The gold set

Each item is a question plus a `(path, line)` that must fall inside a retrieved chunk. Targets are **lines, not chunk IDs**, so the same file scores Phase 1 fixed chunks and Phase 2 AST chunks fairly. Line numbers are only valid at commit `611c616`.

50 items: 35 `semantic` (where/how), 5 `why` (answer depends on comments/reasoning near the code), and 10 `identifier` (bare symbol names). Eval reports recall per type, so you can see which kind of question each change actually helps.

## Phase 1 exit criteria (from the guide)

- [ ] `ask` answers "where is X handled?" correctly more often than not on `requests`
- [ ] Record your baseline: `recall@10` and `MRR` with bge-small, 60/15 chunking. **This is your "before" number.**

### Baseline runs to record

Use a separate `CC_DATA_DIR` per configuration so the indexes don't overwrite each other:

```bash
CC_DATA_DIR=.cc_all codecopilot index requests
CC_DATA_DIR=.cc_all codecopilot eval eval/requests_gold.jsonl -v --save "p1-bge-all"

CC_DATA_DIR=.cc_src codecopilot index requests -x "tests/*" -x "docs/*"
CC_DATA_DIR=.cc_src codecopilot eval eval/requests_gold.jsonl -v --save "p1-bge-no-tests"
```
(Windows PowerShell: `$env:CC_DATA_DIR=".cc_all"` on its own line first.)

`--save` appends to `eval/results.jsonl`: label, recall, MRR, per-type recall, index config, and the missed questions. Commit that file; it is the history of every change you measured.

`eval` refuses to run if gold lines don't exist in the indexed repo, which almost always means the wrong commit is checked out.

## Next: Phase 2

1. Replace `chunk_file` with tree-sitter chunks on function/class boundaries (keep the same `Chunk` fields).
2. Add BM25 (SQLite FTS5) alongside dense and fuse with reciprocal rank fusion.
3. Re-run `eval`. Watch the `identifier` items, which is where BM25 should win.
