# Codebase Intelligence Copilot

Ask questions about a repository; get answers with `path:start-end` citations to the actual code.

| Phase | What | Status |
|---|---|---|
| 1 | Fixed 60-line chunks → dense embeddings → local LLM answer with citations; 50-question gold set | done |
| 2 | Tree-sitter AST chunks + BM25 keyword search + reciprocal rank fusion + symbol lookup | **current** |
| 3 | Cross-encoder reranking, query rewriting | next |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Local LLM (free): https://ollama.com
ollama pull qwen2.5-coder:7b        # ~4.7 GB; use qwen2.5-coder:3b on 8 GB RAM

# Test repo next to this folder, pinned to the commit the gold set was labelled against
git clone https://github.com/psf/requests ../requests && git -C ../requests checkout 611c616
```

The embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) downloads automatically on first `index`.

## Usage

```bash
codecopilot index ../requests -x "tests/*" -x "docs/*"      # AST chunks + embeddings + BM25
codecopilot search "rebuild_auth" -k 5                      # retrieval only; shows dense & BM25 rank per hit
codecopilot ask "Where is the Authorization header removed on a cross-host redirect?"
codecopilot eval eval/requests_gold.jsonl -m all --save p2  # dense, bm25, hybrid in one run
codecopilot report                                          # every saved run as one table
pytest -q
```

Flags: `-c fixed|ast` on `index`; `-m dense|bm25|hybrid` on `search`, `ask`, `eval` (`eval` also takes `all`).
In `search`, a `*` before the symbol means it was pinned by symbol lookup.

Full Phase 2 comparison on Windows: `.\eval\run_phase2.ps1` (builds both indexes, runs all 6 evals, prints the table).

## How retrieval works (Phase 2)

```
question ─┬─ dense: bge-small embedding → cosine top-50 ───┐
          ├─ BM25: code-aware tokens → Okapi BM25 top-50 ──┼─ reciprocal rank fusion ─→ top-k ─→ LLM
          └─ symbol lookup: identifier in the question? ───┘   (pinned symbols go first)
```

**AST chunking** (`chunking.py`, tree-sitter). One chunk per function / method / small class instead of arbitrary windows:
- decorators and the comment block directly above a definition stay attached (that comment is often the "why")
- `@overload` stubs merge into their implementation; runs of tiny getters are packed into one `group` chunk
- a class over 120 lines becomes a `class_header` chunk (docstring, attributes, method list) plus one chunk per method, linked by `parent`
- a function over 120 lines is split between top-level statements, never mid-statement
- module-level code (imports, constants) becomes `module` chunks; every code line is covered
- embedded text is prefixed with `# file:` and `# method: Class.name`, so a method body that never names its class still matches

**BM25** (`bm25.py`). `rebuild_auth` is indexed as `rebuild_auth`, `rebuild`, `auth`, so both the exact identifier and its words match. Symbol-name tokens get extra weight so the definition outranks its callers. It's implemented directly (~40 lines of Okapi BM25) rather than via SQLite FTS5, so it doesn't depend on how Python's bundled SQLite was compiled.

**Reciprocal rank fusion.** `score = Σ 1/(60 + rank)` across retrievers. It uses ranks, not raw scores, so cosine (~0.7) and BM25 (~5–30) combine without calibration.

**Symbol lookup.** If the question contains something identifier-shaped (`snake_case`, `CamelCase`, `` `backticked` ``, or a single bare word) that names a definition, that chunk is pinned to rank 1. Fusion alone can demote an exact match that only BM25 found. Plain English words don't trigger it.

## Results on `requests` @ 611c616 (50 questions, recall@10)

Fill this in from `codecopilot report` after running `.\eval\run_phase2.ps1`:

| Chunks | Retrieval | recall@10 | MRR | identifier hit@1 |
|---|---|---|---|---|
| fixed | dense (Phase 1 baseline) | 0.860 | 0.609 | |
| fixed | hybrid | | | |
| ast | dense | | | |
| ast | bm25 | | | |
| ast | hybrid (Phase 2) | | | |

**Honest-measurement note:** a larger chunk contains more lines, so it makes line-containment recall easier. AST chunks are *smaller* (mean ~23 lines vs 60), so any recall gain they show is not from inflating chunk size. `eval` prints mean chunk size next to every result for this reason.

## Configuration

All settings live in `codecopilot/config.py` and can be overridden with `CC_*` env vars or a `.env` file:

| Variable | Default | Notes |
|---|---|---|
| `CC_CHUNKER` | `ast` | `fixed` reproduces Phase 1 |
| `CC_RETRIEVAL_MODE` | `hybrid` | `dense` reproduces Phase 1 |
| `CC_AST_MAX_LINES` / `CC_AST_MIN_LINES` | `120` / `6` | split threshold / tiny-sibling packing threshold |
| `CC_SYMBOL_PIN` | `true` | symbol lookup stage on/off (ablate it) |
| `CC_RRF_K` / `CC_FUSION_DEPTH` | `60` / `50` | fusion constant / candidates per retriever |
| `CC_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Try `jinaai/jina-embeddings-v2-base-code` for a code-tuned A/B |
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
  ingest.py     repo walk, Chunk model, fixed-window chunker (Phase 1)
  chunking.py   tree-sitter AST chunker (Phase 2)
  embed.py      Embedder interface: sentence-transformers | hashing stub
  index.py      flat cosine index on disk; content-addressed reuse of unchanged chunks
  bm25.py       code-aware tokenizer, BM25 index, reciprocal rank fusion
  llm.py        LLM client wrapper: retries + backoff, timeouts, streaming (Ollama / OpenAI-compatible)
  prompts/      versioned prompt files; prompts.py loads them by id and hashes them
  pipeline.py   retrieve (dense | bm25 | hybrid + symbol pin) → context budget → generate → citation check → run log
  evaluate.py   recall@k, MRR, hit@1 per question type
  cli.py
eval/requests_gold.jsonl   50 labelled questions (35 semantic, 5 why, 10 identifier)
eval/run_phase2.ps1        full fixed-vs-AST × dense/bm25/hybrid comparison
eval/results.jsonl         every saved eval run (commit this)
```

## The gold set

Each item is a question plus a `(path, line)` that must fall inside a retrieved chunk. Targets are **lines, not chunk IDs**, so the same file scores fixed and AST chunks fairly. Line numbers are only valid at commit `611c616`; `eval` refuses to run if they don't match the indexed repo.

50 items: 35 `semantic` (where/how), 5 `why` (answer depends on comments near the code), and 10 `identifier` (bare symbol names). Eval reports recall and hit@1 per type.

## Phase 2 exit criterion (from the guide)

- [ ] "Exact symbol queries always surface the defining chunk": identifier hit@1 = 1.00 with `ast` + `hybrid`.
