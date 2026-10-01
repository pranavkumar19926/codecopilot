# Codebase Intelligence Copilot

Ask questions about a repository; get answers with `path:start-end` citations to the actual code.

| Phase | What | Status |
|---|---|---|
| 1 | Fixed 60-line chunks → dense embeddings → local LLM answer with citations; 50-question gold set | done |
| 2 | Tree-sitter AST chunks + BM25 keyword search + reciprocal rank fusion + symbol lookup | done |
| 3 | Hard gold set, cross-encoder reranking, LLM query rewriting, response cache, latency | done |
| 4 | Call-graph expansion, strict citation enforcement | **next** |

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
Full Phase 3 comparison: `powershell -ExecutionPolicy Bypass -File .\eval\run_phase3.ps1` (needs Ollama running).

Phase 3 flags on `search`, `ask`, `eval`: `--rerank/--no-rerank`, `--rewrite/--no-rewrite`. Rewriting is on by default and needs Ollama running; add `--no-rewrite` to search without it.

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

## Results on `requests` @ 611c616 (50 questions, k=10, bge-small, tests/ and docs/ excluded)

| Chunks | Retrieval | recall@10 | MRR | hit@1 | identifier hit@1 |
|---|---|---|---|---|---|
| fixed (60/15) | dense — **Phase 1 baseline** | 0.860 | 0.609 | 0.480 | 0.70 |
| fixed | bm25 | 0.960 | 0.650 | 0.500 | 0.40 |
| fixed | hybrid | 1.000 | 0.715 | 0.600 | 0.70 |
| AST | dense | 1.000 | 0.895 | 0.820 | 0.90 |
| AST | bm25 | 0.940 | 0.764 | 0.680 | 1.00 |
| AST | hybrid + symbol lookup — **Phase 2 default** | 0.980 | 0.882 | 0.820 | **1.00** |

**What this shows**
- AST chunking is the main win: with the same embedding model, MRR 0.609 → 0.895 and hit@1 0.48 → 0.82, while chunks got *smaller* (mean 56.5 → 23.2 lines), so the gain isn't from bigger chunks containing more lines.
- Hybrid + symbol lookup is what puts exact identifier queries at rank 1 (10/10). On plain-English questions it is on par with AST + dense.
- AST + dense vs AST + hybrid differ by one question (0.02). That is noise at n=50, not a result.
- Recall@10 is now saturated, so this gold set can no longer rank further changes. Phase 3 (reranking) needs harder questions: vaguer wording, multi-hop, and hit@1/MRR as the headline metric.

Reproduce: `.\eval\run_phase2.ps1` then `codecopilot report`.

## Phase 3: reranking and query rewriting

```
question ─→ [LLM rewrite → 3 code-vocabulary queries] ─→ hybrid retrieval per query ─→ RRF (original ×2)
         ─→ symbol pins ─→ [cross-encoder re-scores top 30] ─→ top-k
```

**Why a harder gold set first.** Phase 2 saturated the original 50 questions (recall@10 0.98), so they can't tell
whether anything helps any more. `eval/requests_gold_hard.jsonl` adds 24 questions written the way a user
asks, with no function names: *"What error do I get if I forget the http:// part of the address?"* (answer:
`raise MissingSchema(` in `prepare_url`). Types: `paraphrase`, `behavior`, `why`. BM25 alone drops from
0.94 recall to 0.58 on it, so there is room to measure.

**Cross-encoder reranking** (`rerank.py`). A bi-encoder embeds question and chunk separately, so the two never
interact. A cross-encoder reads `[question] [SEP] [chunk]` as one sequence, so every question token attends to
every code token. It's more precise, but it needs one forward pass per pair and nothing can be precomputed,
so it only re-scores the top 30. Default model `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB). It was trained on
web search, not code, which is exactly what the eval should reveal. Symbol-pinned chunks stay on top.

**Query rewriting** (`rewrite.py`, prompt `prompts/rewrite_v1.md`). The local LLM turns the question into 3
queries in the code's own vocabulary (restatement, likely identifiers, a description of the code). Each is
retrieved separately and the lists are fused with RRF, with the original question weighted ×2 so a bad rewrite can add
candidates but can't displace what the original found. Rewrites are shown by `search`/`ask` and logged.

**Response cache** (`llm.py`). Non-streaming LLM calls are cached on disk under `.cache/`, keyed by provider,
model, temperature, context size and the full messages. The first rewrite eval is slow (one LLM call per
question); reruns are free and deterministic.

**Latency.** Every eval now reports p50/p95 retrieval latency, because reranking and rewriting cost time and
the trade-off has to be visible.

### Results (AST chunks + hybrid; reranker = ms-marco-MiniLM-L-6-v2; rewriter = qwen2.5-coder:7b; CPU)

| Gold set | Stages | recall@10 | MRR | hit@1 | p50 latency |
|---|---|---|---|---|---|
| hard (24) | none (Phase 2) | 0.875 | 0.458 | 0.292 | 20 ms |
| hard | rerank | 0.792 | 0.463 | 0.333 | 1.5 s |
| hard | **rewrite — Phase 3 default** | **0.958** | **0.606** | **0.417** | 2.1 s |
| hard | rewrite + rerank | 0.833 | 0.475 | 0.333 | 1.6 s |
| standard (50) | none (Phase 2) | 0.980 | 0.882 | 0.820 | 21 ms |
| standard | rerank | 0.980 | 0.829 | 0.740 | 1.6 s |
| standard | rewrite | 1.000 | 0.886 | 0.820 | 1.9 s |
| standard | rewrite + rerank | 0.980 | 0.829 | 0.740 | 1.8 s |

**Decisions, from the numbers**
- **Query rewriting: on.** On questions phrased in user vocabulary it lifts MRR 0.458 → 0.606 and hit@1
  7/24 → 10/24, and it is neutral on the standard set. The cost is one local-LLM call (~2 s on CPU), small
  next to answer generation.
- **Cross-encoder reranking: off.** MS MARCO MiniLM was trained on web passages, not code. It lowers standard
  MRR (0.882 → 0.829) and recall on the hard set, and undoes most of the rewriting gain when combined. That's a
  domain-mismatch result: the reranking *stage* is in place and measured, and the off-the-shelf *model* is the
  wrong one. Next candidates: a code-aware reranker, or fine-tuning a cross-encoder on (question, chunk) pairs.
- **Sample size.** 24 questions is small: recall 0.875 → 0.958 is 2 questions. The MRR and hit@1 shifts are
  the more meaningful signal. More hard questions would tighten this.

A/B a stronger reranker: `$env:CC_RERANK_MODEL="BAAI/bge-reranker-base"` (~1.1 GB, slower on CPU), then re-run
the rerank evals with a new `--save` label.

## Configuration

All settings live in `codecopilot/config.py` and can be overridden with `CC_*` env vars or a `.env` file:

| Variable | Default | Notes |
|---|---|---|
| `CC_CHUNKER` | `ast` | `fixed` reproduces Phase 1 |
| `CC_RETRIEVAL_MODE` | `hybrid` | `dense` reproduces Phase 1 |
| `CC_AST_MAX_LINES` / `CC_AST_MIN_LINES` | `120` / `6` | split threshold / tiny-sibling packing threshold |
| `CC_SYMBOL_PIN` | `true` | symbol lookup stage on/off (ablate it) |
| `CC_RRF_K` / `CC_FUSION_DEPTH` | `60` / `50` | fusion constant / candidates per retriever |
| `CC_RERANK` / `CC_QUERY_REWRITE` | `false` / `true` | Phase 3 stages, set from the results above |
| `CC_RERANK_MODEL` / `CC_RERANK_DEPTH` | MiniLM-L6 / `30` | cross-encoder and how many candidates it re-scores |
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
  llm.py        LLM client wrapper: retries + backoff, timeouts, streaming, on-disk response cache
  prompts/      versioned prompt files; prompts.py loads them by id and hashes them
  rerank.py     cross-encoder reranker (+ offline stub for tests)
  rewrite.py    LLM query rewriting → multi-query retrieval
  pipeline.py   retrieve ([rewrite] → dense/bm25 → RRF → symbol pin → [rerank]) → context → generate → citations → run log
  evaluate.py   recall@k, MRR, hit@1 per question type
  cli.py
eval/requests_gold.jsonl   50 labelled questions (35 semantic, 5 why, 10 identifier)
eval/requests_gold_hard.jsonl  24 hard questions, no identifiers (paraphrase / behavior / why)
eval/run_phase2.ps1        full fixed-vs-AST × dense/bm25/hybrid comparison
eval/run_phase3.ps1        rerank × rewrite comparison on both gold sets
eval/results.jsonl         every saved eval run (commit this)
```

## The gold set

Each item is a question plus a `(path, line)` that must fall inside a retrieved chunk. Targets are **lines, not chunk IDs**, so the same file scores fixed and AST chunks fairly. Line numbers are only valid at commit `611c616`; `eval` refuses to run if they don't match the indexed repo.

50 items: 35 `semantic` (where/how), 5 `why` (answer depends on comments near the code), and 10 `identifier` (bare symbol names). Eval reports recall and hit@1 per type.

## Phase 2 exit criterion (from the guide)

- [x] "Exact symbol queries always surface the defining chunk": identifier hit@1 = 1.00 (10/10) with `ast` + `hybrid`.
