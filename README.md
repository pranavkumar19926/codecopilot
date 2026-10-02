# Codebase Intelligence Copilot

Ask questions about a repository; get answers with `path:start-end` citations to the actual code.

| Phase | What | Status |
|---|---|---|
| 1 | Fixed 60-line chunks → dense embeddings → local LLM answer with citations; 50-question gold set | done |
| 2 | Tree-sitter AST chunks + BM25 keyword search + reciprocal rank fusion + symbol lookup | done |
| 3 | Hard gold set, cross-encoder reranking, LLM query rewriting, response cache, latency | done |
| 4 | Call graph (callers / callees / impact), graph expansion, strict citations with repair | done |
| 5 | Website: chat with clickable citations, code viewer, callers/callees panel, add repos from GitHub | **current** |
| — | Free deployment (Hugging Face Spaces + Groq) | next |

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

Full Phase 4 run: `powershell -ExecutionPolicy Bypass -File .\eval\run_phase4.ps1`.

Phase 3 flags on `search`, `ask`, `eval`: `--rerank/--no-rerank`, `--rewrite/--no-rewrite`. Rewriting is on by default and needs Ollama running; add `--no-rewrite` to search without it.

## Website

```bash
codecopilot serve                    # http://127.0.0.1:8000, answers by Ollama (local)
$env:GROQ_API_KEY="gsk_..."          # Windows PowerShell; free key at console.groq.com/keys
codecopilot serve --llm groq         # answers by llama-3.3-70b-versatile on Groq, in seconds
```

* **Repositories:** every `.cc_*` index in the current folder shows up automatically. Add more by pasting a
  public GitHub URL (cloned shallowly, size-capped) or a local folder path; indexing runs in the background.
* **Chat:** answers stream in. Citations are highlighted; clicking one opens the file in the code viewer with
  the same lines highlighted. If the strict-citation check fails, the first draft is kept (collapsed, with
  the problems listed) and the revision streams in below it.
* **Callers and callees:** click any `function` name in an answer, or look one up, to see who calls it,
  what it calls, and how many definitions a change could affect.
* **Search only:** retrieval without a written answer; takes about a second.

Backend: FastAPI (`web.py`); answers stream as NDJSON events (`sources` → `token`… → `repair`? → `done`).
Frontend: one HTML page with vanilla JS (`static/`), no build step. API docs at `/api/docs`.
Safety: GitHub URLs only for remote repos, clone size and time limits, file viewer confined to the repo
folder, one generation and one indexing job at a time.

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

## Phase 4: call graph and strict citations

Retrieval finds code by **content**. "What calls `rebuild_auth`?" is about a **relationship**: the caller
(`resolve_redirects`) never mentions auth or headers, so no similarity search will rank it. Phase 4 adds a static
call graph and uses it twice: as commands, and to widen the context given to the LLM.

### The call graph (`graph.py`)
Built with tree-sitter at index time (`graph.json` next to the index). Nodes are every function, method and class
(including methods of small classes that share one chunk). Edges:

| edge | example |
|---|---|
| call | `self.rebuild_auth(...)` inside `resolve_redirects` |
| decorator | `@admin_required` on `dashboard` → `dashboard` depends on `admin_required` |
| instantiate | `Session()` → the `Session` class |
| inherit | `class Session(SessionRedirectMixin)` |

Python is dynamically typed, so resolution is **conservative**: `self.x` → the enclosing class or its bases;
bare `x()` → same file, then imports, then a unique top-level definition; `Name.x` → class `Name`; any other
`obj.x()` only if exactly one definition has that name and it isn't a generic name like `get` / `send`.
Missing an edge is preferred over inventing one. On `requests`: 320 definitions, 259 edges (174 calls,
48 instantiations, 37 inheritance), 109 ambiguous references dropped.

```bash
codecopilot callers get_auth_from_url        # who calls it, and on which line
codecopilot callees Session.send             # what it calls, and where those are defined
codecopilot impact should_strip_auth         # transitive callers = what could break if it changes
```

### Graph expansion in `ask`
After retrieval, the top hits' 1-hop neighbours are added to the LLM context (marked "related"), steered by
the question: *what calls / uses / breaks* → callers, *what does X call* → callees, anything else → both
(callees weighted higher). Neighbours that retrieval also ranked well win ties. This is off for `eval` on the
old gold sets, so Phase 2–3 numbers are unchanged.

### Strict citations (`citations.py`)
1. Context lines are numbered (`  216| raise TooManyRedirects(`), so the model can cite exact lines.
2. Prompt `answer_v3` requires a narrow `[path:start-end]` on every factual sentence.
3. Every sentence is checked: **invalid** (range not shown to the model), **unsupported** (a `name` in the
   sentence doesn't occur in the cited lines), **uncited** (names code, cites nothing), **broad** (>40 lines).
4. If the check fails, the problems are sent back to the model **once** and the revision is kept only if it
   scores better. `ask` shows the draft, the problems and the revision.

### Evaluation
* `eval/requests_gold_trace.jsonl`: 25 questions with 49 required locations (all callers of X, or what X
  calls). The answer key was built with `grep`, **not** with the graph. Metric: share of required locations in
  the context. The comparison is fair: the no-graph baseline gets the same number of extra chunks from plain retrieval.
* `codecopilot eval-answers`: script-checkable answer metrics, with no LLM judge: gold line cited, valid-citation rate,
  claims cited, unsupported names per answer, mean citation width.

### Results (requests @ 611c616, bge-small + qwen2.5-coder:7b on CPU)

| Trace set (25 q, 49 required locations) | target recall | all targets found |
|---|---|---|
| top-8 + 2.6 more chunks from retrieval, no graph | 0.777 | 0.680 |
| **top-8 + 2.6 chunks from graph expansion** | **1.000** | **1.000** |

At the same context size, the graph found every caller and callee for every question. Callee questions gain the most
(0.50 → 1.00): "what does X call" is invisible to similarity search.

| Answers (6 q, standard set) | gold line cited | valid citations | claims cited | unsupported / answer | mean width |
|---|---|---|---|---|---|
| plain (Phase 3 prompt) | 0.833 | 1.000 | 0.583 | 0.33 | 34.6 lines |
| **strict (Phase 4)** | **1.000** | 1.000 | **1.000** | 0.33 | **16.4 lines** |

Strict mode cites every claim, with ranges half as wide; the repair round fired on 3/6 answers. Unsupported names did
not improve. n=6 is small: this shows direction, not a precise effect size.

**Regression finding: LLM non-determinism.** The Phase 4 rerun of the old gold sets came out slightly lower
(standard MRR 0.886 → 0.857, hard 0.606 → 0.581; 1–2 questions each). Phase 4 retrieval code didn't cause it:
raising the answer model's context window changed the cache key, so query rewrites were regenerated at temperature
0.1 and came out differently. Rewrites now use temperature 0, a fixed seed and their own fixed context size, so
they're reproducible and independent of answer settings. Takeaway: the rewriting gain on the hard set is real
(MRR 0.46 → ~0.58–0.61), but single-run figures with a sampling LLM carry about ±1–2 questions of noise.

**Deterministic reference numbers** (temperature 0, seed 42; two consecutive runs gave identical results; the
cached rerun took 149 ms p50 vs 2357 ms):

| Gold set | recall@10 | MRR | hit@1 |
|---|---|---|---|
| standard (50) | 1.000 | 0.865 | 0.780 |
| hard (24) | 0.958 | 0.592 | 0.375 |

Against Phase 2 on the hard set (MRR 0.458, hit@1 0.292), rewriting still adds +0.13 MRR.

**Limits:** static analysis can't follow dynamic dispatch (`adapter.send(...)` on an unknown type), callbacks
passed as values, or `getattr`. The identifier check proves a name is *present* in the cited lines, which is
necessary for support but doesn't prove the claim is true.

## Configuration

All settings live in `codecopilot/config.py` and can be overridden with `CC_*` env vars or a `.env` file:

| Variable | Default | Notes |
|---|---|---|
| `CC_CHUNKER` | `ast` | `fixed` reproduces Phase 1 |
| `CC_RETRIEVAL_MODE` | `hybrid` | `dense` reproduces Phase 1 |
| `CC_AST_MAX_LINES` / `CC_AST_MIN_LINES` | `120` / `6` | split threshold / tiny-sibling packing threshold |
| `CC_SYMBOL_PIN` | `true` | symbol lookup stage on/off (ablate it) |
| `CC_RRF_K` / `CC_FUSION_DEPTH` | `60` / `50` | fusion constant / candidates per retriever |
| `CC_GRAPH_EXPAND` / `CC_GRAPH_MAX_EXTRA` | `true` / `4` | call-graph context expansion (doubled for caller/callee questions) |
| `CC_STRICT_CITATIONS` / `CC_MAX_REPAIRS` | `true` / `1` | claim-level citation check + repair rounds |
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
  graph.py      tree-sitter call graph: definitions, call/decorator/instantiate/inherit edges, impact
  citations.py  sentence-level citation checker (invalid / unsupported / uncited / broad)
  indexer.py    builds vector index + BM25 + call graph (shared by CLI and website)
  web.py        FastAPI app: repos, streaming ask, search, file viewer, graph API
  static/       the website (index.html, style.css, app.js)
  pipeline.py   retrieve ([rewrite] → dense/bm25 → RRF → symbol pin → [rerank]) → context → generate → citations → run log
  evaluate.py   recall@k, MRR, hit@1 per question type
  cli.py
eval/requests_gold.jsonl   50 labelled questions (35 semantic, 5 why, 10 identifier)
eval/requests_gold_hard.jsonl  24 hard questions, no identifiers (paraphrase / behavior / why)
eval/run_phase2.ps1        full fixed-vs-AST × dense/bm25/hybrid comparison
eval/run_phase3.ps1        rerank × rewrite comparison on both gold sets
eval/requests_gold_trace.jsonl 25 caller/callee questions, 49 grep-verified target lines
eval/run_phase4.ps1        graph + strict-citation evaluation
eval/results.jsonl         every saved eval run (commit this)
```

## The gold set

Each item is a question plus a `(path, line)` that must fall inside a retrieved chunk. Targets are **lines, not chunk IDs**, so the same file scores fixed and AST chunks fairly. Line numbers are only valid at commit `611c616`; `eval` refuses to run if they don't match the indexed repo.

50 items: 35 `semantic` (where/how), 5 `why` (answer depends on comments near the code), and 10 `identifier` (bare symbol names). Eval reports recall and hit@1 per type.

## Phase 2 exit criterion (from the guide)

- [x] "Exact symbol queries always surface the defining chunk": identifier hit@1 = 1.00 (10/10) with `ast` + `hybrid`.
