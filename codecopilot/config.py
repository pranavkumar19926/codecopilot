"""Typed configuration. Override any field via env vars prefixed CC_ (e.g. CC_LLM_MODEL=qwen2.5-coder:7b) or a .env file."""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CC_", env_file=".env", extra="ignore")

    # Ingestion
    include_ext: tuple[str, ...] = (".py", ".ipynb")   # notebooks are read as their code + markdown cells
    exclude_dirs: tuple[str, ...] = (
        ".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist", ".tox", ".mypy_cache",
    )
    max_file_bytes: int = 500_000
    max_notebook_bytes: int = 20_000_000   # raw .ipynb incl. outputs; its code text must still fit max_file_bytes
    # fnmatch globs on repo-relative posix paths, e.g. ("tests/*", "docs/*")
    exclude_globs: tuple[str, ...] = ()

    # Chunking. "ast" = Phase 2 tree-sitter chunks (Python); "fixed" = Phase 1 baseline line windows.
    # Non-Python files always fall back to fixed windows.
    chunker: Literal["ast", "fixed"] = "ast"
    chunk_lines: int = 60        # fixed windows
    chunk_overlap: int = 15      # fixed windows
    ast_max_lines: int = 120     # bigger classes are split into header + methods; bigger functions split by statement
    ast_min_lines: int = 6       # consecutive functions/methods shorter than this are packed into one group chunk

    # Embeddings
    embed_backend: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = 32

    # Retrieval. "hybrid" = dense + BM25 fused with reciprocal rank fusion (Phase 2); "dense" = Phase 1 baseline.
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    fusion_depth: int = 50       # candidates taken from each retriever before fusion
    rrf_k: int = 60              # RRF constant from Cormack et al. 2009; larger = flatter rank weighting
    bm25_symbol_boost: int = 3   # symbol-name tokens are counted this many extra times
    symbol_pin: bool = True      # identifier in the query that names a definition → that chunk ranks first
    symbol_pin_max: int = 3      # cap, e.g. `send` is defined in 3 classes
    top_k: int = 8

    # Phase 3, set from measurements (eval/run_phase3.ps1): rewriting helps on hard questions (MRR 0.46→0.61);
    # the MS MARCO cross-encoder hurts on code (std MRR 0.88→0.83), so reranking stays off.
    rerank: bool = False
    rerank_backend: Literal["cross-encoder", "stub"] = "cross-encoder"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # ~90 MB, CPU-friendly; try BAAI/bge-reranker-base
    rerank_depth: int = 30       # candidates the cross-encoder re-scores
    rerank_max_length: int = 512
    query_rewrite: bool = True
    rewrite_prompt_id: str = "rewrite_v1"
    context_token_budget: int = 6000

    # Phase 4: call graph + strict citations
    graph_expand: bool = True     # add callers/callees of the top hits to the context
    graph_seeds: int = 3          # how many top hits to expand from
    graph_max_extra: int = 4      # extra chunks added (doubled for "what calls X" questions)
    impact_depth: int = 3         # transitive depth for `codecopilot impact`
    strict_citations: bool = True # check every claim; one repair round if the check fails
    context_line_numbers: bool = True
    max_repairs: int = 1
    broad_lines: int = 40         # citations wider than this are reported as "broad"
    repair_prompt_id: str = "repair_v1"

    # Generation: Ollama (local) or any OpenAI-compatible endpoint (Groq free tier, etc.)
    llm_provider: Literal["ollama", "openai_compat"] = "ollama"
    llm_model: str = "qwen2.5-coder:7b"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_seed: int = 42             # fixed sampling seed (Ollama / OpenAI-compatible) for reproducible runs
    llm_num_ctx: int = 10240  # Ollama context window; must exceed context_token_budget + prompt + answer
    llm_timeout_s: float = 300.0  # CPU prompt processing of ~6k tokens can take a minute+
    llm_max_retries: int = 3
    llm_max_wait_s: float = 30.0   # longest single wait when the provider says "rate limited, retry after N s"
    llm_reasoning_effort: str = "low"   # gpt-oss models only: less hidden thinking = fewer tokens per minute

    prompt_id: str = "answer_v4"   # answer_v2 = Phase 1-3 prompt (no line numbers)

    # Website (`codecopilot serve`)
    web_dir: Path = Path(".cc_web")      # repos added from the website: clones + indexes
    web_allow_local: bool = True         # allow "add a local folder" (turned off when deployed)
    web_max_repo_mb: int = 60            # refuse bigger clones
    web_clone_timeout_s: int = 180
    groq_model: str = "openai/gpt-oss-120b"   # free tier; llama-3.3-70b-versatile isn't enabled on new accounts
    groq_context_budget: int = 2500           # free tier allows 8k tokens/minute: draft + repair must fit

    # Storage
    data_dir: Path = Path(".codecopilot")
    cache_dir: Path = Path(".cache")   # LLM response cache (rewrites), shared across indexes


settings = Settings()
