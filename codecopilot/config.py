"""Typed configuration. Override any field via env vars prefixed CC_ (e.g. CC_LLM_MODEL=qwen2.5-coder:7b) or a .env file."""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CC_", env_file=".env", extra="ignore")

    # Ingestion
    include_ext: tuple[str, ...] = (".py",)
    exclude_dirs: tuple[str, ...] = (
        ".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist", ".tox", ".mypy_cache",
    )
    max_file_bytes: int = 500_000
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
    context_token_budget: int = 6000

    # Generation: Ollama (local) or any OpenAI-compatible endpoint (Groq free tier, etc.)
    llm_provider: Literal["ollama", "openai_compat"] = "ollama"
    llm_model: str = "qwen2.5-coder:7b"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192  # Ollama context window; must exceed context_token_budget + prompt + answer
    llm_timeout_s: float = 300.0  # CPU prompt processing of ~6k tokens can take a minute+
    llm_max_retries: int = 3

    prompt_id: str = "answer_v2"

    # Storage
    data_dir: Path = Path(".codecopilot")


settings = Settings()
