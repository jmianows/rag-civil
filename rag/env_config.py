"""
Environment detection and per-environment configuration.

Priority (highest to lowest):
  1. CIVIL_ENV environment variable — "production" / "local"
  2. Default: "local"

Usage:
    from rag.env_config import IS_PRODUCTION, OLLAMA_KEEP_ALIVE, RERANKER_DEVICE
"""

import os
from pathlib import Path


def detect_environment() -> str:
    """Return 'production' or 'local' based on CIVIL_ENV env var."""
    val = os.environ.get("CIVIL_ENV", "").strip().lower()
    if val in ("production", "prod"):
        return "production"
    return "local"


# ── Resolved at import time ────────────────────────────────────────────────────

ENVIRONMENT   = detect_environment()
IS_PRODUCTION = ENVIRONMENT == "production"

# Ollama keep_alive: -1 = never unload model; 300 = 5-minute idle timeout
OLLAMA_KEEP_ALIVE = -1 if IS_PRODUCTION else 300

# Cross-encoder reranker device: override via CIVIL_RERANKER_DEVICE (mps/cuda/cpu)
RERANKER_DEVICE = os.environ.get("CIVIL_RERANKER_DEVICE", "cpu")

# LLM model: 8b on production, 4b locally
LLM_MODEL = "qwen3:8b" if IS_PRODUCTION else "qwen3:4b-instruct"

# Whether to ping Ollama and LanceDB at startup to load them before first request
WARM_ON_STARTUP = IS_PRODUCTION

# Vector database path — override via CIVIL_VECTORDB_DIR env var
_default_vectordb = Path(__file__).parent.parent / "vectordb"
VECTORDB_DIR = Path(os.environ.get("CIVIL_VECTORDB_DIR", str(_default_vectordb)))

# Minimum cross-encoder rerank score to accept a result
RERANK_FLOOR = float(os.environ.get("RERANK_FLOOR", "1.0"))

print(
    f"[env] {ENVIRONMENT.upper()} | model={LLM_MODEL}"
    f" | keep_alive={OLLAMA_KEEP_ALIVE}s"
    f" | reranker={RERANKER_DEVICE} | warm_startup={WARM_ON_STARTUP}",
    flush=True,
)
