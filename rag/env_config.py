"""
Environment detection and per-environment configuration.

Priority (highest to lowest):
  1. CIVIL_ENV environment variable — "production" / "local"
  2. AWS EC2 instance metadata endpoint — auto-detected via link-local IP
  3. Default: "local"

Usage:
    from rag.env_config import IS_PRODUCTION, OLLAMA_KEEP_ALIVE, RERANKER_DEVICE
"""

import os
from pathlib import Path


def _is_aws_ec2() -> bool:
    """Return True if the process is running on an AWS EC2 instance.

    Uses the instance metadata endpoint (169.254.169.254) which is only
    reachable from within EC2. Times out in 300ms so local startup is
    not visibly delayed.
    """
    try:
        import urllib.request
        with urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/instance-id",
            timeout=0.3,
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def detect_environment() -> str:
    """Return 'production' or 'local'.

    Checks CIVIL_ENV first for an explicit override, then falls back to
    AWS instance metadata detection.
    """
    val = os.environ.get("CIVIL_ENV", "").strip().lower()
    if val in ("production", "prod"):
        return "production"
    if val in ("local", "dev", "development"):
        return "local"
    return "production" if _is_aws_ec2() else "local"


# ── Resolved at import time ────────────────────────────────────────────────────

ENVIRONMENT   = detect_environment()
IS_PRODUCTION = ENVIRONMENT == "production"

# Ollama keep_alive: -1 = never unload model from VRAM; 300 = 5-minute idle timeout
OLLAMA_KEEP_ALIVE = -1 if IS_PRODUCTION else 300

# Cross-encoder reranker device: GPU on production, CPU locally
RERANKER_DEVICE = "cuda" if IS_PRODUCTION else "cpu"

# LLM model: 8b on production GPU (T4 16GB), 4b locally
LLM_MODEL = "qwen3:8b" if IS_PRODUCTION else "qwen3:4b-instruct"

# Whether to ping Ollama and LanceDB at server startup to load them before
# the first real user request arrives
WARM_ON_STARTUP = IS_PRODUCTION

# Vector database path — override via CIVIL_VECTORDB_DIR env var for non-default layouts
_default_vectordb = Path(__file__).parent.parent / "vectordb"
VECTORDB_DIR = Path(os.environ.get("CIVIL_VECTORDB_DIR", str(_default_vectordb)))

print(
    f"[env] {ENVIRONMENT.upper()} | model={LLM_MODEL}"
    f" | keep_alive={OLLAMA_KEEP_ALIVE}s"
    f" | reranker={RERANKER_DEVICE} | warm_startup={WARM_ON_STARTUP}",
    flush=True,
)
