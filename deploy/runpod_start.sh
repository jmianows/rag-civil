#!/usr/bin/env bash
# RunPod startup script — RTX 4090
# Handles first-time setup and service startup.
# Run from the pod terminal after rsyncing the project.
set -e

cd /workspace/rag-civil

# ── System dependencies ────────────────────────────────────────────────────────

# Install system packages needed for Ollama install and rsync transfers.
# pciutils/lshw let the Ollama installer auto-detect the NVIDIA GPU.
PKGS_NEEDED=""
for pkg in rsync zstd pciutils lshw; do
    dpkg -s "$pkg" &>/dev/null || PKGS_NEEDED="$PKGS_NEEDED $pkg"
done
if [ -n "$PKGS_NEEDED" ]; then
    echo "==> Installing system packages:$PKGS_NEEDED"
    apt-get update -qq
    apt-get install -y $PKGS_NEEDED -qq
fi

# ── Install Ollama if missing ──────────────────────────────────────────────────

if ! command -v ollama &>/dev/null; then
    echo "==> Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# ── First-time setup ───────────────────────────────────────────────────────────

# Install Python deps if venv is missing or freshly created
if [ ! -f ".venv/bin/uvicorn" ]; then
    echo "==> Installing Python dependencies..."
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

# Force PyTorch to CUDA 12.4 build — the default pip install pulls CUDA 13
# which requires a newer driver than RunPod currently ships (12.8 max).
if .venv/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "==> PyTorch CUDA OK"
else
    echo "==> Fixing PyTorch CUDA version (installing cu124 build)..."
    .venv/bin/pip install "torch==2.6.0" \
        --index-url https://download.pytorch.org/whl/cu124 \
        --force-reinstall --quiet
fi

# ── Start Ollama ───────────────────────────────────────────────────────────────

if ! curl -sf http://127.0.0.1:11434/ &>/dev/null; then
    ollama serve &
    sleep 5
else
    echo "==> Ollama already running"
fi

ollama pull mxbai-embed-large
ollama pull qwen3:8b

# ── Start FastAPI ──────────────────────────────────────────────────────────────

CIVIL_ENV=production \
  .venv/bin/uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
