#!/usr/bin/env bash
# RunPod startup script — RTX 4090
# Handles first-time setup and service startup.
# Run from the pod terminal after rsyncing the project.
set -e

cd /workspace/rag-civil

# ── Pod identity ───────────────────────────────────────────────────────────────
# RunPod sets RUNPOD_POD_ID in the environment. Use it to build the proxy URL
# so CIVIL_API_URL is always correct without manual edits after pod replacement.
if [ -z "${CIVIL_API_URL:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    export CIVIL_API_URL="https://${RUNPOD_POD_ID}-8000.proxy.runpod.net"
    echo "==> API URL: ${CIVIL_API_URL}"
fi

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

if ! curl -sf --connect-timeout 2 http://127.0.0.1:11434/ &>/dev/null; then
    ollama serve &
    echo "==> Waiting for Ollama to be ready..."
    for i in $(seq 1 60); do
        curl -sf --connect-timeout 2 http://127.0.0.1:11434/ &>/dev/null && echo "==> Ollama ready (${i}s)" && break
        [ "$i" -eq 60 ] && echo "ERROR: Ollama did not start within 60s" && exit 1
        sleep 1
    done
else
    echo "==> Ollama already running"
fi

ollama pull mxbai-embed-large
ollama pull qwen3:8b

# ── Start FastAPI ──────────────────────────────────────────────────────────────

# Start uvicorn in background, wait for /health, then keep it alive with restart loop.
echo "==> Starting FastAPI..."
CIVIL_ENV=production CIVIL_API_URL="${CIVIL_API_URL:-}" .venv/bin/uvicorn api.main:app \
    --host 0.0.0.0 --port 8000 --workers 1 &
UVICORN_PID=$!

echo "==> Waiting for API to be ready..."
for i in $(seq 1 90); do
    curl -sf http://127.0.0.1:8000/health &>/dev/null && echo "==> API ready (${i}s)" && break
    [ "$i" -eq 90 ] && echo "ERROR: API did not become ready within 90s" && kill "$UVICORN_PID" && exit 1
    sleep 1
done
echo "===== DEPLOYMENT COMPLETE — API IS LIVE ====="

# Wait for uvicorn; restart automatically if it exits
while true; do
    wait "$UVICORN_PID" 2>/dev/null || true
    echo "==> uvicorn exited, restarting in 3s..."
    sleep 3
    CIVIL_ENV=production CIVIL_API_URL="${CIVIL_API_URL:-}" .venv/bin/uvicorn api.main:app \
        --host 0.0.0.0 --port 8000 --workers 1 &
    UVICORN_PID=$!
done
