#!/usr/bin/env bash
# deploy/ec2_start.sh — first-time setup and service startup for AWS EC2 g4dn.xlarge (T4 GPU)
# Run on the instance after ec2_transfer.sh.
# Safe to re-run — skips steps that are already complete.
set -e

REMOTE_DIR=/home/ubuntu/rag-civil
cd "$REMOTE_DIR"

# ── API URL ────────────────────────────────────────────────────────────────────
# Set CIVIL_API_URL before running if you want to override.
# Default: bare IP on port 8000. Update to domain once DNS/nginx are configured.
if [ -z "${CIVIL_API_URL:-}" ]; then
    PUBLIC_IP=$(curl -sf --connect-timeout 3 http://169.254.169.254/latest/meta-data/public-ipv4 || echo "")
    if [ -n "$PUBLIC_IP" ]; then
        export CIVIL_API_URL="http://${PUBLIC_IP}:8000"
    fi
fi
echo "==> API URL: ${CIVIL_API_URL:-not set}"

# ── System packages + NVIDIA drivers ──────────────────────────────────────────
PKGS_NEEDED=""
for pkg in rsync zstd pciutils lshw python3.12-venv nvidia-driver-570; do
    dpkg -s "$pkg" &>/dev/null || PKGS_NEEDED="$PKGS_NEEDED $pkg"
done
if [ -n "$PKGS_NEEDED" ]; then
    echo "==> Installing system packages:$PKGS_NEEDED"
    sudo apt-get update -qq
    sudo apt-get install -y $PKGS_NEEDED -qq
fi

# Ensure NVIDIA kernel module is loaded BEFORE starting Ollama
if ! nvidia-smi &>/dev/null; then
    echo "==> Loading NVIDIA kernel module..."
    sudo modprobe nvidia || true
fi

if nvidia-smi &>/dev/null; then
    echo "==> NVIDIA driver OK: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    echo "WARNING: nvidia-smi still failing — reboot may be required (sudo reboot)"
fi

# ── Ollama ─────────────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    echo "==> Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Always restart Ollama via systemd so it picks up the GPU
echo "==> Starting Ollama (GPU-aware)..."
sudo systemctl restart ollama
for i in $(seq 1 30); do
    curl -sf --connect-timeout 2 http://127.0.0.1:11434/ &>/dev/null && echo "==> Ollama ready (${i}s)" && break
    [ "$i" -eq 30 ] && echo "ERROR: Ollama did not start within 30s" && exit 1
    sleep 1
done

ollama pull mxbai-embed-large
ollama pull qwen3:8b

# ── Python venv ────────────────────────────────────────────────────────────────
if [ ! -f ".venv/bin/uvicorn" ]; then
    echo "==> Installing Python dependencies..."
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt -q
fi

# T4 GPU uses CUDA 12.x — ensure PyTorch CUDA build is correct
if .venv/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "==> PyTorch CUDA OK"
else
    echo "==> Fixing PyTorch CUDA version (installing cu124 build)..."
    .venv/bin/pip install "torch==2.6.0" \
        --index-url https://download.pytorch.org/whl/cu124 \
        --force-reinstall --quiet
fi

# ── FastAPI ────────────────────────────────────────────────────────────────────
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

# Restart uvicorn automatically if it exits
while true; do
    wait "$UVICORN_PID" 2>/dev/null || true
    echo "==> uvicorn exited, restarting in 3s..."
    sleep 3
    CIVIL_ENV=production CIVIL_API_URL="${CIVIL_API_URL:-}" .venv/bin/uvicorn api.main:app \
        --host 0.0.0.0 --port 8000 --workers 1 &
    UVICORN_PID=$!
done
