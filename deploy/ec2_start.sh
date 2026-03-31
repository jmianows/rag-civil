#!/usr/bin/env bash
# deploy/ec2_start.sh — first-time setup and service startup for AWS EC2 g4dn.xlarge (T4 GPU)
# Run on the instance after ec2_transfer.sh.
# Safe to re-run — skips steps that are already complete.
set -e

REMOTE_DIR=/home/ubuntu/rag-civil
DOMAIN="api.civilsmartdictionary.com"
CERT_EMAIL="jmianows@umich.edu"
cd "$REMOTE_DIR"

# ── System packages + NVIDIA drivers ──────────────────────────────────────────
PKGS_NEEDED=""
for pkg in rsync zstd pciutils lshw python3.12-venv nvidia-driver-570 nginx certbot python3-certbot-nginx; do
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

# ── uvicorn systemd service ────────────────────────────────────────────────────
# Kill any bare uvicorn process started by a previous run of this script
pkill -f "uvicorn api.main:app" 2>/dev/null || true
sleep 1

sudo tee /etc/systemd/system/rag-civil.service > /dev/null <<EOF
[Unit]
Description=Civil RAG API (uvicorn)
After=network.target ollama.service
Wants=ollama.service

[Service]
User=ubuntu
WorkingDirectory=${REMOTE_DIR}
Environment="CIVIL_ENV=production"
Environment="CIVIL_API_URL=https://${DOMAIN}"
ExecStart=${REMOTE_DIR}/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rag-civil
sudo systemctl restart rag-civil

echo "==> Waiting for API to be ready..."
for i in $(seq 1 90); do
    curl -sf http://127.0.0.1:8000/health &>/dev/null && echo "==> API ready (${i}s)" && break
    [ "$i" -eq 90 ] && echo "ERROR: API did not become ready within 90s" && sudo journalctl -u rag-civil -n 30 && exit 1
    sleep 1
done

# ── nginx reverse proxy ────────────────────────────────────────────────────────
# Always rewrite config so timeout changes take effect on AMI-based boots
echo "==> Configuring nginx for ${DOMAIN}..."
sudo tee /etc/nginx/sites-available/${DOMAIN} > /dev/null <<NGINXEOF
server {
    listen 80;
    server_name ${DOMAIN};

    proxy_buffering off;
    proxy_read_timeout 300s;
    proxy_connect_timeout 10s;
    proxy_set_header Host \$host;
    proxy_set_header X-Real-IP \$remote_addr;
    proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto \$scheme;

    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
NGINXEOF
sudo ln -sf /etc/nginx/sites-available/${DOMAIN} /etc/nginx/sites-enabled/${DOMAIN}
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# ── SSL cert ───────────────────────────────────────────────────────────────────
if [ ! -f "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" ]; then
    echo "==> Obtaining Let's Encrypt certificate for ${DOMAIN}..."
    sudo certbot --nginx \
        -d "${DOMAIN}" \
        --non-interactive --agree-tos \
        --email "${CERT_EMAIL}" \
        --redirect
fi

sudo systemctl enable nginx
sudo systemctl reload nginx

echo ""
echo "====================================================="
echo " DEPLOYMENT COMPLETE"
echo " API:    https://${DOMAIN}"
echo " Health: https://${DOMAIN}/health"
echo " Logs:   sudo journalctl -u rag-civil -f"
echo "====================================================="
