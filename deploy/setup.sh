#!/usr/bin/env bash
# deploy/setup.sh — run once on a fresh Ubuntu 22.04 node
# Usage: bash deploy/setup.sh
# Assumes: running as root or with sudo, single NVIDIA GPU present
set -euo pipefail

PROJECT_DIR=/home/ubuntu/rag-civil
PYTHON_VERSION=3.11.9
PYENV_DIR=/home/ubuntu/.pyenv

echo "==> [1/9] Updating apt and installing system dependencies"
apt-get update -y
apt-get install -y \
    nginx \
    tesseract-ocr \
    poppler-utils \
    git \
    curl \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libgl1 \
    libglib2.0-0

echo "==> [2/9] Installing pyenv and Python ${PYTHON_VERSION}"
if [ ! -d "$PYENV_DIR" ]; then
    sudo -u ubuntu bash -c 'curl https://pyenv.run | bash'
fi

# Add pyenv to ubuntu's profile if not already present
sudo -u ubuntu bash -c 'grep -q "pyenv" ~/.bashrc || cat >> ~/.bashrc <<'"'"'EOF'"'"'

# pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF'

# Install Python 3.11.9
sudo -u ubuntu bash -c "
    export PYENV_ROOT=\"$PYENV_DIR\"
    export PATH=\"$PYENV_DIR/bin:\$PATH\"
    eval \"\$(pyenv init -)\"
    pyenv install -s $PYTHON_VERSION
    pyenv global $PYTHON_VERSION
"

PYTHON_BIN="$PYENV_DIR/versions/$PYTHON_VERSION/bin/python3"
echo "    Python binary: $PYTHON_BIN"

echo "==> [3/9] Installing Ollama"
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "    Ollama already installed: $(ollama --version)"
fi

echo "==> [4/9] Pulling Ollama models (starting Ollama temporarily)"
# Start Ollama in background for pulls, stop after
OLLAMA_HOST=127.0.0.1:11434 ollama serve &
OLLAMA_PID=$!
sleep 5

# Wait until Ollama responds
for i in {1..20}; do
    curl -sf http://127.0.0.1:11434/ &>/dev/null && break
    echo "    Waiting for Ollama... ($i)"
    sleep 3
done

ollama pull qwen3:1.7b
ollama pull mxbai-embed-large

kill $OLLAMA_PID 2>/dev/null || true
wait $OLLAMA_PID 2>/dev/null || true
echo "    Models pulled successfully"

echo "==> [5/9] Fixing hardcoded paths in project files"
# Replace /home/justin with /home/ubuntu throughout all Python source files
find "$PROJECT_DIR" -name "*.py" -not -path "*/.venv/*" \
    -exec sed -i 's|/home/justin/rag-civil|/home/ubuntu/rag-civil|g' {} +
echo "    Path substitution complete"

echo "==> [6/9] Creating Python virtual environment and installing requirements"
sudo -u ubuntu bash -c "
    $PYTHON_BIN -m venv $PROJECT_DIR/.venv
    $PROJECT_DIR/.venv/bin/pip install --upgrade pip
    $PROJECT_DIR/.venv/bin/pip install -r $PROJECT_DIR/requirements.txt
"

echo "==> [7/9] Installing systemd service files"
# Template unit: enables ollama@<port>.service instances.
# To add a second Ollama instance later: systemctl enable --now ollama@11435
# and add http://localhost:11435 to OLLAMA_HOSTS in rag-civil-api.service.
cp "$PROJECT_DIR/deploy/ollama@.service" /etc/systemd/system/ollama@.service
cp "$PROJECT_DIR/deploy/api.service"     /etc/systemd/system/rag-civil-api.service

systemctl daemon-reload
systemctl enable ollama@11434
systemctl enable rag-civil-api

echo "==> [8/9] Configuring Nginx"
cp "$PROJECT_DIR/deploy/nginx.conf" /etc/nginx/sites-available/rag-civil
ln -sf /etc/nginx/sites-available/rag-civil /etc/nginx/sites-enabled/rag-civil
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl enable nginx

echo "==> [9/9] Starting all services"
systemctl start ollama@11434
sleep 5
systemctl start rag-civil-api
systemctl restart nginx

echo ""
echo "====================================================="
echo " Setup complete. Service status:"
echo "====================================================="
systemctl is-active --quiet ollama@11434  && echo "  ollama@11434: running" || echo "  ollama@11434: FAILED"
systemctl is-active --quiet rag-civil-api && echo "  rag-civil:    running" || echo "  rag-civil:    FAILED"
systemctl is-active --quiet nginx         && echo "  nginx:        running" || echo "  nginx:        FAILED"
echo ""
echo " Frontend: http://$(hostname -I | awk '{print $1}')"
echo " API:      http://$(hostname -I | awk '{print $1}'):8000/filters"
echo "====================================================="
