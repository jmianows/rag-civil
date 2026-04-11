#!/bin/bash
# deploy/mac_start.sh — First-time setup on Mac Mini.
# Assumes: Python 3.11, nginx, tesseract, poppler, Ollama + qwen3:8b already installed.
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_DEST="$HOME/Library/LaunchAgents/com.civilsmartdictionary.rag.plist"

echo "=== Civil RAG — Mac Mini Setup ==="

# Python venv + dependencies
python3.11 -m venv "$REPO_DIR/.venv"
source "$REPO_DIR/.venv/bin/activate"
pip install --upgrade pip -q
pip install -r "$REPO_DIR/requirements.txt" -q
echo "Dependencies installed."

# Install launchd plist (substitute REPO_DIR and HOME_DIR placeholders)
mkdir -p "$HOME/Library/LaunchAgents"
sed "s|REPO_DIR|$REPO_DIR|g; s|HOME_DIR|$HOME|g" \
    "$REPO_DIR/deploy/rag-civil.plist" > "$PLIST_DEST"
launchctl load "$PLIST_DEST"
echo "Service loaded: $PLIST_DEST"

# nginx config
NGINX_CONF="$(brew --prefix)/etc/nginx/servers/rag-civil.conf"
sed "s|REPO_DIR|$REPO_DIR|g" "$REPO_DIR/deploy/mac_nginx.conf" > "$NGINX_CONF"
brew services restart nginx
echo "nginx configured: $NGINX_CONF"

echo "Waiting for API..."
sleep 6
curl -s http://127.0.0.1:8000/health
echo ""
echo "=== Done. Run 'bash deploy/mac_update.sh' to update in future. ==="
