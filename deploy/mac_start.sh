#!/bin/bash
# deploy/mac_start.sh — First-time setup on Mac Mini.
# Assumes: Python 3.11, nginx, tesseract, poppler, Ollama already installed.
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_DEST="$HOME/Library/LaunchAgents/com.civilsmartdictionary.rag.plist"

echo "=== Civil RAG — Mac Mini Setup ==="

# Ollama models
echo "Pulling Ollama models..."
ollama pull qwen3:8b
ollama pull mxbai-embed-large
echo "Models ready."

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

# Install smoke-check plist (daily 9am health check via WhatsApp)
SMOKE_DEST="$HOME/Library/LaunchAgents/com.civilsmartdictionary.smoke.plist"
sed "s|REPO_DIR|$REPO_DIR|g; s|HOME_DIR|$HOME|g" \
    "$REPO_DIR/deploy/rag-civil-smoke.plist" > "$SMOKE_DEST"
launchctl load "$SMOKE_DEST"
echo "Smoke check scheduled: $SMOKE_DEST"

# nginx config
NGINX_CONF="$(brew --prefix)/etc/nginx/servers/rag-civil.conf"
sed "s|REPO_DIR|$REPO_DIR|g" "$REPO_DIR/deploy/mac_nginx.conf" > "$NGINX_CONF"
brew services restart nginx
echo "nginx configured: $NGINX_CONF"

# Allow nginx through macOS firewall
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$(brew --prefix)/bin/nginx"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp "$(brew --prefix)/bin/nginx"
echo "Firewall rule added for nginx."

echo "Waiting for API..."
sleep 6
curl -s http://127.0.0.1:8000/health
echo ""
echo "=== Done. Run 'bash deploy/mac_update.sh' to update in future. ==="
