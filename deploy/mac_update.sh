#!/bin/bash
# deploy/mac_update.sh — Pull latest code and restart the service.
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.civilsmartdictionary.rag.plist"

cd "$REPO_DIR"
git pull
source .venv/bin/activate
pip install -r requirements.txt -q

launchctl unload "$PLIST"
launchctl load  "$PLIST"

echo "Restarted. Waiting for API..."
sleep 6
curl -s http://127.0.0.1:8000/health
echo ""
