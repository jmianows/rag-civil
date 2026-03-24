#!/usr/bin/env bash
# deploy/update.sh — push code updates without re-running ingestion
# Usage: sudo bash deploy/update.sh
# Run from: /home/ubuntu/rag-civil on the remote node
set -euo pipefail

PROJECT_DIR=/home/ubuntu/rag-civil

echo "==> [1/4] Pulling latest git changes"
cd "$PROJECT_DIR"
sudo -u ubuntu git pull --ff-only

echo "==> [2/4] Installing any new pip dependencies"
sudo -u ubuntu "$PROJECT_DIR/.venv/bin/pip" install -q \
    -r "$PROJECT_DIR/requirements.txt"

echo "==> [3/4] Restarting API service"
systemctl restart rag-civil-api

echo "==> [4/4] Waiting for API to come back up"
for i in {1..20}; do
    curl -sf http://127.0.0.1:8000/filters &>/dev/null && break
    echo "    Waiting... ($i)"
    sleep 3
done

if curl -sf http://127.0.0.1:8000/filters &>/dev/null; then
    echo ""
    echo "====================================================="
    echo " Update complete. API is healthy."
    echo " Note: vectordb and ingestion are unchanged."
    echo "====================================================="
else
    echo ""
    echo "====================================================="
    echo " WARNING: API did not respond after restart."
    echo " Check logs: journalctl -u rag-civil-api -n 50"
    echo "====================================================="
    exit 1
fi
