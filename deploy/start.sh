#!/usr/bin/env bash
# deploy/start.sh — start all services in dependency order with health checks
# Usage: sudo bash deploy/start.sh
set -euo pipefail

wait_for_http() {
    local url="$1"
    local label="$2"
    local max_attempts=30
    local i=0
    echo -n "    Waiting for $label"
    while ! curl -sf "$url" &>/dev/null; do
        i=$((i + 1))
        if [ $i -ge $max_attempts ]; then
            echo " TIMEOUT"
            return 1
        fi
        echo -n "."
        sleep 2
    done
    echo " ready"
}

echo "==> Starting Ollama"
systemctl start ollama@11434
wait_for_http "http://127.0.0.1:11434/" "Ollama"

echo "==> Starting RAG Civil API"
systemctl start rag-civil-api
wait_for_http "http://127.0.0.1:8000/filters" "FastAPI"

echo "==> Starting Nginx"
systemctl start nginx

echo ""
echo "====================================================="
echo " Service status:"
systemctl is-active --quiet ollama@11434  && echo "  ollama@11434: running" || echo "  ollama@11434: FAILED"
systemctl is-active --quiet rag-civil-api && echo "  rag-civil:   running" || echo "  rag-civil:   FAILED"
systemctl is-active --quiet nginx         && echo "  nginx:       running" || echo "  nginx:       FAILED"
echo ""
echo " Frontend: http://$(hostname -I | awk '{print $1}')"
echo "====================================================="
