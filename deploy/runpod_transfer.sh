#!/usr/bin/env bash
# deploy/runpod_transfer.sh — push project to a RunPod pod (SCP/TCP connection)
# Usage: bash deploy/runpod_transfer.sh <ip> <port> [ssh-key-path]
# Example: bash deploy/runpod_transfer.sh 203.57.40.135 10055 ~/.ssh/civildict_runpod
set -euo pipefail

NODE_IP="${1:-}"
NODE_PORT="${2:-}"
SSH_KEY="${3:-~/.ssh/civildict_runpod}"
REMOTE_USER=root
REMOTE_DIR=/workspace/rag-civil

if [ -z "$NODE_IP" ] || [ -z "$NODE_PORT" ]; then
    echo "Usage: bash deploy/runpod_transfer.sh <ip> <port> [ssh-key-path]"
    echo "Example: bash deploy/runpod_transfer.sh 203.57.40.135 10055 ~/.ssh/civildict_runpod"
    exit 1
fi

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Transferring project to ${REMOTE_USER}@${NODE_IP}:${NODE_PORT}:${REMOTE_DIR}"
echo "    Source: $LOCAL_DIR"
echo ""

rsync -avz --progress --no-owner --no-group \
    -e "ssh -p ${NODE_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='docs/' \
    --exclude='tests/' \
    "$LOCAL_DIR/" \
    "${REMOTE_USER}@${NODE_IP}:${REMOTE_DIR}/"

echo ""
echo "====================================================="
echo " Transfer complete."
echo " Next step: ssh root@${NODE_IP} -p ${NODE_PORT} -i ${SSH_KEY}"
echo "   then run: cd ${REMOTE_DIR} && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
echo "   then run: bash deploy/runpod_start.sh"
echo "====================================================="
