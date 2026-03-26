#!/usr/bin/env bash
# deploy/transfer.sh — run from your LOCAL machine to push the project to the cloud node
# Usage: bash deploy/transfer.sh <node-ip> [ssh-key-path]
# Example: bash deploy/transfer.sh 12.34.56.78
#          bash deploy/transfer.sh 12.34.56.78 ~/.ssh/my-key.pem
set -euo pipefail

NODE_IP="${1:-}"
SSH_KEY="${2:-}"
REMOTE_USER=ubuntu
REMOTE_DIR=/home/ubuntu/rag-civil

if [ -z "$NODE_IP" ]; then
    echo "Usage: bash deploy/transfer.sh <node-ip> [ssh-key-path]"
    exit 1
fi

# Build SSH args
SSH_ARGS="-o StrictHostKeyChecking=no"
if [ -n "$SSH_KEY" ]; then
    SSH_ARGS="$SSH_ARGS -i $SSH_KEY"
fi

# Resolve local project root (one level up from this script's location)
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Transferring project to ${REMOTE_USER}@${NODE_IP}:${REMOTE_DIR}"
echo "    Source: $LOCAL_DIR"
echo ""

# Ensure remote directory exists
ssh $SSH_ARGS "${REMOTE_USER}@${NODE_IP}" "mkdir -p $REMOTE_DIR"

rsync -avz --progress \
    $SSH_ARGS \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --include='vectordb/' \
    --exclude='docs/' \
    --exclude='tests/' \
    "$LOCAL_DIR/" \
    "${REMOTE_USER}@${NODE_IP}:${REMOTE_DIR}/"

echo ""
echo "==> Transfer summary"
# Print remote disk usage of the project directory
ssh $SSH_ARGS "${REMOTE_USER}@${NODE_IP}" \
    "du -sh $REMOTE_DIR && echo '  Files transferred successfully to $REMOTE_DIR'"

echo ""
echo "====================================================="
echo " Transfer complete."
echo " Next step (first deploy): ssh ${REMOTE_USER}@${NODE_IP}"
echo "   then run: sudo bash $REMOTE_DIR/deploy/setup.sh"
echo "====================================================="
