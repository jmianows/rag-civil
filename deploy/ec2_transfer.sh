#!/usr/bin/env bash
# deploy/ec2_transfer.sh — push project to an AWS EC2 instance
# Usage: bash deploy/ec2_transfer.sh [ip] [ssh-key-path]
# Example: bash deploy/ec2_transfer.sh 3.14.12.134 ~/.ssh/id_ed25519
set -euo pipefail

NODE_IP="${1:-3.14.12.134}"
SSH_KEY="${2:-~/.ssh/id_ed25519}"
REMOTE_USER=ubuntu
REMOTE_DIR=/home/ubuntu/rag-civil

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Transferring project to ${REMOTE_USER}@${NODE_IP}:${REMOTE_DIR}"
echo "    Source: $LOCAL_DIR"
echo ""

rsync -avz --progress --no-owner --no-group \
    -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new" \
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
echo " Next step: ssh ubuntu@${NODE_IP} -i ${SSH_KEY}"
echo "   then run: bash ~/rag-civil/deploy/ec2_start.sh"
echo "====================================================="
