#!/bin/bash
# deploy/ec2_userdata.sh — EC2 Launch Template user data for Spot auto-recovery
#
# Uses pre-baked AMI (ami-0a31aef711a280f89) — all packages, Ollama, models,
# and Python venv are already installed. Boot time ~2 min vs ~15 min from scratch.
#
# Runs as root on every new Spot instance boot. Handles:
#   1. Elastic IP re-association
#   2. vectordb sync from S3
#   3. SSL cert restore from S3
#   4. Code pull from GitHub
#   5. Start services
set -e
exec > /var/log/ec2_userdata.log 2>&1

# ── Config ────────────────────────────────────────────────────────────────────
ELASTIC_IP_ALLOC_ID="eipalloc-0ec3315566d08a1ac"
S3_BUCKET="civil-smart-dictionary-data"
REPO_URL="https://github.com/jmianows/rag-civil.git"
REMOTE_DIR="/home/ubuntu/rag-civil"
# ──────────────────────────────────────────────────────────────────────────────

# Use IMDSv2 to get instance metadata
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/placement/region)

echo "[userdata] Instance: $INSTANCE_ID  Region: $REGION"

# ── 1. Associate Elastic IP ───────────────────────────────────────────────────
echo "[userdata] Associating Elastic IP..."
aws ec2 associate-address \
    --instance-id "$INSTANCE_ID" \
    --allocation-id "$ELASTIC_IP_ALLOC_ID" \
    --region "$REGION" || echo "[userdata] WARNING: Elastic IP association failed"

# ── 2. Sync vectordb from S3 ──────────────────────────────────────────────────
echo "[userdata] Syncing vectordb from S3..."
mkdir -p "$REMOTE_DIR/vectordb"
chown ubuntu:ubuntu "$REMOTE_DIR/vectordb"
aws s3 sync "s3://$S3_BUCKET/vectordb/" "$REMOTE_DIR/vectordb/" --region "$REGION"
echo "[userdata] vectordb sync complete"

# ── 3. Restore SSL certs from S3 ─────────────────────────────────────────────
echo "[userdata] Restoring SSL certs from S3..."
aws s3 sync "s3://$S3_BUCKET/letsencrypt/" /etc/letsencrypt/ --region "$REGION" \
    || echo "[userdata] WARNING: no certs in S3 yet"

# ── 4. Pull latest code from GitHub ──────────────────────────────────────────
echo "[userdata] Pulling latest code..."
if [ -d "$REMOTE_DIR/.git" ]; then
    cd "$REMOTE_DIR" && sudo -u ubuntu git pull --ff-only
else
    # AMI may not have .git — clone fresh alongside existing files
    echo "[userdata] No .git found, cloning fresh..."
    sudo -u ubuntu git clone "$REPO_URL" /tmp/rag-civil-fresh
    rsync -a --exclude='vectordb' /tmp/rag-civil-fresh/ "$REMOTE_DIR/"
    mv /tmp/rag-civil-fresh/.git "$REMOTE_DIR/.git"
    rm -rf /tmp/rag-civil-fresh
    chown -R ubuntu:ubuntu "$REMOTE_DIR"
fi

# ── 5. Start services ─────────────────────────────────────────────────────────
echo "[userdata] Starting services..."
systemctl restart ollama
systemctl restart rag-civil

# Re-run certbot to wire SSL into nginx config (idempotent)
certbot --nginx -d api.civilsmartdictionary.com --non-interactive --agree-tos \
    --email jmianows@umich.edu --redirect 2>/dev/null || true
systemctl reload nginx

# Wait for API to be healthy
echo "[userdata] Waiting for API..."
for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:8000/health &>/dev/null && echo "[userdata] API ready (${i}s)" && break
    [ "$i" -eq 60 ] && echo "[userdata] ERROR: API did not become ready" && exit 1
    sleep 2
done

# ── 6. Back up SSL certs to S3 ───────────────────────────────────────────────
echo "[userdata] Backing up SSL certs to S3..."
aws s3 sync /etc/letsencrypt/ "s3://$S3_BUCKET/letsencrypt/" --region "$REGION"

echo "[userdata] DONE"
