#!/bin/bash
# deploy/ec2_userdata_scratch_install.sh
#
# ARCHIVED — Original userdata script written during Spot auto-recovery setup
# (2026-03-30). Ran ec2_start.sh from scratch on every boot (~15 min).
# Replaced by ec2_userdata.sh which uses pre-baked AMI ami-0a31aef711a280f89
# for ~2 min boots. EBS volume approach was also tried but abandoned — EBS is
# AZ-locked and Spot instances span all 3 us-east-2 zones. S3 used instead.
# DO NOT use as Launch Template user data — use ec2_userdata.sh instead.
#
# Runs as root on every new instance boot. Handles:
#   1. Elastic IP re-association
#   2. Persistent EBS volume attachment + mount
#   3. Code pull from GitHub
#   4. Full setup via ec2_start.sh
set -e
exec > /var/log/ec2_userdata.log 2>&1

# ── Config ────────────────────────────────────────────────────────────────────
ELASTIC_IP_ALLOC_ID="eipalloc-0ec3315566d08a1ac"
S3_BUCKET="civil-smart-dictionary-data"
REPO_URL="https://github.com/jmianows/rag-civil.git"
REMOTE_DIR="/home/ubuntu/rag-civil"
# ──────────────────────────────────────────────────────────────────────────────

# Install AWS CLI if missing
if ! command -v aws &>/dev/null; then
    echo "[userdata] Installing AWS CLI..."
    apt-get install -y awscli -qq
fi

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
    --region "$REGION" || echo "[userdata] WARNING: Elastic IP association failed (may already be associated)"

# ── 2. Sync vectordb from S3 ──────────────────────────────────────────────────
echo "[userdata] Syncing vectordb from S3..."
mkdir -p "$REMOTE_DIR/vectordb"
chown ubuntu:ubuntu "$REMOTE_DIR/vectordb"
aws s3 sync "s3://$S3_BUCKET/vectordb/" "$REMOTE_DIR/vectordb/" --region "$REGION"
echo "[userdata] vectordb sync complete"

# Restore SSL certs from S3 if present
echo "[userdata] Restoring SSL certs from S3..."
aws s3 sync "s3://$S3_BUCKET/letsencrypt/" /etc/letsencrypt/ --region "$REGION" || echo "[userdata] WARNING: no certs in S3 yet"

# ── 3. Pull latest code ───────────────────────────────────────────────────────
echo "[userdata] Syncing code..."
if [ -d "$REMOTE_DIR/.git" ]; then
    cd "$REMOTE_DIR"
    sudo -u ubuntu git pull --ff-only
else
    sudo -u ubuntu git clone "$REPO_URL" "$REMOTE_DIR"
fi

# ── 4. (data already in place from S3 sync above) ────────────────────────────

# ── 5. Run full setup ─────────────────────────────────────────────────────────
echo "[userdata] Running ec2_start.sh..."
sudo -u ubuntu bash "$REMOTE_DIR/deploy/ec2_start.sh"

# Back up SSL certs to S3 so they survive next reclaim
echo "[userdata] Backing up SSL certs to S3..."
aws s3 sync /etc/letsencrypt/ "s3://$S3_BUCKET/letsencrypt/" --region "$REGION"

echo "[userdata] DONE"
