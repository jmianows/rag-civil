#!/bin/bash
# deploy/spot_watcher.sh — polls for Spot termination notice and archives logs to S3
# Installed as a systemd service by ec2_start.sh. Runs as root.

S3_BUCKET="civil-smart-dictionary-data"
REMOTE_DIR="/home/ubuntu/rag-civil"
LOG_FILES=(
    "$REMOTE_DIR/query_log.jsonl"
    "$REMOTE_DIR/analytics.json"
    "$REMOTE_DIR/code_requests.log"
    "$REMOTE_DIR/rate_limit.log"
)

TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/placement/region)

echo "[spot-watcher] Started. Instance: $INSTANCE_ID"

while true; do
    STATUS=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
        http://169.254.169.254/latest/meta-data/spot/termination-time 2>/dev/null)

    if [ -n "$STATUS" ] && [ "$STATUS" != "404" ]; then
        echo "[spot-watcher] Termination notice received: $STATUS — archiving logs to S3"
        TS=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
        for f in "${LOG_FILES[@]}"; do
            if [ -f "$f" ]; then
                BASENAME=$(basename "$f")
                EXT="${BASENAME##*.}"
                NAME="${BASENAME%.*}"
                DEST="s3://$S3_BUCKET/logs/${TS}_${INSTANCE_ID}_${NAME}.${EXT}"
                aws s3 cp "$f" "$DEST" --region "$REGION" \
                    && echo "[spot-watcher] Uploaded $BASENAME → $DEST" \
                    || echo "[spot-watcher] WARNING: failed to upload $BASENAME"
            fi
        done
        echo "[spot-watcher] Archive complete."
        exit 0
    fi

    sleep 30
done
