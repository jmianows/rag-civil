#!/usr/bin/env bash
# deploy/ec2_nginx.sh — nginx reverse proxy + Let's Encrypt SSL + uvicorn systemd service
# Run on the EC2 instance AFTER the DNS A record for api.civilsmartdictionary.com points here.
# Usage: bash deploy/ec2_nginx.sh [domain]
set -euo pipefail

DOMAIN="${1:-api.civilsmartdictionary.com}"
CERT_EMAIL="jmianows@umich.edu"
REMOTE_DIR=/home/ubuntu/rag-civil

echo "==> Setting up nginx + SSL + systemd for ${DOMAIN}"

# ── System packages ─────────────────────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y nginx certbot python3-certbot-nginx -qq

# ── uvicorn systemd service ─────────────────────────────────────────────────
# Replace the bash restart-loop with a proper systemd service.
# Kill any existing uvicorn started by ec2_start.sh first.
pkill -f "uvicorn api.main:app" 2>/dev/null || true
sleep 2

sudo tee /etc/systemd/system/rag-civil.service > /dev/null <<EOF
[Unit]
Description=Civil RAG API (uvicorn)
After=network.target ollama.service
Wants=ollama.service

[Service]
User=ubuntu
WorkingDirectory=${REMOTE_DIR}
Environment="CIVIL_ENV=production"
Environment="CIVIL_API_URL=https://api.civilsmartdictionary.com"
ExecStart=${REMOTE_DIR}/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rag-civil
sudo systemctl restart rag-civil

echo "==> Waiting for API to be ready..."
for i in $(seq 1 60); do
    curl -sf --connect-timeout 2 http://127.0.0.1:8000/health &>/dev/null && echo "==> API ready (${i}s)" && break
    [ "$i" -eq 60 ] && echo "ERROR: API did not start within 60s" && sudo journalctl -u rag-civil -n 30 && exit 1
    sleep 1
done

# ── nginx config (HTTP first — certbot will add HTTPS) ──────────────────────
sudo tee /etc/nginx/sites-available/${DOMAIN} > /dev/null <<'NGINXEOF'
server {
    listen 80;
    server_name DOMAIN_PLACEHOLDER;

    # SSE needs unbuffered streaming
    proxy_buffering off;
    proxy_read_timeout 120s;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
NGINXEOF

sudo sed -i "s/DOMAIN_PLACEHOLDER/${DOMAIN}/g" /etc/nginx/sites-available/${DOMAIN}

sudo ln -sf /etc/nginx/sites-available/${DOMAIN} /etc/nginx/sites-enabled/${DOMAIN}
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# ── SSL cert ─────────────────────────────────────────────────────────────────
echo "==> Obtaining Let's Encrypt certificate for ${DOMAIN}..."
sudo certbot --nginx \
    -d "${DOMAIN}" \
    --non-interactive --agree-tos \
    --email "${CERT_EMAIL}" \
    --redirect

sudo systemctl enable nginx
echo ""
echo "======================================================="
echo " Setup complete!"
echo " API:     https://${DOMAIN}"
echo " Health:  https://${DOMAIN}/health"
echo " Logs:    sudo journalctl -u rag-civil -f"
echo "======================================================="
