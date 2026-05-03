#!/bin/bash
# deploy/smoke_check.sh — Daily smoke test for the RAG service.
# Usage: smoke_check.sh [--notify]   --notify forces a WhatsApp message regardless of result
OPENCLAW="/opt/homebrew/bin/openclaw"
LOG="$HOME/rag-civil/smoke.log"
TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
FORCE_NOTIFY=0
export PATH="/opt/homebrew/bin:$PATH"

if [ "$1" = "--notify" ]; then FORCE_NOTIFY=1; fi

RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:8000/smoke)
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "$TIMESTAMP OK $BODY" >> "$LOG"
    if [ "$FORCE_NOTIFY" = "1" ]; then
        "$OPENCLAW" message send \
            --target +17346342588 \
            --message "Civil RAG smoke check OK: $BODY"
    fi
else
    echo "$TIMESTAMP FAIL (HTTP $HTTP_CODE) $BODY" >> "$LOG"
    "$OPENCLAW" message send \
        --target +17346342588 \
        --message "Civil RAG smoke check FAILED (HTTP $HTTP_CODE): $BODY"
fi
