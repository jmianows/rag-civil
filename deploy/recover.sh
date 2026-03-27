#!/usr/bin/env bash
# deploy/recover.sh — pull critical data FROM RunPod back to local machine
# Usage: bash deploy/recover.sh <ip> <port> [ssh-key-path]
# Example: bash deploy/recover.sh 203.57.40.135 10055 ~/.ssh/civildict_runpod
set -euo pipefail

NODE_IP="${1:-}"
NODE_PORT="${2:-}"
SSH_KEY="${3:-~/.ssh/civildict_runpod}"
REMOTE_USER=root

if [ -z "$NODE_IP" ] || [ -z "$NODE_PORT" ]; then
    echo "Usage: bash deploy/recover.sh <ip> <port> [ssh-key-path]"
    echo "Example: bash deploy/recover.sh 203.57.40.135 10055 ~/.ssh/civildict_runpod"
    exit 1
fi

SSH_OPTS="-p ${NODE_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new"
RSYNC_SSH="ssh ${SSH_OPTS}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Detect remote project root — check both known paths
echo "==> Detecting remote project root..."
if ssh ${SSH_OPTS} "${REMOTE_USER}@${NODE_IP}" "test -d /workspace/rag-civil" 2>/dev/null; then
    REMOTE_DIR=/workspace/rag-civil
elif ssh ${SSH_OPTS} "${REMOTE_USER}@${NODE_IP}" "test -d /root/rag-civil" 2>/dev/null; then
    REMOTE_DIR=/root/rag-civil
elif ssh ${SSH_OPTS} "${REMOTE_USER}@${NODE_IP}" "test -d /home/ubuntu/rag-civil" 2>/dev/null; then
    REMOTE_DIR=/home/ubuntu/rag-civil
else
    echo "ERROR: Could not find rag-civil directory on remote. Looked in:"
    echo "  /workspace/rag-civil"
    echo "  /root/rag-civil"
    echo "  /home/ubuntu/rag-civil"
    echo "Run: ssh root@${NODE_IP} -p ${NODE_PORT} -i ${SSH_KEY} 'find / -name rag-civil -type d 2>/dev/null'"
    exit 1
fi

echo "    Remote: ${REMOTE_USER}@${NODE_IP}:${REMOTE_DIR}"
echo "    Local:  ${LOCAL_DIR}"
echo ""

# ── Counters ──────────────────────────────────────────────────────────────────
RECOVERED=0
SKIPPED=0
FAILED=0

rsync_item() {
    local label="$1"
    local src="$2"
    local dst="$3"
    shift 3
    local extra_flags=("$@")

    echo "──────────────────────────────────────────────"
    echo "  Recovering: ${label}"
    echo "  From: ${REMOTE_USER}@${NODE_IP}:${src}"
    echo "  To:   ${dst}"

    # Check remote path exists
    if ! ssh ${SSH_OPTS} "${REMOTE_USER}@${NODE_IP}" "test -e '${src}'" 2>/dev/null; then
        echo "  [SKIP] Not found on remote — skipping."
        (( SKIPPED++ )) || true
        return
    fi

    local size
    size=$(ssh ${SSH_OPTS} "${REMOTE_USER}@${NODE_IP}" "du -sh '${src}' 2>/dev/null | cut -f1" || echo "?")
    echo "  Remote size: ${size}"

    mkdir -p "$(dirname "${dst}")"

    if rsync -avz --progress \
        -e "${RSYNC_SSH}" \
        "${extra_flags[@]}" \
        "${REMOTE_USER}@${NODE_IP}:${src}" \
        "${dst}"; then
        echo "  [OK] ${label} recovered."
        (( RECOVERED++ )) || true
    else
        echo "  [FAIL] rsync failed for ${label}."
        (( FAILED++ )) || true
    fi
}

# ── 1. vectordb — most critical ───────────────────────────────────────────────
echo ""
echo "████  STEP 1/5 — vectordb (LanceDB)  ████"
rsync_item "vectordb/" \
    "${REMOTE_DIR}/vectordb/" \
    "${LOCAL_DIR}/vectordb/" \
    --exclude='__pycache__/'

# ── 2. Correction / failure logs ─────────────────────────────────────────────
echo ""
echo "████  STEP 2/5 — ingestion logs  ████"
rsync_item "ingestion/corrections.jsonl" \
    "${REMOTE_DIR}/ingestion/corrections.jsonl" \
    "${LOCAL_DIR}/ingestion/corrections.jsonl"

rsync_item "ingestion/failed_chunks.jsonl" \
    "${REMOTE_DIR}/ingestion/failed_chunks.jsonl" \
    "${LOCAL_DIR}/ingestion/failed_chunks.jsonl"

# ── 3. Python source files ────────────────────────────────────────────────────
echo ""
echo "████  STEP 3/5 — Python source files  ████"

PY_PATHS=(
    "rag/query_engine.py"
    "rag/env_config.py"
    "ingestion/ingest.py"
    "ingestion/retag.py"
    "ingestion/metadata.py"
    "ingestion/section_parser.py"
    "ingestion/__init__.py"
    "ingestion/enrich_file.py"
    "api/main.py"
)

for rel_path in "${PY_PATHS[@]}"; do
    rsync_item "${rel_path}" \
        "${REMOTE_DIR}/${rel_path}" \
        "${LOCAL_DIR}/${rel_path}"
done

# ── 4. frontend (excluding CSS already recovered) ────────────────────────────
echo ""
echo "████  STEP 4/5 — frontend (skip CSS)  ████"
rsync_item "frontend/" \
    "${REMOTE_DIR}/frontend/" \
    "${LOCAL_DIR}/frontend/" \
    --exclude='*.css' \
    --exclude='__pycache__/'

# ── 5. docs (only if missing locally) ────────────────────────────────────────
echo ""
echo "████  STEP 5/5 — docs/ (only if missing locally)  ████"
if [ -d "${LOCAL_DIR}/docs" ] && [ "$(ls -A "${LOCAL_DIR}/docs" 2>/dev/null)" ]; then
    echo "  [SKIP] Local docs/ directory exists and is non-empty — skipping large transfer."
    (( SKIPPED++ )) || true
else
    echo "  Local docs/ missing or empty — recovering from remote..."
    rsync_item "docs/" \
        "${REMOTE_DIR}/docs/" \
        "${LOCAL_DIR}/docs/"
fi

# ── Show git diff for recovered Python files ──────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Git diff after Python file recovery"
echo "════════════════════════════════════════════════"
cd "${LOCAL_DIR}"
if git diff --quiet HEAD -- "${PY_PATHS[@]}" 2>/dev/null; then
    echo "  All recovered Python files match git HEAD — no untracked changes."
else
    git diff HEAD -- "${PY_PATHS[@]}" || true
    echo ""
    echo "  Run 'git add' and 'git commit' to preserve any recovered changes."
fi

# ── vectordb integrity check ──────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  vectordb integrity check"
echo "════════════════════════════════════════════════"
VENV_PYTHON=""
if [ -f "${LOCAL_DIR}/.venv/bin/python" ]; then
    VENV_PYTHON="${LOCAL_DIR}/.venv/bin/python"
elif [ -f "${LOCAL_DIR}/venv/bin/python" ]; then
    VENV_PYTHON="${LOCAL_DIR}/venv/bin/python"
fi

if [ -d "${LOCAL_DIR}/vectordb" ]; then
    echo "  vectordb/ exists locally."
    echo "  Size: $(du -sh "${LOCAL_DIR}/vectordb" | cut -f1)"

    if [ -n "${VENV_PYTHON}" ]; then
        echo "  Running quick LanceDB connectivity check..."
        ${VENV_PYTHON} - <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.expanduser("~/rag-civil"))
try:
    import lancedb
    db = lancedb.connect(os.path.expanduser("~/rag-civil/vectordb"))
    tables = db.table_names()
    print(f"  Tables found: {tables}")
    if tables:
        tbl = db.open_table(tables[0])
        count = tbl.count_rows()
        print(f"  Row count in '{tables[0]}': {count:,}")
        if count > 40000:
            print("  [OK] vectordb looks healthy (>40k rows).")
        else:
            print(f"  [WARN] Expected ~43,000 rows, found {count:,}.")
    else:
        print("  [WARN] No tables found in vectordb.")
except Exception as e:
    print(f"  [ERROR] LanceDB check failed: {e}")
PYEOF
    else
        echo "  [INFO] No venv found — skipping Python integrity check."
        echo "         Activate your venv and run: python -c \"import lancedb; db=lancedb.connect('vectordb'); print(db.table_names())\""
    fi
else
    echo "  [WARN] vectordb/ not present locally after recovery."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  RECOVERY SUMMARY"
echo "════════════════════════════════════════════════"
echo "  Recovered : ${RECOVERED}"
echo "  Skipped   : ${SKIPPED}"
echo "  Failed    : ${FAILED}"
echo ""
if [ "${FAILED}" -gt 0 ]; then
    echo "  WARNING: ${FAILED} item(s) failed. Check output above."
    exit 1
else
    echo "  All items recovered successfully."
fi
