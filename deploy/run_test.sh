#!/usr/bin/env bash
# deploy/run_test.sh — push backend to RunPod and run the 100-prompt test suite
# Usage: bash deploy/run_test.sh <ip> <port> [ssh-key-path]
# Example: bash deploy/run_test.sh 203.57.40.175 10240 ~/.ssh/id_ed25519
set -euo pipefail

NODE_IP="${1:-}"
NODE_PORT="${2:-}"
SSH_KEY="${3:-~/.ssh/id_ed25519}"
REMOTE_USER=root
REMOTE_DIR=/workspace/rag-civil

if [ -z "$NODE_IP" ] || [ -z "$NODE_PORT" ]; then
    echo "Usage: bash deploy/run_test.sh <ip> <port> [ssh-key-path]"
    echo "Example: bash deploy/run_test.sh 203.57.40.175 10240 ~/.ssh/id_ed25519"
    exit 1
fi

SSH_OPTS="-p ${NODE_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=no"
RSYNC_E="ssh ${SSH_OPTS}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── 1. Push backend (uses runpod_transfer.sh excludes) ─────────────────────────
echo "==> Pushing backend code to ${REMOTE_USER}@${NODE_IP}:${NODE_PORT}..."
bash "${LOCAL_DIR}/deploy/runpod_transfer.sh" "${NODE_IP}" "${NODE_PORT}" "${SSH_KEY}"

# Push tests/ separately (runpod_transfer.sh excludes it)
echo "==> Pushing tests/..."
rsync -avz --progress --no-owner --no-group \
    -e "${RSYNC_E}" \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    "${LOCAL_DIR}/tests/" \
    "${REMOTE_USER}@${NODE_IP}:${REMOTE_DIR}/tests/"

# ── 2. Detect next run number ──────────────────────────────────────────────────
echo "==> Detecting next run number..."
LAST_RUN=$(ssh ${SSH_OPTS} "${REMOTE_USER}@${NODE_IP}" \
    "ls ${REMOTE_DIR}/tests/run_*.json 2>/dev/null | grep -oP 'run_\K[0-9]+' | sort -n | tail -1 || echo 0")
NEXT_RUN=$((LAST_RUN + 1))
OUT_PREFIX="${REMOTE_DIR}/tests/run_${NEXT_RUN}"
COMPARE_ARG=""
if [ "${LAST_RUN}" -gt 0 ]; then
    COMPARE_ARG="--compare ${REMOTE_DIR}/tests/run_${LAST_RUN}.json"
fi
echo "    Last run: ${LAST_RUN}  →  This run: run_${NEXT_RUN}"

# ── 3. Verify API is ready ────────────────────────────────────────────────────
echo "==> Checking API health before running prompts..."
ssh ${SSH_OPTS} "${REMOTE_USER}@${NODE_IP}" "
for i in \$(seq 1 30); do
    curl -sf http://127.0.0.1:8000/health &>/dev/null && echo \"API ready (\${i}s)\" && exit 0
    sleep 1
done
echo 'ERROR: API not ready after 30s — is runpod_start.sh running?' && exit 1
"

# ── 4. Run prompts on pod ──────────────────────────────────────────────────────
echo ""
echo "==> Running 100 prompts on pod (run_${NEXT_RUN})..."
echo "    Output: ${OUT_PREFIX}.{log,json}"
echo ""
ssh ${SSH_OPTS} "${REMOTE_USER}@${NODE_IP}" \
    "cd ${REMOTE_DIR} && CIVIL_ENV=production \
     .venv/bin/python tests/run_prompts.py \
       --out ${OUT_PREFIX} \
       ${COMPARE_ARG}"

# ── 5. Sync results back to local tests/ ──────────────────────────────────────
echo ""
echo "==> Syncing run_${NEXT_RUN} results to local tests/..."
rsync -avz \
    -e "${RSYNC_E}" \
    "${REMOTE_USER}@${NODE_IP}:${OUT_PREFIX}.log" \
    "${REMOTE_USER}@${NODE_IP}:${OUT_PREFIX}.json" \
    "${LOCAL_DIR}/tests/"

LOCAL_LOG="${LOCAL_DIR}/tests/run_${NEXT_RUN}.log"
LOCAL_JSON="${LOCAL_DIR}/tests/run_${NEXT_RUN}.json"

# ── 6. Quick pass/fail summary ─────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  run_${NEXT_RUN} SUMMARY"
echo "════════════════════════════════════════════════"
if [ -f "${LOCAL_JSON}" ]; then
    python3 - <<PYEOF
import json
d = json.load(open("${LOCAL_JSON}"))
runs   = d.get("results", [])
total  = len(runs)
passed = sum(1 for r in runs if r.get("correct"))
failed = total - passed
avg_t  = d.get("mean_time_s", 0)
fails  = d.get("fail_prompts", [])
print(f"  Model  : {d.get('llm_model','?')}")
print(f"  Total  : {total}")
print(f"  Passed : {passed}  ({100*passed//total if total else 0}%)")
print(f"  Failed : {failed}  {fails}")
print(f"  Avg t  : {avg_t:.1f}s")
PYEOF
fi
echo ""
echo "  Log  : ${LOCAL_LOG}"
echo "  JSON : ${LOCAL_JSON}"
echo ""
echo "  To commit: git add tests/run_${NEXT_RUN}.{log,json} && git commit -m 'Add run_${NEXT_RUN} results'"
