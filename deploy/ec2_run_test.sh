#!/usr/bin/env bash
# deploy/ec2_run_test.sh — run the 100-prompt test suite on EC2
# Usage: bash deploy/ec2_run_test.sh [ip] [ssh-key-path]
set -euo pipefail

NODE_IP="${1:-16.58.218.79}"
SSH_KEY="${2:-~/.ssh/id_ed25519}"
REMOTE_USER=ubuntu
REMOTE_DIR=/home/ubuntu/rag-civil
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=accept-new"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── 1. Push tests/ to EC2 ──────────────────────────────────────────────────
echo "==> Pushing tests/ to EC2..."
rsync -avz --progress \
    -e "ssh ${SSH_OPTS}" \
    --exclude='__pycache__/' --exclude='*.pyc' \
    "${LOCAL_DIR}/tests/" \
    "${REMOTE_USER}@${NODE_IP}:${REMOTE_DIR}/tests/"

# ── 2. Detect next run number ─────────────────────────────────────────────
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

# ── 3. Verify API is ready ────────────────────────────────────────────────
echo "==> Checking API health..."
ssh ${SSH_OPTS} "${REMOTE_USER}@${NODE_IP}" "
for i in \$(seq 1 30); do
    curl -sf http://127.0.0.1:8000/health &>/dev/null && echo \"API ready (\${i}s)\" && exit 0
    sleep 1
done
echo 'ERROR: API not ready' && exit 1
"

# ── 4. Run prompts ────────────────────────────────────────────────────────
echo ""
echo "==> Running 100 prompts on EC2 (run_${NEXT_RUN})..."
echo "    Output: ${OUT_PREFIX}.{log,json}"
echo ""
ssh ${SSH_OPTS} "${REMOTE_USER}@${NODE_IP}" \
    "cd ${REMOTE_DIR} && CIVIL_ENV=production \
     .venv/bin/python tests/run_prompts.py \
       --out ${OUT_PREFIX} \
       ${COMPARE_ARG}"

# ── 5. Sync results back ──────────────────────────────────────────────────
echo ""
echo "==> Syncing results to local tests/..."
rsync -avz \
    -e "ssh ${SSH_OPTS}" \
    "${REMOTE_USER}@${NODE_IP}:${OUT_PREFIX}.log" \
    "${REMOTE_USER}@${NODE_IP}:${OUT_PREFIX}.json" \
    "${LOCAL_DIR}/tests/"

LOCAL_LOG="${LOCAL_DIR}/tests/run_${NEXT_RUN}.log"
LOCAL_JSON="${LOCAL_DIR}/tests/run_${NEXT_RUN}.json"

# ── 6. Summary ────────────────────────────────────────────────────────────
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
avg_ttfl = d.get("mean_ttfl_s")
fails  = d.get("fail_prompts", [])
slowest = sorted(d.get("results", []), key=lambda r: r.get("ttfl_s") or 0, reverse=True)[:5]
print(f"  Model  : {d.get('llm_model','?')}")
print(f"  Total  : {total}")
print(f"  Passed : {passed}  ({100*passed//total if total else 0}%)")
print(f"  Failed : {failed}  {fails}")
print(f"  Avg t  : {avg_t:.1f}s")
print(f"  Avg TTFL: {avg_ttfl:.1f}s" if avg_ttfl is not None else "  Avg TTFL: n/a")
print(f"  Slowest TTFL: {[(r['n'], r.get('ttfl_s')) for r in slowest]}")
PYEOF
fi
echo ""
echo "  Log  : ${LOCAL_LOG}"
echo "  JSON : ${LOCAL_JSON}"
