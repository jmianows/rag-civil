#!/usr/bin/env bash
# RunPod startup script — RTX 4090
# RunPod's proxy handles HTTPS/SSL; no nginx or systemd needed.
# Run this from the pod terminal or paste into the pod's "On Start" command field.
set -e

# Start Ollama in background
ollama serve &
sleep 5

# Pull required models if not already cached
ollama pull mxbai-embed-large
ollama pull qwen3:8b

# Start FastAPI — listen on all interfaces so RunPod proxy can reach it
cd /workspace/rag-civil
CIVIL_ENV=production \
  .venv/bin/uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
