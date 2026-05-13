#!/usr/bin/env bash
# ------------------------------------------------------------------
# myTTS — Start TTS model server + FastAPI backend
#
# Requires:
#   - Docker with NVIDIA GPU support (--gpus all)
#   - iss/sps:v1.0  (TTS base image with CosyVoice installed)
#   - mytts-backend:v1.0 built from Dockerfile.backend
#
# Volume layout expected on host:
#   ${PWD}/cosyvoice3_0/   → /app/cosyvoice3_0   (model files)
#   ${PWD}/CosyVoice/       → /CosyVoice           (CosyVoice source)
#   ${PWD}/storage/         → /app/storage         (runtime audio storage)
# ------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  myTTS — Starting services"
echo "========================================"

# ---- TTS container (CosyVoice model server, background) ----
echo "[1/2] Starting TTS container (iss/sps:v1.0) ..."
docker run -d \
  --rm \
  --gpus all \
  --network mytts-net \
  --name my-tts \
  -v "${SCRIPT_DIR}/cosyvoice3_0:/app/cosyvoice3_0" \
  -v "${SCRIPT_DIR}/CosyVoice:/CosyVoice" \
  -v "${SCRIPT_DIR}/storage:/app/storage" \
  -w /app \
  iss/sps:v1.0 \
  sleep infinity

# Wait for TTS container to be ready
echo "  Waiting for TTS container to start ..."
sleep 3

# ---- Backend container (FastAPI, background) ----
echo "[2/2] Starting FastAPI backend (mytts-backend:v1.0) ..."
docker run -d \
  --rm \
  --gpus all \
  --network mytts-net \
  --name my-tts-backend \
  --env COSYVOICE_DIR=/CosyVoice \
  --env MODEL_DIR=cosyvoice3_0 \
  --env STORAGE_DIR=/app/storage \
  -v "${SCRIPT_DIR}/backend:/app/backend" \
  -v "${SCRIPT_DIR}/storage:/app/storage" \
  -p 8000:8000 \
  -w /app \
  mytts-backend:v1.0

# ---- Done ----
echo ""
echo "========================================"
echo "  ✅ Services started"
echo ""
echo "  TTS model server:  (container: my-tts)"
echo "  FastAPI backend:    http://localhost:8000"
echo ""
echo "  To stop:"
echo "    docker stop my-tts-backend my-tts"
echo "========================================"
