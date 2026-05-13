#!/usr/bin/env bash
# ------------------------------------------------------------------
# myTTS — Start backend + frontend containers
#
# Usage:
#   ./start.sh
#
# Requires:
#   - Docker with NVIDIA GPU (--gpus all)
#   - mytts-backend built: docker build -t mytts-backend -f Dockerfile.backend .
#   - Streamlit image: iss/streamlit:v1.0 (your frontend image)
# ------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_IMAGE="${BACKEND_IMAGE:-mytts-backend}"
FRONTEND_IMAGE="${FRONTEND_IMAGE:-iss/streamlit:v1.0}"

# ---- Stop existing containers ----
echo "Stopping existing containers ..."
docker stop mytts-backend mytts-frontend 2>/dev/null || true
docker rm mytts-backend mytts-frontend 2>/dev/null || true

echo ""
echo "========================================"
echo "  myTTS — Starting services"
echo "========================================"

# ---- Backend: FastAPI + CosyVoice ----
echo "[1/2] Starting FastAPI backend ($BACKEND_IMAGE) ..."
docker run -d \
  --gpus all \
  --network host \
  --name mytts-backend \
  --shm-size 32G \
  -v "${SCRIPT_DIR}/cosyvoice3_0:/app/cosyvoice3_0" \
  -v "${SCRIPT_DIR}/CosyVoice:/CosyVoice" \
  -v "${SCRIPT_DIR}/storage:/app/storage" \
  -v "${SCRIPT_DIR}/backend:/app/backend" \
  -e COSYVOICE_DIR=/CosyVoice \
  -e MODEL_DIR=cosyvoice3_0 \
  -e STORAGE_DIR=/app/storage \
  -p 8000:8000 \
  -w /app/backend \
  $BACKEND_IMAGE \
  uvicorn main:app --host 0.0.0.0 --port 8000

# ---- Frontend: Streamlit ----
echo "[2/2] Starting Streamlit frontend ($FRONTEND_IMAGE) ..."
docker run -d \
  --network host \
  --name mytts-frontend \
  -v "${SCRIPT_DIR}/frontend:/app" \
  -e BACKEND_URL=http://localhost:8000 \
  -p 8501:8501 \
  -w /app \
  $FRONTEND_IMAGE \
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501

echo ""
echo "========================================"
echo "  ✅ myTTS started"
echo ""
echo "  Backend (FastAPI):  http://localhost:8000"
echo "  Frontend (Streamlit):  http://localhost:8501"
echo ""
echo "  To stop:"
echo "    docker stop mytts-backend mytts-frontend"
echo "========================================"
