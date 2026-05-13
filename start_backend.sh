docker run -it \
  --rm \
  --gpus all \
  --network host \
  --name mytts-backend \
  --shm-size 32G \
  -v ${PWD}/cosyvoice3_0:/app/cosyvoice3_0 \
  -v ${PWD}/storage:/app/storage \
  -v ${PWD}/backend:/app/backend \
  -p 8000:8000 \
  -w /app/backend \
  iss/tts:v1.0 \
  uvicorn main:app --host 0.0.0.0 --port 7723