docker run \
  -d \
  --restart always \ 
  --gpus "device=1" \
  --shm-size 32G \
  --name index_tts \
  --network host \
  -w /index-tts \
  iss/tts:v1.0 \
  uv run webui.py --fp16 --deepspeed