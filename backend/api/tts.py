"""TTS API routes."""

import logging

from fastapi import APIRouter, Request, HTTPException

from backend.models.schemas import TTSRequest, TTSResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tts", tags=["tts"])

DEFAULT_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
)


@router.post("/text-to-speech", response_model=TTSResponse)
async def text_to_speech(req: TTSRequest, request: Request):
    """Synthesize speech from text using the default voice or a registered voice."""
    engine = request.app.state.engine
    store = request.app.state.audio_store

    # Resolve prompt audio path
    if req.voice_id == "default":
        prompt_audio_path = str(store.audio_dir / ".default_prompt.wav")
        # Create a minimal default prompt if missing (silent)
        _ensure_default_prompt(prompt_audio_path, store)
        prompt_text = DEFAULT_PROMPT_TEXT
    else:
        prompt_audio_path = store.get_prompt_audio_path(req.voice_id)
        if prompt_audio_path is None:
            raise HTTPException(status_code=404, detail=f"Voice '{req.voice_id}' not found")
        prompt_text = DEFAULT_PROMPT_TEXT

    # Run inference
    try:
        if req.instruction:
            chunks = list(engine.tts_instruct(
                req.text, req.instruction, prompt_audio_path, stream=req.stream,
            ))
        else:
            chunks = list(engine.tts_zero_shot(
                req.text, prompt_text, prompt_audio_path, stream=req.stream,
            ))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("TTS inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    if not chunks:
        raise HTTPException(status_code=500, detail="Model produced no audio output")

    # Concatenate chunks and save
    import numpy as np

    try:
        import torch
    except ImportError:
        torch = None

    audio_list = []
    for ch in chunks:
        t = ch["tts_speech"]
        if torch is not None and isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        audio_list.append(np.squeeze(np.asarray(t)))

    combined = np.concatenate(audio_list) if len(audio_list) > 1 else audio_list[0]
    audio_id = store.save_audio(combined, engine.sample_rate)

    duration = len(combined) / engine.sample_rate
    return TTSResponse(
        success=True,
        audio_url=f"/api/audio/{audio_id}",
        duration=round(duration, 2),
    )


def _ensure_default_prompt(path: str, store) -> None:
    """Create a minimal silent WAV as default prompt if none exists."""
    import os
    if os.path.isfile(path):
        return
    import wave
    import struct
    sample_rate = 24000
    num_samples = sample_rate  # 1 sec silence
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for _ in range(num_samples):
            wf.writeframes(struct.pack("<h", 0))
