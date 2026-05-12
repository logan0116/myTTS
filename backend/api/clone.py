"""Voice cloning API routes."""

import io
import os
import wave
import logging
import tempfile

import numpy as np

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException

from backend.models.schemas import SynthesizeCloneRequest, TTSResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/clone", tags=["clone"])

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a",
    "audio/ogg", "audio/flac",
}


@router.post("/register-voice")
async def register_voice(
    request: Request,
    audio_file: UploadFile = File(...),
    voice_name: str = Form(...),
    prompt_text: str = Form(""),
):
    """Upload a reference audio clip to register a new voice profile."""
    store = request.app.state.audio_store

    content_type = audio_file.content_type or ""
    if content_type and content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {content_type}. "
                   f"Supported: WAV, MP3, M4A, OGG, FLAC",
        )

    raw = await audio_file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    audio_array, sample_rate = _decode_audio(raw, content_type)

    if sample_rate != 24000:
        logger.info("Resampling from %d to 24000", sample_rate)
        audio_array = _resample(audio_array, sample_rate, 24000)
        sample_rate = 24000

    voice_id = store.register_voice(voice_name, audio_array, sample_rate, prompt_text)
    return {
        "success": True,
        "voice_id": voice_id,
        "voice_name": voice_name,
        "message": "声音学习成功",
    }


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize(req: SynthesizeCloneRequest, request: Request):
    """Synthesize speech using a previously registered cloned voice."""
    engine = request.app.state.engine
    store = request.app.state.audio_store

    prompt_audio_path = store.get_prompt_audio_path(req.voice_id)
    if prompt_audio_path is None:
        raise HTTPException(status_code=404, detail=f"Voice '{req.voice_id}' not found")

    try:
        if req.instruction:
            chunks = list(engine.tts_instruct(
                req.text, req.instruction, prompt_audio_path, stream=False,
            ))
        else:
            chunks = list(engine.tts_zero_shot(
                req.text,
                "You are a helpful assistant.<|endofprompt|>",
                prompt_audio_path,
                stream=False,
            ))
    except Exception as e:
        logger.exception("Clone synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis error: {e}")

    if not chunks:
        raise HTTPException(status_code=500, detail="Model produced no audio output")

    audio_id, duration = _save_audio_chunks(chunks, engine.sample_rate, store, f"clone_{req.voice_id}")
    return TTSResponse(
        success=True,
        audio_url=f"/api/audio/{audio_id}",
        duration=round(duration, 2),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_audio_chunks(chunks, sample_rate, store, prefix: str = "tts") -> tuple[str, float]:
    """Concatenate audio chunks, save via AudioStore, return (audio_id, duration)."""
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
    audio_id = store.save_audio(combined, sample_rate)
    duration = len(combined) / sample_rate
    return audio_id, duration


def _decode_audio(raw: bytes, content_type: str) -> tuple[np.ndarray, int]:
    """Decode raw audio bytes into mono float32 (-1..1) numpy array + sample_rate."""
    if content_type and "wav" in content_type:
        try:
            with wave.open(io.BytesIO(raw), "rb") as wf:
                sr = wf.getframerate()
                nc = wf.getnchannels()
                sw = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            fmt = {1: "int8", 2: "int16", 4: "int32"}[sw]
            arr = np.frombuffer(frames, dtype=fmt).astype(np.float32)
            if nc > 1:
                arr = arr.reshape(-1, nc).mean(axis=1)
            peak = max(abs(arr.max()), abs(arr.min()), 1)
            return arr / peak, sr
        except Exception:
            pass

    # Fallback to pydub
    try:
        from pydub import AudioSegment

        ct_map = {
            "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
            "audio/mp4": ".m4a", "audio/m4a": ".m4a",
            "audio/ogg": ".ogg", "audio/flac": ".flac",
        }
        suffix = ct_map.get(content_type, ".wav")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        seg = AudioSegment.from_file(tmp_path)
        os.unlink(tmp_path)

        arr = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels > 1:
            arr = arr.reshape(-1, seg.channels).mean(axis=1)
        peak = max(abs(arr.max()), abs(arr.min()), 1)
        return arr / peak, seg.frame_rate
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="Cannot decode audio. Install pydub for MP3/M4A/OGG support.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}")


def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Simple linear interpolation resample."""
    duration = len(audio) / orig_rate
    new_len = int(duration * target_rate)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
