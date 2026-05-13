"""myTTS — FastAPI application entry point."""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import settings
from engine.cosyvoice_engine import CosyVoiceEngine, ModelNotFoundError
from storage.audio_store import AudioStore
from api.tts import router as tts_router
from api.clone import router as clone_router
from api.voices import router as voices_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — model warm-up / teardown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting myTTS backend ...")

    # Initialize engine (skip load if model dir doesn't exist — for dev)
    engine = CosyVoiceEngine(
        cosyvoice_dir=settings.cosyvoice_dir,
        model_dir=settings.model_dir,
    )
    try:
        engine.load()
    except (ModelNotFoundError, ImportError) as e:
        logger.warning("Model not available: %s. Backend will start in degraded mode.", e)

    audio_store = AudioStore(storage_dir=settings.storage_dir)
    audio_store._ensure_dirs()

    app.state.engine = engine
    app.state.audio_store = audio_store
    logger.info("Backend ready.")

    yield

    engine.unload()
    logger.info("Backend shut down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="myTTS",
    description="Local TTS & voice cloning service based on Fun-CosyVoice3",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(tts_router)
app.include_router(clone_router)
app.include_router(voices_router)


# ---------------------------------------------------------------------------
# Audio serving
# ---------------------------------------------------------------------------

@app.get("/api/audio/{audio_id}")
async def serve_audio(audio_id: str, request: Request = None):
    """Serve a generated audio file by ID."""
    from fastapi import Request as _R

    # Use app.state.audio_store if available (supports test injection),
    # otherwise fall back to settings.storage_dir
    try:
        store_path = app.state.audio_store.get_audio_path(audio_id)
        if store_path is not None:
            return FileResponse(str(store_path), media_type="audio/wav")
    except Exception:
        pass

    # Fallback: read from configured storage dir
    path = os.path.join(settings.storage_dir, "audio", audio_id)
    if os.path.isfile(path):
        return FileResponse(path, media_type="audio/wav")
    return JSONResponse({"detail": "Audio not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": app.state.engine.is_loaded if hasattr(app.state, 'engine') else False,
    }

