"""TTS API routes."""

from fastapi import APIRouter, Request
from backend.models.schemas import TTSRequest, TTSResponse

router = APIRouter(prefix="/api/tts", tags=["tts"])


@router.post("/text-to-speech", response_model=TTSResponse)
async def text_to_speech(req: TTSRequest, request: Request):
    raise NotImplementedError
