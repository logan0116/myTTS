"""Voice cloning API routes."""

from fastapi import APIRouter, Request, UploadFile, File, Form
from backend.models.schemas import SynthesizeCloneRequest, TTSResponse, RegisterVoiceRequest

router = APIRouter(prefix="/api/clone", tags=["clone"])


@router.post("/register-voice")
async def register_voice(
    audio_file: UploadFile = File(...),
    voice_name: str = Form(...),
    prompt_text: str = Form(""),
):
    raise NotImplementedError


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize(req: SynthesizeCloneRequest, request: Request):
    raise NotImplementedError
