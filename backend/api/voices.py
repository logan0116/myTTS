"""Voice management API routes."""

from fastapi import APIRouter, Request
from backend.models.schemas import VoiceListResponse

router = APIRouter(prefix="/api", tags=["voices"])


@router.get("/voices", response_model=VoiceListResponse)
async def list_voices(request: Request):
    raise NotImplementedError
