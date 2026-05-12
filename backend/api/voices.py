"""Voice management API routes."""

from fastapi import APIRouter, Request, HTTPException

from backend.models.schemas import VoiceListResponse, VoiceInfo

router = APIRouter(prefix="/api", tags=["voices"])


@router.get("/voices", response_model=VoiceListResponse)
async def list_voices(request: Request):
    """List all registered voice profiles."""
    store = request.app.state.audio_store
    voices_data = store.list_voices()
    return VoiceListResponse(
        voices=[VoiceInfo(**v) for v in voices_data]
    )


@router.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str, request: Request):
    """Delete a registered voice profile."""
    store = request.app.state.audio_store
    if store.delete_voice(voice_id):
        return {"success": True, "message": f"Voice '{voice_id}' deleted"}
    raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
