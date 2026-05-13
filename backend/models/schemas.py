"""Pydantic request/response schemas."""
from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: str = Field(default="default")
    language: str = Field(default="zh")
    instruction: str = Field(default="")
    stream: bool = Field(default=False)


class TTSResponse(BaseModel):
    success: bool
    audio_url: str
    duration: float
    format: str = "wav"
    sample_rate: int = 24000


class RegisterVoiceRequest(BaseModel):
    voice_name: str = Field(..., min_length=1, max_length=50)
    prompt_text: str = Field(default="")


class SynthesizeCloneRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: str = Field(...)
    language: str = Field(default="zh")
    instruction: str = Field(default="")


class VoiceInfo(BaseModel):
    voice_id: str
    voice_name: str
    created_at: str
    duration: float


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]
