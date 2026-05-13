"""Application configuration."""

import os
from pydantic import BaseModel


class Settings(BaseModel):
    # CosyVoice installation directory (inside Docker: /CosyVoice)
    cosyvoice_dir = "/CosyVoice"
    # Model directory relative to cosyvoice_dir (inside Docker: cosyvoice3_0)
    model_dir = "/app/cosyvoice3_0"
    # Local storage for generated audio and registered voices
    storage_dir = "/app/storage"
    default_sample_rate: int = 24000
    max_text_length: int = 5000
    audio_ttl_hours: int = 24


settings = Settings()
