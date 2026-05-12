"""Application configuration."""
from pydantic import BaseModel


class Settings(BaseModel):
    model_dir: str = "pretrained_models/Fun-CosyVoice3-0.5B"
    storage_dir: str = "storage"
    default_sample_rate: int = 24000
    max_text_length: int = 5000
    audio_ttl_hours: int = 24  # Temp audio auto-cleanup


settings = Settings()
