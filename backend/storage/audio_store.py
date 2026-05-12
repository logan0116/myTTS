"""Audio file storage manager."""
import os
import json
import uuid
from pathlib import Path


class AudioStore:
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.audio_dir = self.storage_dir / "audio"
        self.voices_dir = self.storage_dir / "voices"
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def save_audio(self, audio, sample_rate: int, audio_id: str | None = None) -> str:
        raise NotImplementedError

    def get_audio(self, audio_id: str) -> bytes | None:
        path = self.audio_dir / audio_id
        if path.exists() and path.is_file():
            return path.read_bytes()
        return None

    def register_voice(self, voice_name: str, audio, sample_rate: int) -> str:
        raise NotImplementedError

    def list_voices(self) -> list[dict]:
        voices = []
        if not self.voices_dir.exists():
            return voices
        for vdir in sorted(self.voices_dir.iterdir()):
            if not vdir.is_dir():
                continue
            meta_path = vdir / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    voices.append(json.load(f))
        return voices

    def delete_voice(self, voice_id: str) -> bool:
        vdir = self.voices_dir / voice_id
        if vdir.exists() and vdir.is_dir():
            import shutil
            shutil.rmtree(vdir)
            return True
        return False
