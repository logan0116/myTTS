"""Audio file storage manager."""

import os
import io
import json
import uuid
import struct
import wave
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


class AudioStore:
    """Manages TTS output audio and registered voice profiles on disk."""

    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.audio_dir = self.storage_dir / "audio"
        self.voices_dir = self.storage_dir / "voices"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Audio save / get
    # ------------------------------------------------------------------

    def save_audio(
        self,
        audio,               # numpy array or torch Tensor
        sample_rate: int,
        audio_id: str | None = None,
    ) -> str:
        """Save audio as 16-bit mono WAV and return the audio_id (filename).

        Args:
            audio: numpy array (shape [num_samples] or [1, num_samples]) or torch Tensor.
            sample_rate: sample rate in Hz.
            audio_id: optional filename; auto-generated if None.

        Returns:
            The audio_id (filename stem + .wav) used to retrieve the file.
        """
        if audio_id is None:
            audio_id = f"{uuid.uuid4().hex}.wav"
        if not audio_id.endswith(".wav"):
            audio_id += ".wav"

        arr = np.squeeze(np.asarray(audio)).astype(np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1-D audio, got shape {arr.shape}")

        # Normalize to int16
        peak = max(abs(arr.max()), abs(arr.min()), 1e-8)
        arr_int16 = (arr / peak * 32767).astype(np.int16)

        path = self.audio_dir / audio_id
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(arr_int16.tobytes())

        logger.debug("Saved audio %s (%.2fs)", audio_id, len(arr) / sample_rate)
        return audio_id

    def get_audio(self, audio_id: str) -> bytes | None:
        """Read audio file as raw bytes."""
        path = self.audio_dir / audio_id
        if path.exists() and path.is_file():
            return path.read_bytes()
        return None

    def get_audio_path(self, audio_id: str) -> Path | None:
        """Return the absolute path to an audio file, or None."""
        path = self.audio_dir / audio_id
        return path if path.exists() else None

    # ------------------------------------------------------------------
    # Voice registration
    # ------------------------------------------------------------------

    def register_voice(
        self,
        voice_name: str,
        audio,               # reference audio (numpy array or torch Tensor)
        sample_rate: int,
        prompt_text: str = "",
    ) -> str:
        """Register a new voice from reference audio.

        Saves the reference audio as WAV and writes meta.json.

        Returns:
            voice_id string.
        """
        voice_id = f"voice_{uuid.uuid4().hex[:8]}"
        vdir = self.voices_dir / voice_id
        vdir.mkdir(parents=True, exist_ok=True)

        # Save reference audio
        prompt_path = vdir / "prompt.wav"
        arr = np.squeeze(np.asarray(audio)).astype(np.float32)
        peak = max(abs(arr.max()), abs(arr.min()), 1e-8)
        arr_int16 = (arr / peak * 32767).astype(np.int16)

        with wave.open(str(prompt_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(arr_int16.tobytes())

        duration = len(arr) / sample_rate

        # Write metadata
        meta = {
            "voice_id": voice_id,
            "voice_name": voice_name,
            "prompt_text": prompt_text,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration": round(duration, 3),
            "sample_rate": sample_rate,
        }
        with open(vdir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("Registered voice '%s' -> %s", voice_name, voice_id)
        return voice_id

    def get_prompt_audio_path(self, voice_id: str) -> str | None:
        """Return the path to the prompt.wav for a registered voice, or None."""
        prompt = self.voices_dir / voice_id / "prompt.wav"
        return str(prompt) if prompt.exists() else None

    # ------------------------------------------------------------------
    # Voice listing & deletion
    # ------------------------------------------------------------------

    def list_voices(self) -> list[dict]:
        """Return all registered voice metadata, newest first."""
        voices: list[dict] = []
        if not self.voices_dir.exists():
            return voices
        for vdir in sorted(self.voices_dir.iterdir(), reverse=True):
            if not vdir.is_dir():
                continue
            meta_path = vdir / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        voices.append(json.load(f))
                except (json.JSONDecodeError, OSError):
                    logger.warning("Skipping unreadable meta.json in %s", vdir)
        return voices

    def delete_voice(self, voice_id: str) -> bool:
        """Delete a registered voice directory. Returns True if found."""
        vdir = self.voices_dir / voice_id
        if vdir.exists() and vdir.is_dir():
            shutil.rmtree(vdir)
            logger.info("Deleted voice %s", voice_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_old_audio(self, ttl_hours: int = 24) -> int:
        """Remove temporary audio files older than *ttl_hours*. Returns count removed."""
        cutoff = datetime.now(timezone.utc).timestamp() - ttl_hours * 3600
        removed = 0
        for f in self.audio_dir.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        if removed:
            logger.info("Cleaned up %d old audio files.", removed)
        return removed
