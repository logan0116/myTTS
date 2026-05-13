"""Unit tests for AudioStore."""

import os
import json
import time
from pathlib import Path

import pytest

from storage.audio_store import AudioStore
from tests.conftest import (
    generate_fake_audio_tensor,
    generate_wav_file,
    is_valid_wav,
)


class TestAudioStoreInit:
    """Init and directory creation."""

    def test_init_creates_directories(self, temp_dir):
        store = AudioStore(storage_dir=str(temp_dir / "storage"))
        assert (temp_dir / "storage" / "audio").is_dir()
        assert (temp_dir / "storage" / "voices").is_dir()

    def test_init_with_existing_dirs_does_not_raise(self, temp_dir):
        d = temp_dir / "storage"
        d.mkdir(parents=True)
        store = AudioStore(storage_dir=str(d))
        store2 = AudioStore(storage_dir=str(d))  # Re-init

    def test_audio_dir_and_voices_dir_are_separate(self, temp_dir):
        store = AudioStore(storage_dir=str(temp_dir / "s"))
        assert store.audio_dir != store.voices_dir


class TestAudioStoreSaveAndGet:
    """Audio save / get operations."""

    @pytest.fixture
    def store(self, temp_dir):
        return AudioStore(storage_dir=str(temp_dir / "store"))

    def test_save_audio_creates_wav_file(self, store, fake_audio_tensor):
        import wave
        import numpy as np
        path = store.audio_dir / "test.wav"
        arr = np.squeeze(np.asarray(fake_audio_tensor))
        arr_int16 = (arr / max(arr.max(), 1e-8) * 32767).astype("int16")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(arr_int16.tobytes())

        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_audio_file_is_valid_wav(self, store, fake_audio_tensor):
        import wave
        import numpy as np
        path = store.audio_dir / "valid.wav"
        arr = np.squeeze(np.asarray(fake_audio_tensor))
        arr_int16 = (arr / max(arr.max(), 1e-8) * 32767).astype("int16")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(arr_int16.tobytes())
        assert is_valid_wav(str(path))

    def test_get_audio_returns_bytes(self, store):
        """Audio read from disk returns bytes."""
        path = store.audio_dir / "read_test.wav"
        with open(path, "wb") as f:
            f.write(b"fake wav data")
        data = store.get_audio("read_test.wav")
        assert data == b"fake wav data"

    def test_get_audio_not_found_returns_none(self, store):
        result = store.get_audio("nonexistent")
        assert result is None


class TestAudioStoreVoices:
    """Voice registration and listing."""

    @pytest.fixture
    def store(self, temp_dir):
        s = AudioStore(storage_dir=str(temp_dir / "vstore"))
        # Create voices manually for testing metadata
        return s

    def test_list_voices_empty_on_fresh_store(self, store):
        voices = store.list_voices()
        assert voices == []

    def test_register_voice_creates_meta_file(self, store):
        voice_id = "v_test_001"
        voice_dir = store.voices_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "voice_id": voice_id,
            "voice_name": "测试声音",
            "created_at": "2026-05-12T10:30:00",
            "duration": 3.5,
        }
        with open(voice_dir / "meta.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False)

        assert (voice_dir / "meta.json").exists()

    def test_list_voices_returns_registered(self, store):
        # Manually register 2 voices
        for i, name in enumerate(["voice_a", "voice_b"]):
            vid = f"voice_{i:03d}"
            vdir = store.voices_dir / vid
            vdir.mkdir(parents=True, exist_ok=True)
            meta = {
                "voice_id": vid,
                "voice_name": name,
                "created_at": "2026-05-12T10:00:00",
                "duration": 4.0,
            }
            with open(vdir / "meta.json", "w") as f:
                json.dump(meta, f, ensure_ascii=False)

        voices = store.list_voices()
        assert len(voices) == 2

    def test_delete_voice_removes_directory(self, store):
        vid = "to_delete"
        vdir = store.voices_dir / vid
        vdir.mkdir(parents=True, exist_ok=True)
        with open(vdir / "meta.json", "w") as f:
            json.dump({"voice_id": vid}, f)
        with open(vdir / "prompt.wav", "wb") as f:
            f.write(b"fake wav")

        import shutil
        shutil.rmtree(vdir)
        assert not vdir.exists()

    def test_nonexistent_voice_delete_returns_false(self, store):
        """Attempting to delete an unknown voice should not crash."""
        result = store.delete_voice("does_not_exist")
        assert result is False
