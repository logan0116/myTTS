"""Pytest fixtures and configuration.

All heavy ML dependencies (torch, torchaudio) are imported lazily so that
unit and API tests can run without a GPU or the CosyVoice model installed.
"""

import io
import os
import sys
import json
import struct
import wave
import uuid
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Lazy torch helpers — only used when torch is installed
# ---------------------------------------------------------------------------

_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch as t
        _torch = t
    return _torch


# ---------------------------------------------------------------------------
# Generated test audio helpers
# ---------------------------------------------------------------------------

def generate_silent_wav_bytes(duration_sec: float = 3.0, sample_rate: int = 24000) -> bytes:
    """Generate silent WAV bytes for fixture audio files."""
    num_samples = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for _ in range(num_samples):
            wf.writeframes(struct.pack("<h", 0))
    return buf.getvalue()


def generate_wav_file(path: str, duration_sec: float = 3.0, sample_rate: int = 24000) -> None:
    """Write a silent WAV file to disk."""
    data = generate_silent_wav_bytes(duration_sec, sample_rate)
    with open(path, "wb") as f:
        f.write(data)


def generate_fake_audio_tensor(duration_sec: float = 1.0, sample_rate: int = 24000):
    """Return a fake audio tensor.

    Returns numpy array if torch is unavailable, torch.Tensor otherwise.
    """
    num_samples = int(sample_rate * duration_sec)
    try:
        return _get_torch().randn(1, num_samples)
    except ImportError:
        return np.random.randn(1, num_samples).astype(np.float32)


def npy_to_wav_bytes(arr, sample_rate: int = 24000) -> bytes:
    """Convert numpy audio array to WAV bytes (int16 PCM)."""
    arr = np.squeeze(arr)
    if arr.max() > 0:
        arr_int16 = (arr / arr.max() * 32767).astype(np.int16)
    else:
        arr_int16 = arr.astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(arr_int16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Temp directories
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Create a temporary directory, cleaned up after test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_storage_dir(temp_dir):
    """Temporary storage directory with audio/ and voices/ subdirs."""
    (temp_dir / "audio").mkdir(parents=True, exist_ok=True)
    (temp_dir / "voices").mkdir(parents=True, exist_ok=True)
    return temp_dir


# ---------------------------------------------------------------------------
# Fixture audio files (on-disk)
# ---------------------------------------------------------------------------

@pytest.fixture
def prompt_zh_wav(temp_dir):
    """A 3-second silent WAV used as Chinese prompt audio."""
    path = temp_dir / "prompt_zh.wav"
    generate_wav_file(str(path), duration_sec=3.0)
    return str(path)


@pytest.fixture
def prompt_en_wav(temp_dir):
    """A 3-second silent WAV used as English prompt audio."""
    path = temp_dir / "prompt_en.wav"
    generate_wav_file(str(path), duration_sec=3.0)
    return str(path)


@pytest.fixture
def silence_wav(temp_dir):
    """A 1-second silent WAV."""
    path = temp_dir / "silence.wav"
    generate_wav_file(str(path), duration_sec=1.0)
    return str(path)


@pytest.fixture
def long_prompt_wav(temp_dir):
    """A 15-second prompt audio."""
    path = temp_dir / "long_prompt.wav"
    generate_wav_file(str(path), duration_sec=15.0)
    return str(path)


# ---------------------------------------------------------------------------
# Audio tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_audio_tensor():
    """Return a 1-second fake mono audio tensor (torch or numpy)."""
    return generate_fake_audio_tensor(1.0)


@pytest.fixture
def fake_audio_tensor_3s():
    """Return a 3-second fake mono audio tensor."""
    return generate_fake_audio_tensor(3.0)


@pytest.fixture
def fake_audio_bytes(fake_audio_tensor):
    """WAV bytes of a generated audio tensor."""
    arr = np.squeeze(np.asarray(fake_audio_tensor))
    return npy_to_wav_bytes(arr, 24000)


# ---------------------------------------------------------------------------
# Mock CosyVoice engine
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_engine():
    """Mock CosyVoiceEngine: load/unload succeed, inference returns fake audio."""
    from backend.engine.cosyvoice_engine import CosyVoiceEngine

    engine = CosyVoiceEngine(model_dir="/fake/model")

    engine.load = MagicMock()
    engine.unload = MagicMock()
    engine.model = MagicMock()

    def _mock_tts(text, prompt_text, prompt_audio_path, stream=False):
        tensor = generate_fake_audio_tensor(2.0)
        yield {"tts_speech": tensor}

    engine.tts_zero_shot = MagicMock(side_effect=_mock_tts)
    engine.tts_cross_lingual = MagicMock(side_effect=_mock_tts)
    engine.tts_instruct = MagicMock(side_effect=_mock_tts)
    engine.sample_rate = 24000

    yield engine


# ---------------------------------------------------------------------------
# AudioStore backed by temp storage
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_audio_store(temp_storage_dir):
    """AudioStore backed by temporary directory."""
    from backend.storage.audio_store import AudioStore
    store = AudioStore(storage_dir=str(temp_storage_dir))
    return store


# ---------------------------------------------------------------------------
# FastAPI TestClient (without loading the real model)
# ---------------------------------------------------------------------------

@pytest.fixture
def fastapi_app_with_mocks(mock_engine, mock_audio_store):
    """Create FastAPI app with mocked engine and audio store on app.state."""
    from fastapi import FastAPI

    app = FastAPI()
    app.state.engine = mock_engine
    app.state.audio_store = mock_audio_store
    return app


@pytest.fixture
def async_client(fastapi_app_with_mocks):
    """httpx AsyncClient bound to FastAPI app via ASGI transport."""
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=fastapi_app_with_mocks)
    return AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# Helpers for reading WAV metadata
# ---------------------------------------------------------------------------

def get_wav_duration(path: str) -> float:
    """Return duration in seconds of a WAV file."""
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate if rate > 0 else 0.0


def is_valid_wav(path: str) -> bool:
    """Check whether file is a valid WAV file."""
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# pytest markers registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that require real model or GPU")
    config.addinivalue_line("markers", "api: FastAPI endpoint tests")
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: end-to-end pipeline tests")
