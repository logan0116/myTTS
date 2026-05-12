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
# Lazy torch helpers
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
    data = generate_silent_wav_bytes(duration_sec, sample_rate)
    with open(path, "wb") as f:
        f.write(data)


def generate_fake_audio_tensor(duration_sec: float = 1.0, sample_rate: int = 24000):
    num_samples = int(sample_rate * duration_sec)
    try:
        return _get_torch().randn(1, num_samples)
    except ImportError:
        return np.random.randn(1, num_samples).astype(np.float32)


def npy_to_wav_bytes(arr, sample_rate: int = 24000) -> bytes:
    arr = np.squeeze(arr)
    peak = max(abs(arr.max()), abs(arr.min()), 1e-8)
    arr_int16 = (arr / peak * 32767).astype(np.int16)
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
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_storage_dir(temp_dir):
    (temp_dir / "audio").mkdir(parents=True, exist_ok=True)
    (temp_dir / "voices").mkdir(parents=True, exist_ok=True)
    return temp_dir


# ---------------------------------------------------------------------------
# Fixture audio files (on-disk)
# ---------------------------------------------------------------------------

@pytest.fixture
def prompt_zh_wav(temp_dir):
    path = temp_dir / "prompt_zh.wav"
    generate_wav_file(str(path), duration_sec=3.0)
    return str(path)


@pytest.fixture
def prompt_en_wav(temp_dir):
    path = temp_dir / "prompt_en.wav"
    generate_wav_file(str(path), duration_sec=3.0)
    return str(path)


@pytest.fixture
def silence_wav(temp_dir):
    path = temp_dir / "silence.wav"
    generate_wav_file(str(path), duration_sec=1.0)
    return str(path)


@pytest.fixture
def long_prompt_wav(temp_dir):
    path = temp_dir / "long_prompt.wav"
    generate_wav_file(str(path), duration_sec=15.0)
    return str(path)


# ---------------------------------------------------------------------------
# Audio tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_audio_tensor():
    return generate_fake_audio_tensor(1.0)


@pytest.fixture
def fake_audio_tensor_3s():
    return generate_fake_audio_tensor(3.0)


@pytest.fixture
def fake_audio_bytes(fake_audio_tensor):
    arr = np.squeeze(np.asarray(fake_audio_tensor))
    return npy_to_wav_bytes(arr, 24000)


# ---------------------------------------------------------------------------
# Mock CosyVoice engine
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_engine(prompt_zh_wav):
    """Mock CosyVoiceEngine with fake but realistic behavior."""
    from backend.engine.cosyvoice_engine import CosyVoiceEngine
    from backend.engine.cosyvoice_engine import ModelNotFoundError

    engine = CosyVoiceEngine(model_dir="/fake/model")

    # Simulate load
    engine.model = MagicMock()
    engine._ensure_loaded = MagicMock()

    def _fake_tts(*args, **kwargs):
        yield {"tts_speech": generate_fake_audio_tensor(2.0)}

    engine.tts_zero_shot = MagicMock(side_effect=_fake_tts)
    engine.tts_cross_lingual = MagicMock(side_effect=_fake_tts)
    engine.tts_instruct = MagicMock(side_effect=_fake_tts)
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
# Full FastAPI app with mocked engine + store on app.state
# ---------------------------------------------------------------------------

@pytest.fixture
def app_with_mocks(mock_engine, mock_audio_store):
    """Import the real FastAPI app, inject mocks onto app.state.

    Uses ASGI transport (no lifespan events fired), so the real model loader
    is never called.
    """
    from backend.main import app

    app.state.engine = mock_engine
    app.state.audio_store = mock_audio_store
    return app


@pytest.fixture
def async_client(app_with_mocks):
    """httpx AsyncClient bound to the real FastAPI app."""
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=app_with_mocks)
    return AsyncClient(transport=transport, base_url="http://test", timeout=60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate if rate > 0 else 0.0


def is_valid_wav(path: str) -> bool:
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() > 0
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that require real model or GPU")
    config.addinivalue_line("markers", "api: FastAPI endpoint tests")
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: end-to-end pipeline tests")
