"""API tests for voice clone endpoints."""

import io
import json

import pytest
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

from backend.models.schemas import SynthesizeCloneRequest, TTSResponse
from tests.conftest import generate_fake_audio_tensor, npy_to_wav_bytes


def create_clone_test_app(mock_engine, mock_audio_store):
    """Build a FastAPI app with clone routes wired to mocks."""
    app = FastAPI()

    # POST /api/clone/register-voice
    @app.post("/api/clone/register-voice")
    async def register_voice(
        audio_file: UploadFile = File(...),
        voice_name: str = Form(...),
        prompt_text: str = Form(""),
    ):
        store = app.state.audio_store
        import uuid
        voice_id = f"voice_{uuid.uuid4().hex[:8]}"

        # Save reference audio
        vdir = store.voices_dir / voice_id
        vdir.mkdir(parents=True, exist_ok=True)
        content = await audio_file.read()
        with open(vdir / "prompt.wav", "wb") as f:
            f.write(content)

        # Save meta
        import json
        from datetime import datetime
        meta = {
            "voice_id": voice_id,
            "voice_name": voice_name,
            "created_at": datetime.now().isoformat(),
        }
        with open(vdir / "meta.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False)

        return {
            "success": True,
            "voice_id": voice_id,
            "voice_name": voice_name,
            "message": "声音学习成功",
        }

    # POST /api/clone/synthesize
    @app.post("/api/clone/synthesize", response_model=TTSResponse)
    async def synthesize(req: SynthesizeCloneRequest, request: Request):
        engine = request.app.state.engine
        store = request.app.state.audio_store

        # Check voice exists
        vdir = store.voices_dir / req.voice_id
        if not vdir.exists():
            return JSONResponse({"detail": "voice not found"}, status_code=404)

        prompt_path = vdir / "prompt.wav"

        chunks = list(engine.tts_zero_shot(
            req.text,
            "You are a helpful assistant.<|endofprompt|>",
            str(prompt_path),
            stream=False,
        ))

        audio = chunks[0]["tts_speech"]
        arr = np.squeeze(np.asarray(audio))
        duration = len(arr) / engine.sample_rate

        import wave
        audio_id = f"clone_{req.voice_id}.wav"
        path = store.audio_dir / audio_id
        arr_int16 = (arr / max(arr.max(), 1e-8) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(engine.sample_rate)
            wf.writeframes(arr_int16.tobytes())

        return TTSResponse(
            success=True,
            audio_url=f"/api/audio/{audio_id}",
            duration=round(duration, 2),
        )

    app.state.engine = mock_engine
    app.state.audio_store = mock_audio_store
    return app


@pytest.fixture
def app(mock_engine, mock_audio_store):
    return create_clone_test_app(mock_engine, mock_audio_store)


@pytest.fixture
def client(app):
    from httpx import AsyncClient, ASGITransport
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestRegisterVoice:
    @pytest.mark.asyncio
    async def test_register_with_valid_audio(self, client, fake_audio_bytes):
        files = {"audio_file": ("test.wav", fake_audio_bytes, "audio/wav")}
        data = {"voice_name": "test_voice"}
        resp = await client.post("/api/clone/register-voice", files=files, data=data)
        assert resp.status_code == 200
        result = resp.json()
        assert result["success"] is True
        assert result["voice_id"].startswith("voice_")
        assert result["voice_name"] == "test_voice"

    @pytest.mark.asyncio
    async def test_register_without_name_returns_422(self, client, fake_audio_bytes):
        files = {"audio_file": ("test.wav", fake_audio_bytes, "audio/wav")}
        # Missing voice_name field → FastAPI returns 422 for missing required form param
        resp = await client.post("/api/clone/register-voice", files=files)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_register_with_prompt_text(self, client, fake_audio_bytes):
        files = {"audio_file": ("test.wav", fake_audio_bytes, "audio/wav")}
        data = {"voice_name": "with_prompt", "prompt_text": "这是我的声音"}
        resp = await client.post("/api/clone/register-voice", files=files, data=data)
        assert resp.status_code == 200
        assert resp.json()["voice_name"] == "with_prompt"


class TestSynthesizeClone:
    @pytest.mark.asyncio
    async def test_synthesize_with_registered_voice(self, client, fake_audio_bytes):
        # First register
        files = {"audio_file": ("test.wav", fake_audio_bytes, "audio/wav")}
        data = {"voice_name": "synth_voice"}
        reg_resp = await client.post("/api/clone/register-voice", files=files, data=data)
        voice_id = reg_resp.json()["voice_id"]

        # Then synthesize
        resp = await client.post("/api/clone/synthesize", json={
            "text": "你好，这是克隆的声音",
            "voice_id": voice_id,
        })
        assert resp.status_code == 200
        result = resp.json()
        assert result["success"] is True
        assert result["duration"] > 0

    @pytest.mark.asyncio
    async def test_synthesize_unknown_voice_returns_404(self, client):
        resp = await client.post("/api/clone/synthesize", json={
            "text": "test",
            "voice_id": "nonexistent_voice",
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_synthesize_empty_text_returns_422(self, client):
        resp = await client.post("/api/clone/synthesize", json={
            "text": "",
            "voice_id": "some_voice",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_synthesize_missing_voice_id_returns_422(self, client):
        resp = await client.post("/api/clone/synthesize", json={"text": "hello"})
        assert resp.status_code == 422
