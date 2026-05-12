"""API tests for TTS endpoints."""

import io

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from backend.api.tts import router as tts_router
from backend.models.schemas import TTSRequest, TTSResponse


def create_test_app(mock_engine, mock_audio_store):
    """Build a FastAPI app wired to mocked engine and audio store."""
    app = FastAPI()

    async def get_engine(request: Request):
        return mock_engine

    async def get_store(request: Request):
        return mock_audio_store

    # Override the route with one that actually uses mocked deps
    @app.post("/api/tts/text-to-speech", response_model=TTSResponse)
    async def text_to_speech(req: TTSRequest, request: Request):
        engine = request.app.state.engine
        store = request.app.state.audio_store

        # Gather audio chunks
        chunks = list(engine.tts_zero_shot(
            req.text,
            "You are a helpful assistant.<|endofprompt|>希望你以后能做得更好。",
            str(store.audio_dir / "default_prompt.wav"),
            stream=req.stream,
        ))

        # Save audio
        import numpy as np
        import wave

        audio = chunks[0]["tts_speech"]
        arr = np.squeeze(np.asarray(audio))
        duration = len(arr) / engine.sample_rate

        audio_id = "tts_test_output.wav"
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

    # Audio serving endpoint
    @app.get("/api/audio/{audio_id}")
    async def get_audio(audio_id: str):
        from backend.storage.audio_store import AudioStore
        path = app.state.audio_store.audio_dir / audio_id
        if path.exists():
            return FileResponse(str(path), media_type="audio/wav")
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "not found"}, status_code=404)

    app.state.engine = mock_engine
    app.state.audio_store = mock_audio_store
    return app


@pytest.fixture
def app(mock_engine, mock_audio_store):
    return create_test_app(mock_engine, mock_audio_store)


@pytest.fixture
def client(app):
    from httpx import AsyncClient, ASGITransport
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestTTSEndpoint:
    @pytest.mark.asyncio
    async def test_tts_with_default_voice_returns_200(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={"text": "你好世界"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "audio_url" in data
        assert data["duration"] > 0

    @pytest.mark.asyncio
    async def test_tts_empty_text_returns_422(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={"text": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_tts_missing_text_returns_422(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_tts_audio_url_is_accessible(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={"text": "test"})
        audio_url = resp.json()["audio_url"]
        audio_resp = await client.get(audio_url)
        assert audio_resp.status_code == 200
        assert audio_resp.headers["content-type"] == "audio/wav"

    @pytest.mark.asyncio
    async def test_tts_response_has_expected_fields(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={"text": "hello"})
        data = resp.json()
        assert "success" in data
        assert "audio_url" in data
        assert "duration" in data
        assert "format" in data
        assert "sample_rate" in data

    @pytest.mark.asyncio
    async def test_tts_with_instruction(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "今天是个好日子",
            "instruction": "请用开心语气朗读",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_tts_with_stream_flag(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "hello", "stream": True,
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_tts_special_characters(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "数字123，English混合!@#$%",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_tts_very_short_text(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={"text": "嗯"})
        assert resp.status_code == 200
        assert resp.json()["duration"] > 0

    @pytest.mark.asyncio
    async def test_tts_long_text(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={"text": "测试" * 500})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_tts_text_too_long_returns_422(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={"text": "a" * 5001})
        assert resp.status_code == 422


class TestAudioServing:
    @pytest.mark.asyncio
    async def test_get_nonexistent_audio_returns_404(self, client):
        resp = await client.get("/api/audio/nonexistent.wav")
        assert resp.status_code == 404
