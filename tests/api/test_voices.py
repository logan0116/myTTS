"""API tests for voice management endpoints."""

import json

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.models.schemas import VoiceListResponse, VoiceInfo


def create_voices_test_app(mock_engine, mock_audio_store):
    """Build a FastAPI app with voices routes wired to mocks."""
    app = FastAPI()

    @app.get("/api/voices", response_model=VoiceListResponse)
    async def list_voices(request: Request):
        store = request.app.state.audio_store
        voices_data = store.list_voices()
        return VoiceListResponse(
            voices=[VoiceInfo(**v) for v in voices_data]
        )

    app.state.engine = mock_engine
    app.state.audio_store = mock_audio_store
    return app


@pytest.fixture
def app(mock_engine, mock_audio_store):
    return create_voices_test_app(mock_engine, mock_audio_store)


@pytest.fixture
def client(app):
    from httpx import AsyncClient, ASGITransport
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestVoicesEndpoint:
    @pytest.mark.asyncio
    async def test_list_voices_returns_array(self, client):
        resp = await client.get("/api/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert "voices" in data
        assert isinstance(data["voices"], list)

    @pytest.mark.asyncio
    async def test_list_voices_empty_on_fresh_start(self, client):
        resp = await client.get("/api/voices")
        assert resp.status_code == 200
        assert resp.json()["voices"] == []

    @pytest.mark.asyncio
    async def test_list_voices_contains_registered(self, client, mock_audio_store):
        # Pre-register a voice manually
        vdir = mock_audio_store.voices_dir / "v_test"
        vdir.mkdir(parents=True, exist_ok=True)
        meta = {
            "voice_id": "v_test",
            "voice_name": "hello",
            "created_at": "2026-05-12T10:00:00",
            "duration": 3.0,
        }
        with open(vdir / "meta.json", "w") as f:
            json.dump(meta, f)

        resp = await client.get("/api/voices")
        assert resp.status_code == 200
        voices = resp.json()["voices"]
        assert len(voices) == 1
        assert voices[0]["voice_id"] == "v_test"
