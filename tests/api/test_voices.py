"""API tests for voice management endpoints — using real backend routes."""

import json

import pytest


@pytest.fixture
def client(async_client):
    return async_client


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

    @pytest.mark.asyncio
    async def test_delete_voice_success(self, client, mock_audio_store):
        # Pre-register a voice
        vdir = mock_audio_store.voices_dir / "to_del"
        vdir.mkdir(parents=True, exist_ok=True)
        meta = {"voice_id": "to_del", "voice_name": "x", "created_at": "x", "duration": 1.0}
        with open(vdir / "meta.json", "w") as f:
            json.dump(meta, f)

        resp = await client.delete("/api/voices/to_del")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_voice_returns_404(self, client):
        resp = await client.delete("/api/voices/nonexistent")
        assert resp.status_code == 404
