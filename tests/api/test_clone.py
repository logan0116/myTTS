"""API tests for voice clone endpoints — using real backend routes."""

import pytest


@pytest.fixture
def client(async_client):
    return async_client


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
        # Register first
        files = {"audio_file": ("test.wav", fake_audio_bytes, "audio/wav")}
        data = {"voice_name": "synth_voice"}
        reg_resp = await client.post("/api/clone/register-voice", files=files, data=data)
        voice_id = reg_resp.json()["voice_id"]

        # Then synthesize
        resp = await client.post("/api/clone/synthesize", json={
            "text": "你好，这是克隆的声音", "voice_id": voice_id,
        })
        assert resp.status_code == 200
        result = resp.json()
        assert result["success"] is True
        assert result["duration"] > 0

    @pytest.mark.asyncio
    async def test_synthesize_unknown_voice_returns_404(self, client):
        resp = await client.post("/api/clone/synthesize", json={
            "text": "test", "voice_id": "nonexistent_voice",
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_synthesize_empty_text_returns_422(self, client):
        resp = await client.post("/api/clone/synthesize", json={
            "text": "", "voice_id": "some_voice",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_synthesize_missing_voice_id_returns_422(self, client):
        resp = await client.post("/api/clone/synthesize", json={"text": "hello"})
        assert resp.status_code == 422
