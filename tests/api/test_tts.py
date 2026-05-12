"""API tests for TTS endpoints — using real backend routes."""

import pytest


@pytest.fixture
def client(async_client):
    """Use the async_client from conftest (real app + mocked engine/store)."""
    return async_client


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
        for field in ("success", "audio_url", "duration", "format", "sample_rate"):
            assert field in data, f"missing field: {field}"

    @pytest.mark.asyncio
    async def test_tts_with_instruction(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "今天是个好日子", "instruction": "请用开心语气朗读",
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
