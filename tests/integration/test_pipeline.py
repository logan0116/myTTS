"""Integration tests — end-to-end TTS & clone pipelines with real backend routes."""

import io
import json
import wave

import pytest


@pytest.fixture
def client(async_client):
    return async_client


# ---------------------------------------------------------------------------
# TTS pipeline
# ---------------------------------------------------------------------------

class TestTTSPipeline:
    @pytest.mark.asyncio
    async def test_full_tts_flow(self, client):
        """Text → API → engine → WAV file → playable audio."""
        resp = await client.post("/api/tts/text-to-speech", json={"text": "你好世界"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

        audio_resp = await client.get(data["audio_url"])
        assert audio_resp.status_code == 200
        assert audio_resp.headers["content-type"] == "audio/wav"

        buf = io.BytesIO(audio_resp.content)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 24000
            assert wf.getnframes() > 0

    @pytest.mark.asyncio
    async def test_full_cross_lingual_flow(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "Hello, this is a cross-lingual test.", "language": "en",
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    @pytest.mark.asyncio
    async def test_full_instruct_flow(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "今天很开心", "instruction": "请用开心语气朗读",
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Clone pipeline
# ---------------------------------------------------------------------------

class TestClonePipeline:
    @pytest.mark.asyncio
    async def test_register_then_synthesize(self, client, fake_audio_bytes):
        # Register
        files = {"audio_file": ("ref.wav", fake_audio_bytes, "audio/wav")}
        data = {"voice_name": "pipeline_voice"}
        reg_resp = await client.post("/api/clone/register-voice", files=files, data=data)
        assert reg_resp.status_code == 200
        voice_id = reg_resp.json()["voice_id"]

        # Synthesize
        synth_resp = await client.post("/api/clone/synthesize", json={
            "text": "这是流水线测试", "voice_id": voice_id,
        })
        assert synth_resp.status_code == 200

        # Verify audio
        audio_resp = await client.get(synth_resp.json()["audio_url"])
        assert audio_resp.status_code == 200

    @pytest.mark.asyncio
    async def test_voice_appears_in_list_after_register(self, client, fake_audio_bytes):
        files = {"audio_file": ("ref.wav", fake_audio_bytes, "audio/wav")}
        data = {"voice_name": "listed_voice"}
        reg_resp = await client.post("/api/clone/register-voice", files=files, data=data)
        assert reg_resp.status_code == 200

        list_resp = await client.get("/api/voices")
        voices = list_resp.json()["voices"]
        assert any(v["voice_name"] == "listed_voice" for v in voices)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_tts_requests(self, client):
        import asyncio

        async def req(text):
            resp = await client.post("/api/tts/text-to-speech", json={"text": text})
            return resp.status_code

        results = await asyncio.gather(req("请求一"), req("请求二"))
        assert all(s == 200 for s in results)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

class TestOutputFormat:
    @pytest.mark.asyncio
    async def test_output_is_valid_wav(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={"text": "test"})
        audio_resp = await client.get(resp.json()["audio_url"])
        buf = io.BytesIO(audio_resp.content)
        with wave.open(buf, "rb") as wf:
            assert wf.getframerate() == 24000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() > 0

    @pytest.mark.asyncio
    async def test_output_duration_matches_metadata(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={"text": "测试时长"})
        data = resp.json()
        audio_resp = await client.get(data["audio_url"])
        buf = io.BytesIO(audio_resp.content)
        with wave.open(buf, "rb") as wf:
            actual_duration = wf.getnframes() / wf.getframerate()
        assert abs(actual_duration - data["duration"]) < 0.1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCasesPipeline:
    @pytest.mark.asyncio
    async def test_multilingual_mixed_text(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "今天天气nice，我们去shopping吧",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_numbers_and_symbols(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "订单号#12345，价格￥99.80，电话138-0000-1234",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_long_text_under_limit(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "这是一段较长的测试文本。" * 100,
        })
        assert resp.status_code == 200
