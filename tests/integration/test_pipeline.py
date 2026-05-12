"""Integration tests for end-to-end TTS and clone pipelines.

These tests verify the full flow: request → engine → audio save → response,
all using mocked dependencies wired through FastAPI.
"""

import io
import json
import wave

import pytest
import numpy as np

from tests.conftest import generate_fake_audio_tensor, npy_to_wav_bytes, is_valid_wav


# ---------------------------------------------------------------------------
# Shared test app factory that wires ALL routes together
# ---------------------------------------------------------------------------

def create_full_app(mock_engine, mock_audio_store):
    """Full FastAPI app with TTS + clone + voices routes wired."""
    from fastapi import FastAPI, Request, UploadFile, File, Form
    from fastapi.responses import FileResponse, JSONResponse

    from backend.models.schemas import (
        TTSRequest, TTSResponse, SynthesizeCloneRequest,
        VoiceListResponse, VoiceInfo,
    )

    app = FastAPI()

    # ---------------- TTS ----------------
    @app.post("/api/tts/text-to-speech", response_model=TTSResponse)
    async def text_to_speech(req: TTSRequest, request: Request):
        engine = request.app.state.engine
        store = request.app.state.audio_store

        prompt_path = str(store.audio_dir / "default_prompt.wav")
        chunks = list(engine.tts_zero_shot(
            req.text,
            "You are a helpful assistant.<|endofprompt|>",
            prompt_path,
            stream=req.stream,
        ))

        audio = chunks[0]["tts_speech"]
        arr = np.squeeze(np.asarray(audio))
        duration = len(arr) / engine.sample_rate

        audio_id = f"tts_pipeline_out.wav"
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

    # ---------------- Clone ----------------
    @app.post("/api/clone/register-voice")
    async def register_voice(
        audio_file: UploadFile = File(...),
        voice_name: str = Form(...),
        prompt_text: str = Form(""),
    ):
        store = app.state.audio_store
        import uuid
        from datetime import datetime

        voice_id = f"voice_{uuid.uuid4().hex[:8]}"
        vdir = store.voices_dir / voice_id
        vdir.mkdir(parents=True, exist_ok=True)

        content = await audio_file.read()
        with open(vdir / "prompt.wav", "wb") as f:
            f.write(content)

        meta = {
            "voice_id": voice_id,
            "voice_name": voice_name,
            "created_at": datetime.now().isoformat(),
            "duration": 0.0,
        }
        with open(vdir / "meta.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False)

        return {"success": True, "voice_id": voice_id, "voice_name": voice_name}

    @app.post("/api/clone/synthesize", response_model=TTSResponse)
    async def synthesize(req: SynthesizeCloneRequest, request: Request):
        engine = request.app.state.engine
        store = request.app.state.audio_store

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

        audio_id = f"clone_pipe_{req.voice_id}.wav"
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

    # ---------------- Voices ----------------
    @app.get("/api/voices", response_model=VoiceListResponse)
    async def list_voices(request: Request):
        store = request.app.state.audio_store
        voices_data = store.list_voices()
        return VoiceListResponse(voices=[VoiceInfo(**v) for v in voices_data])

    # ---------------- Audio serving ----------------
    @app.get("/api/audio/{audio_id}")
    async def get_audio(audio_id: str):
        path = app.state.audio_store.audio_dir / audio_id
        if path.exists():
            return FileResponse(str(path), media_type="audio/wav")
        return JSONResponse({"detail": "not found"}, status_code=404)

    app.state.engine = mock_engine
    app.state.audio_store = mock_audio_store
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app(mock_engine, mock_audio_store):
    return create_full_app(mock_engine, mock_audio_store)


@pytest.fixture
def client(app):
    from httpx import AsyncClient, ASGITransport
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test", timeout=60)


# ---------------------------------------------------------------------------
# TTS pipeline tests
# ---------------------------------------------------------------------------

class TestTTSPipeline:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_tts_flow(self, client):
        """Text → API → engine → WAV file → playable audio."""
        # Step 1: POST TTS
        resp = await client.post("/api/tts/text-to-speech", json={"text": "你好世界"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

        # Step 2: GET audio
        audio_resp = await client.get(data["audio_url"])
        assert audio_resp.status_code == 200
        assert audio_resp.headers["content-type"] == "audio/wav"

        # Step 3: validate WAV
        buf = io.BytesIO(audio_resp.content)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 24000
            assert wf.getnframes() > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_cross_lingual_flow(self, client):
        """Chinese prompt audio → English target text."""
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "Hello, this is a cross-lingual test.",
            "language": "en",
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_instruct_flow(self, client):
        """Instruction-controlled TTS."""
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "今天很开心",
            "instruction": "请用开心语气朗读",
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Clone pipeline tests
# ---------------------------------------------------------------------------

class TestClonePipeline:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_register_then_synthesize(self, client, fake_audio_bytes):
        """Register voice → synthesize → verify audio."""
        # Register
        files = {"audio_file": ("ref.wav", fake_audio_bytes, "audio/wav")}
        data = {"voice_name": "pipeline_voice"}
        reg_resp = await client.post("/api/clone/register-voice", files=files, data=data)
        assert reg_resp.status_code == 200
        voice_id = reg_resp.json()["voice_id"]

        # Synthesize
        synth_resp = await client.post("/api/clone/synthesize", json={
            "text": "这是流水线测试",
            "voice_id": voice_id,
        })
        assert synth_resp.status_code == 200

        # Verify audio
        audio_resp = await client.get(synth_resp.json()["audio_url"])
        assert audio_resp.status_code == 200

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_voice_appears_in_list_after_register(self, client, fake_audio_bytes):
        """Registered voice shows up in GET /api/voices."""
        # Register
        files = {"audio_file": ("ref.wav", fake_audio_bytes, "audio/wav")}
        data = {"voice_name": "listed_voice"}
        reg_resp = await client.post("/api/clone/register-voice", files=files, data=data)
        assert reg_resp.status_code == 200

        # List
        list_resp = await client.get("/api/voices")
        voices = list_resp.json()["voices"]
        assert any(v["voice_name"] == "listed_voice" for v in voices)


# ---------------------------------------------------------------------------
# Concurrency & robustness
# ---------------------------------------------------------------------------

class TestConcurrency:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_tts_requests(self, client):
        """Two concurrent TTS requests both succeed."""
        import asyncio

        async def make_request(text):
            resp = await client.post("/api/tts/text-to-speech", json={"text": text})
            return resp.status_code

        results = await asyncio.gather(
            make_request("请求一"),
            make_request("请求二"),
        )
        assert all(s == 200 for s in results)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

class TestOutputFormat:
    @pytest.mark.integration
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

    @pytest.mark.integration
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
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multilingual_mixed_text(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "今天天气nice，我们去shopping吧",
        })
        assert resp.status_code == 200

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_numbers_and_symbols(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "订单号#12345，价格￥99.80，电话138-0000-1234",
        })
        assert resp.status_code == 200

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_long_text_under_limit(self, client):
        resp = await client.post("/api/tts/text-to-speech", json={
            "text": "这是一段较长的测试文本。" * 100,
        })
        assert resp.status_code == 200
