"""Unit tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from models.schemas import (
    TTSRequest,
    TTSResponse,
    RegisterVoiceRequest,
    SynthesizeCloneRequest,
    VoiceInfo,
    VoiceListResponse,
)


# ---------------------------------------------------------------------------
# TTSRequest
# ---------------------------------------------------------------------------

class TestTTSRequest:
    def test_minimal_valid_request_passes_validation(self):
        req = TTSRequest(text="你好世界")
        assert req.text == "你好世界"
        assert req.voice_id == "default"
        assert req.language == "zh"
        assert req.instruction == ""
        assert req.stream is False

    def test_full_request_passes_validation(self):
        req = TTSRequest(
            text="Hello, this is a test.",
            voice_id="voice_abc",
            language="en",
            instruction="Please speak cheerfully.",
            stream=True,
        )
        assert req.voice_id == "voice_abc"
        assert req.language == "en"
        assert req.instruction == "Please speak cheerfully."
        assert req.stream is True

    def test_empty_text_fails_validation(self):
        with pytest.raises(ValidationError):
            TTSRequest(text="")

    def test_text_exactly_5000_chars_passes(self):
        text = "a" * 5000
        req = TTSRequest(text=text)
        assert len(req.text) == 5000

    def test_text_over_5000_chars_fails_validation(self):
        text = "a" * 5001
        with pytest.raises(ValidationError):
            TTSRequest(text=text)

    def test_missing_text_field_fails(self):
        with pytest.raises(ValidationError):
            TTSRequest()

    def test_stream_defaults_to_false(self):
        req = TTSRequest(text="test")
        assert req.stream is False

    def test_language_defaults_to_zh(self):
        req = TTSRequest(text="test")
        assert req.language == "zh"

    def test_voice_id_defaults_to_default(self):
        req = TTSRequest(text="test")
        assert req.voice_id == "default"

    def test_instruction_defaults_to_empty(self):
        req = TTSRequest(text="test")
        assert req.instruction == ""


# ---------------------------------------------------------------------------
# TTSResponse
# ---------------------------------------------------------------------------

class TestTTSResponse:
    def test_valid_response_passes(self):
        resp = TTSResponse(
            success=True,
            audio_url="/api/audio/tts_abc.wav",
            duration=3.5,
        )
        assert resp.success is True
        assert resp.audio_url == "/api/audio/tts_abc.wav"
        assert resp.duration == 3.5
        assert resp.format == "wav"
        assert resp.sample_rate == 24000

    def test_failure_response_passes(self):
        resp = TTSResponse(
            success=False,
            audio_url="",
            duration=0.0,
        )
        assert resp.success is False

    def test_duration_must_be_float(self):
        with pytest.raises(ValidationError):
            TTSResponse(success=True, audio_url="/x", duration="not_a_float")


# ---------------------------------------------------------------------------
# RegisterVoiceRequest
# ---------------------------------------------------------------------------

class TestRegisterVoiceRequest:
    def test_valid_request_passes(self):
        req = RegisterVoiceRequest(voice_name="my voice")
        assert req.voice_name == "my voice"
        assert req.prompt_text == ""

    def test_prompt_text_is_optional(self):
        req = RegisterVoiceRequest(voice_name="test", prompt_text="你好")
        assert req.prompt_text == "你好"

    def test_empty_voice_name_fails(self):
        with pytest.raises(ValidationError):
            RegisterVoiceRequest(voice_name="")

    def test_voice_name_over_50_chars_fails(self):
        with pytest.raises(ValidationError):
            RegisterVoiceRequest(voice_name="n" * 51)


# ---------------------------------------------------------------------------
# SynthesizeCloneRequest
# ---------------------------------------------------------------------------

class TestSynthesizeCloneRequest:
    def test_valid_request_passes(self):
        req = SynthesizeCloneRequest(text="Hello", voice_id="voice_123")
        assert req.text == "Hello"
        assert req.voice_id == "voice_123"
        assert req.language == "zh"
        assert req.instruction == ""

    def test_empty_text_fails(self):
        with pytest.raises(ValidationError):
            SynthesizeCloneRequest(text="", voice_id="v1")

    def test_missing_voice_id_fails(self):
        with pytest.raises(ValidationError):
            SynthesizeCloneRequest(text="Hello")

    def test_text_too_long_fails(self):
        with pytest.raises(ValidationError):
            SynthesizeCloneRequest(text="a" * 5001, voice_id="v1")


# ---------------------------------------------------------------------------
# VoiceInfo & VoiceListResponse
# ---------------------------------------------------------------------------

class TestVoiceInfo:
    def test_valid_voice_info_passes(self):
        vi = VoiceInfo(
            voice_id="v001",
            voice_name="test voice",
            created_at="2026-05-12T10:30:00",
            duration=10.2,
        )
        assert vi.voice_id == "v001"


class TestVoiceListResponse:
    def test_empty_list_passes(self):
        resp = VoiceListResponse(voices=[])
        assert resp.voices == []

    def test_with_multiple_voices(self):
        resp = VoiceListResponse(
            voices=[
                VoiceInfo(voice_id="1", voice_name="a", created_at="x", duration=1.0),
                VoiceInfo(voice_id="2", voice_name="b", created_at="y", duration=2.0),
            ]
        )
        assert len(resp.voices) == 2

    def test_missing_voices_field_fails(self):
        with pytest.raises(ValidationError):
            VoiceListResponse()
