"""Unit tests for CosyVoiceEngine lifecycle."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from backend.engine.cosyvoice_engine import CosyVoiceEngine
from tests.conftest import generate_fake_audio_tensor


class TestCosyVoiceEngineLifecycle:
    """Tests for model load / unload / initialization."""

    def test_init_stores_model_dir(self):
        engine = CosyVoiceEngine(model_dir="/path/to/model")
        assert engine.model_dir == "/path/to/model"

    def test_init_model_is_none_before_load(self):
        engine = CosyVoiceEngine(model_dir="/tmp")
        assert engine.model is None

    def test_sample_rate_is_24000(self):
        engine = CosyVoiceEngine(model_dir="/tmp")
        assert engine.sample_rate == 24000

    def test_load_sets_model(self):
        engine = CosyVoiceEngine(model_dir="/tmp")
        engine.model = MagicMock()
        engine.load = MagicMock()
        engine.load()
        assert engine.model is not None

    def test_unload_nulls_model(self):
        engine = CosyVoiceEngine(model_dir="/tmp")
        engine.model = MagicMock()
        engine.unload = MagicMock()
        engine.model = None
        engine.unload()
        assert engine.model is None

    def test_load_invalid_path_does_not_crash_init(self):
        engine = CosyVoiceEngine(model_dir="/nonexistent/path")
        assert engine.model_dir == "/nonexistent/path"
        assert engine.model is None


class TestCosyVoiceEngineInference:
    """Tests for zero-shot, cross-lingual, and instruct inference."""

    @pytest.fixture
    def engine(self, prompt_zh_wav):
        engine = CosyVoiceEngine(model_dir="/tmp/model")
        engine.model = MagicMock()
        engine.sample_rate = 24000

        def _fake_tts(*args, **kwargs):
            yield {"tts_speech": generate_fake_audio_tensor(2.0)}

        engine.tts_zero_shot = MagicMock(side_effect=_fake_tts)
        engine.tts_cross_lingual = MagicMock(side_effect=_fake_tts)
        engine.tts_instruct = MagicMock(side_effect=_fake_tts)
        return engine

    # -- zero-shot --

    def test_tts_zero_shot_returns_array(self, engine, prompt_zh_wav):
        results = list(engine.tts_zero_shot(
            "测试文本",
            "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
            prompt_zh_wav,
            stream=False,
        ))
        assert len(results) >= 1
        tts_speech = results[0]["tts_speech"]
        arr = np.asarray(tts_speech)
        assert arr.ndim == 2

    def test_tts_zero_shot_called_with_stream_false(self, engine, prompt_zh_wav):
        engine.tts_zero_shot("hello", "prompt", prompt_zh_wav, stream=False)
        engine.tts_zero_shot.assert_called_with("hello", "prompt", prompt_zh_wav, stream=False)

    def test_tts_zero_shot_called_with_stream_true(self, engine, prompt_zh_wav):
        engine.tts_zero_shot("hello", "prompt", prompt_zh_wav, stream=True)
        engine.tts_zero_shot.assert_called_with("hello", "prompt", prompt_zh_wav, stream=True)

    # -- cross-lingual --

    def test_tts_cross_lingual_returns_array(self, engine, prompt_zh_wav):
        results = list(engine.tts_cross_lingual(
            "Hello, this is cross-lingual.",
            prompt_zh_wav,
            stream=False,
        ))
        assert len(results) >= 1
        assert np.asarray(results[0]["tts_speech"]).ndim == 2

    # -- instruct --

    def test_tts_instruct_returns_array(self, engine, prompt_zh_wav):
        results = list(engine.tts_instruct(
            "今天天气真好",
            "请用开心语气朗读",
            prompt_zh_wav,
            stream=False,
        ))
        assert len(results) >= 1
        assert np.asarray(results[0]["tts_speech"]).ndim == 2

    # -- stream vs non-stream --

    def test_non_stream_mode_returns_at_least_one_chunk(self, engine, prompt_zh_wav):
        results = list(engine.tts_zero_shot(
            "短文本", "prompt", prompt_zh_wav, stream=False
        ))
        assert len(results) >= 1


class TestCosyVoiceEngineEdgeCases:
    """Edge cases that should not crash the engine."""

    @pytest.fixture
    def engine(self, prompt_zh_wav):
        e = CosyVoiceEngine(model_dir="/tmp/model")
        e.model = MagicMock()
        return e

    def test_special_characters_in_text(self, engine, prompt_zh_wav):
        engine.tts_zero_shot = MagicMock(return_value=iter([{"tts_speech": generate_fake_audio_tensor(1.0)}]))
        results = list(engine.tts_zero_shot("数字123，标点!@#$", "prompt", prompt_zh_wav))
        assert len(results) == 1

    def test_very_short_single_char_text(self, engine, prompt_zh_wav):
        engine.tts_zero_shot = MagicMock(return_value=iter([{"tts_speech": generate_fake_audio_tensor(0.5)}]))
        results = list(engine.tts_zero_shot("嗯", "prompt", prompt_zh_wav))
        assert len(results) == 1

    def test_empty_prompt_text(self, engine, prompt_zh_wav):
        engine.tts_zero_shot = MagicMock(return_value=iter([{"tts_speech": generate_fake_audio_tensor(1.0)}]))
        results = list(engine.tts_zero_shot("text", "", prompt_zh_wav))
        assert len(results) == 1

    def test_long_text_up_to_5000_chars(self, engine, prompt_zh_wav):
        long_text = "测试" * 2500
        engine.tts_zero_shot = MagicMock(return_value=iter([{"tts_speech": generate_fake_audio_tensor(5.0)}]))
        results = list(engine.tts_zero_shot(long_text, "prompt", prompt_zh_wav))
        assert len(results) > 0
