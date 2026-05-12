"""CosyVoice model engine wrapper."""

import os
import logging
from typing import Generator

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when the CosyVoice model directory does not exist."""


class CosyVoiceEngine:
    """Wraps Fun-CosyVoice3 AutoModel for TTS and voice cloning."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.sample_rate = 24000

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the CosyVoice AutoModel from local disk into memory."""
        if not os.path.isdir(self.model_dir):
            raise ModelNotFoundError(
                f"Model directory not found: {self.model_dir}. "
                f"Download it via huggingface_hub.snapshot_download."
            )

        # CosyVoice expects the Matcha-TTS third_party on sys.path
        # We handle this inside load() so the import only happens here.
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as e:
            raise ImportError(
                "cosyvoice is not installed. "
                "Clone https://github.com/FunAudioLLM/CosyVoice and install it."
            ) from e

        logger.info("Loading CosyVoice model from %s ...", self.model_dir)
        self.model = AutoModel(model_dir=self.model_dir)
        logger.info("CosyVoice model loaded (sample_rate=%d).", self.sample_rate)

    def unload(self) -> None:
        """Release model resources."""
        self.model = None
        logger.info("CosyVoice model unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call engine.load() first.")

    # ------------------------------------------------------------------
    # Zero-shot TTS (also used for voice cloning)
    # ------------------------------------------------------------------

    def tts_zero_shot(
        self,
        text: str,
        prompt_text: str,
        prompt_audio_path: str,
        stream: bool = False,
    ) -> Generator:
        """Zero-shot TTS / voice cloning inference.

        Args:
            text: Target text to synthesize.
            prompt_text: Prompt text for the reference audio.
            prompt_audio_path: Path to the reference audio file.
            stream: If True, enable streaming mode (lower first-packet latency).

        Yields:
            dict with key 'tts_speech' (torch.Tensor, shape [1, num_samples]).
        """
        self._ensure_loaded()
        if not os.path.isfile(prompt_audio_path):
            raise FileNotFoundError(f"Prompt audio not found: {prompt_audio_path}")

        yield from self.model.inference_zero_shot(
            text, prompt_text, prompt_audio_path, stream=stream
        )

    # ------------------------------------------------------------------
    # Cross-lingual TTS
    # ------------------------------------------------------------------

    def tts_cross_lingual(
        self,
        text: str,
        prompt_audio_path: str,
        stream: bool = False,
    ) -> Generator:
        """Cross-lingual voice cloning.

        Uses a reference audio in one language to speak text in another.
        """
        self._ensure_loaded()
        if not os.path.isfile(prompt_audio_path):
            raise FileNotFoundError(f"Prompt audio not found: {prompt_audio_path}")

        yield from self.model.inference_cross_lingual(
            text, prompt_audio_path, stream=stream
        )

    # ------------------------------------------------------------------
    # Instruction-controlled TTS
    # ------------------------------------------------------------------

    def tts_instruct(
        self,
        text: str,
        instruction: str,
        prompt_audio_path: str,
        stream: bool = False,
    ) -> Generator:
        """Instruction-controlled TTS.

        Supports emotion, speed, dialect, volume control via natural language.
        """
        self._ensure_loaded()
        if not os.path.isfile(prompt_audio_path):
            raise FileNotFoundError(f"Prompt audio not found: {prompt_audio_path}")

        yield from self.model.inference_instruct2(
            text, instruction, prompt_audio_path, stream=stream
        )
