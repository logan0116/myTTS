"""CosyVoice model engine wrapper."""


class CosyVoiceEngine:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.sample_rate = 24000

    def load(self):
        self.model = None

    def unload(self):
        self.model = None

    def tts_zero_shot(self, text, prompt_text, prompt_audio_path, stream=False):
        raise NotImplementedError

    def tts_cross_lingual(self, text, prompt_audio_path, stream=False):
        raise NotImplementedError

    def tts_instruct(self, text, instruction, prompt_audio_path, stream=False):
        raise NotImplementedError
