import sys
sys.path.append('/CosyVoice')
sys.path.append('/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

""" CosyVoice3 Usage, check https://funaudiollm.github.io/cosyvoice3/ for more details
"""
cosyvoice = AutoModel(model_dir='cosyvoice3_0')
# en zero_shot usage
for i, j in enumerate(cosyvoice.inference_zero_shot('CosyVoice is undergoing a comprehensive upgrade, providing more accurate, stable, faster, and better voice generation capabilities.', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                    '/CosyVoice/asset/zero_shot_prompt.wav', stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# zh zero_shot usage
for i, j in enumerate(cosyvoice.inference_zero_shot('八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                    '/CosyVoice/asset/zero_shot_prompt.wav', stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L280
for i, j in enumerate(cosyvoice.inference_cross_lingual('You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]',
                                                        '/CosyVoice/asset/zero_shot_prompt.wav', stream=False)):
    torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage, for supported control, check cosyvoice/utils/common.py#L28
for i, j in enumerate(cosyvoice.inference_instruct2('好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。', 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
                                                    '/CosyVoice/asset/zero_shot_prompt.wav', stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', 'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',
                                                    '/CosyVoice/asset/zero_shot_prompt.wav', stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# hotfix usage
for i, j in enumerate(cosyvoice.inference_zero_shot('高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                    '/CosyVoice/asset/zero_shot_prompt.wav', stream=False)):
    torchaudio.save('hotfix_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
