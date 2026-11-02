"""Shared, dependency-free contract for Voice Studio and its remote worker."""
import re

PROTOCOL = 1
MAX_TEXT = 12000
MAX_REFERENCE = 8 * 1024 * 1024
MAX_AUDIO = 64 * 1024 * 1024
MAX_TAKES = 6
QWEN_LANGUAGES = dict(en='English', zh='Chinese', ja='Japanese', ko='Korean',
                      de='German', fr='French', ru='Russian', pt='Portuguese',
                      es='Spanish', it='Italian')

# Environments are independent; only the selected worker imports ML libraries.
ENGINES = {
    'voxcpm': dict(name='VoxCPM2', group='voxcpm', module='voxcpm', reference='optional',
                   direction=True, license='Apache 2.0', model='openbmb/VoxCPM2',
                   package='voxcpm==2.0.3', url='https://github.com/OpenBMB/VoxCPM'),
    'qwen-clone': dict(name='Qwen voice clone', group='qwen', module='qwen_tts', reference='required',
                       transcript=True, direction=False, license='Apache 2.0',
                       model='Qwen/Qwen3-TTS-12Hz-1.7B-Base', package='qwen-tts==0.1.1',
                       languages=list(QWEN_LANGUAGES), url='https://github.com/QwenLM/Qwen3-TTS'),
    'qwen-preset': dict(name='Qwen expressive presets', group='qwen', module='qwen_tts', reference='unused',
                        direction=True, license='Apache 2.0', model='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
                        package='qwen-tts==0.1.1', languages=list(QWEN_LANGUAGES),
                        url='https://github.com/QwenLM/Qwen3-TTS'),
    'omnivoice': dict(name='OmniVoice', group='omnivoice', module='omnivoice', reference='optional',
                      transcript=True, direction=True, license='Apache 2.0', model='k2-fsa/OmniVoice',
                      package='omnivoice==0.2.1', url='https://github.com/k2-fsa/OmniVoice'),
    'chatterbox': dict(name='Chatterbox Turbo', group='chatterbox', module='chatterbox', reference='required',
                       direction=False, tags=True, license='MIT', model='ResembleAI/chatterbox-turbo',
                       package='chatterbox-tts==0.1.7', languages=['en'],
                       url='https://github.com/resemble-ai/chatterbox'),
    'cosyvoice': dict(name='CosyVoice 3', group='cosyvoice', module='cosyvoice', reference='required',
                      transcript=True, direction=True, license='Apache 2.0',
                      model='FunAudioLLM/Fun-CosyVoice3-0.5B-2512', package='',
                      languages=['en', 'zh', 'ja', 'ko', 'de', 'es', 'fr', 'it', 'ru'],
                      url='https://github.com/QwenAudio/CosyVoice'),
    'kokoro': dict(name='Kokoro presets', group='kokoro', module='kokoro', reference='unused',
                   direction=False, cpu=True, license='Apache 2.0', model='hexgrad/Kokoro-82M',
                   package='kokoro==0.9.4', languages=['en', 'zh', 'ja', 'es', 'fr', 'hi', 'it', 'pt'],
                   url='https://github.com/hexgrad/kokoro'),
    'pocket': dict(name='Pocket TTS', group='pocket', module='pocket_tts', reference='optional',
                   direction=False, cpu=True, license='MIT code; CC BY 4.0 weights',
                   model='kyutai/pocket-tts', package='pocket-tts==3.3.0', languages=['en'],
                   url='https://github.com/kyutai-labs/pocket-tts'),
}


class VoiceError(Exception):
    pass


def validate(data, reference=False):
    """Validate the original contract on both sides before starting a model."""
    if not isinstance(data, dict):
        raise VoiceError('The voice request must be an object.')
    engine = data.get('engine')
    if engine not in ENGINES:
        raise VoiceError('Choose a supported voice engine.')
    spec = ENGINES[engine]
    text = data.get('text')
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise VoiceError('Enter a script of 1 to %d characters.' % MAX_TEXT)
    language = data.get('language', 'en')
    if not isinstance(language, str) or not re.fullmatch('[a-z]{2,3}', language):
        raise VoiceError('Use a language code such as en or fr.')
    if spec.get('languages') and language not in spec['languages']:
        raise VoiceError('%s does not support the selected language in this integration.' % spec['name'])
    for field, limit in (('direction', 1000), ('reference_text', 2000), ('speaker', 100)):
        value = data.get(field, '')
        if not isinstance(value, str) or len(value) > limit:
            raise VoiceError('%s must be text of at most %d characters.' % (field, limit))
    if spec['reference'] == 'required' and not reference:
        raise VoiceError('%s needs a reference WAV.' % spec['name'])
    if reference and spec['reference'] != 'unused' and spec.get('transcript') and not data.get('reference_text', '').strip():
        raise VoiceError('%s needs the words spoken in the reference WAV.' % spec['name'])
    if type(data.get('seed', 0)) is not int or not 0 <= data.get('seed', 0) <= 2**32 - 1:
        raise VoiceError('The seed must be an integer between 0 and 4294967295.')
    if type(data.get('allow_download', False)) is not bool:
        raise VoiceError('Model-download permission must be true or false.')
    return dict(data, text=text.strip(), language=language)


def chunks(text, limit=450):
    """Keep the script in order; split at punctuation/space without dropping text."""
    rest = text.strip()
    out = []
    while len(rest) > limit:
        end = max((m.end() for m in re.finditer(r'[.!?。！？]\s*', rest[:limit + 1])), default=0)
        if end < limit // 3:
            end = rest.rfind(' ', 0, limit + 1)
        if end <= 0:
            end = limit
        out.append(rest[:end].strip())
        rest = rest[end:].strip()
    if rest:
        out.append(rest)
    return out
