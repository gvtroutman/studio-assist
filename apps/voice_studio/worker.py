"""One remote inference job. Third-party imports are confined to this process.

The desktop and remote HTTP server never import ML dependencies. Separate
environments are required for incompatible engines. Raw takes stay separate;
sentence chunks receive a short silence, with no crossfade over spoken words.
"""
import json
import os
from pathlib import Path
import sys
import traceback

from .catalog import ENGINES, QWEN_LANGUAGES, chunks, validate


def loader(request):
    """Return a chunk callable and its sample rate using upstream Python APIs."""
    import numpy as np
    import torch
    engine = request['engine']
    config = request.get('config', {})
    device = config.get('device', 'cpu' if ENGINES[engine].get('cpu') else 'cuda:0')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this engine environment. Install its GPU PyTorch build on the LLM PC, or configure device=cpu.')
    model_id = config.get('models', {}).get(engine, ENGINES[engine]['model'])
    reference = request.get('reference_path')
    transcript = request.get('reference_text', '')
    direction = request.get('direction', '').strip()
    language = request.get('language', 'en')
    dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
    if engine.startswith('qwen-'):
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype,
                                            attn_implementation='sdpa')
        if engine == 'qwen-clone':
            prompt = model.create_voice_clone_prompt(ref_audio=reference, ref_text=transcript,
                                                     x_vector_only_mode=False)
            def generate(text):
                audio, rate = model.generate_voice_clone(text=text, language=QWEN_LANGUAGES[language],
                                                         voice_clone_prompt=prompt)
                return audio[0], rate
        else:
            def generate(text):
                audio, rate = model.generate_custom_voice(text=text, language=QWEN_LANGUAGES[language],
                                                          speaker=request.get('speaker') or 'Ryan', instruct=direction or None)
                return audio[0], rate
        return generate
    if engine == 'voxcpm':
        from voxcpm import VoxCPM
        model = VoxCPM.from_pretrained(model_id, load_denoiser=False, optimize=False, device=device,
                                      local_files_only=not request.get('allow_download', False))
        def generate(text):
            options = dict(text=('(' + direction + ')' if direction else '') + text,
                           cfg_value=2.0, inference_timesteps=10, seed=request['seed'])
            if reference:
                options['reference_wav_path'] = reference
                if transcript:
                    options.update(prompt_wav_path=reference, prompt_text=transcript)
            return model.generate(**options), model.tts_model.sample_rate
        return generate
    if engine == 'omnivoice':
        from omnivoice import OmniVoice
        model = OmniVoice.from_pretrained(model_id, device_map=device,
                                          dtype=torch.float16 if device.startswith('cuda') else torch.float32)
        prompt = model.create_voice_clone_prompt(ref_audio=reference, ref_text=transcript) if reference else None
        def generate(text):
            options = dict(text=text, num_step=32)
            if prompt is not None:
                options['voice_clone_prompt'] = prompt
                # Upstream's style instruction is a voice-design mode, not
                # an extra control on a cloned prompt. Do not promise both.
            elif direction:
                options['instruct'] = direction
            return model.generate(**options)[0], 24000
        return generate
    if engine == 'chatterbox':
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        model.prepare_conditionals(reference)
        def generate(text):
            return model.generate(text).detach().cpu().numpy().reshape(-1), model.sr
        return generate
    if engine == 'cosyvoice':
        from cosyvoice.cli.cosyvoice import AutoModel
        model = AutoModel(model_dir=model_id)
        def generate(text):
            if direction:
                result = model.inference_instruct2(text, 'You are a helpful assistant. ' + direction + '<|endofprompt|>', reference, stream=False)
            else:
                result = model.inference_zero_shot(text, 'You are a helpful assistant.<|endofprompt|>' + transcript, reference, stream=False)
            return np.concatenate([part['tts_speech'].detach().cpu().numpy().reshape(-1) for part in result]), model.sample_rate
        return generate
    if engine == 'kokoro':
        from kokoro import KPipeline
        codes = dict(en='a', zh='z', ja='j', es='e', fr='f', hi='h', it='i', pt='p')
        voices = dict(en='af_heart', zh='zf_xiaobei', ja='jf_alpha', es='ef_dora', fr='ff_siwis', hi='hf_alpha', it='if_sara', pt='pf_dora')
        model = KPipeline(lang_code=codes[language], repo_id=model_id, device=device)
        def generate(text):
            pieces = [np.asarray(audio) for _, _, audio in model(text, voice=request.get('speaker') or voices[language])]
            return np.concatenate(pieces), 24000
        return generate
    if engine == 'pocket':
        from pocket_tts import TTSModel
        model = TTSModel.load_model()
        voice = model.get_state_for_audio_prompt(reference or 'hf://kyutai/tts-voices/alba-mackenna/casual.wav')
        def generate(text):
            return model.generate_audio(voice, text).detach().cpu().numpy().reshape(-1), model.sample_rate
        return generate
    raise RuntimeError('Unknown inference engine.')


def run(request_path, load=loader):
    import numpy as np
    import soundfile as sf
    import torch
    path = Path(request_path)
    request = json.loads(path.read_text(encoding='utf8'))
    request = validate(request, reference=bool(request.get('reference_path')))
    torch.manual_seed(request['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(request['seed'])
    generate = load(request)
    pieces = chunks(request['text'])
    initial_seed = request['seed']
    audio = []
    sample_rate = None
    for index, text in enumerate(pieces):
        request['seed'] = (initial_seed + index) % 2**32
        torch.manual_seed(request['seed'])
        progress = path.parent / 'progress.json'
        temp = progress.with_suffix('.tmp')
        temp.write_text(json.dumps({'progress': 'Generating passage %d of %d' % (index + 1, len(pieces))}), encoding='utf8')
        temp.replace(progress)
        wave, rate = generate(text)
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        if not wave.size or not np.all(np.isfinite(wave)):
            raise RuntimeError('The engine returned empty or invalid audio.')
        if sample_rate is not None and rate != sample_rate:
            raise RuntimeError('The engine changed sample rate between passages.')
        sample_rate = int(rate)
        if audio:
            audio.append(np.zeros(round(sample_rate * .08), dtype=np.float32))
        audio.append(wave)
        if sum(len(part) for part in audio) / sample_rate > 1200:
            raise RuntimeError('Speech exceeded the 20-minute take limit.')
    result = np.concatenate(audio)
    # No peak/loudness normalization or denoising: keep raw performances.
    sf.write(str(path.parent / 'audio.wav'), result, sample_rate, subtype='PCM_16')


if __name__ == '__main__':
    try:
        run(sys.argv[1])
    except Exception as error:
        traceback.print_exc()
        if len(sys.argv) > 1:
            target = Path(sys.argv[1]).parent / 'error.json'
            target.write_text(json.dumps({'error': str(error)[:600]}), encoding='utf8')
        sys.exit(1)
