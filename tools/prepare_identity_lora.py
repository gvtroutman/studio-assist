"""Prepare a local FLUX training trial from existing weights and identity photos.

Run in the AI Toolkit venv. No downloads or source-photo edits. The diffusers
copy is reusable; each completed component gets a marker for safe resumption.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', type=Path, required=True)
    ap.add_argument('--trial', type=Path, required=True)
    ap.add_argument('--references', type=Path, required=True)
    args = ap.parse_args()
    import torch
    from safetensors.torch import load_file
    from transformers import CLIPTextModel, CLIPTextConfig, T5EncoderModel, T5Config, CLIPTokenizer, T5TokenizerFast
    from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers, convert_ldm_vae_checkpoint

    source = Path('D:/ComfyUI-models')
    comfy = Path('D:/ComfyUI/comfy')
    args.models.mkdir(parents=True, exist_ok=True)
    def save_component(name, factory, state):
        folder = args.models / name
        if (folder / '.complete').exists():
            print('Reusing', name, flush=True)
            return
        print('Converting', name, flush=True)
        with torch.device('meta'):
            model = factory()
        weights = state(model)
        model.load_state_dict(weights, strict=True, assign=True)
        del weights
        model.to(dtype=torch.bfloat16)
        model.save_pretrained(folder, max_shard_size='4GB')
        (folder / '.complete').touch()
        del model
        gc.collect()
    save_component('transformer', lambda: FluxTransformer2DModel(guidance_embeds=True),
        lambda model: convert_flux_transformer_checkpoint_to_diffusers(load_file(str(source / 'diffusion_models/flux1-dev.safetensors'))))
    def clip_weights(model):
        weights = load_file(str(source / 'text_encoders/clip_l.safetensors'))
        return {k: v for k, v in weights.items() if k in model.state_dict()}
    save_component('text_encoder', lambda: CLIPTextModel(CLIPTextConfig(**json.loads(
        (comfy / 'sd1_clip_config.json').read_text()))), clip_weights)
    def t5_weights(model):
        weights = load_file(str(source / 'text_encoders/t5xxl_fp16.safetensors'))
        if 'encoder.embed_tokens.weight' not in weights:
            weights['encoder.embed_tokens.weight'] = weights['shared.weight']
        return weights
    t5_config = json.loads((comfy / 'text_encoders/t5_config_xxl.json').read_text())
    t5_config['feed_forward_proj'] = 'gated-gelu'
    save_component('text_encoder_2', lambda: T5EncoderModel(T5Config(**t5_config)), t5_weights)
    save_component('vae', lambda: AutoencoderKL(in_channels=3, out_channels=3,
        down_block_types=('DownEncoderBlock2D',)*4, up_block_types=('UpDecoderBlock2D',)*4,
        block_out_channels=(128,256,512,512), layers_per_block=2, latent_channels=16,
        norm_num_groups=32, sample_size=1024, scaling_factor=0.3611, shift_factor=0.1159,
        use_quant_conv=False, use_post_quant_conv=False),
        lambda model: convert_ldm_vae_checkpoint(load_file(str(source / 'vae/ae.safetensors')), model.config))
    CLIPTokenizer.from_pretrained(str(comfy / 'sd1_tokenizer'), local_files_only=True,
        model_max_length=77).save_pretrained(args.models / 'tokenizer')
    T5TokenizerFast.from_pretrained(str(comfy / 'text_encoders/t5_tokenizer'), local_files_only=True,
        model_max_length=512).save_pretrained(args.models / 'tokenizer_2')
    FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0,
        use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
        base_image_seq_len=256, max_image_seq_len=4096).save_pretrained(args.models / 'scheduler')

    args.trial.mkdir(parents=True, exist_ok=True)
    dataset = args.trial / 'dataset'
    dataset.mkdir(exist_ok=True)
    refs = json.loads(args.references.read_text(encoding='utf-8'))['references']
    # Two other images are near-duplicate frames from the same studio shoot.
    # Keep one frontal, one profile and the distinct winter photograph.
    captions = [
        'Photo of lylperson woman, smiling toward the camera, reddish brown hair tied back, narrow rectangular glasses, coral dress and cream cardigan, pale studio background, soft indoor light.',
        'Photo of lylperson woman, left-facing profile, reddish brown hair in a ponytail, narrow rectangular glasses, coral dress and cream cardigan, pale studio background, soft indoor light.',
        'Photo of lylperson woman, smiling in a three-quarter view, black winter headband, rectangular glasses, pale blue winter jacket and dark gloves, pale background, cool blue outdoor light.'
    ]
    for index, (path, caption) in enumerate(zip(refs[:3], captions), 1):
        shutil.copy2(path, dataset / ('photo%02d.png' % index))
        (dataset / ('photo%02d.txt' % index)).write_text(caption, encoding='utf-8')
    config = {'job': 'extension', 'config': {'name': 'lilya_flux_identity_v1', 'process': [{
        'type': 'sd_trainer', 'training_folder': str(args.trial / 'output'), 'device': 'cuda:0',
        'trigger_word': 'lylperson', 'network': {'type': 'lora', 'linear': 16, 'linear_alpha': 16},
        'save': {'dtype': 'float16', 'save_every': 250, 'max_step_saves_to_keep': 3, 'push_to_hub': False},
        'datasets': [{'folder_path': str(dataset), 'caption_ext': 'txt', 'caption_dropout_rate': 0,
            'shuffle_tokens': False, 'cache_latents_to_disk': True, 'cache_text_embeddings': True,
            'resolution': [512]}],
        'train': {'batch_size': 1, 'steps': 500, 'gradient_accumulation_steps': 1,
            'train_unet': True, 'train_text_encoder': False, 'gradient_checkpointing': True,
            'noise_scheduler': 'flowmatch', 'optimizer': 'adamw8bit', 'lr': 0.00005,
            'dtype': 'bf16', 'skip_first_sample': True, 'disable_sampling': True,
            'cache_text_embeddings': True, 'unload_text_encoder': True},
        'model': {'name_or_path': str(args.models), 'is_flux': True, 'quantize': True,
            'quantize_te': True, 'low_vram': True},
        'sample': {'sampler': 'flowmatch', 'sample_every': 500, 'width': 768, 'height': 768,
            'prompts': ['A photograph of lylperson woman in a green dirndl in a mountain meadow.'],
            'seed': 240927, 'walk_seed': False, 'guidance_scale': 4, 'sample_steps': 25}
    }]}, 'meta': {'name': 'lilya_flux_identity_v1', 'version': '1.0'}}
    (args.trial / 'train.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (args.trial / 'dataset-manifest.json').write_text(json.dumps({'sources': refs[:3],
        'excluded_near_duplicates': refs[3:], 'captions': captions,
        'limitation': 'Three distinct views from five photos; tiny trial, not a robust held-out benchmark.'}, indent=2), encoding='utf-8')
    print('Prepared', args.trial / 'train.json', flush=True)


if __name__ == '__main__':
    main()
