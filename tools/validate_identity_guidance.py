"""Controlled local comparison of primary, repeated tokens and identity pooling.

Uses the installed ComfyUI dependencies and vendor code, but the repository's
node implementation. No installation, server restart or profile changes. Run
with D:/ComfyUI/venv/Scripts/python.exe. Refuses a busy backend. Outputs are
experimental and need visual likeness review; execution is not likeness proof.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comfy', type=Path, default=Path('D:/ComfyUI'))
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--identity', required=True)
    ap.add_argument('--primary', type=int, default=1, help='One-based photo index.')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--seed', type=int, default=240927)
    ap.add_argument('--size', type=int, default=768)
    ap.add_argument('--description', type=Path, help='Optional visual identity description to test.')
    ap.add_argument('--variants', nargs='+', choices=['primary-only', 'separate-tokens', 'identity-consensus'],
                    default=['primary-only', 'separate-tokens', 'identity-consensus'])
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Choose a new output folder.')
    profile = next(p for p in json.loads((args.library / 'identities.json').read_text(
        encoding='utf-8')) if p['id'] == args.identity)
    paths = list(dict.fromkeys(profile['references']))
    if not 1 <= args.primary <= len(paths) or not 2 <= len(paths) <= 8:
        ap.error('Choose a valid primary and two to eight references.')
    paths.insert(0, paths.pop(args.primary - 1))
    with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=10) as response:
        queue = json.load(response)
    if queue.get('queue_running') or queue.get('queue_pending'):
        raise RuntimeError('ComfyUI is busy; no validation run started.')
    request = urllib.request.Request('http://127.0.0.1:8188/free',
        data=json.dumps({'unload_models': True, 'free_memory': True}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30):
        pass

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import studio_imagegen as ig
    description = args.description.read_text(encoding='utf-8').strip() if args.description else ''
    profile['description'] = description
    prompt = ' '.join([args.prompt] + ig.identity_description_text({}, None, [(profile, 1)]))
    sys.path.insert(0, str(args.comfy.resolve()))
    # ComfyUI parses argv at import; keep this script's flags out of it.
    sys.argv = [sys.argv[0]]
    import torch
    import numpy as np
    from PIL import Image, ImageOps
    import nodes
    import comfy.model_management as mm
    from utils.extra_config import load_extra_path_config
    load_extra_path_config(str(args.comfy / 'extra_model_paths.yaml'))
    import folder_paths
    folder_paths.add_model_folder_path('insightface', str(args.comfy / 'models/insightface'))
    node_dir = root / 'comfy_nodes/studio_withanyone'
    spec = importlib.util.spec_from_file_location('studio_identity_trial', node_dir / '__init__.py',
        submodule_search_locations=[str(node_dir), str(args.comfy / 'custom_nodes/studio_withanyone')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    photos = []
    for path in paths:
        with Image.open(path) as image:
            photos.append(torch.from_numpy(np.array(ImageOps.exif_transpose(image).convert('RGB'))
                .astype(np.float32) / 255)[None])
    args.output.mkdir(parents=True)
    report = {'identity': args.identity, 'references': paths, 'primary': paths[0],
              'scene': args.prompt, 'description': description, 'prompt': prompt,
              'seed': args.seed, 'size': args.size,
              'steps': 25, 'siglip_weight': 1.0, 'variants': []}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with torch.inference_mode():
        clip = nodes.DualCLIPLoader().load_clip('clip_l.safetensors',
            't5xxl_fp16.safetensors', 'flux', device='cpu')[0]
        conditioning = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        del clip
        gc.collect()
        for name, extras, mode in [('primary-only', None, 'separate_tokens'),
                                   ('separate-tokens', tuple(photos[1:]), 'separate_tokens'),
                                   ('identity-consensus', tuple(photos[1:]), 'identity_consensus')]:
            if name not in args.variants:
                continue
            print('START', name, flush=True)
            start = time.monotonic()
            latent = module.StudioWithAnyone().generate(conditioning,
                'flux1-dev.safetensors', 'withanyone.safetensors',
                'siglip-base-patch16-256-i18n', photos[0], '[[0.32,0.15,0.68,0.55]]',
                args.size, args.size, 25, 4.0, 1.0, args.seed,
                references1=extras, reference_mode=mode)[0]
            vae = nodes.VAELoader().load_vae('ae.safetensors')[0]
            result = nodes.VAEDecodeTiled().decode(vae, latent, 512)[0]
            path = args.output / (name + '.png')
            Image.fromarray((result[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)).save(path)
            report['variants'].append({'name': name, 'output': str(path),
                                      'seconds': time.monotonic() - start})
            (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            del vae, latent, result
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()
            print('DONE', name, path, flush=True)


if __name__ == '__main__':
    main()
