"""FaceFusion face-only swap using a saved Studio Assist identity.

Run with .runtime/facefusion-venv/Scripts/python.exe. FaceFusion stays in
its separate environment; the Studio Assist/ComfyUI environments are untouched.
The official pipeline (including its content checks) runs unchanged apart from
observing its final face mask. Original pixels outside that mask are restored
and the lossless saved result is checked before success is reported.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', default='partner')
    parser.add_argument('--sources', nargs='+', help='Explicit profile reference photos.')
    parser.add_argument('--face-index', type=int, help='Target face index, left to right, starting at zero.')
    parser.add_argument('--face-count', type=int, default=1)
    parser.add_argument('--target-region', type=float, nargs=4)
    parser.add_argument('--target-point', type=float, nargs=2)
    parser.add_argument('--target', help='If omitted, choose the picture in a file dialog.')
    parser.add_argument('--output', help='Defaults to a new PNG in .work/facefusion.')
    parser.add_argument('--open-result', action='store_true')
    parser.add_argument('--model', default='hyperswap_1a_256')
    parser.add_argument('--weight', type=float, default=0.5,
                        help='FaceFusion\'s face swapper weight: 0.5 is the reference face as it '
                             'is, higher pushes it further from the face it replaces.')
    parser.add_argument('--reference', type=int, help='Use just this reference (1-based).')
    parser.add_argument('--provider', default='cpu')
    args = parser.parse_args()
    if not args.target:
        import tkinter as tk
        from tkinter import filedialog
        window = tk.Tk()
        window.withdraw()
        args.target = filedialog.askopenfilename(title='Choose the picture to put Partner into',
                                                filetypes=[('Pictures', '*.png *.jpg *.jpeg *.webp')])
        window.destroy()
        if not args.target:
            return
    target = Path(args.target).resolve()
    output = (Path(args.output).resolve() if args.output else
              root / '.work/facefusion' / (target.stem + '-' + args.identity + '-' +
                                         datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.png'))
    if output == target or output.exists():
        parser.error('Choose a new output path; originals and previous results are never overwritten.')
    if output.suffix.lower() != '.png':
        parser.error('The verified output must be a lossless PNG.')
    refs = args.sources
    if not refs:
        identity_file = Path(os.environ['APPDATA']) / 'StudioAssistant/image-studio/identities.json'
        identity = next((i for i in json.loads(identity_file.read_text(encoding='utf-8'))
                         if i['id'] == args.identity), None)
        if not identity or not identity.get('references'):
            parser.error('Identity has no saved reference photos.')
        refs = identity['references']
    if args.reference:
        if not 1 <= args.reference <= len(refs):
            parser.error('Reference number is outside this identity\'s photo list.')
        refs = [refs[args.reference - 1]]
    for path in [str(target)] + refs:
        if not Path(path).is_file():
            parser.error('Missing image: ' + path)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['OMP_NUM_THREADS'] = '1'
    ff_root = root / '.runtime/facefusion'
    sys.path.insert(0, str(ff_root))
    os.chdir(ff_root)
    import cv2
    import numpy as np
    from PIL import Image
    from facefusion import conda, core, face_helper, state_manager
    from facefusion.processors.modules.face_swapper import choices as swapper_choices
    from facefusion.processors.modules.face_swapper import core as swapper

    original_pil = Image.open(target)
    original = np.array(original_pil.convert('RGB'))
    captured = {}
    paste_back = swapper.paste_back

    def observe_paste(frame, crop, mask, matrix):
        result = paste_back(frame, crop, mask, matrix)
        box, inverse = face_helper.calculate_paste_area(frame, crop, matrix)
        x1, y1, x2, y2 = box
        support = np.zeros(frame.shape[:2], dtype=bool)
        support[y1:y2, x1:x2] = cv2.warpAffine(mask, inverse, (x2-x1, y2-y1)) > 0
        captured['mask'] = captured.get('mask', np.zeros_like(support)) | support
        captured['frame'] = result.copy()
        captured['count'] = captured.get('count', 0) + 1
        return result

    swapper.paste_back = observe_paste
    sys.path.insert(0, str(root))
    from apps.image_studio.facefusion import target_face
    from facefusion.face_creator import get_static_faces
    from facefusion.face_selector import sort_faces_by_order
    def select_target(reference, sources, targets):
        faces = sort_faces_by_order(get_static_faces(targets), 'left-right')
        height, width = original.shape[:2]
        boxes = [[float(v) / (width if i % 2 == 0 else height)
                  for i, v in enumerate(face.bounding_box)] for face in faces]
        index = target_face(boxes, region=args.target_region, point=args.target_point,
                            index=args.face_index, count=args.face_count)
        # The model draws at 256 px. A face bigger than that came back as a
        # soft, generic 256 px face scaled up; pixel boost swaps it in tiles
        # at the size the face really is (its warped crop is ~1.5x its box).
        box = faces[index].bounding_box
        side = 1.5 * max(box[2] - box[0], box[3] - box[1])
        sizes = swapper_choices.face_swapper_set.get(args.model) or []
        boost = next((s for s in sizes if int(s.split('x')[0]) >= side), sizes[-1] if sizes else None)
        if boost:
            state_manager.set_item('face_swapper_pixel_boost', boost)
            captured['pixel_boost'] = boost
        return [faces[index]]
    swapper.select_faces = select_target
    raw_output = output.with_name(output.stem + '-facefusion.png')
    sys.argv = [str(ff_root / 'facefusion.py'), 'headless-run',
                '--source-paths', *refs, '--target-path', str(target),
                '--output-path', str(raw_output), '--processors', 'face_swapper',
                '--face-swapper-model', args.model, '--face-selector-mode', 'one',
                '--face-swapper-weight', str(round(round(args.weight * 20) / 20, 2)),
                '--face-mask-types', 'box', 'region',
                '--execution-providers', args.provider, '--execution-thread-count', '4',
                '--output-image-quality', '100', '--output-image-scale', '1.0',
                '--jobs-path', str(root / '.runtime/facefusion-jobs'),
                '--temp-path', str(root / '.runtime/facefusion-temp'), '--log-level', 'info']
    conda.setup()
    try:
        core.cli()
    except SystemExit as exc:
        if exc.code:
            raise
    if not captured.get('count'):
        raise RuntimeError('No face was swapped; no verified output was saved.')
    frame = captured['frame'][:, :, ::-1]
    mask = captured['mask']
    if frame.shape != original.shape:
        raise RuntimeError('FaceFusion changed the dimensions; refusing the result.')
    final = original.copy()
    final[mask] = frame[mask]
    if not np.any(final != original):
        raise RuntimeError('The swap made no pixel changes.')
    save_args = {}
    if original_pil.info.get('icc_profile'):
        save_args['icc_profile'] = original_pil.info['icc_profile']
    Image.fromarray(final).save(output, **save_args)
    check = np.array(Image.open(output).convert('RGB'))
    outside_changes = int(np.any(check != original, axis=2)[~mask].sum())
    if outside_changes:
        raise RuntimeError('Saved output failed the outside-face pixel check.')
    Image.fromarray(mask.astype('uint8') * 255).save(output.with_name(output.stem + '-mask.png'))
    report = {'target': str(target), 'output': str(output), 'identity': args.identity,
              'references': refs, 'model': args.model, 'faces_swapped': captured['count'],
              'weight': round(round(args.weight * 20) / 20, 2),
              'pixel_boost': captured.get('pixel_boost'),
              'mask_pixels': int(mask.sum()), 'outside_mask_changed_pixels': outside_changes,
              'changed_pixels': int(np.any(check != original, axis=2).sum()),
              'dimensions': list(original_pil.size)}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if args.open_result:
        os.startfile(str(output))


if __name__ == '__main__':
    main()
