"""FaceFusion face-only swap using a saved Studio Assist identity.

Run with .runtime/facefusion-venv/Scripts/python.exe. FaceFusion stays in
its separate environment; the Studio Assist/ComfyUI environments are untouched.
The official pipeline (including its content checks) runs unchanged apart from
observing its final face mask and whether its content check let the picture
through. Original pixels outside that mask are restored and the lossless saved
result is checked before success is reported.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

# What the worker says, as its last error, of a picture FaceFusion would not take.
REFUSED = 'FaceFusion\'s content check refused this picture, so no face was swapped.'
# The person's face, averaged over their reference photos, is kept between
# runs. FaceFusion reads every photo for its face on every run, about a second
# and a half each on the CPU: with forty photos 58 of a swap's 77 seconds
# (2026-09-29). The average depends on the photos alone, so it is worked out
# once for a set of them and read back after, and the picture that comes of
# it is the same picture, byte for byte.
SOURCES = '.work/facefusion-sources'
SOURCES_KEPT = 24                     # sets of photos; the oldest go
LANDMARKS = {'5': 'lm_5', '5/68': 'lm_5_68', '68': 'lm_68', '68/5': 'lm_68_5'}
LENS_SOFT = 0.04                      # of the crop's height: the lens line's soft edge


def source_key(refs, version):
    """What a set of reference photos' averaged face depends on: FaceFusion's
    version, and each photo in order - where it is, how big and how new."""
    h = hashlib.sha256(('sources 1|%s' % version).encode('utf-8'))
    for path in refs:
        s = os.stat(path)
        h.update(('|%s|%d|%d' % (os.path.normcase(os.path.abspath(path)), s.st_size,
                                 s.st_mtime_ns)).encode('utf-8'))
    return h.hexdigest()[:32]


def keep_source(path, face, first, np):
    """The averaged face and the photo its landmarks are from, written whole
    or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(face.landmark_set[key])
              for key, name in LANDMARKS.items() if face.landmark_set.get(key) is not None}
    meta = {'origin': face.origin, 'angle': int(face.angle), 'gender': face.gender,
            'race': face.race, 'age': [face.age.start, face.age.stop],
            'scores': {k: float(v) for k, v in face.score_set.items()}, 'first': first}
    part = path.with_name(path.name + '.part')
    with open(part, 'wb') as file:
        np.savez(file, embedding=np.asarray(face.embedding),
                 embedding_norm=np.asarray(face.embedding_norm),
                 bounding_box=np.asarray(face.bounding_box),
                 meta=np.array(json.dumps(meta)), **arrays)
    os.replace(part, path)
    old = sorted(path.parent.glob('*.npz'), key=lambda p: p.stat().st_mtime)
    for stale in old[:-SOURCES_KEPT]:
        stale.unlink()


def kept_source(path, Face, np):
    """-> (face, the photo to hand FaceFusion), or None when there is none
    kept that can be read and whose photo is still there."""
    with np.load(path, allow_pickle=False) as kept:
        meta = json.loads(str(kept['meta']))
        face = Face(origin=meta['origin'], bounding_box=kept['bounding_box'],
                    score_set=meta['scores'],
                    landmark_set={key: kept[name] for key, name in LANDMARKS.items()
                                  if name in kept.files},
                    angle=meta['angle'], embedding=kept['embedding'],
                    embedding_norm=kept['embedding_norm'],
                    age=range(*meta['age']), gender=meta['gender'], race=meta['race'])
    if face.embedding.shape != face.embedding_norm.shape or not os.path.isfile(meta['first']):
        return None
    return face, meta['first']


def even(crop, total, amount, np):
    """The swapped face with pixel boost's weave evened out, `amount` (0-1)
    of the way. Pixel boost swaps a face bigger than the model's 128 px as
    `total` x `total` faces of 128, each every `total`th pixel of it, and
    weaves them back. The model does not draw them quite alike, so the weave
    shows: a comb of streaks `total` px apart down a cheek, a grid behind
    the glasses, teeth in blocks (live, 2026-09-29). A box as wide as the
    weave takes out what repeats every `total` px and nothing coarser: the
    same likeness (ArcFace 0.887 and 0.887), no comb."""
    if amount <= 0 or total < 2:
        return crop
    height, width = crop.shape[:2]
    before, after = (total - 1) // 2, total // 2
    wide = np.pad(crop, ((before, after), (before, after), (0, 0)), mode='edge')
    box = sum(wide[y:y + height, x:x + width]
              for y in range(total) for x in range(total)) / float(total * total)
    return crop + (box - crop) * min(1.0, amount)


def under_lenses(mask, glasses, line, np):
    """The swap's mask (its own crop's: the face upright, the eyes at 0.40
    down it) less what is behind `glasses` below `line`, over LENS_SOFT of
    the crop's height. Behind a lens the eyes are swapped and the cheek
    under them stays the picture's: a cheek seen through a lens, where
    inswapper paints a bare one - a pink patch with a hard edge."""
    if not line:
        return mask
    rows = np.linspace(0, 1, mask.shape[0], dtype=np.float32)[:, None]
    below = np.clip((rows - (line - LENS_SOFT / 2)) / LENS_SOFT, 0, 1)
    return mask * (1 - glasses * below)


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
    parser.add_argument('--tone', type=float, default=0.0,
                        help='How far (0-1) the swapped face\'s colour is moved to that of '
                             'the face it replaced, inside the mask.')
    parser.add_argument('--masks', nargs='+', default=['box', 'occlusion', 'region'],
                        choices=['box', 'occlusion', 'area', 'region'],
                        help='FaceFusion\'s face mask types. "occlusion" keeps what is in '
                             'front of the face - a strand of hair, a hand, a glass.')
    parser.add_argument('--boost', default='auto',
                        help='The size the face is swapped at, as FaceFusion\'s pixel boost '
                             '("256x256"); "auto" is the first that holds the face.')
    parser.add_argument('--regions', nargs='+', default=None,
                        help='FaceFusion\'s face mask regions, the parts of the face that are '
                             'swapped; all of them when not given. Without "mouth" the teeth '
                             'stay the picture\'s own.')
    parser.add_argument('--lens-line', type=float, default=0.0,
                        help='Behind glasses, what lies below this line of the swap\'s own '
                             'crop (0-1 down it; the eyes are at 0.40) stays the picture\'s '
                             'own. 0: all that is behind them is swapped.')
    parser.add_argument('--deweave', type=float, default=0.0,
                        help='How much (0-1) of pixel boost\'s weave is evened out of the '
                             'swapped face.')
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
    # FaceFusion's content check ends the run without a word: the log stops at
    # "processing step 1 of 1" and the exit code is all there is. It is only
    # watched here, never answered for: a picture it refuses is not swapped.
    from facefusion.workflows import image_to_image
    analyse_image = image_to_image.analyse_image

    def observe_analysis():
        code = analyse_image()
        captured['refused'] = bool(code)
        return code

    image_to_image.analyse_image = observe_analysis
    sys.path.insert(0, str(root))
    from apps.image_studio.facefusion import target_face
    from facefusion.face_creator import get_static_faces
    from facefusion.face_selector import sort_faces_by_order
    def select_target(reference, sources, targets):
        faces = sort_faces_by_order(get_static_faces(targets), 'left-right')
        height, width = original.shape[:2]
        boxes = [[float(v) / (width if i % 2 == 0 else height)
                  for i, v in enumerate(face.bounding_box)] for face in faces]
        print('Faces found, left to right: %s' % [[round(v, 3) for v in b] for b in boxes])
        index = target_face(boxes, region=args.target_region, point=args.target_point,
                            index=args.face_index, count=args.face_count)
        # The model draws at 256 px. A face bigger than that came back as a
        # soft, generic 256 px face scaled up; pixel boost swaps it in tiles
        # at the size the face really is (its warped crop is ~1.5x its box).
        box = faces[index].bounding_box
        side = 1.5 * max(box[2] - box[0], box[3] - box[1])
        sizes = swapper_choices.face_swapper_set.get(args.model) or []
        boost = next((s for s in sizes if int(s.split('x')[0]) >= side), sizes[-1] if sizes else None)
        if args.boost in sizes:
            boost = args.boost
        if boost:
            state_manager.set_item('face_swapper_pixel_boost', boost)
            captured['pixel_boost'] = boost
        return [faces[index]]
    swapper.select_faces = select_target
    # The lens line and the weave are FaceFusion's own two steps with one of
    # ours after each. A FaceFusion that has neither under these names swaps
    # as it does, and the report says what was not done.
    region_mask = getattr(swapper, 'create_region_mask', None)
    lens_line = args.lens_line if region_mask else 0.0

    def region_mask_above(crop, regions):
        mask = region_mask(crop, regions)
        if 'glasses' in regions:
            mask = under_lenses(mask, region_mask(crop, ['glasses']), lens_line, np)
        return mask
    if lens_line:
        swapper.create_region_mask = region_mask_above
    explode = getattr(swapper, 'explode_pixel_boost', None)
    deweave = min(1.0, max(0.0, args.deweave)) if explode else 0.0
    if deweave:
        swapper.explode_pixel_boost = lambda frames, total, model_size, boost_size: even(
            explode(frames, total, model_size, boost_size), total, deweave, np)
    # The person's averaged face: read back when this set of photos has been
    # worked out before, else worked out by FaceFusion as ever and kept. On
    # any trouble with what is kept, FaceFusion works it out: a swap is never
    # lost to its own shortcut.
    from facefusion import metadata
    from facefusion.types import Face
    kept_at = root / SOURCES / (source_key(refs, metadata.get('version')) + '.npz')
    sources, known = refs, None
    try:
        known = kept_source(kept_at, Face, np) if kept_at.is_file() else None
    except Exception as error:
        print('The kept face could not be read (%s); reading the photos.' % error)
    extract_source_face = swapper.extract_source_face
    if known:
        sources = [known[1]]
        swapper.extract_source_face = lambda frames: known[0]
        captured['sources'] = 'kept'
    else:
        def observe_source(frames):
            face = extract_source_face(frames)
            if face is not None and 'sources' not in captured:
                captured['sources'] = 'read'
                try:
                    first = next(path for path, frame in zip(refs, frames)
                                 if get_static_faces([frame]))
                    keep_source(kept_at, face, first, np)
                except Exception as error:
                    print('The face could not be kept for the next run (%s).' % error)
            return face
        swapper.extract_source_face = observe_source
    raw_output = output.with_name(output.stem + '-facefusion.png')
    sys.argv = [str(ff_root / 'facefusion.py'), 'headless-run',
                '--source-paths', *sources, '--target-path', str(target),
                '--output-path', str(raw_output), '--processors', 'face_swapper',
                '--face-swapper-model', args.model, '--face-selector-mode', 'one',
                '--face-swapper-weight', str(round(round(args.weight * 20) / 20, 2)),
                '--face-mask-types', *args.masks,
                *(['--face-mask-regions', *args.regions] if args.regions else []),
                '--execution-providers', args.provider, '--execution-thread-count', '4',
                '--output-image-quality', '100', '--output-image-scale', '1.0',
                '--jobs-path', str(root / '.runtime/facefusion-jobs'),
                '--temp-path', str(root / '.runtime/facefusion-temp'), '--log-level', 'info']
    conda.setup()
    try:
        core.cli()
    except SystemExit as exc:
        if exc.code and captured.get('refused'):
            raise RuntimeError(REFUSED) from None
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
    tone = min(1.0, max(0.0, args.tone))
    if tone:
        # The swap model paints the references' skin: a studio photo's pale pink
        # face in a picture lit by a low sun. Inside the mask the new face takes
        # the mean and spread (Lab) of the face it replaced, which the picture's
        # own light fell on. The spread is held near its own, so a face in
        # hard light does not posterize a smooth one.
        new = cv2.cvtColor(final, cv2.COLOR_RGB2LAB).astype(np.float32)
        old = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
        for c in range(3):
            a, b = new[..., c][mask], old[..., c][mask]
            gain = min(1.3, max(0.7, float(b.std()) / max(float(a.std()), 1e-3)))
            moved = (a - a.mean()) * gain + b.mean()
            new[..., c][mask] = a + (moved - a) * tone
        toned = cv2.cvtColor(np.clip(new, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        final[mask] = toned[mask]
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
              'weight': round(round(args.weight * 20) / 20, 2), 'tone': tone,
              'masks': list(args.masks), 'sources': captured.get('sources'),
              'regions': list(args.regions) if args.regions else None, 'deweave': deweave,
              'lens_line': lens_line,
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
