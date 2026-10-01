"""The Image Studio's local, final face swap. Stdlib only in the GUI process."""
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import math
import threading

import core.procs as studio_procs

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / '.runtime/facefusion-venv/Scripts/python.exe'
SCRIPT = ROOT / 'tools/facefusion_swap.py'
# FaceFusion's face swapper weight. 0.5 hands the model the reference face's
# identity as it is; above it the identity is pushed away from the face being
# replaced, so less of the generated person survives. The swap ran at 0.5
# until 2026-09-27 (the user: "not as strong as i'd like"). A profile's own
# `swap_strength` wins.
SWAP_STRENGTH = 0.8
# The swap model. HyperSwap 1a was the first approved recipe; on a small,
# turned, full-body face it barely changed it (ArcFace against Partner's photos
# 0.17 -> 0.20, 2026-09-27) where inswapper_128 reached 0.81, and after a
# Klein head swap (`headswap`) inswapper reached 0.77-0.85 on both trial
# pictures at 0.5, 0.65 and 0.8 alike (2026-09-29, the user: "klein with
# inswapper seems to work well enough"). A profile's own `swap_model` wins.
SWAP_MODEL = 'inswapper_128'
SWAP_MODELS = ('inswapper_128', 'hyperswap_1a_256', 'hyperswap_1b_256', 'hyperswap_1c_256',
               'ghost_2_256', 'simswap_256', 'blendswap_256')
# The weight past which a model's likeness falls, and a strength is held to.
# Above 0.5 the person's face is pushed past their own, away from the face
# replaced, to take the last of the stranger out. With inswapper it took the
# person out: ArcFace against Partner's photos, the swap alone, on five heads
# Klein had drawn 0.848 at 0.5, 0.831 at 0.8 and 0.813 at 1.0, and on two
# generated faces with no head swap 0.836 at 0.5 against 0.798 at 1.0 - lower
# at the higher weight on every picture (2026-09-29). SWAP_STRENGTH 0.8 and a
# profile's 1.0 were set when the model was HyperSwap, which has no peak here:
# it was not measured.
SWAP_PEAK = {'inswapper_128': 0.5}
# How far the swapped face's colour is moved to that of the face it replaced
# (tools/facefusion_swap.py --tone). Without it Partner's face came back pale
# pink on a body in a low sun, after a head swap that had the light right
# (live, 2026-09-29).
SWAP_TONE = 0.8
# FaceFusion's face masks. "occlusion" is what is in front of the face: the
# swap goes behind it. Without it (box and region alone, until 2026-09-29)
# the new face was painted over glasses frames, which came back mottled and
# half rubbed out, and over a strand of hair across a cheek. With it the
# frames are the picture's own, to the pixel, on thin metal and thick frames
# alike, for 0.02-0.04 of ArcFace likeness (0.87 to 0.84) - the glasses being
# the picture's, not hers.
SWAP_MASKS = ('box', 'occlusion', 'region')
# The parts of the face that are swapped: FaceFusion's face mask regions, all
# but "mouth" - the inside of it. inswapper draws a face at 128 px, and its
# teeth came back as yellowed blocks in a smile that Klein's head had drawn
# whole (live, 2026-09-29); the teeth are not what the likeness is in (ArcFace
# 0.887 with them swapped, 0.886 without). The lips are swapped.
SWAP_REGIONS = ('skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'glasses',
                'nose', 'upper-lip', 'lower-lip')
# Behind glasses the eyes are swapped and the cheek under them is not: what
# lies below this line of the swap's own crop (0-1 down it, the eyes at 0.40
# and the tip of the nose at 0.56) stays the picture's. inswapper paints a
# bare cheek where the picture has one seen through a lens, and it showed as
# a pink patch with a hard edge under each eye (live, 2026-09-29). Leaving
# all that is behind the glasses cost the likeness 0.05 (the eyes are most of
# it), this 0.01-0.03. 0.44 cut into the lower lids.
SWAP_LENS_LINE = 0.47
# How much of pixel boost's weave is evened out (tools/facefusion_swap.py
# `even`): all of it.
SWAP_DEWEAVE = 1.0
# FaceFusion's face enhancer after the swap, on the same face and through
# the swap's own mask ('gfpgan_1.4'; '' is none), and how much of it is
# taken (0-100). inswapper's face is soft; GFPGAN 1.4 sharpens the eyes,
# lashes and lips and leaves the skin smooth. Its likeness cost grows with
# the blend - the swap alone 0.846 on five heads, 0.839 at 20, 0.832 at 40,
# 0.818 at 60, 0.802 at 80 - and after the eye pass, which redraws the eyes
# it sharpened, little of it shows: 0.812 at the end without it, 0.797 at
# 40, 0.786 at 60 (2026-09-29). So it is off. Sitter's to turn on; it runs
# only when its model is installed (`enhancer`), never downloaded by the app.
SWAP_ENHANCE = ''
SWAP_ENHANCE_BLEND = 60
MODELS = ROOT / '.runtime/facefusion/.assets/models'
_SWAP_LOCK = threading.Lock()  # FaceFusion's jobs/temp directories are shared across backend lanes.
# An error's own line in the worker's log: "RuntimeError: what went wrong".
_ERROR = re.compile(r'^[A-Za-z_][\w.]*(?:Error|Exception): (.+)$')


def available():
    return PYTHON.is_file() and (ROOT / '.runtime/facefusion/facefusion.py').is_file()


def preview_path(path):
    try:
        key = '%s:%s:%s' % (path, os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        key = str(path)
    return str(ROOT / '.work/profile-previews' / (hashlib.sha256(key.encode()).hexdigest() + '.png'))


def prepare_previews(profiles):
    """Make missing previews. True when there were any to make."""
    paths = {p for profile in profiles for p in
             list(profile.get('references') or []) + [profile.get('avatar')] if p and os.path.isfile(p)}
    pending = [(p, preview_path(p)) for p in paths if not os.path.isfile(preview_path(p))]
    if not pending:
        return False
    folder = ROOT / '.work/profile-previews'
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as request:
        json.dump(pending, request)
    child = None
    try:
        child = studio_procs.spawn([str(PYTHON), str(ROOT / 'tools/facefusion_previews.py'), request.name],
                                   creationflags=studio_procs.NO_WINDOW,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child.proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass                          # whatever it finished is still used
    finally:
        if child:
            child.stop(grace=0)
        os.unlink(request.name)
    return True


def selected(library, settings):
    """Profiles chosen on the form; the avatar is deliberately never a reference."""
    result = []
    scene = settings.get('scene_faces')
    if scene is not None:
        for person in scene.get('people') or []:
            ident = library.get('identities', person.get('identity'))
            if ident and ident.get('face_swap', True) and (ident.get('references') or not ident.get('lora')):
                result.append(dict(ident, target_region=person.get('region'),
                                   person_id=person.get('id')))
        return result
    for item in settings.get('identities') or []:
        ident = library.get('identities', item.get('id') if isinstance(item, dict) else item)
        if ident and ident.get('face_swap', True) and (ident.get('references') or not ident.get('lora')):
            if ident['id'] not in {r['id'] for r in result}:
                result.append(ident)
    return result


def profile_errors(profiles):
    errors = []
    if profiles and not available():
        errors.append('FaceFusion is not installed. Set up the local FaceFusion runtime, '
                      'or turn off the final face swap in Image references on the People tab.')
    for profile in profiles:
        refs = profile.get('references') or []
        if not refs:
            errors.append('%s has no reference picture. Add one in Image references on the People tab.' % profile['name'])
        elif any(not os.path.isfile(p) for p in refs):
            errors.append('%s needs existing reference photos. Update them in Image references on the People tab.'
                          % profile['name'])
        region = profile.get('target_region')
        if 'target_region' in profile and not valid_region(region):
            errors.append('%s needs a visible face target. Reframe the person in Scene Builder.'
                          % profile['name'])
    return errors


def valid_region(region):
    return (isinstance(region, (list, tuple)) and len(region) == 4
            and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in region)
            and region[0] < region[2] and region[1] < region[3])


def target_face(boxes, region=None, point=None, index=None, count=1):
    """Select only an unambiguous face. Boxes and targets use normalized coordinates."""
    if point is not None:
        hits = [i for i, b in enumerate(boxes) if b[0] <= point[0] <= b[2]
                and b[1] <= point[1] <= b[3]]
    elif region is not None:
        if not valid_region(region):
            raise RuntimeError('Invalid face target region.')
        hits = [i for i, b in enumerate(boxes)
                if region[0] <= (b[0] + b[2]) / 2 <= region[2]
                and region[1] <= (b[1] + b[3]) / 2 <= region[3]]
    elif index is not None and len(boxes) == count:
        hits = [index] if 0 <= index < len(boxes) else []
    else:
        hits = [0] if len(boxes) == 1 and count == 1 else []
    if len(hits) != 1:
        raise RuntimeError('The target face is ambiguous or missing. Open Fix a spot, '
                           'choose the identity, then use Choose face and click its face.')
    return hits[0]


def strength(identity):
    """The profile's face swap strength, on FaceFusion's 0-1 scale in its 0.05
    steps, and no more than its swap model's peak (SWAP_PEAK)."""
    value = identity.get('swap_strength')
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        value = SWAP_STRENGTH
    value = min(value, SWAP_PEAK.get(model(identity), 1.0))
    return round(min(1.0, max(0.0, value)) * 20) / 20


def model(identity):
    """The profile's swap model when it names one FaceFusion has, else SWAP_MODEL."""
    value = identity.get('swap_model')
    return value if value in SWAP_MODELS else SWAP_MODEL


def enhancer():
    """The face enhancer a swap may run after it: SWAP_ENHANCE when its model
    and hash are installed in FaceFusion's own folder, else None. FaceFusion
    would download a missing model on its own; it is never asked to."""
    if SWAP_ENHANCE and all((MODELS / (SWAP_ENHANCE + ext)).is_file() for ext in ('.onnx', '.hash')):
        return SWAP_ENHANCE
    return None


def swap(data, identity, stop=None, face_index=None, face_count=1):
    while not _SWAP_LOCK.acquire(timeout=0.1):
        if stop and stop():
            raise RuntimeError('Face swap cancelled.')
    try:
        if stop and stop():
            raise RuntimeError('Face swap cancelled.')
        return _swap(data, identity, stop, face_index, face_count)
    finally:
        _SWAP_LOCK.release()


def failure(log):
    """Why the worker failed, out of its log: the last error it raised, in
    its own words and without the traceback above it; else the log's last
    lines; else that it said nothing."""
    lines = [x.strip() for x in log.splitlines() if x.strip()]
    for line in reversed(lines):
        said = _ERROR.match(line)
        if said:
            return said.group(1)
    return ' '.join(lines[-3:])[-600:] or 'FaceFusion ended without saying why.'


def _clear(folder, tries=30, wait=0.1):
    """Remove a swap's folder, the picture in it and all. On Windows a worker
    that was stopped lets go of its log a moment after it ends (what it
    started - ffmpeg - holds the same file), and a removal that fails then
    must not stand in for why the swap ended: a cancel read "[WinError 32]
    ... run.log" and left the folder behind (2026-09-29)."""
    for _ in range(tries):
        shutil.rmtree(folder, ignore_errors=True)
        if not os.path.exists(folder):
            return True
        time.sleep(wait)
    return False


def _swap(data, identity, stop=None, face_index=None, face_count=1):
    if not available():
        raise RuntimeError('FaceFusion is not installed. The selected face was not applied.')
    refs = identity.get('references') or []
    if not refs or any(not os.path.isfile(p) for p in refs):
        raise RuntimeError('%s needs existing reference photos in Identities.' % identity['name'])
    folder = tempfile.mkdtemp(prefix='studio-facefusion-')
    child = None
    try:
        # FaceFusion keeps its working copy under .runtime/facefusion-temp in
        # a folder named after the target file, and clears that folder before
        # and after a run. Two swaps at once with the same file name - another
        # Studio process, the phone server's - cleared each other's, and one
        # ended "copying image failed" (2026-09-29). So the target's name is
        # this swap's own.
        target = Path(folder) / (Path(folder).name + '.png')
        output = Path(folder) / 'result.png'
        target.write_bytes(data)
        args = [str(PYTHON), str(SCRIPT), '--identity', identity['id'], '--sources', *refs,
                '--target', str(target), '--output', str(output),
                '--model', model(identity), '--weight', str(strength(identity)),
                '--tone', str(SWAP_TONE), '--masks', *SWAP_MASKS,
                '--regions', *SWAP_REGIONS, '--lens-line', str(SWAP_LENS_LINE),
                '--deweave', str(SWAP_DEWEAVE)]
        if enhancer():
            args += ['--enhance', enhancer(), '--enhance-blend', str(SWAP_ENHANCE_BLEND)]
        if face_index is not None:
            args += ['--face-index', str(face_index), '--face-count', str(face_count)]
        if identity.get('target_region') is not None:
            args += ['--target-region', *map(str, identity['target_region'])]
        if identity.get('target_point') is not None:
            args += ['--target-point', *map(str, identity['target_point'])]
        with open(Path(folder) / 'run.log', 'w+', encoding='utf-8') as log:
            child = studio_procs.spawn(args, cwd=str(ROOT), stdout=log, stderr=log,
                                       creationflags=studio_procs.NO_WINDOW)
            deadline = time.monotonic() + 300
            while child.proc.poll() is None:
                if stop and stop():
                    raise RuntimeError('Face swap cancelled.')
                if time.monotonic() > deadline:
                    raise RuntimeError('FaceFusion did not finish within five minutes.')
                time.sleep(0.1)
            if child.proc.returncode or not output.is_file():
                log.seek(0)
                raise RuntimeError('%s\'s face swap failed: %s' % (identity['name'],
                                                                  failure(log.read())))
            report = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
            if report['outside_mask_changed_pixels'] != 0:
                raise RuntimeError('The face swap changed pixels outside its mask.')
            report = {k: v for k, v in report.items() if k not in ('target', 'output')}
            return output.read_bytes(), report
    finally:
        if child is not None:
            child.stop(grace=0)
        _clear(folder)
