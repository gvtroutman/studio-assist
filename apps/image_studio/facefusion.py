"""The Image Studio's local, final face swap. Stdlib only in the GUI process."""
import json
import hashlib
import os
from pathlib import Path
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
# How far the swapped face's colour is moved to that of the face it replaced
# (tools/facefusion_swap.py --tone). Without it Partner's face came back pale
# pink on a body in a low sun, after a head swap that had the light right
# (live, 2026-09-29).
SWAP_TONE = 0.8
_SWAP_LOCK = threading.Lock()  # FaceFusion's jobs/temp directories are shared across backend lanes.


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
    """The profile's face swap strength, on FaceFusion's 0-1 scale in its 0.05 steps."""
    value = identity.get('swap_strength')
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        value = SWAP_STRENGTH
    return round(min(1.0, max(0.0, value)) * 20) / 20


def model(identity):
    """The profile's swap model when it names one FaceFusion has, else SWAP_MODEL."""
    value = identity.get('swap_model')
    return value if value in SWAP_MODELS else SWAP_MODEL


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


def _swap(data, identity, stop=None, face_index=None, face_count=1):
    if not available():
        raise RuntimeError('FaceFusion is not installed. The selected face was not applied.')
    refs = identity.get('references') or []
    if not refs or any(not os.path.isfile(p) for p in refs):
        raise RuntimeError('%s needs existing reference photos in Identities.' % identity['name'])
    with tempfile.TemporaryDirectory(prefix='studio-facefusion-') as folder:
        target, output = Path(folder) / 'target.png', Path(folder) / 'result.png'
        target.write_bytes(data)
        args = [str(PYTHON), str(SCRIPT), '--identity', identity['id'], '--sources', *refs,
                '--target', str(target), '--output', str(output),
                '--model', model(identity), '--weight', str(strength(identity)),
                '--tone', str(SWAP_TONE)]
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
            try:
                while child.proc.poll() is None:
                    if stop and stop():
                        raise RuntimeError('Face swap cancelled.')
                    if time.monotonic() > deadline:
                        raise RuntimeError('FaceFusion did not finish within five minutes.')
                    time.sleep(0.1)
                if child.proc.returncode or not output.is_file():
                    log.seek(0)
                    detail = log.read()[-2000:]
                    raise RuntimeError('%s\'s face swap failed: %s' % (identity['name'], detail))
                report = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
                if report['outside_mask_changed_pixels'] != 0:
                    raise RuntimeError('The face swap changed pixels outside its mask.')
                report = {k: v for k, v in report.items() if k not in ('target', 'output')}
                return output.read_bytes(), report
            finally:
                child.stop(grace=0)
