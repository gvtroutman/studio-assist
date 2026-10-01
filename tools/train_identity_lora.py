"""Train one person's Klein head LoRA with ai-toolkit: the Build LoRA button's
worker.

Run in the ai-toolkit venv by `lora_train.Build` with a spec.json written by
`lora_train.write_spec`. The source photos are never changed: each is copied
into <work>/dataset as an upright RGB PNG cut to a square round the head
(`head_square`; OpenCV finds the face), with a one-line caption. A photo with
no face it can find goes in whole. Then ai-toolkit trains on <work>/train.json
and its final .safetensors is copied to `lora_out`.

Prints, one per line, for the app: `KEPT n total` (the photos that could be
read), `FACES n total` (those cut to the head), `STEP n total`, then
`DONE <path>` or `ERROR <words>`. Fewer readable photos than the spec's
`min_photos` is an ERROR before any training. Everything ai-toolkit says goes
to <work>/train.log.

`--aitk <train.json> <text encoder>` is ai-toolkit's run.py with Klein's text
encoder read from the local folder (its class names the Hugging Face repo) and
the hub offline: a missing file fails, nothing is downloaded.
"""
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

LONGEST = 768       # ai-toolkit buckets the crops down to the spec's resolution
MARGIN = 2.4        # the square's side, in face sizes: head, hair, a little shoulder
LOWER = 0.15        # of a face, how far below the face's centre the square's is


def say(text):
    print(text, flush=True)


def head_square(width, height, face):
    """The square round `face` (x, y, w, h) that goes into the dataset: MARGIN
    faces wide, centred LOWER of a face below it, made smaller where the photo
    is and slid back inside it (white padding showed as bars on photos not on
    white) -> (left, top, right, bottom)."""
    x, y, w, h = face
    side = int(min(max(w, h) * MARGIN, width, height))
    cx, cy = x + w / 2.0, y + h / 2.0 + h * LOWER
    left = int(min(max(cx - side / 2.0, 0), width - side))
    top = int(min(max(cy - side / 2.0, 0), height - side))
    return left, top, left + side, top + side


def find_face(im):
    """The biggest face OpenCV's cascades find in a PIL image: frontal, else
    profile facing either way -> (x, y, w, h) or None."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
    small = max(24, min(gray.shape) // 12)
    width = gray.shape[1]
    for name, flip in (("haarcascade_frontalface_default.xml", False),
                       ("haarcascade_profileface.xml", False),
                       ("haarcascade_profileface.xml", True)):
        cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))
        faces = cascade.detectMultiScale(cv2.flip(gray, 1) if flip else gray, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(small, small))
        if len(faces):
            x, y, w, h = (int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
            return (width - x - w if flip else x), y, w, h
    return None


def prepare(spec, dataset, find=None):
    """-> (photos in the dataset, those cut to the head, paths that could not be read).
    `find` is `find_face` unless given."""
    from PIL import Image, ImageOps
    find = find or find_face
    dataset.mkdir(parents=True, exist_ok=True)
    kept, faces, skipped = 0, 0, []
    for i, src in enumerate(spec['photos'], 1):
        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                face = find(im)
                if face is not None:
                    im = im.crop(head_square(im.size[0], im.size[1], face))
                im.thumbnail((LONGEST, LONGEST), Image.LANCZOS)
                im.save(dataset / ('photo%03d.png' % i))
        except Exception as exc:
            say('photo skipped: %s: %s' % (src, exc))
            skipped.append(src)
            continue
        if face is None:
            say('no face found, kept whole: %s' % src)
        (dataset / ('photo%03d.txt' % i)).write_text(spec['caption'], encoding='utf-8')
        kept += 1
        faces += face is not None
    return kept, faces, skipped


def train(spec, work):
    """ai-toolkit (through `--aitk`), its output to train.log; tqdm's `n/total`
    for this job's step count becomes STEP lines. -> exit code."""
    total = int(spec['steps'])
    step_re = re.compile(r'(\d+)/%d\b' % total)
    last = -1
    with open(work / 'train.log', 'wb') as log:
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), '--aitk',
                                 str(work / 'train.json'), spec['text_encoder']],
                                cwd=spec['toolkit'], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        buf = b''
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, 'read1') else proc.stdout.read(4096)
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            buf = (buf + chunk)[-4096:]
            found = step_re.findall(buf.decode('utf-8', 'replace'))
            if found and int(found[-1]) != last:
                last = int(found[-1])
                say('STEP %d %d' % (last, total))
        return proc.wait()


def aitk(job, text_encoder):
    """ai-toolkit's run.py on `job`, in this process, with Klein 9B's text
    encoder at `text_encoder` and the Hugging Face hub offline."""
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    toolkit = os.getcwd()
    sys.path.insert(0, toolkit)
    import extensions_built_in.diffusion_models.flux2.flux2_klein_model as klein
    klein.Flux2Klein9BModel.flux2_klein_te_path = text_encoder
    sys.argv = ['run.py', job]
    runpy.run_path(os.path.join(toolkit, 'run.py'), run_name='__main__')


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    work = Path(spec['work'])
    kept, faces, skipped = prepare(spec, work / 'dataset')
    say('KEPT %d %d' % (kept, len(spec['photos'])))
    say('FACES %d %d' % (faces, kept))
    need = max(1, int(spec.get('min_photos', 1)))
    if kept < need:
        names = [Path(p).name for p in skipped]
        if len(names) > 5:
            names = names[:5] + ['%d more' % (len(names) - 5)]
        say('ERROR Only %d of %d photos could be read; a LoRA needs at least %d.%s'
            % (kept, len(spec['photos']), need,
               ' Unreadable: %s.' % ', '.join(names) if names else ''))
        return 1
    say('STEP 0 %d' % spec['steps'])
    code = train(spec, work)
    final = work / 'output' / spec['name'] / (spec['name'] + '.safetensors')
    if code != 0 or not final.is_file():
        say('ERROR ai-toolkit stopped (exit %s) without a LoRA; see %s'
            % (code, work / 'train.log'))
        return 1
    out = Path(spec['lora_out'])
    tmp = out.with_suffix('.part')
    shutil.copyfile(final, tmp)
    os.replace(tmp, out)
    say('DONE %s' % out)
    return 0


if __name__ == '__main__':
    if sys.argv[1:2] == ['--aitk']:
        aitk(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main())
