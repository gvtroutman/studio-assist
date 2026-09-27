"""Train one identity LoRA with ai-toolkit: the Build LoRA button's worker.

Run in the ai-toolkit venv by `studio_lora_train.Build` with a spec.json
written by `studio_lora_train.write_spec`. The source photos are never
changed: each is copied into <work>/dataset as an upright RGB PNG (longest
side at most 1536 px) with a one-line caption. Then ai-toolkit's run.py trains
on <work>/train.json, and its final .safetensors is copied to `lora_out`.

Prints, one per line, for the app: `STEP n total`, then `DONE <path>` or
`ERROR <words>`. Everything ai-toolkit says goes to <work>/train.log.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

LONGEST = 1536


def say(text):
    print(text, flush=True)


def prepare(spec, dataset):
    from PIL import Image, ImageOps
    dataset.mkdir(parents=True, exist_ok=True)
    kept = 0
    for i, src in enumerate(spec['photos'], 1):
        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((LONGEST, LONGEST), Image.LANCZOS)
                im.save(dataset / ('photo%03d.png' % i))
        except Exception as exc:
            say('photo skipped: %s: %s' % (src, exc))
            continue
        (dataset / ('photo%03d.txt' % i)).write_text(spec['caption'], encoding='utf-8')
        kept += 1
    return kept


def train(spec, work):
    """ai-toolkit's run.py, its output to train.log; tqdm's `n/total` for
    this job's step count becomes STEP lines. -> exit code."""
    total = int(spec['steps'])
    step_re = re.compile(r'(\d+)/%d\b' % total)
    last = -1
    with open(work / 'train.log', 'wb') as log:
        proc = subprocess.Popen([sys.executable, 'run.py', str(work / 'train.json')],
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


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    work = Path(spec['work'])
    kept = prepare(spec, work / 'dataset')
    if kept < 1:
        say('ERROR None of the photos could be read.')
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
    sys.exit(main())
