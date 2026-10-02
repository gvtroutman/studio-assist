"""Matched-seed pose-strength experiment, isolated from the user's library."""
import copy
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apps.image_studio import imagegen as ig
from apps.image_studio.scene import scene as sc
from tools.run_scene_likeness import ensure_backend, isolated, save


def wait_for_backend(studio, backend):
    deadline = time.monotonic() + 600
    while True:
        try:
            return ensure_backend(studio, backend, start=True)
        except RuntimeError as error:
            if 'Backend is busy' not in str(error) or time.monotonic() >= deadline:
                raise
            print('WAIT: existing backend jobs are still running', flush=True)
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corrected', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'Scene Math Measurements' / ('pose-contact-comparison' if args.corrected else 'pose-strength-comparison')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        raise RuntimeError('Comparison already exists; keep its evidence intact')
    studio = isolated(output)
    try:
        backend = studio.lib.get('backends', '5090')
        client = wait_for_backend(studio, backend)
        studio.check(backend, full=True)
        source = (ROOT / 'Scene Math Measurements/pose-contact-audit/corrected.scene.json' if args.corrected else
                  ROOT / 'Scene Enhancements/prepared/oktoberfest - playing accordion.scene.json')
        scene, problems = sc.load(str(source))
        scene['real_faces'] = False
        characters = {r['id']: r for r in studio.lib.all('characters')}
        identities = {r['id']: r for r in studio.lib.all('identities')}
        maps, notes = sc.scene_maps(scene, {'pose', 'composition'}, str(output))
        words, extra = sc.generation(scene, maps, characters, identities)
        settings = dict(ig.default_settings(), **extra)
        settings.update(model='z-image-turbo', backend='5090', scene=words.text,
                        seed_mode='fixed', auto_refine=False, face_detail=False,
                        hand_pass=False, head_swap=False, glasses_pass=False,
                        refine=False, batch=1, critic=False)
        shot, factor = sc.framing(scene)
        report = {'scene': scene, 'scene_warnings': problems, 'maps': maps,
                  'notes': notes, 'shot': shot, 'framing_factor': factor,
                  'method': 'Only pose strength changes. Depth, prompt, maps, model and seeds identical; finishing disabled.',
                  'runs': []}
        save(output / 'report.json', report)
        arms = ['pose085', 'pose100'] if args.corrected else ['scaled', 'full']
        for index, seed in enumerate([261002, 261003]):
            for arm in (arms if index == 0 else list(reversed(arms))):
                wait_for_backend(studio, backend)
                folder = output / f'{arm}_{seed}'
                folder.mkdir()
                one = copy.deepcopy(settings)
                one['seed'] = seed
                one['pose']['strength'] = ({'pose085': 0.85, 'pose100': 1.0}[arm] if args.corrected else
                                          round(scene['pose_strength'] * (factor if arm == 'scaled' else 1), 3))
                plan = studio.preview(one, backend)
                if plan.errors:
                    raise RuntimeError('Preflight: ' + '; '.join(plan.errors))
                if 'pose_image' not in plan.images or 'composition_image' not in plan.images:
                    raise RuntimeError('Both conditioning maps required')
                save(folder / 'settings.json', one)
                save(folder / 'plan.json', {'values': plan.values, 'warnings': plan.warnings})
                print('RENDER', arm, seed, 'pose', plan.values.get('pose_strength'),
                      'depth', plan.values.get('composition_strength'), flush=True)
                job = ig.Job(one, backend)
                last = [None, 0]
                def progress(job):
                    now = time.monotonic()
                    if job.status != last[0] or now - last[1] > 20:
                        print('PROGRESS', arm, seed, job.status, job.detail, flush=True)
                        last[:] = [job.status, now]
                start = time.perf_counter()
                studio.run_job(job, progress)
                row = {'arm': arm, 'seed': seed, 'pose_strength': one['pose']['strength'],
                       'depth_strength': one['composition']['strength'], 'status': job.status,
                       'detail': job.detail, 'seconds': time.perf_counter() - start,
                       'outputs': job.outputs, 'record': job.record}
                report['runs'].append(row)
                save(output / 'report.json', report)
                print('DONE', arm, seed, job.status, row['seconds'], job.outputs, flush=True)
                if job.status != 'complete':
                    raise RuntimeError('Render failed: ' + job.detail)
    finally:
        studio.close()


if __name__ == '__main__':
    main()
