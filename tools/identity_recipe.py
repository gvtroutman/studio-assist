"""Run staged identity evaluations through Image Studio, without changing profiles.

python tools/identity_recipe.py run recipe.json portrait output-folder
python tools/identity_recipe.py run recipe.json scene output-folder
See docs/identity-recipe.md for the recipe and review contract.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fingerprint(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def cases(recipe, stage, portrait_report=None):
    if stage not in ('portrait', 'scene'):
        raise ValueError('Choose portrait or scene.')
    seeds = recipe['seeds']
    if len(set(seeds)) < 2:
        raise ValueError('Use at least two distinct seeds to check repeatability.')
    candidates = recipe['candidates']
    ids = [c['id'] for c in candidates]
    if not ids or len(set(ids)) != len(ids) or 'baseline' in ids:
        raise ValueError('Give each candidate a unique ID other than baseline.')
    if stage == 'scene':
        if (not portrait_report or portrait_report.get('stage') != 'portrait'
                or portrait_report.get('recipe_hash') != fingerprint(recipe)):
            raise ValueError('Scene tests require a portrait report for this exact recipe.')
        approved = []
        for candidate in candidates:
            rows = [r for r in portrait_report['results'] if r['candidate'] == candidate['id']]
            if all(any(r['seed'] == seed and r['status'] == 'complete'
                       and r.get('outputs') and r.get('prompt') == recipe['portrait']
                       and [candidate['file'], candidate.get('strength', 1.0)] in r.get('applied_loras', [])
                       and r.get('review', {}).get('likeness') == 'pass'
                       and r.get('review', {}).get('reviewer', '').strip()
                       and r.get('review', {}).get('notes', '').strip()
                       for r in rows) for seed in seeds):
                approved.append(candidate)
        if not approved:
            raise ValueError('No checkpoint passed portrait likeness review at every seed.')
        candidates = approved
    prompts = [recipe['portrait']] if stage == 'portrait' else recipe['scenes']
    if not prompts or any(not p.strip() for p in prompts):
        raise ValueError('The stage needs a nonempty prompt.')
    return [(candidate, seed, prompt) for prompt in prompts for seed in seeds
            for candidate in [{'id': 'baseline', 'file': None}] + candidates]


def run(recipe, stage, output, portrait_report=None):
    import apps.image_studio.imagegen as ig
    planned = cases(recipe, stage, portrait_report)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'recipe.json').write_text(json.dumps(recipe, indent=2), encoding='utf-8')
    report = {'recipe_hash': fingerprint(recipe), 'stage': stage,
              'likeness_validated': False, 'results': []}
    studio = ig.Studio(root=str(output / 'studio'))
    backend = dict(recipe['backend'])
    try:
        studio.check(backend, full=True)
        for candidate, seed, prompt in planned:
            queue = studio.client(backend).get_queue()
            if queue.get('queue_running') or queue.get('queue_pending'):
                raise RuntimeError('Backend is busy; retry in a new output folder when idle.')
            profile = {'id': 'subject', 'name': recipe['name'],
                       'trigger': recipe['trigger'], 'description': recipe.get('description', ''),
                       'references': [], 'face_swap': False, 'use_references': False}
            if candidate['file']:
                studio.lib.save('loras', [{'id': candidate['id'], 'file': candidate['file'],
                    'family': 'flux1', 'name': candidate['id'],
                    'strength': candidate.get('strength', 1.0)}])
                profile.update(lora=candidate['id'], strength=candidate.get('strength', 1.0))
            studio.lib.save('identities', [profile])
            settings = dict(ig.default_settings(), model='flux-dev', identities=['subject'],
                scene=prompt, seed=seed, seed_mode='fixed', steps=25, guidance=4.0,
                width=768, height=768, anatomy=False, face_detail=False,
                auto_refine=False, hand_pass=False, refine=False)
            plan = studio.preview(settings, backend)
            if plan.errors or (candidate['file'] and not any(
                    item[0] == candidate['file'] for item in plan.loras)):
                raise RuntimeError('Preflight failed: ' + '; '.join(plan.errors + plan.warnings))
            job = ig.Job(settings, backend)
            print(stage, candidate['id'], seed, flush=True)
            studio.run_job(job, lambda j: None)
            report['results'].append({'candidate': candidate['id'], 'seed': seed,
                'prompt': prompt, 'settings': settings, 'status': job.status,
                'detail': job.detail, 'outputs': job.outputs,
                'applied_loras': job.plan.loras if job.plan else [],
                'review': {'likeness': 'pending', 'reviewer': '', 'notes': ''}})
            (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            if job.status != 'complete':
                raise RuntimeError('Render did not complete: ' + job.detail)
    finally:
        studio.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run'])
    parser.add_argument('recipe', type=Path)
    parser.add_argument('stage', choices=['portrait', 'scene'])
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--portrait-report', type=Path)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding='utf-8'))
    report = json.loads(args.portrait_report.read_text(encoding='utf-8')) if args.portrait_report else None
    if args.command == 'plan':
        print(json.dumps(cases(recipe, args.stage, report), indent=2))
    elif args.output is None:
        parser.error('run requires a new output folder')
    else:
        run(recipe, args.stage, args.output, report)


if __name__ == '__main__':
    main()
