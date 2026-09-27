import copy
import unittest

from tools.identity_recipe import cases, fingerprint


class IdentityRecipeTests(unittest.TestCase):
    def setUp(self):
        self.recipe = dict(seeds=[1, 2], candidates=[dict(id='a', file='a.safetensors')],
                           portrait='portrait', scenes=['new clothes', 'new background'])
        self.report = dict(stage='portrait', recipe_hash=fingerprint(self.recipe), results=[
            dict(candidate='a', seed=seed, prompt='portrait', status='complete', outputs=['image.png'],
                 applied_loras=[['a.safetensors', 1]],
                 review=dict(likeness='pass', reviewer='User', notes='Matches references'))
            for seed in (1, 2)])

    def test_portrait_pairs_baseline_and_candidate_at_each_seed(self):
        self.assertEqual([(c['id'], seed) for c, seed, _ in cases(self.recipe, 'portrait')],
                         [('baseline', 1), ('a', 1), ('baseline', 2), ('a', 2)])

    def test_render_success_does_not_approve_likeness(self):
        self.report['results'][1]['review']['likeness'] = 'pending'
        with self.assertRaises(ValueError):
            cases(self.recipe, 'scene', self.report)

    def test_missing_adapter_failed_job_and_missing_review_block(self):
        for key, value in [('applied_loras', []), ('status', 'failed'), ('review', {})]:
            report = copy.deepcopy(self.report)
            report['results'][0][key] = value
            with self.assertRaises(ValueError):
                cases(self.recipe, 'scene', report)

    def test_changed_recipe_invalidates_review(self):
        self.recipe['portrait'] = 'different prompt'
        with self.assertRaises(ValueError):
            cases(self.recipe, 'scene', self.report)

    def test_reviewed_candidate_advances(self):
        self.assertEqual(len(cases(self.recipe, 'scene', self.report)), 8)

    def test_single_seed_rejected(self):
        self.recipe['seeds'] = [1, 1]
        with self.assertRaises(ValueError):
            cases(self.recipe, 'portrait')

    def test_wrong_adapter_or_prompt_cannot_advance(self):
        for key, value in [('applied_loras', [['other.safetensors', 1]]),
                           ('prompt', 'meadow')]:
            report = copy.deepcopy(self.report)
            report['results'][0][key] = value
            with self.assertRaises(ValueError):
                cases(self.recipe, 'scene', report)
