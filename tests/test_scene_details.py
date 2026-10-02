"""Flag-pattern and lettering finish: no models or network."""
import json
import unittest
from unittest.mock import patch

from apps.image_studio import imagegen as ig
from apps.image_studio.scene import details as dt
from test_imagegen import FLUX_FILES, TempStudioMixin, settle
from test_wear import WornClient


class TestDetails(unittest.TestCase):
    def test_only_described_designs_are_requested_and_wording_is_short(self):
        self.assertEqual(dt.requests({"scene": "A man at Oktoberfest"}), [])
        self.assertEqual(dt.requests({"scene": "A red flag on a boat"}), [])
        self.assertEqual(dt.requests({"scene": "German flags at Oktoberfest"}), [])
        req = dt.requests({"scene": "Oktoberfest flags and a welcome sign"})
        self.assertEqual([(r['noun'], r['text']) for r in req], [('flag', ''), ('sign', 'Oktoberfest')])
        self.assertEqual(dt.requests({"scene": "Oktoberfest flags", "scene_details_pass": False}), [])
        self.assertEqual(dt.requests({"scene": "Oktoberfest flags", "mode": "fix"}), [])
        self.assertEqual(dt.requests({'scene': 'A wooden sign reading “Biergarten”'})[0]['text'],
                         'Biergarten')
        self.assertEqual(dt.requests({'scene': 'A sign reading "A" and "B"'}), [])

    def test_scene_prop_wording_does_not_copy_a_quoted_character(self):
        req = dt.requests({'scene': 'Oktoberfest, a sign and "Max"', 'scene_layout': {
            'details': 'Oktoberfest', 'objects': [
                {'asset': 'person', 'description': 'A man called "Max"'},
                {'asset': 'box', 'description': 'A wooden sign reading "Biergarten"'},
                {'asset': 'cylinder', 'dressing': [{'text': 'Bavarian flags'}]}]}})
        self.assertEqual([r['text'] for r in req if r['text']], ['Biergarten'])
        self.assertEqual(req[0]['pattern'], 'bavarian')

    def test_boxes_are_bounded_deduplicated_and_tiny_or_global_hits_are_skipped(self):
        wanted = dt.requests({'scene': 'Bavarian flags and an Oktoberfest sign'})
        boxes = [(10, 20, 80, 40, 'flag'), (11, 21, 80, 40, 'flag'),
                 (0, 0, 10, 8, 'flag'), (0, 0, 1024, 1024, 'flag'),
                 (500, 300, 200, 80, 'sign'), (0, 0, 50, 50, 'face')]
        planned, notes = dt.plan(1024, 1024, wanted, boxes)
        self.assertEqual([p['noun'] for p in planned], ['flag', 'sign'])
        self.assertEqual(notes, [])
        self.assertTrue(all(p['crop']['x'] >= 0 and p['crop']['y'] >= 0 for p in planned))
        many, _ = dt.plan(1024, 1024, wanted[:1], [(x * 80, 0, 60, 40, 'flag') for x in range(12)])
        self.assertEqual(len(many), dt.MAX_TARGETS)
        both, _ = dt.plan(1024, 1024, wanted, [(x * 80, 0, 60, 40, 'flag') for x in range(12)] +
                         [(400, 400, 150, 60, 'sign')])
        self.assertEqual(both[-1]['noun'], 'sign')

    def test_ambiguous_sign_wording_is_not_assigned_by_detection_order(self):
        wanted = dt.requests({'scene': 'Signs', 'scene_layout': {'objects': [
            {'description': 'A sign reading "A"'}, {'description': 'A sign reading "B"'}]}})
        planned, notes = dt.plan(1024, 1024, wanted, [(10, 10, 100, 50, 'sign')])
        self.assertEqual(planned, [])
        self.assertTrue(notes)

    def test_the_design_reference_has_area_sampled_blue_and_white_pixels(self):
        data = dt.pattern_png(64)
        from core import icons
        rgba, w, h = icons.png_to_rgba(data)
        self.assertEqual((w, h), (64, 64))
        colours = {tuple(rgba[i:i + 4]) for i in range(0, len(rgba), 4)}
        self.assertIn((35, 112, 192, 255), colours)
        self.assertIn((255, 255, 255, 255), colours)
        self.assertTrue(all(c[3] == 255 and c[0] <= c[1] <= c[2] for c in colours))
        self.assertLessEqual(len(colours), 5)

    def test_lattice_periods_and_reference_aspects(self):
        # Moving by either full repeat vector keeps phase; a half repeat
        # exchanges blue and white. Same pixel pitch means the same rhombi
        # in a portrait and landscape reference, rather than stretched ones.
        for x, y in ((1.3, 2.7), (22.1, 17.4), (-6.2, 5.5)):
            a = dt.lattice(x, y, 10)
            self.assertEqual(dt.lattice(x + 10, y, 10), a)
            self.assertEqual(dt.lattice(x + 4, y + 16, 10), a)
            self.assertEqual(dt.lattice(x + 5, y + 8, 10), 1 - a)
        self.assertEqual(dt.reference_size((0, 0, 200, 100)), (512, 256))
        self.assertEqual(dt.reference_size((0, 0, 100, 200)), (256, 512))
        from core import icons
        wide, _, _ = icons.png_to_rgba(dt.pattern_png(128, 64))
        tall, _, _ = icons.png_to_rgba(dt.pattern_png(64, 128))
        for y in range(64):
            self.assertEqual(wide[y * 128 * 4:(y * 128 + 64) * 4],
                             tall[y * 64 * 4:(y + 1) * 64 * 4])
        with self.assertRaises(ValueError):
            dt.pattern_png(0)

    def test_graph_uses_a_design_reference_and_masks_only_the_target_box(self):
        wanted = dt.requests({'scene': 'Oktoberfest flags and sign reading "Biergarten"'})
        items, _ = dt.plan(1024, 1024, wanted, [(100, 100, 100, 60, 'flag'),
                                              (500, 400, 150, 60, 'sign')])
        g = dt.detail_graph('source.png', items, ig.MAX_SEED, 'out', 'sam3.pt',
                            (1024, 1024), 'diamonds.png')
        self.assertEqual(g['i1_photo']['inputs']['image'], 'diamonds.png')
        self.assertIn('image 2', g['i1_text']['inputs']['text'])
        self.assertNotIn('i2_photo', g)
        self.assertEqual(g['i2_guide']['inputs']['positive'], ['i2_pos1', 0])
        self.assertIn('"Biergarten"', g['i2_text']['inputs']['text'])
        self.assertEqual(g['i2_cut']['inputs']['image'], ['i1_put', 0])
        self.assertEqual(g['i1_put']['inputs']['mask'], ['i1_ink5', 0])
        self.assertEqual(g['i1_ink4']['inputs']['source'], ['i1_ma', 0])
        self.assertEqual(g['i1_ink5']['inputs']['source'], ['i1_limit2', 0])
        self.assertEqual(g['i1_ink0']['inputs']['expand'], -1)
        self.assertEqual(g['i1_limit1']['inputs']['width'], 100)
        self.assertEqual(g['i1_word']['inputs']['text'], 'flag')
        self.assertEqual(g['i2_noise']['inputs']['noise_seed'], 0)
        # Every linked node exists after removing the unused second reference.
        for node in g.values():
            for value in node['inputs'].values():
                if ig._is_link(value):
                    self.assertIn(value[0], g)


class DetailClient(WornClient):
    fail_at = None

    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
        g = self.graphs[int(pid[3:]) - 1]
        if 'p0d' in g and 'i1_ks' not in g:
            found = {'flag': [(100, 100, 100, 60)], 'sign': [(500, 400, 150, 60)]}
            outputs = {'7': {'text': ['1024']}, '8': {'text': ['1024']}}
            for k in (k for k in g if k.endswith('t') and k.startswith('p')):
                word = g[k]['inputs']['text'].split(':')[0]
                outputs[k[:-1] + 'v'] = {'text': [json.dumps([[
                    dict(x=x, y=y, width=w, height=h) for x, y, w, h in found.get(word, [])]])]}
            return {'status': {'completed': True}, 'outputs': outputs}
        if 'i1_ks' in g and g['save']['inputs']['filename_prefix'].endswith(self.fail_at or 'NEVER'):
            raise ig.ComfyError('detail failed')
        return super().listen_for_progress(pid, on_event, stop, timeout)


class TestRun(TempStudioMixin, unittest.TestCase):
    client_factory = DetailClient

    def test_generate_runs_the_details_finish_and_records_it(self):
        self.studio.client_factory = DetailClient
        # Headswap's fake backend includes the installed Klein weights.
        from test_headswap import KLEIN
        inventory = {k: set(v) | KLEIN.get(k, set()) for k, v in FLUX_FILES.items()}
        with patch('test_imagegen.FLUX_FILES', inventory):
            jobs = self.studio.submit(dict(ig.default_settings(), model='z-image-turbo',
                backend='5090', scene='Oktoberfest flags and a wooden sign', hand_pass=False))
            settle(jobs)
        job = jobs[0]
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertEqual([p['label'] for p in job.passes], [dt.LABEL, dt.LABEL])
        self.assertIn('Oktoberfest', ' '.join(job.record['notes']))
        self.assertIn((dt.STATUS, dt.LABEL), ig.pipeline_stages(self.studio.lib, job.settings))
        self.assertNotIn((dt.STATUS, dt.LABEL), ig.pipeline_stages(
            self.studio.lib, dict(job.settings, scene_details_pass=False)))

    def test_failed_second_detail_keeps_the_first_and_drops_the_failed_graph(self):
        self.studio.client_factory = DetailClient
        from test_headswap import KLEIN
        DetailClient.fail_at = '_0_1'
        try:
            inventory = {k: set(v) | KLEIN.get(k, set()) for k, v in FLUX_FILES.items()}
            with patch('test_imagegen.FLUX_FILES', inventory):
                jobs = self.studio.submit(dict(ig.default_settings(), model='z-image-turbo',
                    backend='5090', scene='Oktoberfest flags and a sign', hand_pass=False))
                settle(jobs)
            job = jobs[0]
            self.assertEqual(job.status, 'complete', job.detail)
            self.assertEqual(len(job.passes), 1)
            self.assertTrue(any('last finished picture' in n for n in job.record['notes']))
            self.assertFalse(any('lettering requested' in n for n in job.record['notes']))
        finally:
            DetailClient.fail_at = None

    def test_a_backend_without_klein_keeps_the_generated_picture(self):
        jobs = self.studio.submit(dict(ig.default_settings(), model='z-image-turbo',
            backend='5090', scene='Oktoberfest flags and a sign', hand_pass=False))
        settle(jobs)
        job = jobs[0]
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertEqual(job.passes, [])
        self.assertTrue(any('No flags and lettering pass' in n for n in job.record['notes']))


if __name__ == '__main__':
    unittest.main()
