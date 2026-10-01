"""Profile selection through to the saved result, without models or network."""
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import apps.image_studio.facefusion as ff
import apps.image_studio.imagegen as ig
from test_imagegen import TempStudioMixin, PNG, settle


class TestProfiles(TempStudioMixin, unittest.TestCase):
    def profile(self):
        path = os.path.join(self.dir, 'reference.png')
        with open(path, 'wb') as file:
            file.write(PNG)
        self.studio.lib.save('identities', [{'id': 'partner', 'name': 'Partner',
            'references': [path], 'use_references': False, 'avatar': 'generated-avatar.png'}])
        return self.studio.lib.get('identities', 'partner')

    def test_selected_profile_applies_after_refinement_and_is_recorded(self):
        profile = self.profile()
        order = []
        def refine(job, client, plan, values, files, say):
            order.append('refine')
            return files
        def swap(data, who, **kw):
            order.append('swap')
            self.assertEqual(who['references'], profile['references'])
            self.assertNotIn(who['avatar'], who['references'])
            return PNG, {'outside_mask_changed_pixels': 0, 'identity': who['id']}
        with patch.object(self.studio, '_refine', refine), patch.object(ff, 'available', return_value=True), \
                patch.object(ff, 'swap', side_effect=swap):
            jobs = self.studio.submit(dict(ig.default_settings(), scene='Dancing',
                model='z-image-turbo', backend='5090', identities=[{'id': 'partner'}], auto_refine=True))
            settle(jobs)
        self.assertEqual(jobs[0].status, 'complete', jobs[0].detail)
        self.assertEqual(order, ['refine', 'swap'])
        self.assertEqual(jobs[0].record['facefusion'][0]['identity'], 'partner')

    def test_missing_facefusion_fails_before_rendering(self):
        self.profile()
        with patch.object(ff, 'available', return_value=False):
            jobs = self.studio.submit(dict(ig.default_settings(), scene='Dancing',
                model='z-image-turbo', backend='5090', identities=['partner']))
            settle(jobs)
        self.assertEqual(jobs[0].status, 'failed')
        self.assertIsNone(jobs[0].graph)
        self.assertIn('FaceFusion', jobs[0].detail)

    def test_no_selection_does_not_apply_a_profile(self):
        self.profile()
        self.assertEqual(ff.selected(self.studio.lib, {'identities': []}), [])
        self.assertEqual(len(ff.selected(self.studio.lib, {'identities': ['partner', 'partner']})), 1)

    def test_swap_strength_defaults_strong_and_follows_the_profile(self):
        profile = dict(self.profile(), swap_model='hyperswap_1a_256')
        self.assertEqual(profile['swap_strength'], ff.SWAP_STRENGTH)
        self.assertGreater(ff.strength(profile), 0.5)   # 0.5 is FaceFusion's neutral
        self.assertEqual(ff.strength(dict(profile, swap_strength=0.62)), 0.6)
        self.assertEqual(ff.strength(dict(profile, swap_strength=7)), 1.0)
        self.assertEqual(ff.strength({'swap_strength': 'lots', 'swap_model': 'hyperswap_1a_256'}),
                         ff.SWAP_STRENGTH)
        self.assertEqual(ig.clean_identity({'name': 'X', 'swap_strength': 3})['swap_strength'], 1.0)

    def test_a_strength_is_held_to_its_models_peak(self):
        # inswapper's likeness fell past 0.5 on every picture (2026-09-29).
        self.assertEqual(ff.SWAP_MODEL, 'inswapper_128')
        profile = self.profile()                        # no model of its own: inswapper
        self.assertEqual(ff.strength(profile), 0.5)
        self.assertEqual(ff.strength(dict(profile, swap_strength=1.0)), 0.5)
        self.assertEqual(ff.strength(dict(profile, swap_strength=0.3)), 0.3)     # less is less
        self.assertEqual(ff.strength({'swap_strength': 'lots'}), 0.5)
        self.assertEqual(ff.strength(dict(profile, swap_strength=1.0,
                                          swap_model='hyperswap_1a_256')), 1.0)
        # The profile keeps the number it was given.
        self.assertEqual(profile['swap_strength'], ff.SWAP_STRENGTH)
        args = self.swap_args(swap_strength=1.0)
        self.assertEqual(args[args.index('--weight') + 1], '0.5')

    def test_swap_hands_facefusion_the_profile_strength(self):
        profile = dict(self.profile(), swap_strength=0.95, swap_model='hyperswap_1a_256')
        seen = []
        def spawn(args, **kw):
            seen.append(args)
            raise RuntimeError('stop here')
        with patch.object(ff, 'available', return_value=True), \
                patch.object(ff.studio_procs, 'spawn', side_effect=spawn):
            with self.assertRaisesRegex(RuntimeError, 'stop here'):
                ff.swap(PNG, profile)
        args = seen[0]
        self.assertEqual(args[args.index('--weight') + 1], '0.95')
        # The swap goes behind what is in front of the face (glasses frames
        # came back mottled without it, live 2026-09-29).
        at = args.index('--masks')
        self.assertEqual(args[at + 1:at + 4], ['box', 'occlusion', 'region'])

    def swap_args(self, **profile):
        seen = []
        def spawn(args, **kw):
            seen.append(args)
            raise RuntimeError('stop here')
        with patch.object(ff, 'available', return_value=True), \
                patch.object(ff.studio_procs, 'spawn', side_effect=spawn):
            with self.assertRaisesRegex(RuntimeError, 'stop here'):
                ff.swap(PNG, dict(self.profile(), **profile))
        return seen[0]

    def test_the_teeth_stay_the_pictures_own_and_the_weave_is_evened_out(self):
        # Live, 2026-09-29: inswapper's teeth were yellowed blocks, and pixel
        # boost's weave a comb of streaks down a cheek.
        args = self.swap_args()
        at = args.index('--regions')
        said = args[at + 1:args.index('--lens-line')]
        self.assertEqual(said, list(ff.SWAP_REGIONS))
        self.assertNotIn('mouth', said)
        for part in ('skin', 'nose', 'upper-lip', 'lower-lip', 'left-eye', 'right-eye',
                     'glasses'):
            self.assertIn(part, said)
        self.assertEqual(args[args.index('--deweave') + 1], str(ff.SWAP_DEWEAVE))
        self.assertEqual(ff.SWAP_DEWEAVE, 1.0)

    def test_behind_glasses_the_eyes_are_swapped_and_the_cheek_under_them_is_not(self):
        # Live, 2026-09-29: a pink patch with a hard edge under each eye.
        args = self.swap_args()
        self.assertEqual(args[args.index('--lens-line') + 1], str(ff.SWAP_LENS_LINE))
        self.assertIn('glasses', ff.SWAP_REGIONS)
        # Below the eyes (0.40 down the swap's crop), above the tip of the nose (0.56).
        self.assertTrue(0.44 < ff.SWAP_LENS_LINE < 0.56)

    def test_the_face_enhancer_runs_only_when_its_model_is_installed(self):
        models = Path(self.dir) / 'models'
        models.mkdir()
        with patch.object(ff, 'MODELS', models), patch.object(ff, 'SWAP_ENHANCE', 'gfpgan_1.4'):
            self.assertIsNone(ff.enhancer())            # nothing installed: no download
            self.assertNotIn('--enhance', self.swap_args())
            (models / 'gfpgan_1.4.onnx').write_bytes(b'x')
            self.assertIsNone(ff.enhancer())            # no hash file: FaceFusion would fetch
            (models / 'gfpgan_1.4.hash').write_text('5a6c6364')
            self.assertEqual(ff.enhancer(), 'gfpgan_1.4')
            args = self.swap_args()
            at = args.index('--enhance')
            self.assertEqual(args[at + 1:at + 4],
                             ['gfpgan_1.4', '--enhance-blend', str(ff.SWAP_ENHANCE_BLEND)])
            with patch.object(ff, 'SWAP_ENHANCE', ''):
                self.assertIsNone(ff.enhancer())
                self.assertNotIn('--enhance', self.swap_args())
        # Off as shipped: after the eye pass little of it shows, for 0.015-0.026.
        self.assertEqual(ff.SWAP_ENHANCE, '')
        self.assertTrue(0 < ff.SWAP_ENHANCE_BLEND <= 100)

    def test_a_failed_swap_says_why_in_the_workers_own_words(self):
        refused = ('[FACEFUSION.CORE] processing step 1 of 1\n'
                   'Traceback (most recent call last):\n'
                   '  File "tools\\facefusion_swap.py", line 1, in <module>\n'
                   '    main()\n'
                   'RuntimeError: FaceFusion\'s content check refused this picture, so no '
                   'face was swapped.\n')
        self.assertEqual(ff.failure(refused), 'FaceFusion\'s content check refused this '
                                              'picture, so no face was swapped.')
        unclear = ('  File "core.py", line 307, in process_step\n'
                   '    error_code = conditional_process()\n'
                   '                 ^^^^^^^^^^^^^^^^^^^^^\n'
                   'RuntimeError: The target face is ambiguous or missing. Open Fix a spot, '
                   'choose the identity, then use Choose face and click its face.\n\n')
        self.assertTrue(ff.failure(unclear).startswith('The target face is ambiguous'))
        self.assertNotIn('Traceback', ff.failure(unclear))
        self.assertNotIn('core.py', ff.failure(unclear))
        # No error of its own: the log's end. Nothing at all: said.
        self.assertEqual(ff.failure('[FACEFUSION.CORE] processing step 1 of 1\n'),
                         '[FACEFUSION.CORE] processing step 1 of 1')
        self.assertIn('without saying why', ff.failure('\n\n'))

    def fake_child(self, code=None, write=None):
        """A worker that is running (`code` None) or has ended with `code`,
        having written `write` to the output folder first."""
        made = []

        def spawn(args, **kw):
            folder = os.path.dirname(args[args.index('--output') + 1])
            for name, data in (write or {}).items():
                with open(os.path.join(folder, name), 'wb') as f:
                    f.write(data)
            kw['stdout'].write('RuntimeError: No face was swapped; no verified output '
                               'was saved.\n')
            kw['stdout'].flush()
            child = SimpleNamespace(proc=SimpleNamespace(poll=lambda: code, returncode=code),
                                    stop=lambda grace=0: made.append('stopped'),
                                    folder=folder)
            made.append(child)
            return child
        return spawn, made

    def test_a_cancelled_swap_says_so_and_leaves_nothing_behind(self):
        # Live, 2026-09-29: three cancels read "[WinError 32] The process
        # cannot access the file ... run.log" and left their folders, the
        # picture in each, in the temp folder.
        profile = self.profile()
        spawn, made = self.fake_child()
        with patch.object(ff, 'available', return_value=True), \
                patch.object(ff.studio_procs, 'spawn', side_effect=spawn):
            with self.assertRaisesRegex(RuntimeError, '^Face swap cancelled.$'):
                ff.swap(PNG, profile, stop=lambda: True)
        self.assertEqual(made, [])                     # cancelled before it was started
        spawn, made = self.fake_child()
        asked = iter([False, False, True])
        with patch.object(ff, 'available', return_value=True), \
                patch.object(ff.studio_procs, 'spawn', side_effect=spawn):
            with self.assertRaisesRegex(RuntimeError, '^Face swap cancelled.$'):
                ff.swap(PNG, profile, stop=lambda: next(asked))
        self.assertEqual(made[1:], ['stopped'])
        self.assertFalse(os.path.exists(made[0].folder))

    def test_a_folder_still_held_is_tried_again_and_never_the_error(self):
        folder = os.path.join(self.dir, 'held')
        os.makedirs(folder)
        calls = []

        def held(path, ignore_errors=False):
            calls.append(path)
            if len(calls) >= 3:
                os.rmdir(path)
        with patch.object(ff.shutil, 'rmtree', side_effect=held):
            self.assertTrue(ff._clear(folder, tries=5, wait=0))
        self.assertEqual(len(calls), 3)
        os.makedirs(folder)
        with patch.object(ff.shutil, 'rmtree', side_effect=lambda *a, **k: None):
            self.assertFalse(ff._clear(folder, tries=3, wait=0))

    def test_a_failed_swap_is_the_workers_error_and_a_finished_one_its_report(self):
        profile = self.profile()
        spawn, made = self.fake_child(code=1)
        with patch.object(ff, 'available', return_value=True), \
                patch.object(ff.studio_procs, 'spawn', side_effect=spawn):
            with self.assertRaisesRegex(RuntimeError, r"^Partner's face swap failed: No face "
                                                      r"was swapped; no verified output"):
                ff.swap(PNG, profile)
        self.assertFalse(os.path.exists(made[0].folder))
        report = {'outside_mask_changed_pixels': 0, 'masks': list(ff.SWAP_MASKS),
                  'sources': 'kept', 'target': 'x', 'output': 'y'}
        spawn, made = self.fake_child(code=0, write={
            'result.png': PNG, 'result.json': json.dumps(report).encode()})
        with patch.object(ff, 'available', return_value=True), \
                patch.object(ff.studio_procs, 'spawn', side_effect=spawn):
            data, said = ff.swap(PNG, profile)
        self.assertEqual(data, PNG)
        self.assertEqual(said, {'outside_mask_changed_pixels': 0,
                                'masks': list(ff.SWAP_MASKS), 'sources': 'kept'})
        self.assertFalse(os.path.exists(made[0].folder))


class TestKeptSources(unittest.TestCase):
    """tools/facefusion_swap.py keeps the person's averaged face between runs
    (it runs in FaceFusion's own Python; its helpers need only numpy)."""

    def setUp(self):
        try:
            import numpy
        except ImportError:
            self.skipTest('numpy is not installed for this Python')
        import collections
        import tempfile
        import tools.facefusion_swap as tool
        self.np, self.tool = numpy, tool
        self.dir = Path(tempfile.mkdtemp())
        self.Face = collections.namedtuple('Face', [
            'origin', 'bounding_box', 'score_set', 'landmark_set', 'angle', 'embedding',
            'embedding_norm', 'age', 'gender', 'race'])

    def face(self):
        np = self.np
        return self.Face(origin='detect', bounding_box=np.array([1.0, 2.0, 30.0, 40.0]),
                         score_set={'detector': 0.91, 'landmarker': 0.8},
                         landmark_set={'5': np.ones((5, 2)), '5/68': np.ones((5, 2)) * 2,
                                       '68': np.ones((68, 2)), '68/5': np.ones((68, 2)) * 3},
                         angle=0, embedding=np.arange(512, dtype=np.float64) / 7,
                         embedding_norm=np.arange(512, dtype=np.float64) / 900,
                         age=range(25, 33), gender='female', race='white')

    def photos(self, n=3):
        out = []
        for i in range(n):
            path = self.dir / ('photo%d.png' % i)
            path.write_bytes(PNG + bytes([i]))
            out.append(str(path))
        return out

    def test_the_face_comes_back_as_it_was_kept(self):
        refs = self.photos()
        path = self.dir / 'kept' / (self.tool.source_key(refs, '3.9.0') + '.npz')
        self.tool.keep_source(path, self.face(), refs[1], self.np)
        face, first = self.tool.kept_source(path, self.Face, self.np)
        self.assertEqual(first, refs[1])
        kept = self.face()
        for name in ('embedding', 'embedding_norm', 'bounding_box'):
            self.assertTrue(self.np.array_equal(getattr(face, name), getattr(kept, name)), name)
            self.assertEqual(getattr(face, name).dtype, getattr(kept, name).dtype, name)
        self.assertEqual(sorted(face.landmark_set), sorted(kept.landmark_set))
        self.assertTrue(self.np.array_equal(face.landmark_set['5/68'],
                                            kept.landmark_set['5/68']))
        self.assertEqual((face.origin, face.angle, face.gender, face.race, face.age),
                         ('detect', 0, 'female', 'white', range(25, 33)))
        self.assertEqual(face.score_set, kept.score_set)
        self.assertEqual([p.name for p in path.parent.iterdir()], [path.name])   # no .part left

    def test_another_set_of_photos_is_another_face(self):
        refs = self.photos()
        key = self.tool.source_key(refs, '3.9.0')
        self.assertEqual(key, self.tool.source_key(list(refs), '3.9.0'))
        self.assertNotEqual(key, self.tool.source_key(refs[:2], '3.9.0'))         # one fewer
        self.assertNotEqual(key, self.tool.source_key(refs[::-1], '3.9.0'))       # the first differs
        self.assertNotEqual(key, self.tool.source_key(refs, '3.9.1'))             # FaceFusion's
        Path(refs[0]).write_bytes(PNG + b'changed')
        self.assertNotEqual(key, self.tool.source_key(refs, '3.9.0'))             # a photo replaced

    def test_a_kept_face_whose_photo_is_gone_is_not_used(self):
        refs = self.photos()
        path = self.dir / 'kept' / 'a.npz'
        self.tool.keep_source(path, self.face(), refs[0], self.np)
        os.remove(refs[0])
        self.assertIsNone(self.tool.kept_source(path, self.Face, self.np))
        path.write_bytes(b'not a kept face')
        with self.assertRaises(Exception):
            self.tool.kept_source(path, self.Face, self.np)   # main() reads the photos then

    def test_pixel_boosts_weave_is_evened_out_and_nothing_coarser(self):
        np, total = self.np, 6
        ys, xs = np.mgrid[0:96, 0:96]
        # A face that changes slowly, and on it what 6 x 6 swaps that are not
        # quite alike leave: each its own offset, every sixth pixel.
        face = np.repeat((100 + ys * 0.5 + xs * 0.25)[..., None], 3, axis=2)
        offsets = np.random.RandomState(7).uniform(-12, 12, (total, total))
        offsets -= offsets.mean()
        woven = face + offsets[ys % total, xs % total][..., None]
        self.assertGreater(np.abs(woven - face).max(), 8)
        evened = self.tool.even(woven, total, 1.0, np)
        inner = (slice(total, -total), slice(total, -total))
        # The box is half a pixel off centre: a slope of 0.75 a pixel at most.
        self.assertLess(np.abs(evened - face)[inner].max(), 0.5)
        self.assertEqual(evened.shape, woven.shape)
        half = self.tool.even(woven, total, 0.5, np)
        self.assertTrue(np.allclose(half, (woven + evened) / 2))
        # Nothing asked, or a face swapped in one piece: as it was.
        self.assertIs(self.tool.even(woven, total, 0.0, np), woven)
        self.assertIs(self.tool.even(woven, 1, 1.0, np), woven)

    def test_under_the_lens_line_what_is_behind_glasses_is_left(self):
        np = self.np
        mask = np.ones((100, 100), dtype=np.float32)
        glasses = np.zeros((100, 100), dtype=np.float32)
        glasses[30:60, 10:90] = 1.0               # frames and lenses, eyes and cheek
        left = self.tool.under_lenses(mask, glasses, 0.47, np)
        self.assertEqual(left.shape, mask.shape)
        self.assertTrue((left[30:44, 10:90] == 1.0).all())      # the eyes: swapped
        self.assertTrue((left[51:60, 10:90] == 0.0).all())      # the cheek under them: left
        self.assertTrue((left[60:, :] == 1.0).all())            # below the glasses: swapped
        self.assertTrue((left[:, :10] == 1.0).all())            # beside them
        band = left[44:51, 50]
        self.assertTrue((np.diff(band) <= 0).all() and band[0] > band[-1])   # a soft edge
        # Glasses half seen are half left; no line asked is no change.
        self.assertAlmostEqual(float(self.tool.under_lenses(
            mask, glasses * 0.5, 0.47, np)[55, 50]), 0.5)
        self.assertIs(self.tool.under_lenses(mask, glasses, 0.0, np), mask)

    def test_the_enhancers_change_comes_through_the_swaps_own_mask(self):
        np = self.np
        frame = np.full((4, 4, 3), 100, dtype=np.uint8)
        enhanced = np.full((4, 4, 3), 200, dtype=np.uint8)
        enhanced[0, 0] = 0
        soft = np.zeros((4, 4), dtype=np.float32)
        soft[1, 1], soft[2, 2], soft[0, 0] = 1.0, 0.5, 1.0
        out = self.tool.through(frame, enhanced, soft, np)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out[1, 1].tolist(), [200, 200, 200])     # inside the swap: enhanced
        self.assertEqual(out[2, 2].tolist(), [150, 150, 150])     # its soft edge: half
        self.assertEqual(out[3, 3].tolist(), [100, 100, 100])     # outside: the swap's own
        self.assertEqual(out[0, 0].tolist(), [0, 0, 0])           # darker is taken too

    def test_only_the_newest_sets_are_kept(self):
        refs = self.photos()
        folder = self.dir / 'kept'
        for i in range(self.tool.SOURCES_KEPT + 3):
            path = folder / ('%02d.npz' % i)
            self.tool.keep_source(path, self.face(), refs[0], self.np)
            os.utime(path, (1000 + i, 1000 + i))
        names = sorted(p.name for p in folder.iterdir())
        self.assertEqual(len(names), self.tool.SOURCES_KEPT)
        self.assertNotIn('00.npz', names)


if __name__ == '__main__':
    unittest.main()
