"""The person's own face in photos (`apps.image_studio.faces`,
`tools/identity_faces.py`): who they are among a crowd, the crops made on
import, Build LoRA's head squares round them - without InsightFace, ComfyUI
or Tk."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import apps.image_studio.faces as faces
import apps.image_studio.lora_train as lt

ROOT = os.path.dirname(os.path.dirname(__file__))


def load(name):
    path = os.path.join(ROOT, 'tools', name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emb(*xs):
    """A unit embedding pointing mostly along the given axes."""
    v = [0.0] * 8
    for i, x in enumerate(xs):
        v[i] = x
    w = load('identity_faces')
    return w.unit(v)


class WhoTests(unittest.TestCase):
    """identity_faces' pure helpers: the person's mean and their face."""

    def setUp(self):
        self.w = load('identity_faces')

    def test_a_crowd_does_not_pull_the_person_away(self):
        her = [emb(1, 0.1), emb(1, -0.1), emb(1, 0.05, 0.1)]
        crowd = [emb(0, 1), emb(0, 0, 1), emb(0, 0, 0, 1), emb(0, 0, 0, 0, 1)]
        refs = [[her[0]], [her[1]]]                    # photos of her alone
        groups = [[crowd[0], her[2], crowd[1]], crowd[2:] + [emb(0.9, 0.2)]] * 5
        c = self.w.person(refs, groups)
        self.assertGreater(self.w.dot(c, emb(1)), 0.99)
        k, sim = self.w.theirs(groups[0], c)
        self.assertEqual(k, 1)
        self.assertGreater(sim, self.w.SAME)

    def test_with_no_references_photos_of_one_face_seed_it(self):
        c = self.w.person([], [[emb(1, 0.1)], [emb(0, 1), emb(1)], [emb(1, -0.1)]])
        self.assertGreater(self.w.dot(c, emb(1)), 0.99)

    def test_nobody_alone_means_nobody_known(self):
        self.assertIsNone(self.w.person([], [[emb(1), emb(0, 1)], [emb(0, 1), emb(1)]]))

    def test_a_photo_without_them_has_no_face_of_theirs(self):
        k, sim = self.w.theirs([emb(0, 1), emb(0, 0, 1)], emb(1))
        self.assertIsNone(k)
        self.assertLess(sim, self.w.SAME)
        self.assertEqual(self.w.theirs([], emb(1)), (None, 0.0))

    def test_the_square_is_the_cascades_size_round_the_face(self):
        x, y, w, h = self.w.haar_square((100, 200, 200, 330))
        self.assertEqual((w, h), (127, 127))            # 1.27 face widths
        self.assertAlmostEqual(x + w / 2, 150, delta=1)
        self.assertAlmostEqual(y + h / 2, 265, delta=1)

    def test_the_crop_keeps_head_and_shoulders_inside_the_photo(self):
        # a 100-px face in a 6720 x 4480 group photo
        left, top, right, bottom = self.w.keep_box(6720, 4480, (3000, 1000, 3100, 1130))
        self.assertEqual((right - left, bottom - top), (400, 500))
        self.assertEqual((left + right) / 2, 3050)
        self.assertEqual(top, 900)                      # a face width above for the hair
        # at the photo's corner: slid inside, not padded
        self.assertEqual(self.w.keep_box(1000, 1000, (0, 0, 100, 120))[:2], (0, 0))
        self.assertEqual(self.w.keep_box(1000, 1000, (950, 950, 1000, 1000))[2:], (1000, 1000))
        # a face filling the photo: kept whole
        self.assertIsNone(self.w.keep_box(500, 600, (100, 100, 400, 450)))


class AdapterTests(unittest.TestCase):
    def fake(self, d, body):
        script = Path(d, 'fake.py')
        script.write_text(textwrap.dedent(body))
        return mock.patch.multiple(faces, SCRIPT=str(script), python_exe=lambda _c: sys.executable)

    def test_the_check_names_what_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn('venv', faces.problem(d))
            exe = Path(faces.python_exe(d))
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b'')
            self.assertIn('antelopev2', faces.problem(d))
            model = Path(faces.model_root(d), 'models', 'antelopev2', 'glintr100.onnx')
            model.parent.mkdir(parents=True)
            model.write_bytes(b'')
            self.assertIsNone(faces.problem(d))

    def test_lines(self):
        self.assertEqual(faces.parse('FACE 3 10\n'), ('face', (3, 10)))
        self.assertEqual(faces.parse('DONE'), ('done', ''))
        self.assertEqual(faces.parse('ERROR antelopev2 is not at X'),
                         ('error', 'antelopev2 is not at X'))
        self.assertEqual(faces.parse('warming up'), ('note', 'warming up'))

    def test_a_job_reports_progress_and_returns_the_answer(self):
        with tempfile.TemporaryDirectory() as d:
            with self.fake(d, '''
                    import json, sys
                    job = json.load(open(sys.argv[1]))
                    assert job['mode'] == 'find' and job['photos'] == ['a.jpg', 'b.jpg']
                    for i in (1, 2): print('FACE', i, 2, flush=True)
                    json.dump({'a.jpg': {'box': [1, 2, 3, 3]}, 'b.jpg': {'box': None,
                               'why': 'not found'}}, open(job['result'], 'w'))
                    print('DONE')
                    '''):
                seen = []
                got = faces.Job('find', [], ['a.jpg', 'b.jpg'], comfy=d).run(
                    lambda k, v: seen.append((k, v)))
        self.assertEqual(seen, [('face', (1, 2)), ('face', (2, 2))])
        self.assertEqual(got['a.jpg']['box'], [1, 2, 3, 3])
        self.assertIsNone(got['b.jpg']['box'])

    def test_a_failed_job_says_why(self):
        with tempfile.TemporaryDirectory() as d:
            with self.fake(d, "print('ERROR antelopev2 is not at X'); raise SystemExit(1)"):
                with self.assertRaisesRegex(RuntimeError, 'antelopev2'):
                    faces.Job('find', [], ['a.jpg'], comfy=d).run()
            with self.fake(d, "print('Traceback: boom')"):
                with self.assertRaisesRegex(RuntimeError, 'boom'):
                    faces.Job('find', [], ['a.jpg'], comfy=d).run()


class ImportTests(unittest.TestCase):
    def answer(self, paths, scratch):
        return {paths[0]: {'box': [1, 1, 5, 5], 'crop': os.path.join(scratch, 'c0.jpg')},
                paths[1]: {'box': [1, 1, 5, 5], 'crop': None},        # fills the photo
                paths[2]: {'box': None, 'why': 'not found'},
                paths[3]: {'box': None, 'why': 'unreadable: cannot identify'}}

    def test_each_photo_is_imported_as_its_crop_when_it_has_one(self):
        paths = ['wedding.jpg', 'selfie.jpg', 'bridesmaids.jpg', 'broken.jpg']
        with mock.patch.object(faces, 'problem', return_value=None), \
                mock.patch.object(faces.Job, 'run', autospec=True,
                                  side_effect=lambda job, on: self.answer(paths, 'S')) as run:
            got, note = faces.crop_for_import(paths, ['ref.png'], 'cache.json', 'S', 'Partner')
        job = run.call_args[0][0].job
        self.assertEqual((job['mode'], job['refs'], job['out'], job['cache']),
                         ('crop', ['ref.png'], 'S', 'cache.json'))
        self.assertEqual(got, [os.path.join('S', 'c0.jpg'), 'selfie.jpg', 'bridesmaids.jpg',
                               'broken.jpg'])
        self.assertEqual(note, '1 cut to Partner; 1 already close on them; '
                               '2 kept whole (not found: 1, unreadable: 1)')

    def test_without_the_finder_the_photos_go_in_as_they_are_and_it_says_why(self):
        with mock.patch.object(faces, 'problem', return_value="ComfyUI's venv is not at X."):
            got, note = faces.crop_for_import(['a.jpg'], [], 'c', 'S', 'Partner')
        self.assertEqual(got, ['a.jpg'])
        self.assertIn("Not cut to Partner: ComfyUI's venv", note)
        with mock.patch.object(faces, 'problem', return_value=None), \
                mock.patch.object(faces.Job, 'run', side_effect=RuntimeError('boom')):
            got, note = faces.crop_for_import(['a.jpg'], [], 'c', 'S', 'Partner')
        self.assertEqual((got, note), (['a.jpg'], 'Not cut to Partner: boom'))

    def test_the_editors_import_cuts_photos_from_disk_but_not_ones_made_here(self):
        from apps.image_studio.ui import RecordEditor
        import apps.image_studio.imagegen as ig
        with tempfile.TemporaryDirectory() as d:
            lib = ig.Library(os.path.join(d, 'lib'))
            src = Path(d, 'wedding.jpg')
            src.write_bytes(b'\xff\xd8\xff\xe0 whole wedding photo')
            crop = Path(d, 'crop.jpg')
            crop.write_bytes(b'\xff\xd8\xff\xe0 her head and shoulders')
            editor = RecordEditor.__new__(RecordEditor)
            editor.current, editor.records = 0, [{'id': 'partner', 'name': 'Partner'}]
            editor.status, editor._draw_paths = mock.Mock(), mock.Mock()
            editor.win = mock.Mock(winfo_exists=lambda: True)
            editor.owner = mock.Mock(studio=mock.Mock(lib=lib),
                                     _post=lambda kind, fn: fn(),
                                     host=mock.Mock(_spawn=lambda _id, fn: fn()))
            pics = {'paths': ['ref.png'], 'sel': set(), 'grid': mock.Mock(winfo_exists=lambda: True)}
            with mock.patch.object(faces, 'crop_for_import',
                                   return_value=([str(crop)], '1 cut to Partner')) as cut:
                editor._import_paths(pics, [str(src)])
                args = cut.call_args[0]
                self.assertEqual((args[0], args[1], args[2]),
                                 ([str(src)], ['ref.png'],
                                  os.path.join(lib.root, faces.CACHE)))
                self.assertEqual(len(pics['paths']), 2)
                with open(pics['paths'][1], 'rb') as f:
                    self.assertIn(b'head and shoulders', f.read())
                self.assertIn('1 cut to Partner', editor.status.call_args[0][0])
                cut.reset_mock()
                editor._import_paths(pics, [str(src)], crop=False)
                cut.assert_not_called()
            self.assertEqual(len(pics['paths']), 3)


class BuildFindsTheFaceTests(unittest.TestCase):
    def spec(self, d, photos):
        return {'name': 'n', 'person': 'Partner', 'trigger': 't', 'steps': 3, 'photos': photos,
                'work': os.path.join(d, 'work'), 'toolkit': d, 'base': d,
                'vae': os.path.join(d, 'ae.safetensors'), 'resolution': 512,
                'text_encoder': d, 'lora_out': os.path.join(d, 'out.safetensors'),
                'face_cache': os.path.join(d, 'face-cache.json')}

    def fake_trainer(self, d):
        out = os.path.join(d, 'out.safetensors')
        script = Path(d, 'fake.py')
        script.write_text(textwrap.dedent('''
            import json, sys
            spec = json.load(open(sys.argv[1]))
            print('SPECFACES', json.dumps(spec.get('faces')), flush=True)
            open(%r, 'wb').write(b'lora')
            print('DONE ' + %r)
            ''' % (out, out)))
        return mock.patch.multiple(lt, SCRIPT=str(script), python_exe=lambda _t: sys.executable)

    def test_the_persons_own_face_goes_into_the_spec_before_training(self):
        with tempfile.TemporaryDirectory() as d:
            spec = self.spec(d, ['a.jpg', 'b.jpg'])
            found = {'a.jpg': {'box': [10, 20, 30, 30]}, 'b.jpg': {'box': None, 'why': 'not found'}}

            def run(job, on):
                self.assertEqual((job.job['mode'], job.job['refs'], job.job['cache']),
                                 ('find', [], spec['face_cache']))
                on('face', (2, 2))
                return found
            seen = []
            with self.fake_trainer(d), mock.patch.object(faces, 'problem', return_value=None), \
                    mock.patch.object(faces.Job, 'run', autospec=True, side_effect=run):
                lt.Build(spec).run(lambda k, v: seen.append((k, v)))
            self.assertEqual(spec['faces'], {'a.jpg': [10, 20, 30, 30], 'b.jpg': None})
            self.assertIn(('finding', (2, 2)), seen)
            with open(os.path.join(d, 'work', 'spec.json')) as f:
                self.assertEqual(json.load(f)['faces'], spec['faces'])

    def test_without_the_finder_it_says_so_and_trains_on_the_biggest_face(self):
        with tempfile.TemporaryDirectory() as d:
            spec = self.spec(d, ['a.jpg'])
            seen = []
            with self.fake_trainer(d), \
                    mock.patch.object(faces, 'problem', return_value='InsightFace is missing.'):
                lt.Build(spec).run(lambda k, v: seen.append((k, v)))
            self.assertNotIn('faces', spec)
            self.assertIn(('finder', 'InsightFace is missing.'), seen)

    def test_a_plan_without_the_cache_does_not_look(self):
        with tempfile.TemporaryDirectory() as d:
            spec = dict(self.spec(d, ['a.jpg']), face_cache=None)
            with self.fake_trainer(d), mock.patch.object(faces, 'problem') as problem:
                lt.Build(spec).run()
            problem.assert_not_called()

    def test_stopping_while_it_looks_stops_the_build(self):
        with tempfile.TemporaryDirectory() as d:
            spec = self.spec(d, ['a.jpg'])
            build = lt.Build(spec)

            def run(job, on):
                build.stop()
                raise RuntimeError('Stopped.')
            with self.fake_trainer(d), mock.patch.object(faces, 'problem', return_value=None), \
                    mock.patch.object(faces.Job, 'run', autospec=True, side_effect=run):
                with self.assertRaisesRegex(RuntimeError, 'Stopped'):
                    build.run()
            self.assertFalse(os.path.exists(os.path.join(d, 'work', 'spec.json')))

    def test_left_lines(self):
        self.assertEqual(lt.parse('LEFT 4 135'), ('left', (4, 135)))


class TrainerUsesTheFaceTests(unittest.TestCase):
    """tools/train_identity_lora.py with the spec's `faces`: the box is the
    person's, and a photo without them is left out."""

    def test_the_spec_faces_are_cut_round_and_a_photo_without_them_left_out(self):
        sys.path.insert(0, os.path.dirname(__file__))
        import test_lora_train as tlt
        case = tlt.WorkerTests()
        worker = case.worker()
        with tempfile.TemporaryDirectory() as d:
            good = tlt.photos(d, 4)
            work = Path(d, 'work')
            work.mkdir()
            spec = {'photos': good, 'caption': 'a photo of t', 'min_photos': 3, 'steps': 3,
                    'name': 'n', 'work': str(work), 'toolkit': d, 'person': 'Partner',
                    'lora_out': os.path.join(d, 'out.safetensors'),
                    'faces': {good[0]: [100, 100, 50, 50], good[1]: [600, 300, 80, 80],
                              good[2]: None}}         # good[3]: not looked for
            (work / 'spec.json').write_text(json.dumps(spec))
            said = []

            def train(_spec, w):
                final = w / 'output' / 'n' / 'n.safetensors'
                final.parent.mkdir(parents=True)
                final.write_bytes(b'lora')
                return 0
            with case.fake_pil(), mock.patch.object(worker, 'say', said.append), \
                    mock.patch.object(worker, 'find_face', return_value=(1, 2, 30, 30)) as biggest, \
                    mock.patch.object(worker, 'train', side_effect=train), \
                    mock.patch.object(sys, 'argv', ['x', str(work / 'spec.json')]):
                self.assertEqual(worker.main(), 0)
        self.assertIn('LEFT 1 4', said)
        self.assertIn('KEPT 3 4', said)
        self.assertIn('FACES 3 3', said)
        self.assertTrue(any('Partner not found, left out' in s and 'p02.png' in s for s in said))
        square = worker.head_square
        self.assertEqual(tlt.WorkerTests.crops, [
            ('p00.png', square(1000, 800, (100, 100, 50, 50))),
            ('p01.png', square(1000, 800, (600, 300, 80, 80))),
            ('p03.png', square(1000, 800, (1, 2, 30, 30)))])
        biggest.assert_called_once()      # only the photo it had no answer for

    def test_too_few_with_them_names_the_photos_they_are_not_in(self):
        sys.path.insert(0, os.path.dirname(__file__))
        import test_lora_train as tlt
        case = tlt.WorkerTests()
        worker = case.worker()
        with tempfile.TemporaryDirectory() as d:
            good = tlt.photos(d, 3)
            work = Path(d, 'work')
            work.mkdir()
            spec = {'photos': good, 'caption': 'c', 'min_photos': 3, 'steps': 3, 'name': 'n',
                    'work': str(work), 'toolkit': d, 'person': 'Partner',
                    'lora_out': os.path.join(d, 'o.safetensors'),
                    'faces': {good[0]: [1, 1, 9, 9], good[1]: None, good[2]: None}}
            (work / 'spec.json').write_text(json.dumps(spec))
            said = []
            with case.fake_pil(), mock.patch.object(worker, 'say', said.append), \
                    mock.patch.object(worker, 'train') as trained, \
                    mock.patch.object(sys, 'argv', ['x', str(work / 'spec.json')]):
                self.assertEqual(worker.main(), 1)
            trained.assert_not_called()
        error = [s for s in said if s.startswith('ERROR ')][0]
        self.assertIn('Only 1 of 3 photos could be used', error)
        self.assertIn('Partner not found in: p01.png, p02.png', error)


if __name__ == '__main__':
    unittest.main()
