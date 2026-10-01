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


class MeasureHelperTests(unittest.TestCase):
    def setUp(self):
        self.w = load('identity_faces')

    def test_share_inside_and_hash_distance(self):
        self.assertEqual(self.w.share_inside((0, 0, 100, 100), (50, 50, 150, 150)), 0.25)
        self.assertEqual(self.w.share_inside((0, 0, 100, 100), (200, 0, 300, 50)), 0)
        self.assertEqual(self.w.hamming('ff00', 'ff01'), 1)
        self.assertEqual(self.w.hamming('0' * 16, 'f' * 16), 64)

    def test_what_made_a_picture_is_read_from_its_graph(self):
        cut = json.dumps({'1': {'class_type': 'LoadImage'}, '5': {'class_type': 'SAM3_Detect'},
                          '6': {'class_type': 'EmptyImage'},
                          '7': {'class_type': 'ImageCompositeMasked'}})
        edit = json.dumps({'1': {'class_type': 'ReferenceLatent'}})
        self.assertEqual(self.w.kind_of(cut), 'cutout')
        self.assertEqual(self.w.kind_of(edit), 'generated')
        self.assertEqual(self.w.kind_of(None), 'photo')
        self.assertEqual(self.w.kind_of('not json'), 'photo')


def good(**kw):
    """A worker `rate` answer for a clear, close, real photo of the person."""
    m = {'box': [1, 1, 9, 9], 'sim': 0.8, 'side': 700, 'sharp': 300.0, 'luma': 0.5,
         'clip': 0.0, 'others': [], 'kind': 'photo', 'yaw': 0.0}
    m.update(kw)
    return m


class ScoreTests(unittest.TestCase):
    def test_a_clear_close_real_photo_scores_near_full(self):
        r = faces.score(good())
        self.assertGreaterEqual(r['score'], 95)
        self.assertEqual(r['flags'], [])
        self.assertEqual(set(r['parts']), set(faces.WEIGHTS))

    def test_a_photo_they_are_not_found_in_is_left_out(self):
        r = faces.score({'box': None, 'why': 'not found'})
        self.assertEqual(r['score'], 0)
        self.assertIn('not found: left out of a LoRA', r['flags'])

    def test_each_fault_costs_and_is_named(self):
        base = faces.score(good())['score']
        for fault, words in ((dict(side=200), 'enlarged 2.6x'), (dict(sharp=20.0), 'soft face'),
                             (dict(others=[0.9]), '1 other face in the square'),
                             (dict(luma=0.08), 'face too dark'),
                             (dict(kind='cutout'), 'cut out on white'),
                             (dict(kind='generated'), 'made here')):
            r = faces.score(good(**fault))
            self.assertLess(r['score'], base, fault)
            self.assertTrue(any(words in f for f in r['flags']), (fault, r['flags']))

    def test_twins_are_the_same_face_or_a_burst(self):
        m = {'near': [['same', 0.97, 30], ['burst', 0.91, 9], ['moment', 0.91, 25],
                      ['day', 0.88, 5]]}
        self.assertEqual(faces.twins(m), ['same', 'burst'])

    def test_of_each_set_of_twins_the_best_stays_and_the_primary_always(self):
        found = {'primary': good(sharp=40.0, near=[['sharp', 0.97, 3]]),
                 'sharp': good(near=[['primary', 0.97, 3], ['soft', 0.96, 4]]),
                 'soft': good(sharp=30.0, near=[['sharp', 0.96, 4]]),
                 'burst': good(side=400, near=[['other', 0.91, 8]]),
                 'other': good(near=[['burst', 0.91, 8]]),
                 'gone': {'box': None, 'why': 'not found'}}
        paths = list(found)
        rated = faces.rate(found, paths)
        self.assertIsNone(rated['primary']['dup_of'])           # the Primary stays
        self.assertEqual(rated['sharp']['dup_of'], 'primary')
        self.assertEqual(rated['soft']['dup_of'], None)          # its twin went, so it stays
        self.assertEqual(rated['burst']['dup_of'], 'other')     # the smaller face goes
        self.assertIsNone(rated['other']['dup_of'])
        self.assertIsNone(rated['gone']['dup_of'])
        self.assertTrue(rated['other']['top'])
        self.assertFalse(rated['sharp']['top'] or rated['gone']['top'])
        self.assertIn('2 duplicates', faces.summary(rated))
        self.assertIn('1 left out', faces.summary(rated))

    def test_the_top_set_spreads_over_head_angles(self):
        found = {'f%d' % i: good(sim=0.80 + i / 1000.0) for i in range(40)}
        found['left'] = good(sim=0.70, yaw=-30.0)
        with mock.patch.object(faces, 'TOP', 5):
            rated = faces.rate(found, list(found))
        self.assertTrue(rated['left']['top'])
        self.assertEqual(rated['left']['angle'], 'three-quarter left')
        self.assertEqual(sum(r['top'] for r in rated.values()), 5)

    def test_details_in_words(self):
        rated = faces.rate({'a': good(side=200)}, ['a'])['a']
        text = faces.detail(rated)
        self.assertTrue(text.startswith('%d ★ · front · likeness' % rated['score']))
        self.assertIn('enlarged', text)
        self.assertEqual(faces.detail(faces.rate({'b': {'box': None, 'why': 'no face'}},
                                                 ['b'])['b']),
                         '0 · no face: left out of a LoRA')


class EditorRatingTests(unittest.TestCase):
    """RecordEditor's Rate photos and Remove duplicates, without Tk."""

    def editor(self, d, paths):
        from apps.image_studio.ui import RecordEditor
        editor = RecordEditor.__new__(RecordEditor)
        editor.status, editor._draw_paths = mock.Mock(), mock.Mock()
        editor.win = mock.Mock(winfo_exists=lambda: True)
        editor.owner = mock.Mock(studio=mock.Mock(lib=mock.Mock(root=d)),
                                 _post=lambda kind, fn: fn(),
                                 host=mock.Mock(_spawn=lambda _id, fn: fn()))
        pics = {'paths': paths, 'sel': {1}, 'grid': mock.Mock(winfo_exists=lambda: True)}
        return editor, pics

    def test_rating_marks_the_tiles_and_says_how_it_went(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [str(p) for p in (Path(d, 'a.png'), Path(d, 'b.png'))]
            for p in paths:
                Path(p).write_bytes(b'x')
            editor, pics = self.editor(d, paths)
            found = {paths[0]: good(), paths[1]: good(near=[[paths[0], 0.99, 2]])}
            then = mock.Mock()
            with mock.patch.object(faces, 'problem', return_value=None), \
                    mock.patch.object(faces.Job, 'run', autospec=True,
                                      side_effect=lambda job, on: found) as run:
                editor._rate_paths(pics, then=then)
            job = run.call_args[0][0].job
            self.assertEqual((job['mode'], job['photos'], job['cache']),
                             ('rate', paths, os.path.join(d, faces.CACHE)))
            self.assertEqual(pics['ratings'][paths[1]]['dup_of'], paths[0])
            editor._draw_paths.assert_called_with(pics)
            self.assertIn('Rated 2 photos', editor.status.call_args[0][0])
            then.assert_called_once()
            self.assertFalse(pics['rating'])

    def test_without_the_finder_it_says_why(self):
        editor, pics = self.editor('D', [__file__])
        with mock.patch.object(faces, 'problem', return_value='InsightFace is missing.'):
            editor._rate_paths(pics)
        self.assertIn('InsightFace is missing', editor.status.call_args[0][0])
        self.assertNotIn('rating', pics)

    def test_remove_duplicates_takes_out_the_twins_after_asking(self):
        editor, pics = self.editor('D', [__file__, 'b', 'c'])
        pics['ratings'] = {__file__: {'dup_of': None}, 'b': {'dup_of': __file__},
                           'c': {'dup_of': None}}
        with mock.patch('apps.image_studio.ui.messagebox.askyesno', return_value=False):
            editor._remove_duplicates(pics)
        self.assertEqual(pics['paths'], [__file__, 'b', 'c'])
        with mock.patch('apps.image_studio.ui.messagebox.askyesno', return_value=True) as ask:
            editor._remove_duplicates(pics)
        self.assertIn('Remove 1 duplicate photo ', ask.call_args[0][1])
        self.assertEqual(pics['paths'], [__file__, 'c'])
        self.assertEqual(pics['sel'], set())
        self.assertIn('Removed 1 duplicate.', editor.status.call_args[0][0])

    def test_remove_duplicates_rates_first_when_photos_are_not_rated(self):
        editor, pics = self.editor('D', [__file__])
        editor._rate_paths = mock.Mock()
        editor._remove_duplicates(pics)
        editor._rate_paths.assert_called_once()
        self.assertIsNotNone(editor._rate_paths.call_args[1]['then'])


sys.path.insert(0, os.path.dirname(__file__))
import test_imagegen as ti      # the module, so its TestCases are not collected here


class RatingsInTheEditor(unittest.TestCase):
    """The real identity editor (Tk, fake ComfyUI): the tiles' badges."""
    setUpClass = ti.TestImageStudioTab.__dict__['setUpClass']
    tearDownClass = ti.TestImageStudioTab.__dict__['tearDownClass']
    tab = ti.TestImageStudioTab.tab

    def texts(self, widget):
        for w in widget.winfo_children():
            try:
                yield w.cget('text')
            except Exception:
                pass
            yield from self.texts(w)

    def test_each_rated_photo_shows_its_score(self):
        _, ui = self.tab()
        paths = []
        for i in range(4):
            p = os.path.join(self.dir, 'rated%d.png' % i)
            with open(p, 'wb') as f:
                f.write(ti.PNG)
            paths.append(p)
        ui.studio.lib.save('identities', [{'id': 'p', 'name': 'P', 'references': paths}])
        ed = ui.edit_identities()
        self.addCleanup(ed.win.destroy)
        self.app.update()
        pics = ed.widgets['references'][1]
        pics['ratings'] = {paths[0]: {'score': 91, 'dup_of': None, 'top': True},
                           paths[1]: {'score': 64, 'dup_of': None, 'top': False},
                           paths[2]: {'score': 80, 'dup_of': paths[0], 'top': False},
                           paths[3]: {'score': 0, 'dup_of': None, 'top': False}}
        ed._draw_paths(pics)
        self.app.update()
        shown = list(self.texts(pics['grid']))
        for text in ('★ 91', '64', 'duplicate · 80', '0 · left out'):
            self.assertIn(text, shown)
        names = [t for t in self.texts(ed.form) if t in ('Rate photos', 'Remove duplicates…')]
        self.assertEqual(len(names), 2)


if __name__ == '__main__':
    unittest.main()
