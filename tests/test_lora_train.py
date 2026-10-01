"""Build LoRA: planning, the toolkit check, the run's progress, and the editor
button - without ai-toolkit, a GPU or Tk."""
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import apps.image_studio.imagegen as ig
import apps.image_studio.lora_train as lt
from apps.image_studio.ui import ImageStudio, RecordEditor


def photos(folder, n):
    out = []
    for i in range(n):
        p = Path(folder) / ('p%02d.png' % i)
        p.write_bytes(b'x')
        out.append(str(p))
    return out


class PlanTests(unittest.TestCase):
    def test_fewer_than_twenty_usable_photos_is_refused_in_words(self):
        with tempfile.TemporaryDirectory() as d:
            refs = photos(d, 19) + [os.path.join(d, 'gone.png')] + photos(d, 19)[:1]
            with self.assertRaisesRegex(ValueError, '19 usable photos.*at least 20'):
                lt.plan({'id': 'partner', 'name': 'Partner', 'references': refs}, d)

    def test_a_plan_names_its_files_and_keeps_the_persons_trigger(self):
        with tempfile.TemporaryDirectory() as d:
            refs = photos(d, 20)
            spec = lt.plan({'id': 'partner', 'name': 'Partner', 'references': refs,
                            'trigger': 'parperson'}, d, toolkit=d, when=0)
            self.assertEqual(spec['trigger'], 'parperson')
            self.assertEqual(spec['photos'], refs)
            self.assertEqual(spec['min_photos'], lt.MIN_PHOTOS)
            self.assertTrue(spec['lora_out'].startswith(d))
            self.assertTrue(spec['lora_out'].endswith('.safetensors'))
            self.assertIn('partner_head_klein_', spec['name'])
            self.assertEqual(os.path.dirname(spec['work']), os.path.join(d, 'output'))
            self.assertEqual((spec['steps'], spec['resolution']), (lt.STEPS, lt.RESOLUTION))
            self.assertEqual((spec['text_encoder'], spec['vae']),
                             (lt.text_encoder(d), lt.vae(d)))
            fresh = lt.plan({'id': 'g', 'name': 'Sitter T.', 'references': refs}, d, toolkit=d)
            self.assertEqual(fresh['trigger'], 'sitperson')

    def test_the_job_trains_klein_on_the_face_crops(self):
        spec = {'name': 'n', 'work': 'W', 'trigger': 't', 'steps': 750, 'base': 'B',
                'vae': 'V', 'resolution': 512}
        proc = lt.config(spec)['config']['process'][0]
        self.assertEqual((proc['model']['arch'], proc['model']['name_or_path'],
                          proc['model']['vae_path']), ('flux2_klein_4b', 'B', 'V'))
        self.assertFalse(proc['model']['quantize'])
        self.assertEqual(proc['train']['steps'], 750)
        self.assertEqual(proc['datasets'][0]['resolution'], [512])
        self.assertEqual(proc['datasets'][0]['folder_path'], os.path.join('W', 'dataset'))
        self.assertEqual(proc['datasets'][0]['num_workers'], 0)
        self.assertTrue(proc['train']['disable_sampling'])

    def test_the_toolkit_check_names_what_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn('not installed', lt.problem(d))
            exe = Path(lt.python_exe(d))
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b'')
            Path(d, 'run.py').write_text('')
            self.assertIn('flux-2-klein-base-4b', lt.problem(d))
            self.assertIn('prepare_klein_lora', lt.problem(d))
            for part in lt.parts(d)[:-1]:
                Path(part).parent.mkdir(parents=True, exist_ok=True)
                Path(part).touch()
            self.assertIn('ae.safetensors', lt.problem(d))
            Path(lt.vae(d)).parent.mkdir(parents=True)
            Path(lt.vae(d)).touch()
            self.assertIsNone(lt.problem(d))

    def test_lines_and_time_left(self):
        self.assertEqual(lt.parse('STEP 12 2000\n'), ('step', (12, 2000)))
        self.assertEqual(lt.parse('KEPT 18 20'), ('kept', (18, 20)))
        self.assertEqual(lt.parse('FACES 15 18'), ('faces', (15, 18)))
        self.assertEqual(lt.parse('DONE C:\\x.safetensors'), ('done', 'C:\\x.safetensors'))
        self.assertEqual(lt.parse('ERROR boom'), ('error', 'boom'))
        self.assertEqual(lt.parse('loading'), ('note', 'loading'))
        self.assertEqual(lt.eta(0, 2, 100, now=10), '')
        self.assertEqual(lt.eta(0, 50, 100, now=600), 'about 10 min left')
        self.assertEqual(lt.eta(0, 100, 2000, now=600), 'about 3.2 h left')


class BuildTests(unittest.TestCase):
    def fake(self, d, body):
        script = Path(d, 'fake.py')
        script.write_text(textwrap.dedent(body))
        return mock.patch.multiple(lt, SCRIPT=str(script), python_exe=lambda _t: sys.executable)

    def spec(self, d):
        return {'name': 'n', 'person': 'P', 'trigger': 't', 'steps': 3, 'photos': [],
                'work': os.path.join(d, 'work'), 'toolkit': d, 'base': d,
                'vae': os.path.join(d, 'ae.safetensors'), 'resolution': 512,
                'text_encoder': d, 'lora_out': os.path.join(d, 'out.safetensors')}

    def test_steps_are_reported_and_the_lora_returned(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'out.safetensors')
            with self.fake(d, '''
                    out = %r
                    print('loading the model')
                    for n in range(4): print('STEP', n, 3, flush=True)
                    open(out, 'wb').write(b'lora')
                    print('DONE ' + out)
                    ''' % out):
                seen = []
                build = lt.Build(self.spec(d))
                self.assertEqual(build.run(lambda k, v: seen.append((k, v))), out)
            self.assertEqual([v for k, v in seen if k == 'step'][-1], (3, 3))
            self.assertEqual(build.step, (3, 3))
            self.assertTrue(os.path.isfile(os.path.join(d, 'work', 'train.json')))

    def test_a_failed_run_says_why(self):
        with tempfile.TemporaryDirectory() as d:
            with self.fake(d, "print('ERROR out of memory'); raise SystemExit(1)"):
                with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                    lt.Build(self.spec(d)).run()

    def test_the_photos_kept_are_reported_and_named_in_the_record(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'out.safetensors')
            with self.fake(d, '''
                    out = %r
                    print('KEPT 18 20', flush=True)
                    print('FACES 15 18', flush=True)
                    open(out, 'wb').write(b'lora')
                    print('DONE ' + out)
                    ''' % out):
                seen = []
                spec = dict(self.spec(d), photos=['p'] * 20)
                lt.Build(spec).run(lambda k, v: seen.append((k, v)))
            self.assertIn(('kept', (18, 20)), seen)
            self.assertIn(('faces', (15, 18)), seen)
            record = lt.lora_record(spec)
            self.assertIn('Built from 18 photos (15 cut to the head)', record['notes'])
            self.assertEqual((record['family'], record['trigger']), ('flux2', 't'))


class WorkerTests(unittest.TestCase):
    """tools/train_identity_lora.py's photo floor and head crops, with a fake
    PIL (the app has none) that cannot read a file holding b'bad', 1000x800,
    and a fake face finder (OpenCV is not the app's either) that finds a face
    in every photo but the last good one."""
    crops = []

    def worker(self):
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'tools', 'train_identity_lora.py')
        spec = importlib.util.spec_from_file_location('train_identity_lora', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fake_pil(self):
        crops = WorkerTests.crops = []

        class Picture:
            size = (1000, 800)
            def __init__(self, src): self.src = src
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def convert(self, _mode): return self
            def crop(self, box): crops.append((Path(self.src).name, box)); return self
            def thumbnail(self, *a): pass
            def save(self, path): Path(path).write_bytes(b'png')

        def open_(src):
            if Path(src).read_bytes() == b'bad':
                raise OSError('cannot identify image file')
            return Picture(src)
        image = mock.Mock(open=open_, LANCZOS=1)
        ops = mock.Mock(exif_transpose=lambda im: im)
        return mock.patch.dict(sys.modules, {'PIL': mock.Mock(Image=image, ImageOps=ops),
                                             'PIL.Image': image, 'PIL.ImageOps': ops})

    def run_worker(self, d, need):
        good = photos(d, 3)
        bad = []
        for i in range(2):
            p = Path(d, 'bad%d.jpg' % i)
            p.write_bytes(b'bad')
            bad.append(str(p))
        work = Path(d, 'work')
        spec = {'photos': good + bad, 'caption': 'a photo of t', 'min_photos': need,
                'steps': 3, 'name': 'n', 'work': str(work), 'toolkit': d,
                'lora_out': os.path.join(d, 'out.safetensors')}
        work.mkdir()
        (work / 'spec.json').write_text(json.dumps(spec))
        worker, said = self.worker(), []

        def train(_spec, w):
            final = w / 'output' / 'n' / 'n.safetensors'
            final.parent.mkdir(parents=True)
            final.write_bytes(b'lora')
            return 0
        last = Path(good[-1]).name

        def find(im):
            return None if Path(im.src).name == last else (400, 300, 100, 120)
        with self.fake_pil(), mock.patch.object(worker, 'say', said.append), \
                mock.patch.object(worker, 'find_face', find), \
                mock.patch.object(worker, 'train', side_effect=train) as trained, \
                mock.patch.object(sys, 'argv', ['x', str(work / 'spec.json')]):
            code = worker.main()
        return code, said, trained

    def test_too_few_readable_photos_stop_before_training_and_name_the_bad_ones(self):
        with tempfile.TemporaryDirectory() as d:
            code, said, trained = self.run_worker(d, need=4)
        self.assertEqual(code, 1)
        trained.assert_not_called()
        self.assertIn('KEPT 3 5', said)
        error = [s for s in said if s.startswith('ERROR ')]
        self.assertEqual(len(error), 1)
        self.assertEqual(lt.parse(error[0])[0], 'error')
        self.assertIn('Only 3 of 5', error[0])
        self.assertIn('at least 4', error[0])
        self.assertIn('bad0.jpg, bad1.jpg', error[0])

    def test_enough_readable_photos_train_on_those_and_say_how_many(self):
        with tempfile.TemporaryDirectory() as d:
            code, said, trained = self.run_worker(d, need=3)
            self.assertEqual(code, 0)
            trained.assert_called_once()
            self.assertIn('KEPT 3 5', said)
            self.assertIn('FACES 2 3', said)
            self.assertTrue(said[-1].startswith('DONE '))
            self.assertEqual(len(list(Path(d, 'work', 'dataset').glob('*.png'))), 3)
        # Two cut to the head; the photo with no face found goes in whole.
        square = self.worker().head_square(1000, 800, (400, 300, 100, 120))
        self.assertEqual(self.crops, [('p00.png', square), ('p01.png', square)])
        self.assertTrue(any('no face found, kept whole' in s and 'p02.png' in s for s in said))

    def test_the_head_square_has_room_for_the_hair_and_stays_inside_the_photo(self):
        square = self.worker().head_square
        # 2.4 faces wide round a 100x120 face, a little below its centre.
        left, top, right, bottom = square(1000, 800, (400, 300, 100, 120))
        self.assertEqual((right - left, bottom - top), (288, 288))
        self.assertEqual((left + right) / 2.0, 450)
        self.assertGreater((top + bottom) / 2.0, 360)
        # A face filling the photo: the square shrinks to the photo, never past it.
        self.assertEqual(square(600, 800, (0, 100, 600, 600)), (0, 190, 600, 790))
        # At an edge: slid back inside, not padded.
        self.assertEqual(square(1000, 800, (0, 0, 100, 100)), (0, 0, 240, 240))


class ButtonTests(unittest.TestCase):
    def editor(self, n):
        editor = RecordEditor.__new__(RecordEditor)
        editor.current = 0
        editor.records = [{'id': 'partner', 'name': 'Partner'}]
        editor.widgets = {'references': ('paths', {'paths': n})}
        editor.status = mock.Mock()
        editor._save = mock.Mock()
        return editor

    def test_the_button_refuses_fewer_than_twenty_photos_before_saving(self):
        with tempfile.TemporaryDirectory() as d:
            studio = ImageStudio.__new__(ImageStudio)
            studio.lora_build = None
            editor = self.editor(photos(d, 19))
            studio.build_lora(editor)
            self.assertIn('at least 20', editor.status.call_args[0][0])
            editor._save.assert_not_called()
            self.assertIsNone(studio.lora_build)

    def test_a_finished_lora_becomes_the_head_lora_of_the_open_editors_person(self):
        with tempfile.TemporaryDirectory() as d:
            lib = ig.Library(os.path.join(d, 'lib'))
            lib.save('identities', [{'id': 'partner', 'name': 'Partner', 'lora': 'flux-one',
                                     'trigger': 'partner'}])
            studio = ImageStudio.__new__(ImageStudio)
            studio.studio = mock.Mock(lib=lib)
            studio._saved = mock.Mock()
            editor = self.editor([])
            editor.records = [{'id': 'partner', 'name': 'Partner', 'notes': 'unsaved edit',
                               'lora': 'flux-one'}]
            editor.win = mock.Mock(winfo_exists=lambda: True)
            editor.fields = [('name', 'Name', 'text'), ('lora', 'Identity LoRA', ('choice', [])),
                             ('head_lora', 'Head swap LoRA', ('choice', []))]
            editor._store = mock.Mock()
            editor._build_form = mock.Mock()
            spec = {'identity': 'partner', 'person': 'Partner', 'trigger': 'lilperson',
                    'steps': 750, 'photos': ['a'] * 20, 'work': d,
                    'lora_out': os.path.join(d, 'partner_head_klein_x.safetensors')}
            studio._attach_lora(spec, editor)
            rec = lib.lora_by_file('partner_head_klein_x.safetensors')
            self.assertEqual((rec['category'], rec['family'], rec['trigger']),
                             ('Identity', 'flux2', 'lilperson'))
            saved = ig.Library(os.path.join(d, 'lib')).get('identities', 'partner')
            self.assertEqual(saved['head_lora'], rec['id'])
            # The picture's own identity LoRA and trigger are left as they were.
            self.assertEqual((saved['lora'], saved['trigger']), ('flux-one', 'partner'))
            self.assertEqual(editor.records[0]['head_lora'], rec['id'])
            self.assertEqual(editor.records[0]['lora'], 'flux-one')
            self.assertEqual(editor.records[0]['notes'], 'unsaved edit')
            fields = dict((f[0], f[2]) for f in editor.fields)
            self.assertIn(rec['id'], [k for k, _ in fields['head_lora'][1]])
            self.assertIn(rec['id'], [k for k, _ in fields['lora'][1]])
            editor._store.assert_called_once()
            editor._build_form.assert_called_once()

    def test_a_build_that_dies_unexpectedly_still_frees_the_button(self):
        studio = ImageStudio.__new__(ImageStudio)
        studio.lora_build = None
        posted = []
        studio._post = lambda kind, value: posted.append((kind, value))
        studio.say = mock.Mock()
        studio.studio = mock.Mock()
        studio.host = mock.Mock(_spawn=lambda _id, work: work)
        studio.s = mock.Mock(event_id=1)
        with tempfile.TemporaryDirectory() as d:
            editor = self.editor(photos(d, 20))
            editor.records = [{'id': 'partner', 'name': 'Partner'}]
            editor.win = mock.Mock()
            studio.studio.backends.return_value = [{'name': '5090', 'url': 'http://127.0.0.1:8188',
                                                    'lora_dir': d}]
            studio.studio.lib.get.return_value = {'id': 'partner', 'name': 'Partner',
                                                  'references': photos(d, 20)}
            spawned = []
            studio.host._spawn = lambda _id, work: spawned.append(work)
            with mock.patch.object(lt, 'problem', return_value=None), \
                    mock.patch('apps.image_studio.ui.messagebox.askokcancel', return_value=True), \
                    mock.patch.object(lt.Build, 'run', side_effect=KeyError('surprise')):
                studio.build_lora(editor)
                self.assertIsNotNone(studio.lora_build)
                with self.assertRaises(KeyError):
                    spawned[0]()
        calls = [v for k, v in posted if k == 'call']
        self.assertEqual(len(calls), 1)
        calls[0]()
        self.assertIsNone(studio.lora_build)
        self.assertIn('not built', studio.say.call_args[0][0])


sys.path.insert(0, os.path.dirname(__file__))
import test_imagegen as ti      # the module, so its TestCases are not collected here


class ButtonInTheEditor(unittest.TestCase):
    """The real Image Studio tab and identity editor (Tk, fake ComfyUI)."""
    setUpClass = ti.TestImageStudioTab.__dict__['setUpClass']
    tearDownClass = ti.TestImageStudioTab.__dict__['tearDownClass']
    tab = ti.TestImageStudioTab.tab

    def buttons(self, widget):
        for w in widget.winfo_children():
            if hasattr(w, 'invoke') and hasattr(w, 'paint'):
                yield w
            yield from self.buttons(w)

    def test_build_lora_sits_with_the_photos_and_asks_for_twenty(self):
        _, ui = self.tab()
        ui.studio.lib.save('identities', [{'id': 'p', 'name': 'P',
                                           'references': photos(self.dir, 19)}])
        ed = ui.edit_identities()
        self.addCleanup(ed.win.destroy)
        self.app.update()
        (button,) = [b for b in self.buttons(ed.form) if b.cget('text').startswith('Build LoRA')]
        with mock.patch.object(lt, 'problem', return_value=None):
            button.invoke()
        self.assertIn('at least 20', ed.msg.cget('text'))
        self.assertIsNone(ui.lora_build)


if __name__ == '__main__':
    unittest.main()
