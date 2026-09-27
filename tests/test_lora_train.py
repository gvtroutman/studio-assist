"""Build LoRA: planning, the toolkit check, the run's progress, and the editor
button - without ai-toolkit, a GPU or Tk."""
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import studio_imagegen as ig
import studio_lora_train as lt
from studio_images_ui import ImageStudio, RecordEditor


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
                lt.plan({'id': 'lilya', 'name': 'Lilya', 'references': refs}, d)

    def test_a_plan_names_its_files_and_keeps_the_persons_trigger(self):
        with tempfile.TemporaryDirectory() as d:
            refs = photos(d, 20)
            spec = lt.plan({'id': 'lilya', 'name': 'Lilya', 'references': refs,
                            'trigger': 'lylperson'}, d, toolkit=d, when=0)
            self.assertEqual(spec['trigger'], 'lylperson')
            self.assertEqual(spec['photos'], refs)
            self.assertTrue(spec['lora_out'].startswith(d))
            self.assertTrue(spec['lora_out'].endswith('.safetensors'))
            self.assertIn('lilya_identity_', spec['name'])
            self.assertEqual(os.path.dirname(spec['work']), os.path.join(d, 'output'))
            fresh = lt.plan({'id': 'g', 'name': 'Gavin T.', 'references': refs}, d, toolkit=d)
            self.assertEqual(fresh['trigger'], 'gavperson')

    def test_the_job_trains_the_local_flux_copy_on_the_dataset(self):
        spec = {'name': 'n', 'work': 'W', 'trigger': 't', 'steps': 2000, 'base': 'B'}
        proc = lt.config(spec)['config']['process'][0]
        self.assertEqual(proc['model']['name_or_path'], 'B')
        self.assertTrue(proc['model']['is_flux'])
        self.assertEqual(proc['train']['steps'], 2000)
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
            self.assertIn('transformer', lt.problem(d))
            for part in lt.BASE_PARTS:
                folder = Path(lt.base_model(d), part)
                folder.mkdir(parents=True)
                (folder / '.complete').touch()
            self.assertIsNone(lt.problem(d))

    def test_lines_and_time_left(self):
        self.assertEqual(lt.parse('STEP 12 2000\n'), ('step', (12, 2000)))
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
                'lora_out': os.path.join(d, 'out.safetensors')}

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


class ButtonTests(unittest.TestCase):
    def editor(self, n):
        editor = RecordEditor.__new__(RecordEditor)
        editor.current = 0
        editor.records = [{'id': 'lilya', 'name': 'Lilya'}]
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

    def test_a_finished_lora_joins_the_library_and_the_open_editors_person(self):
        with tempfile.TemporaryDirectory() as d:
            lib = ig.Library(os.path.join(d, 'lib'))
            lib.save('identities', [{'id': 'lilya', 'name': 'Lilya'}])
            studio = ImageStudio.__new__(ImageStudio)
            studio.studio = mock.Mock(lib=lib)
            studio._saved = mock.Mock()
            editor = self.editor([])
            editor.records = [{'id': 'lilya', 'name': 'Lilya', 'notes': 'unsaved edit'}]
            editor.win = mock.Mock(winfo_exists=lambda: True)
            editor.fields = [('name', 'Name', 'text'), ('lora', 'Identity LoRA', ('choice', []))]
            editor._store = mock.Mock()
            editor._build_form = mock.Mock()
            spec = {'identity': 'lilya', 'person': 'Lilya', 'trigger': 'lilperson',
                    'steps': 2000, 'photos': ['a'] * 20, 'work': d,
                    'lora_out': os.path.join(d, 'lilya_identity_x.safetensors')}
            studio._attach_lora(spec, editor)
            rec = lib.lora_by_file('lilya_identity_x.safetensors')
            self.assertEqual((rec['category'], rec['family'], rec['trigger']),
                             ('Identity', 'flux1', 'lilperson'))
            self.assertEqual(lib.get('identities', 'lilya')['lora'], rec['id'])
            self.assertEqual(ig.Library(os.path.join(d, 'lib')).get('identities', 'lilya')['lora'],
                             rec['id'])
            self.assertEqual(editor.records[0]['lora'], rec['id'])
            self.assertEqual(editor.records[0]['notes'], 'unsaved edit')
            self.assertIn(rec['id'], [k for k, _ in dict(
                (f[0], f[2]) for f in editor.fields)['lora'][1]])
            editor._store.assert_called_once()
            editor._build_form.assert_called_once()


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
