"""Run via tests/offscreen.py: real widgets; no model or remote service."""
import gc
import io
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from apps.voice_studio import ui
from apps.voice_studio.catalog import ENGINES
from core.audio_review_ui import AudioReviewWindow
from core.ui import DARK
from tests.test_voice_studio import wav, request


class VoiceWindowTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(gc.collect)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = patch.dict(os.environ, STUDIO_SETTINGS=os.path.join(self.temp.name, 'settings.json'))
        self.settings.start()
        self.addCleanup(self.settings.stop)
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.root.geometry('1060x780')
        self.root.C = DARK
        self.root.q = queue.Queue()
        self.root.windows = {}
        self.root.active = 'voice-studio'
        self.root._paint_tab = lambda _sid: None
        self.root._apply_status = lambda: None
        self.root._voice_make_room = lambda _url: None
        self.errors = []
        self.root._report = lambda *args: self.errors.append(args)
        self.prefs = {}
        self.root.prefs = SimpleNamespace(get=lambda key: self.prefs.get(key), set=lambda **values: self.prefs.update(values))
        self.frame = tk.Frame(self.root, bg=DARK['bg'])
        self.frame.pack(fill='both', expand=True)
        self.session = SimpleNamespace(frame=self.frame, id='voice-studio', event_id=('voice-studio', 'generation'), busy=False)
        self.view = ui.VoiceStudio(self.root, self.session)
        self.root._audio_review_window = self.review_window
        self.addCleanup(self.cleanup)
        self.root.update()

    def cleanup(self):
        if hasattr(self, 'review'):
            self.review.close()
        self.view.release(True)
        self.root.update_idletasks()
        self.root.destroy()
        gc.collect()

    def review_window(self):
        self.review = AudioReviewWindow(self.root, palette=DARK)
        self.root.windows['audio-review'] = self.review

    def connected(self):
        self.view.handle(('capabilities', (self.view.url.get(), {'engines': [dict(id=i, configured=True) for i in ENGINES]})))

    def drain(self):
        while True:
            try:
                kind, event_id, payload = self.root.q.get_nowait()
            except queue.Empty:
                break
            self.assertEqual(kind, 'voices')
            self.assertEqual(event_id, self.session.event_id)
            self.view.handle(payload)
        self.root.update()

    def test_form_controls_fit_and_named_profile_survives(self):
        self.assertGreater(self.view.script.winfo_height(), 60)
        self.assertLessEqual(self.view.note.winfo_rooty() + self.view.note.winfo_height(), self.root.winfo_rooty() + self.root.winfo_height())
        self.view.profile.set('Narrator')
        self.view.transcript.insert('1.0', 'Reference words.')
        self.view.save_profile()
        self.view.transcript.delete('1.0', 'end')
        self.view.load_profile()
        self.assertEqual(self.view.transcript.get('1.0', 'end-1c'), 'Reference words.')
        self.assertTrue(self.view.can_close())
        self.assertEqual(self.view.library.data['draft']['profile'], 'Narrator')

    def test_batch_returns_separate_takes_without_saving_token(self):
        self.connected()
        self.view.script.insert('1.0', 'A line to compare.')
        self.view.token.set('never-persist-this')
        for ident, variable in self.view.selected.items():
            variable.set(ident in ('voxcpm', 'qwen-preset'))
        calls = []
        main_thread = threading.get_ident()
        class FakeClient:
            def __init__(self, url, key):
                self.url = url
            def generate(self, request, **options):
                if threading.get_ident() == main_thread:
                    raise AssertionError('Inference ran on the UI thread')
                calls.append(request)
                return wav(), {'status': 'completed'}
        with patch.object(ui, 'Client', FakeClient):
            self.view.generate()
            deadline = time.monotonic() + 3
            while self.view.running and time.monotonic() < deadline:
                self.drain()
                time.sleep(.01)
            self.drain()
        self.assertFalse(self.view.running)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]['seed'], calls[1]['seed'])
        self.assertEqual(len(self.view.library.data['takes']), 2)
        self.assertFalse(self.view.library.data['pending'])
        self.assertNotIn('never-persist-this', Path(self.view.library.path).read_text(encoding='utf8'))

    def test_compare_prefills_audio_without_starting_review(self):
        for engine in ('voxcpm', 'qwen-preset'):
            self.view.library.add_take(wav(), request(engine=engine), profile='Narrator')
        self.view.refresh_takes()
        self.view.takes.selection_set(*self.view.takes.get_children())
        self.view.compare()
        self.assertEqual(len(self.review.takes), 2)
        self.assertFalse(self.review.running)
        self.assertIsNone(self.review.result)
        self.assertEqual(self.review.script.get('1.0', 'end-1c'), 'The same spoken line.')

    def test_incompatible_scripts_and_late_events_do_not_change_widgets(self):
        for text in ('One line.', 'A different line.'):
            self.view.library.add_take(wav(), request(text=text))
        self.view.refresh_takes()
        self.view.takes.selection_set(*self.view.takes.get_children())
        self.view.compare()
        self.assertIn('same script', self.view.status.get())
        before = self.view.status.get()
        self.view.release(True)
        self.view.handle(('error', 'late error'))
        self.assertEqual(self.view.status.get(), before)
        self.assertTrue(self.view.cancel_event.is_set())

    def test_failed_take_does_not_become_success_on_done(self):
        self.view.handle(('error', 'Remote worker failed'))
        self.view.handle(('done', 1))
        self.assertEqual(self.view.status.get(), 'Remote worker failed')

    def test_failed_draft_save_vetoes_close(self):
        with patch.object(self.view.library, 'save', side_effect=OSError('disk full')):
            self.assertFalse(self.view.can_close())
        self.assertIn('disk full', self.view.status.get())

    def test_unreadable_library_leaves_a_closable_error_tab(self):
        frame = tk.Frame(self.root)
        session = SimpleNamespace(frame=frame)
        with patch.object(ui, 'Library', side_effect=ui.VoiceError('Library needs recovery')):
            view = ui.VoiceStudio(self.root, session)
        self.assertIn('recovery', view.status.get())
        self.assertTrue(view.can_close())
        view.release(True)
        frame.destroy()


if __name__ == '__main__':
    unittest.main()
