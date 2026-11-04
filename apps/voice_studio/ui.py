"""The Voice Studio tab. Only handle() and UI commands touch Tk widgets."""
import os
import secrets
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont, ttk
import traceback
import uuid

from core import doctor
from core.local_config import default as local_default
from core.ui import Pill
from .catalog import ENGINES, MAX_TAKES, VoiceError, validate
from .client import Client, Library, check_wav, reference_wav


class VoiceStudio:
    def __init__(self, app, session):
        self.app, self.session = app, session
        self.C = app.C
        self.cancel_event = threading.Event()
        self.closed = False
        self.running = False
        self.checking = False
        self.last_error = None
        self.available = set()
        self.connected_url = None
        self.rows = {}
        self.token = tk.StringVar(master=app, value=os.environ.get('STUDIO_VOICE_TOKEN', ''))
        try:
            self.library = Library(os.path.join(doctor.data_dir(), 'voice-studio'))
        except VoiceError as error:
            self.library = None
            self.status = tk.StringVar(master=app, value=str(error))
            self.note = tk.Label(session.frame, textvariable=self.status, wraplength=700,
                                 bg=self.C['bg'], fg=self.C['text'], justify='left')
            self.note.pack(padx=24, pady=24, anchor='w')
            self._button(session.frame, 'Remote setup...', self.export_setup).pack(padx=24, anchor='w')
            return
        self.saved = app.prefs.get('voice_studio') or {}
        if not isinstance(self.saved, dict):
            self.saved = {}
        draft = self.library.data.get('draft') or {}
        self.url = tk.StringVar(master=app, value=self.saved.get('url') or os.environ.get('STUDIO_VOICE_URL', local_default('STUDIO_VOICE_URL', 'http://llm-pc:17494')))
        self.profile = tk.StringVar(master=app, value=draft.get('profile', 'My voice'))
        self.reference = tk.StringVar(master=app, value=draft.get('reference', ''))
        self.language = tk.StringVar(master=app, value=draft.get('language', 'en'))
        self.speaker = tk.StringVar(master=app, value=draft.get('speaker', 'Ryan'))
        self.per_engine = tk.IntVar(master=app, value=1)
        self.download = tk.BooleanVar(master=app, value=False)
        self.selected = {ident: tk.BooleanVar(master=app, value=ident in ('voxcpm', 'qwen-preset', 'omnivoice')) for ident in ENGINES}
        self.status = tk.StringVar(master=app, value='Connect to the voice service on the LLM PC, or export Remote setup.')
        self._build(draft)
        self.refresh_takes()

    def _label(self, parent, text, **kw):
        options = dict(bg=self.C['bg'], fg=self.C['text'], anchor='w')
        options.update(kw)
        return tk.Label(parent, text=text, **options)

    def _entry(self, parent, variable, width=24, secret=False):
        return tk.Entry(parent, textvariable=variable, width=width, show='*' if secret else '',
                        bg=self.C['card'], fg=self.C['text'], insertbackground=self.C['text'])

    def _button(self, parent, text, command):
        if hasattr(self.app, '_button'):
            return self.app._button(parent, text, command, padx=9, pady=4)
        button = Pill(parent, text, command, tkfont.Font(root=self.app, family='Segoe UI', size=9),
                      ('card', 'text', 'sel', 'card', 'muted'), padx=9, pady=4, bg=self.C['bg'])
        button.paint(self.C)
        return button

    def _build(self, draft):
        body = self.session.frame
        body.columnconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)
        header = tk.Frame(body, bg=self.C['bg'])
        header.grid(row=0, column=0, sticky='ew', padx=18, pady=(12, 8))
        self._label(header, 'Voice Studio', font=('Segoe UI', 20, 'bold')).pack(side='left')
        self._button(header, 'Remote setup...', self.export_setup).pack(side='right')
        connection = tk.Frame(body, bg=self.C['bg'])
        connection.grid(row=1, column=0, sticky='ew', padx=18)
        connection.columnconfigure(1, weight=1)
        self._label(connection, 'Voice server').grid(row=0, column=0, sticky='w')
        self._entry(connection, self.url, 38).grid(row=0, column=1, sticky='ew', padx=8)
        self._label(connection, 'Server token').grid(row=0, column=2)
        self._entry(connection, self.token, 20, secret=True).grid(row=0, column=3, padx=8)
        self.connect_button = self._button(connection, 'Connect', self.connect)
        self.connect_button.grid(row=0, column=4)
        fields = tk.Frame(body, bg=self.C['bg'])
        fields.grid(row=2, column=0, sticky='ew', padx=18, pady=8)
        fields.columnconfigure(1, weight=1)
        self._label(fields, 'Voice profile').grid(row=0, column=0, sticky='w')
        self.profile_menu = ttk.Combobox(fields, textvariable=self.profile,
                                        values=[p.get('name', '') for p in self.library.data['profiles'] if isinstance(p, dict)])
        self.profile_menu.grid(row=0, column=1, sticky='ew', padx=8)
        self.profile_menu.bind('<<ComboboxSelected>>', self.load_profile)
        self._button(fields, 'Save profile', self.save_profile).grid(row=0, column=2)
        self._label(fields, 'Language').grid(row=0, column=3, padx=(14, 4))
        self._entry(fields, self.language, 5).grid(row=0, column=4)
        self._label(fields, 'Qwen speaker').grid(row=0, column=5, padx=(14, 4))
        ttk.Combobox(fields, textvariable=self.speaker, width=11,
                     values=['Ryan', 'Aiden', 'Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric', 'Ono_Anna', 'Sohee']).grid(row=0, column=6)
        self._label(fields, 'Reference WAV').grid(row=1, column=0, sticky='w', pady=6)
        self._entry(fields, self.reference).grid(row=1, column=1, sticky='ew', padx=8)
        self._button(fields, 'Choose...', self.choose_reference).grid(row=1, column=2)
        self._label(fields, 'Use a clean 6–10 second, 16-bit PCM WAV.', fg=self.C['muted']).grid(row=1, column=3, columnspan=4, sticky='w', padx=14)
        self._label(fields, 'Reference words').grid(row=2, column=0, sticky='nw')
        self.transcript = self._text(fields, 2)
        self.transcript.grid(row=2, column=1, columnspan=6, sticky='ew', padx=8)
        self.transcript.insert('1.0', draft.get('reference_text', ''))
        self._label(fields, 'Voice / delivery').grid(row=3, column=0, sticky='nw', pady=6)
        self.direction = self._text(fields, 2)
        self.direction.grid(row=3, column=1, columnspan=6, sticky='ew', padx=8, pady=6)
        self.direction.insert('1.0', draft.get('direction', 'Warm, natural narration, clear pronunciation and relaxed pacing.'))
        engine_box = tk.Frame(body, bg=self.C['bg'])
        engine_box.grid(row=3, column=0, sticky='ew', padx=18, pady=(0, 6))
        engine_box.columnconfigure(0, weight=1)
        engine_box.columnconfigure(1, weight=1)
        self.engine_buttons = {}
        for index, (ident, spec) in enumerate(ENGINES.items()):
            hint = 'reference' if spec['reference'] == 'required' else 'preset' if spec['reference'] == 'unused' else 'clone / create'
            button = tk.Checkbutton(engine_box, variable=self.selected[ident], text='%s · %s' % (spec['name'], hint),
                                    bg=self.C['bg'], fg=self.C['muted'], selectcolor=self.C['card'], anchor='w',
                                    activebackground=self.C['bg'], activeforeground=self.C['text'])
            button.grid(row=index // 2, column=index % 2, sticky='ew')
            self.engine_buttons[ident] = button
        panes = tk.PanedWindow(body, orient='horizontal', bg=self.C['bg'], sashwidth=8, bd=0)
        panes.grid(row=4, column=0, sticky='nsew', padx=18)
        left = tk.Frame(panes, bg=self.C['bg'])
        right = tk.Frame(panes, bg=self.C['bg'])
        panes.add(left, minsize=260, stretch='always')
        panes.add(right, minsize=320, stretch='always')
        self._label(left, 'Script', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        self.script = self._text(left, 7)
        self.script.pack(fill='both', expand=True, pady=5)
        self.script.insert('1.0', draft.get('text', ''))
        self._label(right, 'Takes · select up to six to compare', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        self.takes = ttk.Treeview(right, columns=('engine', 'duration', 'peak'), show='headings', selectmode='extended', height=7)
        for column, label, width in [('engine', 'Engine / voice', 180), ('duration', 'Seconds', 65), ('peak', 'Peak dBFS', 75)]:
            self.takes.heading(column, text=label)
            self.takes.column(column, width=width, minwidth=50)
        self.takes.pack(fill='both', expand=True, pady=5)
        self.takes.bind('<Double-1>', lambda _event: self.play())
        actions = tk.Frame(body, bg=self.C['bg'])
        actions.grid(row=5, column=0, sticky='ew', padx=18, pady=8)
        self.generate_button = self._button(actions, 'Generate takes', self.generate)
        self.generate_button.pack(side='left')
        self.stop_button = self._button(actions, 'Cancel', self.stop)
        self.stop_button.pack(side='left', padx=5)
        self.stop_button.set(state='disabled')
        self._label(actions, 'Takes per engine').pack(side='left', padx=(8, 3))
        tk.Spinbox(actions, from_=1, to=2, textvariable=self.per_engine, width=3).pack(side='left')
        tk.Checkbutton(actions, text='Allow remote model downloads', variable=self.download,
                       bg=self.C['bg'], fg=self.C['text'], selectcolor=self.C['card'],
                       activebackground=self.C['bg']).pack(side='left', padx=8)
        actions2 = tk.Frame(body, bg=self.C['bg'])
        actions2.grid(row=6, column=0, sticky='ew', padx=18)
        for text, command in [('Play', self.play), ('Stop playback', self.stop_playback),
                              ('Compare selected audio', self.compare), ('Export selected...', self.export_selected),
                              ('Recover jobs', self.recover)]:
            self._button(actions2, text, command).pack(side='left', padx=(0, 6))
        self.note = tk.Label(body, textvariable=self.status, wraplength=1050, justify='left', anchor='w',
                             bg=self.C['bg'], fg=self.C['muted'])
        self.note.grid(row=7, column=0, sticky='ew', padx=18, pady=(6, 12))

    def _text(self, parent, height):
        return tk.Text(parent, height=height, wrap='word', exportselection=False, bg=self.C['card'], fg=self.C['text'],
                       insertbackground=self.C['text'], padx=8, pady=6, relief='flat', font=('Segoe UI', 10))

    def snapshot(self):
        return dict(profile=self.profile.get(), reference=self.reference.get(), language=self.language.get().strip(),
                    speaker=self.speaker.get(), reference_text=self.transcript.get('1.0', 'end-1c'),
                    direction=self.direction.get('1.0', 'end-1c'), text=self.script.get('1.0', 'end-1c'))

    def start(self):
        pass  # Connecting and uploading are explicit user actions.

    def post(self, kind, value):
        self.app.q.put(('voices', self.session.event_id, (kind, value)))

    def connect(self):
        if self.running or self.checking:
            return
        try:
            client = Client(self.url.get(), self.token.get())
        except VoiceError as error:
            self.status.set(str(error))
            return
        self.checking = True
        self.connected_url = None
        self.available.clear()
        self.connect_button.set(state='disabled')
        self.status.set('Checking remote engine environments…')
        def work():
            try:
                self.post('capabilities', (client.url, client.capabilities()))
            except Exception as error:
                self.post('error', str(error))
            finally:
                self.post('connected', None)
        threading.Thread(target=work, name='VoiceConnect', daemon=True).start()

    def choose_reference(self):
        path = filedialog.askopenfilename(parent=self.app, title='Choose voice reference', filetypes=[('PCM WAV', '*.wav')])
        if path:
            self.reference.set(path)
            for ident in ('qwen-clone', 'chatterbox'):
                self.selected[ident].set(ident in self.available)
            self.selected['qwen-preset'].set(False)

    def save_profile(self):
        draft = self.snapshot()
        name = draft['profile'].strip()
        if not name:
            self.status.set('Give this voice profile a name.')
            return
        profile = {k: draft[k] for k in ('reference', 'reference_text', 'direction', 'language', 'speaker')}
        profile['name'] = name
        self.library.data['profiles'] = [p for p in self.library.data['profiles'] if p.get('name') != name] + [profile]
        self.library.data['draft'] = draft
        try:
            self.library.save()
        except OSError as error:
            self.status.set('Could not save voice profile: ' + str(error))
            return
        self.profile_menu.config(values=[p['name'] for p in self.library.data['profiles']])
        self.status.set('Saved voice profile ' + name)

    def load_profile(self, _event=None):
        profile = next((p for p in self.library.data['profiles'] if p.get('name') == self.profile.get()), None)
        if profile:
            self.reference.set(profile.get('reference', ''))
            self.language.set(profile.get('language', 'en'))
            self.speaker.set(profile.get('speaker', 'Ryan'))
            for widget, key in ((self.transcript, 'reference_text'), (self.direction, 'direction')):
                widget.delete('1.0', 'end')
                widget.insert('1.0', profile.get(key, ''))

    def generate(self):
        if self.running or self.checking:
            return
        draft = self.snapshot()
        try:
            client = Client(self.url.get(), self.token.get())
            if client.url != self.connected_url:
                raise VoiceError('Connect to this voice server before generating.')
            count = self.per_engine.get()
            chosen = [ident for ident in ENGINES if self.selected[ident].get()]
            if not chosen or count not in (1, 2) or len(chosen) * count > MAX_TAKES:
                raise VoiceError('Choose one to six total takes, with one or two per engine.')
            if any(ident not in self.available for ident in chosen):
                raise VoiceError('Connect first, then select engines configured on the remote PC.')
            requests = []
            for ident in chosen:
                for _index in range(count):
                    request = dict(engine=ident, text=draft['text'], direction=draft['direction'], language=draft['language'],
                                   reference_text=draft['reference_text'], seed=secrets.randbelow(2**32),
                                   speaker=draft['speaker'] if ident == 'qwen-preset' else '', allow_download=self.download.get())
                    validate(request, reference=bool(draft['reference']))
                    requests.append((uuid.uuid4().hex, request))
            self.library.data['draft'] = draft
            # Pending IDs survive a failed POST, so an uncertain submission is
            # inspectable without automatically starting another GPU job.
            self.library.data['pending'] = self.library.data.get('pending', []) + [dict(id=i, engine=r['engine'], request=r,
                                                        profile=draft['profile'], server=client.url) for i, r in requests]
            self.library.save()
            self.app.prefs.set(voice_studio={'url': client.url})
        except (VoiceError, OSError, tk.TclError) as error:
            self.status.set(str(error))
            return
        self.cancel_event.clear()
        self.last_error = None
        self.running = True
        self.session.busy = True
        self.generate_button.set(state='disabled')
        self.connect_button.set(state='disabled')
        self.stop_button.set(state='normal')
        self.status.set('Preparing remote voice generation…')
        self.app._paint_tab(self.session.id)
        if self.app.active == self.session.id:
            self.app._apply_status()
        def work():
            completed = 0
            try:
                reference = reference_wav(draft['reference']) if draft['reference'] else None
                if any(not ENGINES[r['engine']].get('cpu') for _, r in requests):
                    self.app._voice_make_room(client.url)
                for ident, request in requests:
                    if self.cancel_event.is_set():
                        break
                    try:
                        audio, _status = client.generate(request, ident=ident, reference=reference, cancel=self.cancel_event,
                                                        progress=lambda text: self.post('progress', text))
                        self.post('take', (audio, request, ident, draft['profile'], client.url, check_wav(audio)))
                        completed += 1
                    except VoiceError as error:
                        self.post('error', '%s: %s' % (ENGINES[request['engine']]['name'], error))
                        # A failed/uncertain transport must not advance to a
                        # competing GPU job while the first may still run.
                        break
            except VoiceError as error:
                self.post('error', str(error))
            except Exception:
                self.post('unexpected', traceback.format_exc())
            finally:
                self.post('done', completed)
        threading.Thread(target=work, name='VoiceGenerate', daemon=True).start()

    def handle(self, payload):
        if self.closed or self.library is None:
            return
        kind, value = payload
        if kind == 'connected':
            self.checking = False
            self.connect_button.set(state='normal')
        elif kind == 'capabilities':
            url, data = value
            self.connected_url = url
            self.available = {r['id'] for r in data['engines'] if r.get('configured') and r.get('id') in ENGINES}
            for ident, button in self.engine_buttons.items():
                button.config(fg=self.C['text'] if ident in self.available else self.C['faint'])
                if ident not in self.available:
                    self.selected[ident].set(False)
            self.app.prefs.set(voice_studio={'url': url})
            self.status.set('Connected: %d engine choices configured. Models download only when allowed.' % len(self.available))
        elif kind == 'take':
            audio, request, ident, profile, server, stats = value
            try:
                self.library.add_take(audio, request, ident=ident, profile=profile, server=server, stats=stats)
                self.library.data['pending'] = [p for p in self.library.data.get('pending', []) if p['id'] != ident]
                self.library.save()
                self.refresh_takes()
                self.takes.selection_set(ident)
                self.takes.see(ident)
            except (VoiceError, OSError) as error:
                self.cancel_event.set()
                self.last_error = str(error)
                self.status.set('The remote take finished but could not be saved here: ' + str(error))
        elif kind == 'settled':
            self.library.data['pending'] = [p for p in self.library.data.get('pending', []) if p['id'] != value]
            self.library.save()
        elif kind == 'unexpected':
            self.last_error = 'Voice generation failed.'
            self.app._report('generating voice takes', value)
            self.status.set('Voice generation failed. See the Studio Assist error log.')
        elif kind in ('error', 'progress'):
            if kind == 'error':
                self.last_error = value
            self.status.set(value)
        elif kind == 'done':
            self.running = self.session.busy = False
            self.generate_button.set(state='normal')
            self.connect_button.set(state='normal')
            self.stop_button.set(state='disabled')
            self.app._paint_tab(self.session.id)
            if self.app.active == self.session.id:
                self.app._apply_status()
            if value and not self.last_error:
                self.status.set('%d take(s) made. Listen, compare selected audio, then export your choice.' % value)

    def recover(self):
        if self.running or self.checking:
            return
        try:
            client = Client(self.url.get(), self.token.get())
            pending = [p for p in self.library.data.get('pending', []) if p.get('server') == client.url and isinstance(p.get('request'), dict)]
            if not pending:
                raise VoiceError('There are no pending jobs for this server.')
            existing_ids = {take['id'] for take in self.library.data['takes']}
        except VoiceError as error:
            self.status.set(str(error))
            return
        self.running = self.session.busy = True
        self.last_error = None
        self.cancel_event.clear()
        self.generate_button.set(state='disabled')
        self.connect_button.set(state='disabled')
        self.stop_button.set(state='normal')
        self.status.set('Recovering existing job IDs; no new generation is submitted…')
        def work():
            completed = 0
            try:
                for item in pending:
                    if self.cancel_event.is_set():
                        break
                    if item['id'] in existing_ids:
                        self.post('settled', item['id'])
                        continue
                    status = client.request('/jobs/' + item['id'])
                    if status.get('status') in ('failed', 'cancelled'):
                        self.post('settled', item['id'])
                        self.post('error', status.get('error') or status['status'])
                        continue
                    audio, _status = client.collect(item['id'], item['request'], cancel=self.cancel_event,
                                                   progress=lambda text: self.post('progress', text))
                    self.post('take', (audio, item['request'], item['id'], item.get('profile', ''), client.url, check_wav(audio)))
                    completed += 1
            except VoiceError as error:
                self.post('error', str(error))
            except Exception:
                self.post('unexpected', traceback.format_exc())
            finally:
                self.post('done', completed)
        threading.Thread(target=work, name='VoiceRecover', daemon=True).start()

    def refresh_takes(self):
        self.takes.delete(*self.takes.get_children())
        self.rows = {}
        for record in self.library.data['takes'][:200]:
            if not isinstance(record, dict) or record.get('engine') not in ENGINES or not record.get('id'):
                continue
            ident = record['id']
            self.rows[ident] = record
            self.takes.insert('', 'end', iid=ident, values=(ENGINES[record['engine']]['name'] + ' · ' + record.get('profile', ''),
                                                        '%.1f' % record.get('duration', 0), record.get('peak_db') if record.get('peak_db') is not None else 'silent'))

    def selected_takes(self):
        return [self.rows[i] for i in self.takes.selection() if i in self.rows]

    def play(self):
        selected = self.selected_takes()
        if len(selected) != 1:
            self.status.set('Select one take to play.')
            return
        try:
            import winsound
            winsound.PlaySound(selected[0]['path'], winsound.SND_FILENAME | winsound.SND_ASYNC)
        except (ImportError, OSError, RuntimeError):
            self.status.set('This take could not be played. Export the WAV to your audio player.')

    def stop_playback(self):
        try:
            import winsound
            winsound.PlaySound(None, 0)
        except (ImportError, RuntimeError):
            pass

    def compare(self):
        selected = self.selected_takes()
        if not 2 <= len(selected) <= MAX_TAKES:
            self.status.set('Select two to six takes of the same script.')
            return
        if len({r['text'] for r in selected}) != 1:
            self.status.set('Compare takes with the same script. Select matching takes from the list.')
            return
        self.app._audio_review_window()
        win = self.app.windows['audio-review']
        loaded = win.load_takes([dict(path=r['path'], end='60' if r.get('duration', 0) > 60 else '',
                                     label=ENGINES[r['engine']]['name'] + ' · ' + r.get('profile', '')) for r in selected],
                                script=selected[0]['text'], direction=selected[0].get('direction', ''))
        if not loaded:
            self.status.set('Wait for the active Audio Review comparison to finish.')

    def export_selected(self):
        selected = self.selected_takes()
        if len(selected) != 1:
            self.status.set('Select one finished take to export.')
            return
        path = filedialog.asksaveasfilename(parent=self.app, title='Export voice take', defaultextension='.wav', filetypes=[('PCM WAV', '*.wav')])
        if path:
            try:
                self.library.export(selected[0], path)
                self.status.set('Exported the original WAV to ' + path)
            except OSError as error:
                self.status.set('Could not export this take: ' + str(error))

    def export_setup(self):
        from .bundle import export
        path = filedialog.asksaveasfilename(parent=self.app, title='Save setup bundle for the LLM PC',
                                          initialfile='StudioAssist-Voice-Remote.zip', defaultextension='.zip', filetypes=[('ZIP archive', '*.zip')])
        if path:
            try:
                export(path)
                self.status.set('Remote setup saved. Extract it on the LLM PC and follow docs/voice-studio.md.')
            except OSError as error:
                self.status.set('Could not export remote setup: ' + str(error))

    def stop(self):
        self.cancel_event.set()
        self.stop_button.set(state='disabled')
        self.status.set('Cancelling; waiting for the remote worker to release its engine…')

    def can_close(self):
        if self.library is None:
            return True  # Leave an unreadable library untouched for recovery.
        try:
            self.library.data['draft'] = self.snapshot()
            self.library.save()
            return True
        except (OSError, VoiceError) as error:
            self.status.set('Could not save the voice draft: ' + str(error))
            return False

    def release(self, confirmed=False):
        self.closed = True
        self.cancel_event.set()
        self.token.set('')
        self.stop_playback()

    def close(self):
        self.cancel_event.set()  # Session.close may run on a worker thread.
