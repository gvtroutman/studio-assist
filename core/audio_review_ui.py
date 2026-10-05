"""The Audio Review window. Workers touch only queues, never Tk widgets."""
import json
import os
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog

from core import audio_review


class AudioReviewWindow(tk.Toplevel):
    def __init__(self, master, *, palette, settings=None, save_settings=None, report_error=None):
        super().__init__(master)
        self.title('Audio Review')
        self.geometry('1060x740')
        self.minsize(900, 700)
        self.transient(master)
        self.C = palette
        self.configure(bg=self.C['bg'])
        self.settings = settings if isinstance(settings, dict) else {}
        self.save_settings = save_settings
        self.report_error = report_error
        self.takes = []
        self.result = None
        self.events = queue.Queue()
        self.cancel_event = threading.Event()
        self.closed = False
        self.running = False
        self.timer = None
        self.protocol('WM_DELETE_WINDOW', self.close)
        body = tk.Frame(self, bg=self.C['bg'])
        body.pack(fill='both', expand=True, padx=22, pady=18)
        body.columnconfigure(1, weight=1)
        self.host = tk.StringVar(value=self._setting('host', 'STUDIO_AUDIO_BASE_URL'))
        self.model = tk.StringVar(value=self._setting('model', 'STUDIO_AUDIO_MODEL'))
        self.key = tk.StringVar()
        self.start = tk.StringVar(value='0')
        self.end = tk.StringVar()
        self.status = tk.StringVar(value='Choose recordings of the same line to compare.')
        self._label(body, 'Audio Review', 0, font=('Segoe UI', 20, 'bold'))
        self._label(body, 'Compare delivery, pronunciation, pacing and audible edit problems.', 1)
        connection = tk.Frame(body, bg=self.C['bg'])
        connection.grid(row=2,column=0,columnspan=2,sticky='ew',pady=(6,8))
        for column,(title,variable,secret) in enumerate([
                ('Host URL',self.host,False),('Audio model ID',self.model,False),
                ('API key (optional, never saved)',self.key,True)]):
            connection.columnconfigure(column,weight=2 if column==0 else 1)
            tk.Label(connection,text=title,bg=self.C['bg'],fg=self.C['text'],anchor='w').grid(
                row=0,column=column,sticky='ew',padx=(0,10))
            tk.Entry(connection,textvariable=variable,bg=self.C['card'],fg=self.C['text'],
                     insertbackground=self.C['text'],show='•' if secret else '',width=22).grid(
                row=1,column=column,sticky='ew',padx=(0,10),pady=4)
        self._label(body, 'Recordings are sent to this audio-capable host when you click Compare takes.', 3)
        workspace = tk.Frame(body,bg=self.C['bg'])
        workspace.grid(row=4,column=0,columnspan=2,sticky='nsew',pady=(8,4))
        workspace.columnconfigure(0,weight=1,uniform='panes')
        workspace.columnconfigure(1,weight=1,uniform='panes')
        workspace.rowconfigure(0,weight=1)
        left=tk.Frame(workspace,bg=self.C['bg'])
        left.grid(row=0,column=0,sticky='nsew',padx=(0,14))
        left.columnconfigure(1,weight=1)
        right=tk.Frame(workspace,bg=self.C['bg'])
        right.grid(row=0,column=1,sticky='nsew')
        right.columnconfigure(1,weight=1)
        span = tk.Frame(left, bg=self.C['bg'])
        span.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(4, 4))
        tk.Label(span, text='Source range in seconds:', bg=self.C['bg'], fg=self.C['text']).pack(side='left')
        for title, variable in [('Start', self.start), ('End (blank = full)', self.end)]:
            tk.Label(span, text=title, bg=self.C['bg'], fg=self.C['muted']).pack(side='left', padx=(10,4))
            tk.Entry(span, textvariable=variable, width=8, bg=self.C['card'], fg=self.C['text'], insertbackground=self.C['text']).pack(side='left')
        buttons = tk.Frame(left, bg=self.C['bg'])
        buttons.grid(row=1, column=0, columnspan=2, sticky='ew', pady=4)
        self.add_button = self._button(buttons, 'Add files…', self.add_files)
        self.remove_button = self._button(buttons, 'Remove selected', self.remove_selected)
        self._label(left, '2–6 WAV/MP3 takes, up to 60 seconds each. Add the same file again for another range.', 2,
                    wraplength=390,justify='left')
        self.listbox = tk.Listbox(left, height=5, bg=self.C['card'], fg=self.C['text'],
                                 selectbackground=self.C['sel'], selectmode='extended', exportselection=False)
        self.listbox.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(4,12))
        self._label(left, 'Intended words (optional)', 4)
        self.script = self._text(left, 5, 4)
        self._label(left, 'Desired delivery (optional)', 6)
        self.direction = self._text(left, 7, 3)
        left.rowconfigure(5,weight=1)
        left.rowconfigure(7,weight=1)
        self.direction.insert('1.0', 'Natural, confident safety-training narration. Clear and calm, with appropriate emphasis.')
        action = tk.Frame(body, bg=self.C['bg'])
        action.grid(row=5, column=0, columnspan=2, sticky='ew', pady=10)
        self.compare_button = self._button(action, 'Compare takes', self.start_review)
        self.stop_button = self._button(action, 'Stop', self.stop)
        self.stop_button.configure(state='disabled')
        self.save_button = self._button(action, 'Save review…', self.save_review)
        self.save_button.configure(state='disabled')
        tk.Label(body, textvariable=self.status, bg=self.C['bg'], fg=self.C['muted'], anchor='w',
                 wraplength=840).grid(row=6, column=0, columnspan=2, sticky='ew')
        self._label(right,'Comparison',0)
        self.output = self._text(right, 1, 16)
        self.output.configure(state='disabled')
        right.rowconfigure(1,weight=1)
        body.rowconfigure(4, weight=1)

    def _setting(self, name, env):
        value = os.environ.get(env) or self.settings.get(name)
        return value if isinstance(value, str) else ''

    def _label(self, parent, text, row, **kw):
        label = tk.Label(parent, text=text, bg=self.C['bg'], fg=self.C['text'], anchor='w', **kw)
        label.grid(row=row, column=0, columnspan=2, sticky='ew', pady=3)
        return label

    def _entry(self, parent, variable, row, **kw):
        entry = tk.Entry(parent, textvariable=variable, bg=self.C['card'], fg=self.C['text'],
                         insertbackground=self.C['text'], **kw)
        entry.grid(row=row, column=1, sticky='ew', padx=(12,0), pady=3)
        return entry

    def _text(self, parent, row, height):
        frame = tk.Frame(parent, bg=self.C['bg'])
        frame.grid(row=row, column=0, columnspan=2, sticky='nsew', pady=4)
        text = tk.Text(frame, height=height, wrap='word', bg=self.C['card'], fg=self.C['text'],
                       insertbackground=self.C['text'], font=('Segoe UI',10), padx=8, pady=6)
        scroll = tk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        text.pack(fill='both', expand=True)
        return text

    def _button(self, parent, text, command):
        button = tk.Button(parent, text=text, command=command, bg=self.C['card'], fg=self.C['text'],
                           activebackground=self.C['sel'], activeforeground=self.C['text'], padx=12, pady=5)
        button.pack(side='left', padx=(0,8))
        return button

    def add_files(self):
        paths = filedialog.askopenfilenames(parent=self, title='Choose narration takes',
                                           filetypes=[('Audio files','*.wav *.mp3')])
        for path in paths:
            if len(self.takes) >= audio_review.MAX_TAKES:
                self.status.set('At most six takes per comparison.')
                break
            self.takes.append({'path':path, 'start':self.start.get(), 'end':self.end.get()})
        self.refresh_takes()

    def refresh_takes(self):
        self.listbox.delete(0,'end')
        for i,take in enumerate(self.takes,1):
            self.listbox.insert('end','T%d  %s  [%s–%s s]' %
                                (i,os.path.basename(take['path']),take['start'] or '0',take['end'] or 'end'))

    def remove_selected(self):
        for i in reversed(self.listbox.curselection()):
            del self.takes[i]
        self.refresh_takes()

    def start_review(self):
        if self.running:
            return
        try:
            audio_review.endpoint(self.host.get())
            if not self.model.get().strip():
                raise audio_review.AudioReviewError('Enter the ID of an audio-capable model on the selected host.')
            if not 2 <= len(self.takes) <= audio_review.MAX_TAKES:
                raise audio_review.AudioReviewError('Choose two to six takes of the same line.')
        except audio_review.AudioReviewError as e:
            self.status.set(str(e))
            return
        host, model = self.host.get().strip(), self.model.get().strip()
        if self.save_settings:
            self.save_settings({'host':host,'model':model})
        takes = [dict(t) for t in self.takes]
        options = {'script':self.script.get('1.0','end-1c'), 'direction':self.direction.get('1.0','end-1c'),
                   'api_key':self.key.get(), 'cancel':self.cancel_event,
                   'progress':lambda text:self.events.put(('progress',text))}
        self.cancel_event.clear()
        self.result = None
        self.running = True
        self._busy(True)
        self.save_button.configure(state='disabled')
        self._output('')
        self.status.set('Preparing recordings…')
        def work():
            try:
                result = audio_review.compare(takes,host,model,**options)
                self.events.put(('result',result))
            except audio_review.AudioReviewError as e:
                self.events.put(('error',str(e)))
            except Exception:
                self.events.put(('unexpected',traceback.format_exc()))
            finally:
                self.events.put(('done',None))
        threading.Thread(target=work, name='AudioReview', daemon=True).start()
        self.timer = self.after(50,self.drain)

    def _busy(self, busy):
        for button in (self.add_button,self.remove_button,self.compare_button):
            button.configure(state='disabled' if busy else 'normal')
        self.stop_button.configure(state='normal' if busy else 'disabled')

    def _output(self, text):
        self.output.configure(state='normal')
        self.output.delete('1.0','end')
        self.output.insert('1.0',text)
        self.output.configure(state='disabled')

    def drain(self):
        self.timer = None
        if self.closed:
            return
        try:
            while True:
                kind,value = self.events.get_nowait()
                if kind=='done':
                    self.running=False
                    self._busy(False)
                elif kind=='unexpected':
                    self.status.set('Audio review failed. See the Studio Assist error log.')
                    if self.report_error: self.report_error('reviewing narration audio',value)
                elif kind=='error':
                    self.status.set(value)
                elif not self.cancel_event.is_set():
                    if kind=='progress': self.status.set(value)
                    elif kind=='result':
                        self.result=value
                        self._output(audio_review.report_text(value))
                        self.save_button.configure(state='normal')
                        self.status.set('Audio comparison complete. Recommendations are subjective; source files are unchanged.')
        except queue.Empty:
            pass
        finally:
            if not self.closed and self.running:
                self.timer=self.after(50,self.drain)

    def stop(self):
        self.cancel_event.set()
        self.status.set('Review cancelled. Waiting for the active request to finish.')
        self.stop_button.configure(state='disabled')

    def save_review(self):
        if not self.result:
            return
        path=filedialog.asksaveasfilename(parent=self, title='Save audio review', defaultextension='.txt',
                                         filetypes=[('Text report','*.txt'),('Analysis JSON','*.json')])
        if not path:
            return
        try:
            with open(path,'w',encoding='utf-8') as f:
                if path.lower().endswith('.json'): json.dump(self.result,f,indent=2,ensure_ascii=False)
                else: f.write(audio_review.report_text(self.result))
            self.status.set('Review saved to '+path)
        except OSError:
            self.status.set('The review could not be saved to that location.')

    def close(self):
        if self.closed:
            return
        self.closed=True
        self.cancel_event.set()
        self.key.set('')
        if self.timer is not None:
            self.after_cancel(self.timer)
            self.timer=None
        self.destroy()
