"""
studio_nodes_ui - the ComfyUI tab's Nodes view: ComfyUI's own node editor
where the transcript is, with a picture's graph loaded in it.

The tab keeps its conversation. A Chat | Nodes switch above the transcript
swaps it for the editor and back; while Nodes shows, the composer is hidden
(`Chat._select` asks `on`). The editor is `studio_comfy_view.ComfyBrowser`
held as the session's `browser`, so closing the tab takes it out of its
frame and ends it the way the Milanote tab's window is ended. The Image
Studio's Nodes button comes here through `Chat.open_nodes`: the tab opened,
switched to Nodes, and the picture's steps offered on the bar.

A collaborator of `studio_chat.Chat` like `studio_images_ui.ImageStudio`:
widgets through the host's `_skin` and `_button`, work through `_spawn`, and
the way back is a ("nodes", sid, callable) event run on the UI thread.
"""

import threading
import time
import tkinter as tk

import core.agent as eng
import apps.comfyui.view as comfy_view
import apps.image_studio.imagegen as ig

ELLIPSIS = "…"
HINT = ("ComfyUI's own editor. Nodes in the Image Studio opens a picture's graph here. "
        "What Run makes here goes to ComfyUI's output folder, not History.")


class NodesView:
    def __init__(self, host, s):
        self.host, self.s = host, s
        self.on = False                   # the editor shows, not the transcript
        self.home = eng.COMFYUI_URL       # the tab's own ComfyUI, shown first
        self.url = self.home              # the ComfyUI it shows
        self.transcript, self.hero_was = [], False
        self.steps, self.name = [], None  # the picture opened from the Image Studio
        self.pending = None               # (graph, title, url) to load once it is open
        self.lock = threading.Lock()      # one start or load at a time
        f = s.frame
        first = f.pack_slaves()[0] if f.pack_slaves() else None
        bar = self.bar = host._skin(tk.Frame(f), bg="bg")
        bar.pack(side="top", fill="x", padx=host._px(18), pady=(0, host._px(6)),
                 **({"before": first} if first is not None else {}))
        self.b_chat = host._button(bar, "Chat", self.show_chat, kind="option")
        self.b_chat.pack(side="left")
        self.b_nodes = host._button(bar, "Nodes", self.show_nodes, kind="ghost")
        self.b_nodes.pack(side="left", padx=(host._px(6), 0))
        self.tools = host._skin(tk.Frame(bar), bg="bg")
        self.b_backend = host._button(self.tools, self._backend_label() + "  ▾",
                                      self._backend_menu, kind="quiet")
        self.b_backend.pack(side="left")
        self.b_step = host._button(self.tools, "Step  ▾", self._step_menu, kind="quiet")
        self.b_reload = host._button(self.tools, "Reload", self._reload, kind="ghost")
        self.b_reload.pack(side="left", padx=(host._px(6), 0))
        self.note = tk.Label(self.tools, font=host.f_ui, anchor="w", text=HINT)
        host._skin(self.note, bg="bg", fg="muted")
        self.note.pack(side="left", fill="x", expand=True, padx=(host._px(12), 0))
        self.frame = tk.Frame(f, bd=0, highlightthickness=0)
        host._skin(self.frame, bg="card")
        self.frame.bind("<Configure>", lambda ev: s.browser is not None
                        and s.browser.fit(ev.width, ev.height))

    # ------------------------------------------------------------ switching
    def _paint_switch(self):
        for pill, lit in ((self.b_chat, not self.on), (self.b_nodes, self.on)):
            pill.roles = self.host.PILL_ROLES["option" if lit else "ghost"]
            pill.paint(self.host.C)

    def show_nodes(self, load=True):
        """The editor in place of the transcript; the window started (on
        the ComfyUI last shown) unless `load` is False, when the caller
        loads something itself."""
        if self.on:
            return
        self.on = True
        self._paint_switch()
        self.tools.pack(side="left", fill="x", expand=True, padx=(self.host._px(18), 0))
        self.transcript = [w for w in self.s.frame.pack_slaves() if w is not self.bar]
        for w in self.transcript:
            w.pack_forget()
        self.hero_was = self.s.hero is not None and bool(self.s.hero.place_info())
        if self.hero_was:
            self.s.hero.place_forget()
        self.frame.pack(side="top", fill="both", expand=True, padx=self.host._px(14),
                        pady=(0, self.host._px(14)))
        self.host._select(self.s.id)      # the composer goes; the window takes focus
        if load and (self.s.browser is None or not self.s.browser.running()):
            self._load(None, None, self.url)

    def show_chat(self):
        if not self.on:
            return
        self.on = False
        self._paint_switch()
        self.tools.pack_forget()
        self.frame.pack_forget()
        # Back in the order the transcript built them: the bar on the right first.
        for w in self.transcript:
            w.pack(side="right" if isinstance(w, tk.Scrollbar) else "left",
                   fill="y" if isinstance(w, tk.Scrollbar) else "both",
                   expand=not isinstance(w, tk.Scrollbar))
        if self.hero_was:
            self.host._welcome(self.s)
        self.host._select(self.s.id)

    def focus(self):
        if self.on and self.s.browser is not None:
            self.s.browser.focus()

    def say(self, text, role="muted"):
        self.note.config(text=text)
        self.host._skin(self.note, bg="bg", fg=role)

    # ------------------------------------------------------------ pickers
    def _backends(self):
        """(name, url) of each ComfyUI the Image Studio knows, and this tab's."""
        out = [(b.get("name") or b["id"], b["url"]) for b in ig.Library().all("backends")
               if b.get("url")]
        if not any(comfy_view.origin(u) == comfy_view.origin(self.home) for _n, u in out):
            out.append((self.s.app.name, self.home))
        return out

    def _backend_label(self):
        for name, url in self._backends():
            if comfy_view.origin(url) == comfy_view.origin(self.url):
                return name
        return comfy_view.origin(self.url).split("//", 1)[-1]

    def _menu(self, pill, items):
        menu = tk.Menu(pill, tearoff=0)
        self.host._skin(menu, bg="card", fg="text", activebackground="sel",
                        activeforeground="text")
        for label, command in items:
            menu.add_command(label=label, command=command)
        menu.tk_popup(pill.winfo_rootx(), pill.winfo_rooty() + pill.winfo_height())

    def _backend_menu(self):
        self._menu(self.b_backend, [(name, lambda u=url: self._switch(u))
                                    for name, url in self._backends()])

    def _switch(self, url):
        """Another ComfyUI, chosen by hand: the picture's steps were made on
        the one it came from, so they go from the bar."""
        self.steps = []
        self.b_step.pack_forget()
        self._load(None, None, url)

    def _step_menu(self):
        self._menu(self.b_step, [(label, lambda i=i: self.load_step(i))
                                 for i, (label, _g) in enumerate(self.steps)])

    # ------------------------------------------------------------ loading
    def open(self, steps, url, name):
        """A picture from the Image Studio: its steps on the bar, Picture
        (else the first) opened. On the UI thread."""
        self.steps, self.name = steps, name
        self.b_step.pack(side="left", padx=(self.host._px(6), 0), before=self.b_reload)
        self.url = url
        first = next((i for i, (label, _g) in enumerate(steps) if label == "Picture"), 0)
        if not self.on:
            self.show_nodes(load=False)
        self.load_step(first)

    def load_step(self, i):
        label, graph = self.steps[i]
        self.b_step.set(text=label + "  ▾")
        self._load(graph, "%s · %s" % (self.name, label), self.url)

    def _load(self, graph, title, url):
        """The window started if it is not running, gone to `url`, and
        `graph` (if any) opened in it. Off the UI thread."""
        self.url = url
        self.b_backend.set(text=self._backend_label() + "  ▾")
        self.say(("Opening %s" % title if graph else "Opening ComfyUI") + ELLIPSIS)
        self.host._spawn(self.s.event_id, self._work, graph, title, url)

    def _post(self, fn):
        self.host.q.put(("nodes", self.s.event_id, fn))

    def _work(self, graph, title, url):
        def said(text, role="muted"):
            if role == "err":             # an error's own words, made a sentence
                text = text[:1].upper() + text[1:]
            self._post(lambda: self.say(text, role))
        with self.lock:
            s, b = self.s, self.s.browser
            if b is None or not b.running():
                b = s.browser = comfy_view.ComfyBrowser(url)
                try:
                    b.start()
                except Exception as e:
                    s.browser = None
                    b.close(0)
                    return said(str(e), "err")
                if s.closed:
                    s.browser = None
                    return b.close()
                self._post(lambda: self._adopt(b))
            try:
                if graph is None:
                    b.goto(url)
                    return said(HINT)
                n = b.show(graph, title, url)
            except Exception as e:
                return said(str(e), "err")
        said("%s: %s nodes. What Run makes here goes to ComfyUI's output folder, "
             "not History." % (title, n))

    def _adopt(self, b):
        """The new window into the frame, then measured twice as the Milanote
        tab's is, so its own title bar is clipped off."""
        if self.s.closed or b is not self.s.browser:
            return
        self.frame.update_idletasks()
        b.embed(self.frame.winfo_id(), self.frame.winfo_width(), self.frame.winfo_height())
        self.focus()

        def measure():
            for wait in (1.0, 3.0):
                time.sleep(wait)
                if b is not self.s.browser:
                    return
                try:
                    before = b.inset
                    if b.measure() != before:
                        self._post(lambda: b.fit(*b.size))
                except Exception:
                    return            # it keeps its title bar; nothing else is wrong
        self.host._spawn(self.s.event_id, measure)

    def _reload(self):
        b = self.s.browser
        if b is None or not b.running():
            return self._load(None, None, self.url)

        def work():
            try:
                b.reload()
            except Exception as e:
                self._post(lambda: self.say(str(e), "err"))
        self.host._spawn(self.s.event_id, work)
