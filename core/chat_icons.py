"""Preferences > Icons: reading an app's icon out of its .exe, the
upload/reset flow, and the window that lists every mark and button.
Split out of core/chat.py (docs/CODEMAP.md) as a mixin."""
import base64
import hashlib
import os
import re
import tempfile
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed

import core.agent as eng
import core.icon_search as icon_search
import core.icons as icons

from core.chat import clip, BUTTON_SHOWS


class ChatIconsMixin:
    def _read_icons(self, only=None, uploaded=None):
        """
        Off the UI thread: an app's icon lives inside its .exe, and a cold read
        of a 500MB Photoshop binary is not something to do while the window is
        trying to open. Badges are drawn until these land.

        An icon the user uploaded (Preferences > Icons) wins over the .exe's,
        and every tab takes one - Chat and the panel tabs too, which have no
        .exe and otherwise keep their badge. `only` reads one key's again;
        `uploaded` is the prefs' icons as the UI thread saw them.
        """
        row, tab, menu, hero = (self.marks_px["row"], self.marks_px["tab"],
                                self.marks_px["menu"], self.marks_px["hero"])
        uploaded = dict(self.prefs.get("icons") if uploaded is None else uploaded)
        jobs = [(a["id"] or a["name"], a["exe"], row) for a in self.detected]
        for app in list(eng.TABS) + list(eng.custom_bridges()):
            if only is not None and app.id != only:
                continue
            exe = app.exe() if app in eng.APPS else None
            # The row size too: Preferences > Icons shows every tab at it.
            jobs += [(app.id, exe, s) for s in (row, tab, menu, hero)]
        done = set()
        for key, exe, size in jobs:
            if (only is not None and key != only) or (key, size) in done:
                continue
            done.add((key, size))
            custom = uploaded.get(key)
            data = (icons.sized_png(os.path.join(self._icons_dir(), custom), size)
                    if custom else None)
            # No .exe (ComfyUI is on the LLM PC): its mark is drawn instead.
            data = data or icons.icon_png(exe, size) or icons.drawn_png(key, size)
            if data:
                self.q.put(("icon", None, (key, size, data)))

    def _icons_dir(self):
        return os.path.join(self._data_dir(), "icons")

    def _icon_specs(self):
        """[(mark spec, name)] for every mark the window can draw: each kind
        of tab, bridges added by hand, then any app in the rail that is none
        of those. Preferences > Icons lists them in this order."""
        out, seen = [], set()
        for app in list(eng.TABS) + list(eng.custom_bridges()):
            if app.id not in seen:
                seen.add(app.id)
                out.append((self._spec_for(app), app.name))
        for a in self.detected:
            key = a["id"] or a["name"]
            if key not in seen:
                seen.add(key)
                out.append(({"key": key, "code": a["code"], "fg": a["fg"],
                             "bg": a["bg"]}, a["name"]))
        return out

    SLIDER_SETTLE_MS = 200                # a drag applies once it pauses this long

    def _slider(self, body, title, bounds, pref, apply):
        """A Preferences slider over `bounds`, shown as a percentage of the
        designed size, with a Reset to 100%. Dragging applies it once the
        drag pauses (`SLIDER_SETTLE_MS`): a text size change lays out the
        whole window, and doing that for every pixel of a drag stutters."""
        self._cap(body, title, bg="bg").pack(fill="x", pady=(22, 4))
        line = self._skin(tk.Frame(body), bg="bg")
        line.pack(fill="x")
        var = tk.DoubleVar(master=line, value=self.prefs.get(pref))
        shown = tk.Label(line, font=self.f_ui, width=5, anchor="e")
        self._skin(shown, bg="bg", fg="text")
        pending = {"id": None}

        def settle():
            pending["id"] = None
            value = round(var.get(), 2)
            if value != self.prefs.get(pref):
                apply(value)

        def moved(_value=None):
            shown.config(text="%d%%" % round(var.get() * 100))
            if pending["id"] is not None:
                line.after_cancel(pending["id"])
            pending["id"] = line.after(self.SLIDER_SETTLE_MS, settle)

        scale = self._scale(line, var, bounds[0], bounds[1], resolution=0.05,
                             command=moved, length=self._px(240))
        scale.pack(side="left")
        shown.pack(side="left", padx=(10, 0))

        def reset():
            var.set(1.0)
            moved()
        self._button(line, "100%", reset, kind="ghost").pack(side="left", padx=(10, 0))
        shown.config(text="%d%%" % round(var.get() * 100))
        return scale

    def _icons_section(self, body, _prefs):
        """Preferences > Icons: one line and a button. The list itself is a
        window of its own - a dozen rows of it made Preferences taller than a
        laptop's screen, and Preferences does not resize."""
        self._cap(body, "ICONS", bg="bg").pack(fill="x", pady=(22, 8))
        row = self._skin(tk.Frame(body), bg="bg")
        row.pack(fill="x")
        self._button(row, "Change icons...", self._icons_window).pack(side="left")
        n = tk.Label(row, font=self.f_small, anchor="w")
        self._skin(n, bg="bg", fg="faint")
        n.pack(side="left", padx=(10, 0))

        def count():
            k = len(self.prefs.get("icons"))
            try:
                n.config(text="Every app's own icon." if not k else
                         "%d uploaded icon%s." % (k, "" if k == 1 else "s"))
            except tk.TclError:
                pass
        self.windows["icons_count"] = count
        count()

    def _button_labels(self):
        """Every text button's label in the window now (and any that has an
        icon but is not on screen), once each: an icon is per label, so the
        Send of every tab is one row."""
        live = {p.text.strip() for p in self.pills
                if p.anchor == "center" and p.text.strip() and p.winfo_exists()}
        kept = {k[len("button:"):] for k in self.prefs.get("icons")
                if k.startswith("button:")}
        return sorted(live | kept, key=str.lower)

    def _icons_window(self):
        """Every app and tab with its mark as drawn now, then every button,
        each with Upload... to give it a picture and Reset to take it back.
        Scrolls: with the Image Studio built there are dozens of buttons."""
        win = self.windows.get("icons")
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            return
        self._forget()
        labels = self._button_labels()    # before this window adds pills of its own
        win = tk.Toplevel(self)
        self.windows["icons"] = win
        win.title("Icons")
        win.transient(self)
        self._skin(win, bg="bg")
        outer = self._skin(tk.Frame(win), bg="bg")
        outer.pack(fill="both", expand=True, padx=(22, 6), pady=18)
        note = tk.Label(outer, font=self.f_small, anchor="w", justify="left",
                        wraplength=self._px(620),
                        text="PNG, ICO, JPEG, GIF, BMP or TIFF. Square pictures "
                             "look best; others are centred, not stretched. A "
                             "button's picture goes on every button with its label.")
        self._skin(note, bg="bg", fg="faint")
        note.pack(fill="x", pady=(0, 10))
        foot = self._skin(tk.Frame(outer), bg="bg")
        foot.pack(side="bottom", fill="x")
        self._button(foot, "Close", win.destroy, kind="accent").pack(
            anchor="e", pady=(12, 0), padx=(0, 16))
        scroller = tk.Canvas(outer, highlightthickness=0, bd=0)
        self._skin(scroller, bg="bg")
        bar = tk.Scrollbar(outer, orient="vertical", command=scroller.yview)
        scroller.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        scroller.pack(side="left", fill="both", expand=True)
        body = self._skin(tk.Frame(scroller), bg="bg")
        scroller.create_window(0, 0, window=body, anchor="nw")

        def fit(_ev=None):
            scroller.configure(scrollregion=scroller.bbox("all"),
                               width=body.winfo_reqwidth())
        body.bind("<Configure>", fit)
        wheel = lambda ev: scroller.yview_scroll(int(-ev.delta / 120), "units")
        win.bind("<MouseWheel>", wheel)

        def counted():
            if callable(self.windows.get("icons_count")):
                self.windows["icons_count"]()

        def section(title):
            self._cap(body, title, bg="bg").pack(fill="x", pady=(0, 6))
            grid = self._skin(tk.Frame(body), bg="bg")
            grid.pack(fill="x", pady=(0, 16))
            return grid

        size = self.marks_px["row"]
        # (key, default name, mark spec or None). A string is a heading: apps
        # take two columns, the rest one, since each of those also carries
        # the Text / Both / Icon choice.
        rows = ["APPS AND TABS"]
        rows += [(spec["key"], name, spec) for spec, name in self._icon_specs()]
        rows.append("CONNECTIONS")
        rows += [(key, name, None) for key, name in self.CONN_NAMES]
        rows.append("BUTTONS")
        rows += [("glyph:" + g, label, None) for g, label in self.GLYPH_NAMES]
        rows += [("button:" + label, label, None) for label in labels]
        grid = i = cols = None
        states = {}                       # key -> its row's refresh, for renames
        self.windows["icon_rows"] = states    # and for a change from the sidebar
        for entry in rows:
            if isinstance(entry, str):
                grid, i, cols = section(entry), 0, 2 if entry == "APPS AND TABS" else 1
                continue
            key, name, spec = entry
            # Where Text / Both / Icon is saved: a button by its label, a
            # connection row by its key. Glyphs and apps have no choice.
            show_key = (key[len("button:"):] if key.startswith("button:")
                        else key if key.startswith("conn:") else None)
            cell = self._skin(tk.Frame(grid), bg="bg")
            cell.grid(row=i // cols, column=i % cols, sticky="w", padx=(0, 18), pady=3)
            i += 1
            if spec is not None:
                self._mark(cell, spec, size, bg="bg").pack(side="left")
                preview = None
            else:
                # A button's picture as it will be worn, or its glyph, or
                # nothing yet: a blank square the size of a mark.
                preview = tk.Label(cell, font=self.f_glyph, compound="center",
                                   bd=0, padx=0, pady=0)
                self._skin(preview, bg="bg", fg="faint")
                preview.pack(side="left")
            if key.startswith("glyph:"):
                # A glyph shows a character or a picture, never words.
                label = tk.Label(cell, text=clip(name, 22), font=self.f_ui,
                                 anchor="w", width=18)
                self._skin(label, bg="bg", fg="text")
                label.pack(side="left", padx=(8, 6))
            else:
                # The name, editable in place: Enter or leaving the field
                # saves it; emptied, the program's own name comes back.
                var = tk.StringVar(master=cell, value=self._name(key, name))
                field = tk.Entry(cell, textvariable=var, font=self.f_ui, width=20,
                                 relief="flat", bd=0, highlightthickness=1)
                self._skin(field, bg="card", fg="text", insertbackground="accent",
                           highlightbackground="border", highlightcolor="accent")
                field.pack(side="left", padx=(8, 6), ipady=self._px(3))

                def rename(_ev=None, key=key, name=name, var=var):
                    self._rename(key, var.get(), name)
                    var.set(self._name(key, name))
                    states.get(key, lambda: None)()
                field.bind("<Return>", rename)
                field.bind("<FocusOut>", rename)
            reset = self._button(cell, "Reset", lambda: None, kind="ghost")

            def changed(key=key, show_key=show_key):
                return (key in self.prefs.get("icons") or key in self.prefs.get("names")
                        or (show_key is not None
                            and show_key in self.prefs.get("button_show")))

            def state(key=key, reset=reset, preview=preview, changed=changed):
                try:
                    reset.set(state="normal" if changed() else "disabled")
                    if preview is not None:
                        photo = self._icon_photo(key, size)
                        glyph = (self.g.get(key[len("glyph:"):], "")
                                 if key.startswith("glyph:") else "")
                        # An empty image of the mark's size keeps an empty
                        # row as tall as the rest.
                        preview.config(image=photo or self._blank(size),
                                       text="" if photo else glyph)
                except tk.TclError:
                    pass                  # the window closed while an upload ran
                counted()

            def upload(key=key, name=name, state=state):
                self._pick_icon(key, name, parent=win, after=state)

            def find(key=key, name=name, state=state, logo=spec is not None):
                self._search_icons(key, name, parent=win, after=state,
                                   words=icon_search.words_for(name, logo))

            choice = {}

            def forget(key=key, name=name, state=state, show_key=show_key,
                       choice=choice, var=None if key.startswith("glyph:") else var):
                """Everything back as the program made it: icon, name, and
                what the button shows."""
                self._set_icon(key, None)
                self._rename(key, "", name)
                if show_key is not None:
                    self._set_button_show(show_key, "both")
                if var is not None:
                    var.set(name)
                if choice.get("light"):
                    choice["light"]()
                state()

            self._button(cell, "Upload...", upload).pack(side="left")
            self._button(cell, "Search...", find).pack(side="left", padx=(6, 0))
            reset.command = forget
            reset.pack(side="left", padx=(6, 0))
            if show_key is not None:
                choice["light"] = self._show_choice(cell, show_key, after=state)
            states[key] = state
            state()
        win.update_idletasks()
        fit()
        scroller.configure(height=min(body.winfo_reqheight(),
                                      int(self.winfo_screenheight() * 0.6)))

    def _show_choice(self, parent, label, after=None):
        """Text / Both / Icon for buttons labelled `label` (or a connection
        row's key), the chosen one in the accent. Icon alone still shows the
        text while there is no icon. Returns the function that relights it;
        `after` runs once a choice is saved."""
        line = self._skin(tk.Frame(parent), bg="bg")
        line.pack(side="left", padx=(14, 0))
        pills = {}

        def light():
            for mode, pill in pills.items():
                pill.roles = self.PILL_ROLES[
                    "accent" if self._button_show(label) == mode else "quiet"]
                pill.paint(self.C)

        def choose(mode):
            self._set_button_show(label, mode)
            light()
            if after is not None:
                after()

        for mode, word in BUTTON_SHOWS.items():
            # Small: three of these per row, beside Upload and Reset.
            pills[mode] = self._button(line, word, lambda m=mode: choose(m),
                                       font=self.f_small, padx=10, pady=3)
            pills[mode].pack(side="left", padx=(0, 4))
        light()
        return light

    def _blank(self, size):
        """A transparent size x size image: a Label given one is sized in
        pixels, so a row with no picture yet stands as tall as one with."""
        key = ("blank", size, 0)
        if key not in self.photos:
            self.photos[key] = tk.PhotoImage(width=size, height=size, master=self)
        return self.photos[key]

    ICON_TYPES = [("Pictures", "*.png *.ico *.jpg *.jpeg *.gif *.bmp *.tif *.tiff"),
                  ("All files", "*.*")]

    def _pick_icon(self, key, name, parent=None, after=None):
        """Ask for a picture and make it `key`'s icon: the Icons window's
        Upload... and a sidebar row's Change icon... both. A picture that
        will not read says why; `after` runs once it is settled, either way."""
        from tkinter import filedialog, messagebox
        parent = parent or self
        path = filedialog.askopenfilename(parent=parent, title="Icon for %s" % name,
                                          filetypes=self.ICON_TYPES)
        if not path:
            return

        def done(error):
            if error:
                messagebox.showerror("Icon not changed", error,
                                     parent=parent if parent.winfo_exists() else self)
            if after is not None:
                after()
        self._spawn(None, self._upload_icon, key, path, done)

    SEARCH_COLS = 6
    SEARCH_TILE = 72                      # a found picture's preview, before _px
    SEARCH_FETCHERS = 4                   # previews downloaded at once

    def _search_icons(self, key, name, parent=None, after=None, words=None):
        """Find an icon on the web (Wikimedia Commons, `core.icon_search`)
        and make the one clicked `key`'s icon, kept like an upload. Searching
        and downloading are off the UI thread; each preview is made on it,
        and dropped on it when the window closes - a PhotoImage freed on a
        worker is a Tk call there."""
        parent = parent or self
        wkey = ("icon_search", key)
        win = self.windows.get(wkey)
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            return
        win = tk.Toplevel(parent)
        self.windows[wkey] = win
        win.title("Find an icon for %s" % name)
        win.transient(parent)
        self._skin(win, bg="bg")
        outer = self._skin(tk.Frame(win), bg="bg")
        outer.pack(fill="both", expand=True, padx=22, pady=18)
        line = self._skin(tk.Frame(outer), bg="bg")
        line.pack(fill="x")
        field = tk.Entry(line, font=self.f_ui, width=36, relief="flat", bd=0,
                         highlightthickness=1)
        self._skin(field, bg="card", fg="text", insertbackground="accent",
                   highlightbackground="border", highlightcolor="accent")
        field.insert(0, words or icon_search.words_for(name))
        field.pack(side="left", fill="x", expand=True, ipady=self._px(3))
        status = tk.Label(outer, font=self.f_small, anchor="w", justify="left",
                          wraplength=self._px(560))
        self._skin(status, bg="bg", fg="faint")
        grid = self._skin(tk.Frame(outer), bg="bg")
        foot = self._skin(tk.Frame(outer), bg="bg")
        note = tk.Label(foot, font=self.f_small, anchor="w", justify="left",
                        wraplength=self._px(420),
                        text="From Wikimedia Commons, with each picture's licence "
                             "under it. A logo stays its maker's trademark.")
        self._skin(note, bg="bg", fg="faint")
        note.pack(side="left")
        self._button(foot, "Close", win.destroy, kind="accent").pack(side="right")
        status.pack(fill="x", pady=(8, 8))
        grid.pack(fill="both", expand=True)
        foot.pack(fill="x", pady=(14, 0))

        run = {"n": 0}                    # the search on show; older ones stop
        found = {}                        # tile -> the thumbnail's bytes
        tile_px = self._px(self.SEARCH_TILE)

        def say(text):
            try:
                status.config(text=text)
            except tk.TclError:
                pass

        def gone(_ev=None):
            run["n"] += 1
            for k in [k for k in self.photos
                      if isinstance(k, tuple) and k[:2] == ("found", key)]:
                del self.photos[k]

        win.bind("<Destroy>", lambda ev: gone() if ev.widget is win else None)

        def search(_ev=None):
            gone()
            n = run["n"]
            found.clear()
            for w in grid.winfo_children():
                w.destroy()
            text = field.get()
            if not text.strip():
                say("Type what to search for.")
                return
            say("Searching Wikimedia Commons...")
            self._spawn(None, look, text, n)

        def look(text, n):
            """Worker: the search, then every preview, a few at a time."""
            try:
                hits, why = icon_search.search(text), None
            except icon_search.SearchError as e:
                hits, why = [], str(e)
            self.q.put(("call", None, lambda: listed(n, hits, why)))
            if not hits:
                return

            def get(hit):
                if run["n"] != n:
                    return None
                try:
                    return icon_search.fetch(hit["thumb"])
                except icon_search.SearchError:
                    return None
            with ThreadPoolExecutor(self.SEARCH_FETCHERS) as pool:
                jobs = {pool.submit(get, hit): i for i, hit in enumerate(hits)}
                for job in as_completed(jobs):
                    data = job.result()
                    if data and run["n"] == n:
                        self.q.put(("call", None,
                                    lambda i=jobs[job], d=data: shown(n, i, d)))

        def listed(n, hits, why):
            if run["n"] != n or not win.winfo_exists():
                return
            if why:
                say(why)
            elif not hits:
                say("Nothing found. Try other words - the program's name, "
                    "or its maker's.")
            else:
                say("Click a picture to use it. Drawings first.")
            for i, hit in enumerate(hits):
                tile(i, hit)

        def tile(i, hit):
            cell = self._skin(tk.Frame(grid, highlightthickness=1, cursor="hand2"),
                              bg="bg", highlightbackground="bg")
            cell.grid(row=i // self.SEARCH_COLS, column=i % self.SEARCH_COLS,
                      padx=3, pady=3, sticky="n")
            pic = tk.Label(cell, image=self._blank(tile_px), bd=0)
            self._skin(pic, bg="bg")
            pic.pack(padx=4, pady=(4, 2))
            title = tk.Label(cell, text=clip(hit["title"], 14), font=self.f_small)
            self._skin(title, bg="bg", fg="text")
            title.pack()
            lic = tk.Label(cell, text=clip(hit["license"], 14), font=self.f_small)
            self._skin(lic, bg="bg", fg="faint")
            lic.pack(pady=(0, 4))
            cell.pic = pic
            for w in (cell, pic, title, lic):
                w.bind("<Enter>", lambda ev: cell.config(
                    highlightbackground=self.C["accent"]))
                w.bind("<Leave>", lambda ev: cell.config(
                    highlightbackground=self.C["bg"]))
                w.bind("<Button-1>", lambda ev: keep(i, hit))

        def shown(n, i, data):
            """A preview: Tk reads any PNG or GIF, and shrinks it whole times."""
            if run["n"] != n or not win.winfo_exists():
                return
            cells = grid.winfo_children()
            if i >= len(cells):
                return
            try:
                full = tk.PhotoImage(data=base64.b64encode(data).decode("ascii"),
                                     master=self)
                step = max(1, -(-max(full.width(), full.height()) // tile_px))
                photo = full.subsample(step) if step > 1 else full
                self.photos[("found", key, n, i)] = photo
                cells[i].pic.config(image=photo)
            except tk.TclError:
                return                    # not a picture Tk reads: left blank
            found[i] = data

        def keep(i, hit):
            say("Using %s..." % hit["title"])
            self._spawn(None, take, hit, found.get(i))

        def take(hit, data):
            """Worker: the picture to a temp file, kept as an upload is."""
            try:
                data = data or icon_search.fetch(hit["thumb"])
                fd, path = tempfile.mkstemp(
                    suffix=".gif" if data[:3] == b"GIF" else ".png")
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
            except (icon_search.SearchError, OSError) as e:
                why = str(e)
                self.q.put(("call", None, lambda: say(why)))
                return
            try:
                self._upload_icon(key, path, taken)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

        def taken(error):
            if error:
                say("Icon not changed: %s" % error)
                return
            try:
                win.destroy()
            except tk.TclError:
                pass
            if after is not None:
                after()

        self._button(line, "Search", search).pack(side="left", padx=(8, 0))
        field.bind("<Return>", search)
        field.focus_set()
        search()

    def _upload_icon(self, key, path, done):
        """Worker: make `path` the icon for `key`. The picture is kept as a
        square PNG under the icons folder, named by its bytes so a new upload
        is never mistaken for the old one; `done(error)` runs on the UI thread."""
        try:
            data = icons.upload_png(path)
            name = "%s-%s.png" % (re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:40],
                                  hashlib.sha1(data).hexdigest()[:10])
            os.makedirs(self._icons_dir(), exist_ok=True)
            with open(os.path.join(self._icons_dir(), name), "wb") as f:
                f.write(data)
        except (OSError, ValueError) as e:
            why = str(e)                  # `e` itself is gone once the block ends
            self.q.put(("call", None, lambda: done(why)))
            return

        def keep():
            self._set_icon(key, name)
            done(None)
        self.q.put(("call", None, keep))

    def _set_icon(self, key, name):
        """Save `key`'s uploaded icon (None to go back to the app's own), put
        the badge up until the new picture is read, and read it."""
        got = dict(self.prefs.get("icons"))
        old = got.pop(key, None)
        if name:
            got[key] = name
        self.prefs.set(icons=got)
        if old and old != name:
            try:
                os.remove(os.path.join(self._icons_dir(), old))
            except OSError:
                pass
        # Changed from a sidebar row, an open Icons window or Preferences
        # would otherwise show the old state until reopened.
        for refresh in (self.windows.get("icon_rows", {}).get(key),
                        self.windows.get("icons_count")):
            if callable(refresh):
                refresh()
        if key.startswith(("button:", "glyph:", "conn:")):
            # Drawn by the UI thread from the kept file, not read from an .exe.
            for k in [k for k in self.button_photos if k[0] == key]:
                del self.button_photos[k]
            self._repaint_buttons()
            return
        # (icon key, size); the discs' keys are 3-tuples and stay
        for k in [k for k in self.photos
                  if isinstance(k, tuple) and len(k) == 2 and k[0] == key]:
            del self.photos[k]
        self._redraw_marks(key)
        self._spawn(None, self._read_icons, key, got)

