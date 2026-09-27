"""
studio_codeaddons_ui - the OpenCode tab's Add-ons window.

The coding agent's counterpart to the Image Studio's Add-ons (LoRAs per
model): a pill per kind - MCP servers, Plugins, Skills - above two tabs.

- **Installed** - what OpenCode is given. Each can be turned off (kept, left
  out of its config) or removed; an MCP server says whether the running
  OpenCode connected to it, and why not when it did not.
- **Catalog** - the MCP Registry, npm's OpenCode plugins, or a GitHub
  repository of skills, with a search. Install files the add-on here; an MCP
  server that needs a key or a setting asks for it first.

Everything takes effect when OpenCode (re)starts, so the footer offers the
restart once something changed. The engine is studio_codeaddons; network
work runs on threads and comes back through the window's queue as a "call"
event, dropped if this window closed or a newer search replaced it.
"""

import threading
import tkinter as tk
import webbrowser
from tkinter import filedialog

import apps.opencode.codeaddons as ca


class AddonsWindow:
    CONFIRM_MS = 4000

    def __init__(self, host, spec):
        self.host = h = host
        self.spec = spec
        self.kind = "mcp"
        self.tab = "installed"
        self.query = tk.StringVar(master=host)
        self.repo = tk.StringVar(master=host, value=ca.SKILL_REPOS[0])
        self.gen = 0
        self.cursor = None
        self.cards = []
        self.status_live = None           # what the running OpenCode says of its MCP servers
        self.changed = False              # something to restart OpenCode for
        self.armed = None
        self.open_form = None             # the catalog card whose Install form is open
        win = self.win = tk.Toplevel(host)
        win.title("OpenCode Add-ons")
        win.transient(host)
        h._skin(win, bg="bg")
        win.geometry("%dx%d" % (h._px(820), h._px(760)))
        win.minsize(h._px(620), h._px(520))
        win.protocol("WM_DELETE_WINDOW", self.close)
        head = self.frame(win)
        head.pack(side="top", fill="x", padx=h._px(16), pady=(h._px(14), 0))
        self.label(head, "Add-ons", font=h.f_title).pack(side="top", anchor="w")
        self.label(head, "What OpenCode is given beyond its own tools. Each takes effect "
                   "when OpenCode starts, and every step one takes that changes something "
                   "is still yours to allow.", "muted", h.f_small,
                   wraplength=h._px(760)).pack(side="top", fill="x")
        self.kind_row = self.frame(win)
        self.kind_row.pack(side="top", fill="x", padx=h._px(16), pady=(h._px(10), 0))
        self.tab_row = self.frame(win)
        self.tab_row.pack(side="top", fill="x", padx=h._px(16), pady=(h._px(8), 0))
        self.tools = self.frame(win)
        self.tools.pack(side="top", fill="x", padx=h._px(16), pady=(h._px(8), 0))
        foot = self.frame(win)
        foot.pack(side="bottom", fill="x", padx=h._px(16), pady=h._px(12))
        self.button(foot, "Close", self.close, kind="ghost").pack(side="right")
        self.btn_restart = self.button(foot, "Restart OpenCode", self.restart, kind="accent")
        self.msg = self.label(foot, "", "muted", h.f_small, wraplength=h._px(470))
        self.msg.pack(side="left", fill="x", expand=True)
        scroll, self.list = self.scrolled(win)
        scroll.pack(side="top", fill="both", expand=True, padx=h._px(16), pady=(h._px(8), 0))
        self.show()
        self.refresh_live()

    # ------------------------------------------------------------ plumbing
    def frame(self, parent, bg="bg"):
        return self.host._skin(tk.Frame(parent, bd=0, highlightthickness=0), bg=bg)

    def label(self, parent, text="", role="text", font=None, bg="bg", **kw):
        lbl = tk.Label(parent, text=text, font=font or self.host.f_ui, anchor="w",
                       justify="left", **kw)
        return self.host._skin(lbl, bg=bg, fg=role)

    def button(self, parent, text, command, kind="quiet", bg="bg", **kw):
        kw.setdefault("font", self.host.f_ui)
        kw.setdefault("padx", self.host._px(12))
        kw.setdefault("pady", self.host._px(4))
        kw.setdefault("r", self.host._px(10))
        return self.host._button(parent, text, command, kind=kind, bg=bg, **kw)

    def entry(self, parent, var=None, secret=False, bg="bg"):
        e = tk.Entry(parent, textvariable=var, font=self.host.f_ui, relief="flat",
                     highlightthickness=1, show="•" if secret else "")
        self.host._skin(e, bg=bg, fg="text", insertbackground="text",
                        highlightbackground="border", highlightcolor="accent")
        return e

    def scrolled(self, parent, bg="bg"):
        outer = self.frame(parent, bg)
        bar = tk.Scrollbar(outer, highlightthickness=0, bd=0, width=11)
        self.host._skin(bar, bg=bg, troughcolor=bg, activebackground="faint")
        bar.pack(side="right", fill="y")
        canvas = tk.Canvas(outer, highlightthickness=0, bd=0, yscrollcommand=bar.set)
        self.host._skin(canvas, bg=bg)
        canvas.pack(side="left", fill="both", expand=True)
        bar.config(command=canvas.yview)
        inner = self.frame(canvas, bg)
        item = canvas.create_window(0, 0, window=inner, anchor="nw")
        inner.bind("<Configure>", lambda ev: canvas.config(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda ev: canvas.itemconfig(item, width=ev.width))

        def wheel(ev):
            if canvas.winfo_height() < inner.winfo_height():
                canvas.yview_scroll(int(-ev.delta / 120), "units")
        outer.bind("<Enter>", lambda ev: canvas.bind_all("<MouseWheel>", wheel))
        outer.bind("<Leave>", lambda ev: canvas.unbind_all("<MouseWheel>"))
        inner.canvas = canvas
        return outer, inner

    def alive(self):
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def close(self):
        self.gen += 1
        if self.alive():
            self.win.destroy()

    def say(self, text, role="muted"):
        if self.alive():
            self.msg.config(text=text)
            self.host._skin(self.msg, bg="bg", fg=role)

    def later(self, fn):
        """From a worker thread: run `fn` on the UI thread while this window
        is open. Tk is not to be touched from any other thread."""
        self.host.q.put(("call", None, lambda: fn() if self.alive() else None))

    def work(self, fn, done, fail=None):
        """`fn()` on a thread; `done(result)` or `fail(error)` back here, unless
        a newer piece of work replaced it."""
        gen = self.gen

        def run():
            try:
                res = fn()
            except (ca.AddonError, OSError, ValueError, RuntimeError) as e:
                err = e
                self.later(lambda: gen == self.gen and (fail or self.failed)(err))
                return
            self.later(lambda: gen == self.gen and done(res))
        threading.Thread(target=run, daemon=True).start()

    def failed(self, err):
        self.say(str(err), "err")

    def state(self):
        return self.spec.state_dir

    # --------------------------------------------------------------- layout
    def show(self):
        h = self.host
        for row in (self.kind_row, self.tab_row, self.tools):
            for w in row.winfo_children():
                w.destroy()
        for kind in ca.KINDS:
            self.button(self.kind_row, ca.KIND_NAMES[kind], lambda k=kind: self.pick(kind=k),
                        kind="accent" if kind == self.kind else "quiet").pack(
                side="left", padx=(0, h._px(6)))
        for tab, text in (("installed", "Installed"), ("catalog", "Catalog")):
            self.button(self.tab_row, text, lambda t=tab: self.pick(tab=t),
                        kind="accent" if tab == self.tab else "ghost").pack(
                side="left", padx=(0, h._px(6)))
        if self.tab == "catalog":
            if self.kind == "skill":
                self.label(self.tools, "GitHub repository", "muted", h.f_small).pack(
                    side="left", padx=(0, h._px(6)))
                e = self.entry(self.tools, self.repo)
                e.pack(side="left", fill="x", expand=True, ipady=h._px(3))
                e.bind("<Return>", lambda ev: self.search())
                self.button(self.tools, "List skills", self.search).pack(
                    side="left", padx=(h._px(6), 0))
            else:
                e = self.entry(self.tools, self.query)
                e.pack(side="left", fill="x", expand=True, ipady=h._px(3))
                e.bind("<Return>", lambda ev: self.search())
                self.button(self.tools, "Search", self.search).pack(
                    side="left", padx=(h._px(6), 0))
            self.search()
        else:
            hand = {"mcp": "Add a server by hand…", "plugin": "Add a package by name…",
                    "skill": "Add a skill folder…"}[self.kind]
            self.button(self.tools, hand, self.add_by_hand, kind="ghost").pack(side="left")
            self.show_installed()
        self.paint_restart()

    def pick(self, kind=None, tab=None):
        self.kind = kind or self.kind
        self.tab = tab or self.tab
        self.gen += 1
        self.cursor = None
        self.open_form = None
        self.say("")
        self.show()

    def clear(self):
        for w in self.list.winfo_children():
            w.destroy()
        self.list.canvas.yview_moveto(0)

    def card(self, title, meta, description):
        h = self.host
        card = self.frame(self.list, "card")
        card.pack(side="top", fill="x", pady=(0, h._px(8)), ipady=h._px(4))
        body = self.frame(card, "card")
        body.pack(side="top", fill="x", padx=h._px(12), pady=(h._px(8), 0))
        self.label(body, title, font=h.f_bold, bg="card").pack(side="top", fill="x")
        if meta:
            self.label(body, meta, "faint", h.f_small, bg="card").pack(side="top", fill="x")
        if description:
            self.label(body, description[:400], "muted", h.f_small, bg="card",
                       wraplength=h._px(720)).pack(side="top", fill="x", pady=(h._px(2), 0))
        row = self.frame(card, "card")
        row.pack(side="top", fill="x", padx=h._px(12), pady=(h._px(6), h._px(2)))
        return card, body, row

    # ------------------------------------------------------------- installed
    def show_installed(self):
        self.clear()
        rows = [r for r in ca.load(self.state()) if r["kind"] == self.kind]
        if not rows:
            self.label(self.list, "None yet. The Catalog has %s to install."
                       % ca.KIND_NAMES[self.kind].lower(), "muted").pack(side="top", fill="x")
            return
        for r in rows:
            meta = self.installed_meta(r)
            card, body, row = self.card(r.get("title") or r["name"], meta,
                                        r.get("description", ""))
            live = self.live_line(r)
            if live:
                text, role = live
                self.label(body, text, role, self.host.f_small, bg="card",
                           wraplength=self.host._px(720)).pack(side="top", fill="x")
            k = ca.key(r)
            on = r.get("enabled", True)
            self.button(row, "Turn off" if on else "Turn on",
                        lambda k=k, on=on: self.toggle(k, not on), bg="card").pack(side="left")
            self.button(row, "Click again to remove" if self.armed == k else "Remove",
                        lambda k=k: self.remove(k), kind="ghost", bg="card").pack(
                side="left", padx=(self.host._px(6), 0))
            if str(r.get("source", "")).startswith("http"):
                self.button(row, "Page", lambda u=r["source"]: webbrowser.open(u),
                            kind="ghost", bg="card").pack(side="right")

    def installed_meta(self, r):
        bits = []
        if not r.get("enabled", True):
            bits.append("turned off")
        if r["kind"] == "mcp":
            cfg = r.get("config") or {}
            bits.append(" ".join(cfg.get("command") or []) if cfg.get("type") == "local"
                        else cfg.get("url", ""))
        elif r["kind"] == "plugin":
            bits.append(r.get("package", ""))
        else:
            bits.append(r.get("path", ""))
        return "  ·  ".join(b for b in bits if b)

    def live_line(self, r):
        """What the running OpenCode says of an MCP server, as (text, role)."""
        if r["kind"] != "mcp" or not r.get("enabled", True):
            return None
        if self.status_live is None:
            return ("OpenCode is not running; it will start this server when it does.", "faint")
        st = self.status_live.get(r["name"])
        if st is None:
            return ("Not loaded yet - restart OpenCode to start it.", "warn")
        status = st.get("status", "?")
        if status == "connected":
            return ("Connected. Each of its tools asks before it runs.", "ok")
        return ("%s%s" % (status.capitalize(), (": " + st["error"]) if st.get("error") else ""),
                "err")

    def refresh_live(self):
        url, key = self.spec_url(), self.spec.key_path
        self.work(lambda: ca.live_status(url, key), self.got_live, lambda e: None)

    def spec_url(self):
        import core.agent as eng
        return eng.OPENCODE_URL

    def got_live(self, status):
        self.status_live = status
        if self.tab == "installed":
            self.show_installed()
        self.paint_restart()

    def toggle(self, k, on):
        ca.set_enabled(self.state(), k, on)
        self.mark_changed("Turned %s. " % ("on" if on else "off"))
        self.show_installed()

    def remove(self, k):
        if self.armed != k:
            self.armed = k
            self.show_installed()
            self.win.after(self.CONFIRM_MS, lambda: self.alive() and self.armed == k
                           and self.disarm())
            return
        self.armed = None
        try:
            ca.remove(self.state(), k)
        except OSError as e:
            self.say(str(e), "err")
            return
        self.mark_changed("Removed. ")
        self.show_installed()

    def disarm(self):
        self.armed = None
        self.show_installed()

    def add_by_hand(self):
        if self.kind == "skill":
            folder = filedialog.askdirectory(parent=self.win, title="A folder with a SKILL.md")
            if not folder:
                return
            try:
                rec = ca.folder_skill(folder)
            except (ca.AddonError, OSError) as e:
                self.say(str(e), "err")
                return
            ca.add(self.state(), rec)
            self.mark_changed("Added %s. " % rec["name"])
            self.show_installed()
            return
        self.hand_form()

    def hand_form(self):
        """A small form at the top of the list: a name and a command line or
        URL for an MCP server, or an npm package name for a plugin."""
        h = self.host
        self.clear()
        card, body, row = self.card(
            "Add an MCP server" if self.kind == "mcp" else "Add a plugin",
            "A command line (npx -y some-server) or a URL (https://…/mcp)"
            if self.kind == "mcp" else "An npm package, optionally @version", "")
        name = tk.StringVar(master=h)
        line = tk.StringVar(master=h)
        if self.kind == "mcp":
            self.label(body, "Name", "faint", h.f_small, bg="card").pack(side="top", fill="x")
            self.entry(body, name).pack(side="top", fill="x", ipady=h._px(3))
        self.label(body, "Command or URL" if self.kind == "mcp" else "Package", "faint",
                   h.f_small, bg="card").pack(side="top", fill="x", pady=(h._px(4), 0))
        e = self.entry(body, line)
        e.pack(side="top", fill="x", ipady=h._px(3))
        e.focus_set()

        def add():
            try:
                if self.kind == "mcp":
                    words = line.get().split()
                    rec = ca.hand_mcp(name.get() or (words[-1] if words else ""), line.get())
                else:
                    pkg = line.get().strip()
                    if not pkg:
                        raise ca.AddonError("Type the package's name.")
                    base = pkg.rsplit("@", 1)[0] if pkg.count("@") > (1 if pkg.startswith("@")
                                                                     else 0) else pkg
                    rec = {"kind": "plugin", "name": base, "title": base, "package": pkg,
                           "description": "Added by hand.",
                           "source": "https://www.npmjs.com/package/" + base}
            except ca.AddonError as err:
                self.say(str(err), "err")
                return
            ca.add(self.state(), rec)
            self.mark_changed("Added %s. " % rec["name"])
            self.show_installed()

        self.button(row, "Add", add, kind="accent", bg="card").pack(side="left")
        self.button(row, "Cancel", self.show_installed, kind="ghost", bg="card").pack(
            side="left", padx=(h._px(6), 0))

    # --------------------------------------------------------------- catalog
    def search(self, more=False):
        self.gen += 1
        if not more:
            self.cursor = None
            self.cards = []
            self.clear()
        self.say("Looking" + "…")
        kind, query, cursor = self.kind, self.query.get().strip(), self.cursor
        if kind == "mcp":
            fn = lambda: ca.search_mcp(query, cursor)
        elif kind == "plugin":
            fn = lambda: ca.search_plugins(query, cursor or 0)
        else:
            repo = self.repo.get().strip()
            fn = lambda: (ca.list_skills(repo), None)
        self.work(fn, self.got_cards)

    def got_cards(self, res):
        cards, self.cursor = res
        self.cards += cards
        self.say("%d shown." % len(self.cards) if self.cards else "Nothing found.")
        self.paint_cards(cards)
        if self.kind == "skill":
            self.fetch_descriptions(cards)

    def paint_cards(self, cards):
        for w in self.list.winfo_children():
            if getattr(w, "is_more", False):
                w.destroy()
        have = ca.load(self.state())
        sources = {r.get("source") for r in have} | {ca.key(r) for r in have}
        for c in cards:
            self.paint_card(c, sources)
        if self.cursor:
            more = self.button(self.list, "More", lambda: self.search(more=True))
            more.is_more = True
            more.pack(side="top", pady=self.host._px(6))

    def paint_card(self, c, sources):
        h = self.host
        meta = []
        if c.get("version"):
            meta.append("v" + c["version"])
        if c["kind"] == "plugin" and c.get("downloads"):
            meta.append("%s downloads a month" % _count(c["downloads"]))
        if c["kind"] == "mcp":
            if c.get("registry_name") != c["title"]:
                meta.append(c.get("registry_name", ""))
            if not c["ways"]:
                meta.append("no way to run it from here")
        if c["kind"] == "skill":
            meta.append("%s  ·  %d files" % (c["repo"], len(c["files"])))
        card, body, row = self.card(c["title"], "  ·  ".join(m for m in meta if m),
                                    c.get("description", ""))
        c["_desc"] = body
        # An MCP server is known by its registry name (two servers can share
        # a short one); a plugin or skill by where it came from, or its name.
        mark = c.get("registry_name") if c["kind"] == "mcp" else c.get("url")
        installed = mark in sources or (c["kind"] != "mcp" and
                                        "%s:%s" % (c["kind"], c["name"]) in sources)
        if installed:
            self.label(row, "Installed", "ok", h.f_small, bg="card").pack(side="left")
        elif c["kind"] != "mcp" or c["ways"]:
            self.button(row, "Install", lambda: self.install(c, card, row), kind="accent",
                        bg="card").pack(side="left")
        if c.get("url"):
            self.button(row, "Page", lambda u=c["url"]: webbrowser.open(u), kind="ghost",
                        bg="card").pack(side="right")

    def fetch_descriptions(self, cards):
        """A skill's description is in its SKILL.md: one read each, filled in
        as they come."""
        for c in cards:
            def fill(front, c=c):
                c["description"] = front.get("description", "")
                body = c.get("_desc")
                if c["description"] and body is not None and body.winfo_exists():
                    self.label(body, c["description"][:400], "muted", self.host.f_small,
                               bg="card", wraplength=self.host._px(720)).pack(
                        side="top", fill="x", pady=(self.host._px(2), 0))
            self.work(lambda c=c: ca.skill_front(c), fill, lambda e: None)

    def install(self, c, card, row):
        if c["kind"] == "plugin":
            self.finish_install(ca.plugin_record(c))
        elif c["kind"] == "skill":
            self.say("Downloading %s…" % c["name"])
            state = self.state()
            self.work(lambda: ca.install_skill(state, c), self.finish_install)
        else:
            self.mcp_form(c, card, row)

    def mcp_form(self, c, card, row):
        """How to run it, and the values it needs - a key, a setting - before
        it is filed. With one way and nothing to ask, it is filed at once."""
        way_var = tk.IntVar(master=self.host, value=0)
        if len(c["ways"]) == 1 and not c["ways"][0]["fields"]:
            self.file_mcp(c, c["ways"][0], {})
            return
        if self.open_form is not None and self.open_form.winfo_exists():
            self.open_form.destroy()
        h = self.host
        form = self.open_form = self.frame(card, "card")
        form.pack(side="top", fill="x", padx=h._px(12), pady=(h._px(4), h._px(4)),
                  before=row)
        values = {}

        def paint():
            for w in form.winfo_children():
                w.destroy()
            if len(c["ways"]) > 1:
                for i, w in enumerate(c["ways"]):
                    rb = tk.Radiobutton(form, text=w["label"], variable=way_var, value=i,
                                        command=paint, anchor="w", font=h.f_small, bd=0,
                                        highlightthickness=0)
                    h._skin(rb, bg="card", fg="text", selectcolor="bg",
                            activebackground="card", activeforeground="text")
                    rb.pack(side="top", fill="x")
            way = c["ways"][way_var.get()]
            values.clear()
            for f in way["fields"]:
                title = f["name"] + ("" if f["required"] else "  (optional)")
                self.label(form, title, "faint", h.f_small, bg="card").pack(
                    side="top", fill="x", pady=(h._px(4), 0))
                if f["description"]:
                    self.label(form, f["description"][:200], "faint", h.f_small, bg="card",
                               wraplength=h._px(700)).pack(side="top", fill="x")
                var = tk.StringVar(master=h, value=f.get("default") or "")
                self.entry(form, var, secret=f["secret"]).pack(side="top", fill="x",
                                                                ipady=h._px(3))
                values[f["name"]] = var
            if any(f["secret"] for f in way["fields"]):
                self.label(form, "Kept in OpenCode's config in your profile, not in the "
                           "repository.", "faint", h.f_small, bg="card").pack(side="top",
                                                                             fill="x")
            self.button(form, "Install", lambda: self.file_mcp(
                c, c["ways"][way_var.get()], {k: v.get() for k, v in values.items()}),
                kind="accent", bg="card").pack(side="top", anchor="w", pady=(h._px(6), 0))

        paint()

    def file_mcp(self, c, way, values):
        try:
            rec = ca.mcp_record(c, way, values)
        except ca.AddonError as e:
            self.say(str(e), "err")
            return
        self.finish_install(rec)

    def finish_install(self, rec):
        ca.add(self.state(), rec)
        self.mark_changed("Installed %s. " % (rec.get("title") or rec["name"]))
        self.open_form = None
        if self.tab == "catalog":
            self.clear()
            self.paint_cards(self.cards)

    # --------------------------------------------------------------- restart
    def mark_changed(self, said):
        self.changed = True
        running = self.status_live is not None
        self.say(said + ("Restart OpenCode to use it." if running else
                         "It is used when OpenCode starts."), "ok")
        self.paint_restart()

    def paint_restart(self):
        if self.changed and self.status_live is not None:
            self.btn_restart.pack(side="right", padx=(0, self.host._px(6)))
        else:
            self.btn_restart.pack_forget()

    def restart(self):
        s = self.host.sessions.get(self.spec.id)
        if s is not None and s.busy:
            self.say("OpenCode's tab is working; restart it when the work is done.", "warn")
            return
        self.say("Restarting OpenCode…")
        self.btn_restart.pack_forget()

        def go():
            self.spec.launch()
            import time
            for _ in range(30):
                if self.spec.running():
                    break
                time.sleep(1)
            time.sleep(2)                   # give its MCP servers a moment to connect
            return ca.live_status(self.spec_url(), self.spec.key_path)

        def done(status):
            self.changed = False
            self.status_live = status
            self.say("OpenCode restarted with its add-ons.", "ok")
            if self.tab == "installed":
                self.show_installed()
            self.paint_restart()

        self.work(go, done, lambda e: self.say("Restart failed: %s" % e, "err"))


def _count(n):
    return ("%.1fM" % (n / 1e6) if n >= 1e6 else "%.0fk" % (n / 1e3) if n >= 1e4 else
            "%.1fk" % (n / 1e3) if n >= 1e3 else str(n))
