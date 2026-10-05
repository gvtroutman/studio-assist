"""The drawing primitives every window in this app is built from:
buttons, fields, switches, marks, dots, arcs, tooltips, menus. Split
out of core/chat.py (docs/CODEMAP.md) as a mixin - the toolkit nearly
every other mixin calls into."""
import base64
import math
import os
import tkinter as tk

import core.icons as icons
import core.ui as ui

from core.chat import Pill, rounded, blend, clip, SHADES, SWEEP_FRAMES, BUTTON_SHOWS


class ChatWidgetsMixin:
    def _button(self, parent, text, command, kind="quiet", bg="bg", font=None,
                **kw):
        """Every button in this window is a `Pill`. Tk's Button is a rectangle
        and nothing on it bends, so one square control among rounded ones is
        not a style choice, it is the only shape Tk would give. `Pill` draws
        its own, and registering it here is what lets `_theme` repaint it -
        a canvas that plotted its own palette colours cannot be told a new one
        with config()."""
        pill = Pill(parent, text, command, font or self.f_ui,
                    self.PILL_ROLES[kind], **kw)
        pill.disc = self._disc_colour     # for the jumping dots a "…" label draws
        self._skin(pill, bg=bg)
        self.pills.append(pill)
        pill.paint(self.C)
        return pill

    def _repaint_on_theme(self, widget, draw):
        """Register a shape that must be drawn again when the palette changes.

        `_theme` re-reads `skin` and reconfigures widgets, which is enough for
        anything whose colour is a Tk option. It is no use at all to a canvas
        that plotted palette colours into items of its own - a rounded card, a
        chip, a field's outline, the corners masked off a picture. The tab
        chips and the rail rows are already redrawn by `_paint_tab` and
        `_build_apps`; everything else that draws itself belongs here, and is
        swept by widget so a dead one cannot raise mid-switch."""
        self.repaints.append((widget, draw))

    def _entry(self, parent, var, bg="bg"):
        """A text field with a rounded outline: the entry itself on a canvas
        that draws the border, exactly as the composer's input sits inside
        one. Tk's own `highlightthickness` can only draw a rectangle, and a
        square field between rounded buttons is the one shape that gives the
        window away. Returns the entry; its canvas is `entry.master` and that
        is what the caller packs."""
        INSET, R = self._px(6), self._px(10)
        shell = tk.Canvas(parent, highlightthickness=0, bd=0)
        self._skin(shell, bg=bg)
        entry = tk.Entry(shell, textvariable=var, font=self.f_ui, bd=0,
                         highlightthickness=0)
        self._skin(entry, bg="card", fg="text", insertbackground="accent")
        item = shell.create_window(INSET, INSET, window=entry, anchor="nw")
        focused = {"on": False}

        def paint(_ev=None):
            w = shell.winfo_width()
            h = entry.winfo_reqheight() + 2 * INSET
            shell.config(height=h)
            shell.itemconfig(item, width=max(1, w - 2 * INSET))
            shell.delete("box")
            # At rest the field is a raised surface, not a box; the accent
            # ring comes back only to say where typing will go.
            if focused["on"]:
                edge = self.C["accent"]
                rounded(shell, 0, 0, w, h, R, fill=edge, outline=edge, tags="box")
                rounded(shell, 1, 1, w - 1, h - 1, R - 1, fill=self.C["card"],
                        outline=self.C["card"], tags="box")
            else:
                ui.lifted(shell, w, h, R, self.C["card"], self.C[bg], "box")
            shell.tag_lower("box")

        shell.bind("<Configure>", paint)
        entry.bind("<FocusIn>", lambda ev: (focused.__setitem__("on", True), paint()))
        entry.bind("<FocusOut>", lambda ev: (focused.__setitem__("on", False), paint()))
        self._repaint_on_theme(shell, paint)
        return entry

    def _switch(self, parent, var, command=None, bg="bg"):
        """A toggle switch bound to a `tk.BooleanVar`: a pill track with a
        knob that sits at whichever end is on. Drawn rather than a
        Checkbutton for the same reason `_entry` draws its own outline - a
        canvas that plotted the palette itself needs `_repaint_on_theme`,
        not `config()`, to follow a theme switch."""
        w, h = self._px(34), self._px(18)
        pad = self._px(2)
        c = tk.Canvas(parent, width=w, height=h, highlightthickness=0, bd=0,
                      cursor="hand2")
        self._skin(c, bg=bg)

        def paint():
            c.delete("all")
            on = bool(var.get())
            track = self.C["accent"] if on else self.C["faint"]
            rounded(c, 0, 0, w, h, h / 2, fill=track, outline=track)
            r = h / 2 - pad
            cx = w - pad - r if on else pad + r
            c.create_oval(cx - r, pad, cx + r, h - pad, fill=self.C["card"],
                         outline=self.C["card"])

        def toggle(_ev=None):
            var.set(not var.get())
            paint()
            if command:
                command()

        c.bind("<Button-1>", toggle)
        paint()
        self._repaint_on_theme(c, paint)
        return c

    def _cap(self, parent, text, bg="side"):
        lbl = tk.Label(parent, text=text, font=self.f_cap, anchor="w")
        self._skin(lbl, bg=bg, fg="faint")
        return lbl

    def _mark(self, parent, spec, size=26, bg="side"):
        """
        An app's mark: its real icon, read out of its own .exe, and the drawn
        two-letter badge until (or unless) that arrives. Registered by key so
        the icon can be swapped in from the loader thread.
        """
        c = tk.Canvas(parent, width=size, height=size, highlightthickness=0, bd=0)
        self._skin(c, bg=bg)
        self.marks.setdefault(spec["key"], []).append((c, size, spec))
        self._draw_mark(c, size, spec)
        return c

    def _draw_mark(self, c, size, spec):
        c.delete("all")
        photo = self.photos.get((spec["key"], size))
        if photo is not None:
            c.create_image(size / 2, size / 2, image=photo)
            return
        big = size >= self.marks_px["hero"]
        rounded(c, 1, 1, size - 1, size - 1, size * 0.22 if big else 7,
                fill=spec["bg"], outline=spec["bg"])
        c.create_text(size / 2, size / 2 + 1, text=spec["code"], fill=spec["fg"],
                      font=self.f_hero_badge if big else self.f_badge)

    def _dots(self, parent, bg="bg", role="muted", n=3):
        """Three dots jumping in a wave: something is still running, and this
        is where its answer will land. Embedded in the transcript, so it says
        it in the place the user is already looking rather than only in the
        header. The same motion as a status ending in an ellipsis
        (`ui.jumps`), at the transcript's dot size.

        Drawn as the status dot is - `_disc` renders a PNG because a canvas
        oval this small comes out as an octagon. A dot is a touch brighter the
        higher it is, which is what keeps a low one from reading as a stray
        full stop; every shade is cached by colour, so a run of these costs a
        handful of small images and not one per frame. The animation reads
        `self.C` live, which is what makes a theme switch mid-run correct
        without a repaint hook."""
        size = self.dot_px
        gap = size + self._px(5)
        rise = size                       # room above for the hop, and no more
        c = tk.Canvas(parent, width=gap * (n - 1) + size + 2,
                      height=size + rise + 2, highlightthickness=0, bd=0)
        self._skin(c, bg=bg)
        ground = rise + size / 2 + 1      # a resting dot's centre
        ids = [c.create_image(gap * i + size / 2 + 1, ground) for i in range(n)]

        def draw(frame):
            for i, (item, up) in enumerate(zip(ids, ui.jumps(frame, n))):
                c.coords(item, gap * i + size / 2 + 1, ground - up * rise)
                c.itemconfig(item, image=self._shade(role, bg, 0.55 + 0.45 * up,
                                                     size))
        self._animate(("dots", str(c)), draw)
        return c

    def _shade(self, role, bg, level, size):
        """A disc of `role` at `level` brightness over `bg`. Quantised to
        SHADES steps before it is blended, so the cache holds a fixed dozen
        images per pair of roles rather than a new one for every frame of
        every pulse - and at this size the steps are not tellable apart."""
        level = round(min(1.0, max(0.0, level)) * SHADES) / float(SHADES)
        return self._disc_colour(blend(self.C[bg], self.C[role], level), size)

    def _arc(self, parent, bg="side", role="faint"):
        """
        The bridges' mark: a span, drawn. MDL2's chain link says "a link" -
        two things fastened together - and that is not what a bridge is or
        what this row reports. An arc between two banks is, and it is the one
        shape in the window that can also show its own state: a bridge that is
        starting draws itself across from left to right, over and over, and
        stops as a finished span once it is up.

        Hand-plotted rather than a glyph, so it owes nothing to a font being
        installed; a drawn canvas is repainted on a theme switch rather than
        reconfigured, same as the app marks, which is what `self.arcs` is for.
        """
        w, h = self._px(18), self._px(13)
        c = tk.Canvas(parent, width=w, height=h, highlightthickness=0, bd=0)
        self._skin(c, bg=bg)
        self.arcs[c] = role
        self._paint_arc(c, role)
        return c

    def _paint_arc(self, c, role=None, span=1.0):
        """Draw the arc in the palette's `role`, `span` of the way across. The
        piers stay put while the deck is still being built, so a partial arc
        reads as one going up rather than as one that is broken."""
        role = self.arcs[c] = role or self.arcs.get(c, "faint")
        c.delete("all")
        w, h = int(c.cget("width")), int(c.cget("height"))
        pad, thick = self._px(1), max(1, self._px(1.6))
        base = h - self._px(2)            # the line the span lands on
        rise = self._px(6)                # shallow: a deck, not a rooftop
        colour = self.C[role]
        # The span it is going to be, behind the span it is: while the arc
        # draws itself across, the rest of the crossing is already faintly
        # there, which is what makes a partial one read as building rather
        # than as broken.
        ghost = blend(self.C[role], self.C["side"], 0.72)
        # Walked as a polyline rather than drawn as an arc item, because a
        # partial span is then simply a matter of stopping early. Shallow on
        # purpose: at 18 by 13 pixels a tall one reads as a chevron, and the
        # piers and abutments that would fix that only thicken the middle -
        # both were tried at size before this was settled on.
        steps = 28
        pts = [(pad + (w - 2 * pad) * (i / steps),
                base - rise * math.sin(math.pi * (i / steps)))
               for i in range(steps + 1)]
        c.create_line(*[xy for p in pts for xy in p], fill=ghost, width=thick,
                      smooth=True, capstyle="round")
        cut = max(2, int(round(len(pts) * min(1.0, max(0.0, span)))))
        c.create_line(*[xy for p in pts[:cut] for xy in p], fill=colour,
                      width=thick, smooth=True, capstyle="round")

    def _arc_state(self, c, role):
        """Paint the arc in `role`. A role it is not already in draws itself
        across once and settles: the row is told its state on every bridge
        event, and a span building is a better account of one coming up than
        a colour appearing fully formed. One pass, then the animation drops
        itself - there is no state this can sit in spinning, which matters
        because a bridge that never starts would otherwise animate for ever."""
        if self.arcs.get(c) == role:
            return
        self.arcs[c] = role
        start = self.anim_frame

        def draw(frame):
            step = frame - start + 1
            self._paint_arc(c, role, step / float(SWEEP_FRAMES))
            return step < SWEEP_FRAMES

        self._animate(("arc", str(c)), draw)

    def _dot(self, parent, role, size=None, bg="side"):
        size = size or self.dot_px
        c = tk.Canvas(parent, width=size + 2, height=size + 2, highlightthickness=0,
                      bd=0)
        self._skin(c, bg=bg)
        c.create_image(size / 2 + 1, size / 2 + 1, image=self._disc(role, size))
        self.dot_role[c] = role
        return c

    def _disc(self, role, size):
        """The status dot as an antialiased image - a canvas oval this small
        comes out as an octagon. Cached by colour, since a theme switch
        changes what every role means."""
        return self._disc_colour(self.C[role], size)

    def _disc_colour(self, colour, size):
        """The same disc, by literal colour rather than by role: a pulse needs
        the shades between two roles and the palette holds only the ends."""
        key = ("disc", colour, size)
        photo = self.photos.get(key)
        if photo is None:
            data = icons.disc_png(colour, size)
            photo = tk.PhotoImage(data=base64.b64encode(data).decode("ascii"),
                                  master=self)
            self.photos[key] = photo
        return photo

    def _set_dot(self, canvas, role):
        self.dot_role[canvas] = role
        try:
            size = int(canvas.cget("width")) - 2
            canvas.itemconfig(1, image=self._disc(role, size))
        except tk.TclError:
            self.dot_role.pop(canvas, None)

    def _paint_app_dot(self, s):
        """The rail's dot for one app: its bridge's state, breathing while
        that tab is still starting up. `booting` and not the bridge role,
        because a tab can sit at "not started" indefinitely - the host coming
        back resets every stuck tab, and the ones you are not looking at wait
        to be selected. Those are settled, not busy, and must not animate."""
        dot = self.app_dots.get(s.id)
        if dot is not None:
            self._pulse_dot(dot, s.bridge[0], "side", s.booting)

    def _pulse_dot(self, canvas, role, bg, busy):
        """A status dot that breathes while its tab is working, and sits still
        the rest of the time. The dot already carries what the bridge is doing;
        this is the other half - whether that tab is mid-run - and it is worth
        saying on the tab itself, because the run you started is very often not
        the tab you are looking at now.

        `dot_role` still holds the settled role, so a theme switch repaints it
        correctly whether it is pulsing or not."""
        key = ("dot", str(canvas))
        if not busy:
            self._unanimate(key)
            self._set_dot(canvas, role)
            return
        self.dot_role[canvas] = role
        size = int(canvas.cget("width")) - 2
        self._animate(key, lambda frame: canvas.itemconfig(
            1, image=self._shade(role, bg, 0.3 + 0.7 * (
                0.5 + 0.5 * math.sin(frame * 0.3)), size)))

    def _glyph(self, parent, name, command, bg="side", fg="faint", tip=None):
        """A one-character button in Windows' icon font. Cheaper than an image
        and it stays crisp at any DPI, same reasoning as the drawn badges."""
        lbl = tk.Label(parent, text=self.g[name], font=self.f_glyph, cursor="hand2",
                       padx=3)
        self._skin(lbl, bg=bg, fg=fg)
        self._glyph_icon(lbl, name)
        lbl.bind("<Button-1>", lambda ev: (command(lbl), "break")[1])
        lbl.bind("<Enter>", lambda ev: lbl.config(fg=self.C["text"]))
        lbl.bind("<Leave>", lambda ev: lbl.config(fg=self.C[fg]))
        if tip:
            self._tip(lbl, tip)
        return lbl

    # The one-character buttons, by the name Preferences > Icons lists them
    # under. Each can wear an uploaded picture in place of its glyph.
    GLYPH_NAMES = [("add", "Add (+)"), ("close", "Close (x)"), ("pin", "Pin"),
                   ("unpin", "Unpin"), ("folder", "Attach folder"),
                   ("more", "More (v)"), ("update", "Update an app")]

    def _glyph_icon(self, lbl, name):
        """Show `name`'s uploaded picture on a glyph label instead of its
        character, or the character when there is none; remembered, so an
        upload, a Reset or a text size change can do it again."""
        self.glyphs.append((lbl, name))
        photo = self._icon_photo("glyph:" + name, self.f_glyph.metrics("linespace"))
        try:
            if photo is not None:
                lbl.config(image=photo, text="")
            else:
                lbl.config(image="", text=self.g[name])
        except tk.TclError:
            pass

    def _icon_photo(self, key, size):
        """The PhotoImage uploaded for a button (`button:<label>`) or glyph
        (`glyph:<name>`) at size x size, or None. Made on the UI thread from
        the kept 256px PNG and cached by file, so an upload is a new entry
        and never an old picture; a few dozen milliseconds the first time."""
        name = self.prefs.get("icons").get(key)
        if not name or size < 1:
            return None
        cache = (key, size, name)
        if cache not in self.button_photos:
            data = icons.sized_png(os.path.join(self._icons_dir(), name), size)
            try:
                self.button_photos[cache] = data and tk.PhotoImage(
                    data=base64.b64encode(data).decode("ascii"), master=self)
            except tk.TclError:
                self.button_photos[cache] = None
        return self.button_photos[cache]

    def _button_icon(self, label, size):
        """`Pill.icon`: every button with this label wears the same picture."""
        return self._icon_photo("button:" + label.strip(), size) if label else None

    # What every Pill asks of its window (`self._root()`) as it paints: its
    # uploaded icon, whether to show text, both or the icon, and its new name.
    def pill_icon(self, label, size):
        return self._button_icon(label, size)

    def pill_show(self, label):
        return self._button_show(label)

    def pill_rename(self, label):
        return self._renamed("button:" + (label or "").strip())

    def _button_show(self, label):
        """`Pill.show`: "text", "both" (the default) or "icon" for this label."""
        return self.prefs.get("button_show").get((label or "").strip(), "both")

    def _renamed(self, key):
        """What the user renamed `key` to in the Icons window, or None."""
        return self.prefs.get("names").get(key)

    def _name(self, key, default):
        """`key`'s name as shown: the user's words, else the program's."""
        return self._renamed(key) or default

    def _rename(self, key, text, default):
        """Save a rename from the Icons window - blank, or the default itself,
        takes the rename away - and show it everywhere at once."""
        text = " ".join((text or "").split())[:60]
        got = dict(self.prefs.get("names"))
        if not text or text == default:
            got.pop(key, None)
        else:
            got[key] = text
        if got == self.prefs.get("names"):
            return
        self.prefs.set(names=got)
        self._renamed_everywhere()

    def _dress(self, lbl, key, default, clip_n=None, icon=False):
        """A plain label that shows `key`'s name (renamed or not) and, with
        `icon`, the picture uploaded for it before the words - text, both or
        the icon alone, like a button. Remembered, so a rename, upload or
        text size change can dress it again."""
        self.dressed.append((lbl, key, default, clip_n, icon))
        text = self._name(key, default)
        if clip_n:
            text = clip(text, clip_n)
        photo = None
        if icon:
            photo = self._icon_photo(key, self.f_ui.metrics("linespace"))
            if photo is not None and self._button_show(key) == "text":
                photo = None
        try:
            if photo is not None:
                alone = self._button_show(key) == "icon"
                lbl.config(image=photo, compound="left",
                           text="" if alone else " " + text)
            else:
                lbl.config(image="", text=text)
        except tk.TclError:
            pass
        return lbl

    def _redress(self):
        """Every dressed label again, after a rename, an upload or a change of
        what a label shows."""
        live, self.dressed = self.dressed, []
        for entry in live:
            try:
                if entry[0].winfo_exists():
                    self._dress(*entry)
            except tk.TclError:
                pass

    def _renamed_everywhere(self):
        """A rename reaches the rail (rebuilt), the tab chips, the heroes and
        the connection rows (dressed), and every button (repainted)."""
        self._build_apps()
        self._repaint_buttons()           # and every dressed label
        for sid in self.order:
            self._paint_tab(sid)
        self._fit_tabs()

    def _set_button_show(self, label, mode):
        """Save what buttons with this label show, and redraw them."""
        got = dict(self.prefs.get("button_show"))
        if mode == "both":
            got.pop(label, None)          # the default is not worth a line in the file
        elif mode in BUTTON_SHOWS:
            got[label] = mode
        self.prefs.set(button_show=got)
        self._repaint_buttons()
        self._fit_tabs()

    def _repaint_buttons(self):
        """After an upload, a Reset, a rename or a text size change: every
        pill, glyph and dressed label drawn again with whatever picture and
        name it now has."""
        self._forget()
        for pill in self.pills:
            try:
                pill.paint(self.C)
            except tk.TclError:
                pass
        live, self.glyphs = self.glyphs, []
        for lbl, name in live:
            try:
                if lbl.winfo_exists():
                    self._glyph_icon(lbl, name)
            except tk.TclError:
                pass
        self._redress()

    def _tip(self, widget, text):
        """A plain tooltip - these controls are small and their meaning is not
        guessable from a 9pt glyph."""
        state = {"win": None}

        def show(_ev=None):
            if state["win"] or not widget.winfo_exists():
                return
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.configure(bg=self.C["border"])
            tk.Label(win, text=text, font=self.f_small, bg=self.C["card"],
                     fg=self.C["text"], padx=7, pady=3).pack(padx=1, pady=1)
            win.wm_geometry("+%d+%d" % (widget.winfo_rootx(),
                                        widget.winfo_rooty() + widget.winfo_height() + 4))
            state["win"] = win

        def hide(_ev=None):
            if state["win"]:
                state["win"].destroy()
                state["win"] = None

        widget.bind("<Enter>", show, add="+")
        widget.bind("<Leave>", hide, add="+")
        widget.bind("<Button-1>", hide, add="+")

    def _hook_click(self, widget, fn):
        """Children swallow clicks, so bind the whole subtree."""
        widget.bind("<Button-1>", fn)
        for child in widget.winfo_children():
            self._hook_click(child, fn)

    def _hover(self, row, widgets, base, lit, on=None, draw=None):
        """
        Light a whole row on hover. <Leave> also fires when the pointer moves
        onto a child, so check where it actually went before unlighting.
        `on`, if given, is told whether the row is lit - for controls that
        only appear while the pointer is over the row. `draw`, if given, is
        handed the role instead of a widget: a row whose highlight is a
        rounded shape has it drawn on a canvas, and a canvas cannot be told a
        colour it plotted itself with config().

        `row` takes the bindings whether or not it is in `widgets`, so a
        drawn row's own canvas - which must keep its background - still
        notices the pointer arriving over its margin.
        """
        def paint(role):
            for w in widgets:
                try:
                    w.config(bg=self.C[role])
                except tk.TclError:
                    pass
            if draw is not None:
                draw(role)
            if on is not None:
                on(role == lit)

        def leave(ev):
            under = row.winfo_containing(ev.x_root, ev.y_root)
            while under is not None:
                if under is row:
                    return
                under = getattr(under, "master", None)
            paint(base)

        for w in ([row] + list(widgets) if row not in widgets else widgets):
            w.bind("<Enter>", lambda ev: paint(lit), add="+")
            w.bind("<Leave>", leave, add="+")

    def _popup(self, menu, widget):
        """Post a menu under the control that opened it, and always let go."""
        try:
            menu.tk_popup(widget.winfo_rootx(),
                          widget.winfo_rooty() + widget.winfo_height())
        finally:
            menu.grab_release()

    def _menu_item(self, menu, label, key, command):
        """A menu row carrying the app's own icon, once we have read it."""
        extra = {}
        photo = self.photos.get((key, self.marks_px["menu"]))
        if photo is not None:
            extra = {"image": photo, "compound": "left"}
        menu.add_command(label=label, command=command, **extra)

    def _menu(self):
        return tk.Menu(self, tearoff=0, bg=self.C["card"], fg=self.C["text"],
                       activebackground=self.C["accent"],
                       activeforeground=self.C["accent_fg"], bd=0,
                       activeborderwidth=0)

    def _selectable(self, view, editable=False):
        """Text the user can highlight and copy. Tk already selects by dragging
        in a disabled Text, and a click gives it the focus Ctrl+C needs - what
        was missing is seeing it: `sel` is a hover grey, nearly the page's own
        colour in Light, and Windows' default selected text is white on it.
        So: the I-beam, `hilite` under `text`, kept while the menu is up, a
        right-click Copy / Select all, and Ctrl+A for all of it (Tk's own
        Ctrl+A goes to the start of the line). An `editable` Text keeps its
        own Cut / Paste and Ctrl+A, and only takes the colours."""
        roles = dict(self.skin.get(view, {}))
        roles.update(selectbackground="hilite", inactiveselectbackground="hilite",
                     selectforeground="text")
        self._skin(view, **roles)
        if editable:
            return view
        view.config(cursor="xterm")

        def select_all(_ev=None):
            view.tag_add("sel", "1.0", "end-1c")
            view.mark_set("insert", "1.0")
            return "break"

        def copy():
            view.event_generate("<<Copy>>")

        def menu(ev):
            view.focus_set()                  # <<Copy>> reads the focused Text
            m = self._menu()
            m.add_command(label="Copy", accelerator="Ctrl+C", command=copy,
                          state="normal" if view.tag_ranges("sel") else "disabled")
            m.add_command(label="Select all", accelerator="Ctrl+A", command=select_all)
            try:
                m.tk_popup(ev.x_root, ev.y_root)
            finally:
                m.grab_release()
            return "break"

        view.bind("<Button-3>", menu, add="+")
        view.bind("<Control-a>", select_all)
        return view

    # ---------------------------------------------------------------- tab strip
