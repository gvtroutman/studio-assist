"""The palette: applying it, remembering which widgets drew which
colours so a theme switch is a re-read rather than a rebuild. Split
out of core/chat.py (docs/CODEMAP.md) as a mixin - every method still
reads self.* set up in Chat.__init__."""
import tkinter as tk

import core.ui as ui

from core.chat import THEMES


class ChatThemeMixin:
    def _skin(self, widget, **roles):
        """Remember which palette role each colour came from, so switching
        theme is a re-read of the same widgets rather than a rebuild."""
        self.skin[widget] = roles
        try:
            widget.config(**{k: self.C[v] for k, v in roles.items()})
        except tk.TclError:
            pass
        return widget

    def _forget(self):
        """Sweep out destroyed widgets. The app list is rebuilt on every pin,
        hide and theme change, and each rebuild leaves its rows behind."""
        for reg in (self.skin, self.dot_role, self.arcs, self.row_role):
            for widget in list(reg):
                try:
                    if not widget.winfo_exists():
                        del reg[widget]
                except tk.TclError:
                    reg.pop(widget, None)
        for key, entries in list(self.marks.items()):
            self.marks[key] = [e for e in entries if e[0].winfo_exists()]
        # Chips are rebuilt on every attach and every send, and each one
        # registers a repaint; swept here the list tracks the window instead
        # of the session's history of it.
        self.pills = [p for p in self.pills if p.winfo_exists()]
        self.repaints = [(w, d) for w, d in self.repaints if w.winfo_exists()]
        self.glyphs = [(w, n) for w, n in self.glyphs if w.winfo_exists()]
        self.dressed = [e for e in self.dressed if e[0].winfo_exists()]

    def _accent(self, colour):
        """Save the accent (None for the theme's own) and repaint with it."""
        self.prefs.set(accent=colour if ui.is_hex(colour) else None)
        self._theme(self.prefs.get("theme"))

    def _rounding(self, k):
        """Preferences > Corners. `ui.rounded` reads the factor as it draws,
        so a repaint is all it takes - the theme's repaint, which already
        reaches every drawn shape, plus the badges, which it leaves alone."""
        ui.ROUNDING = k
        self.prefs.set(rounding=k)
        self._theme(self.prefs.get("theme"))
        self._redraw_marks()

    def _text_size(self, k):
        """Preferences > Text size: resize the fonts, then lay out again what
        was measured against them - the rail's width, and (by the theme's
        repaint) every pill, chip and field."""
        self.prefs.set(text_size=k)
        self._scale_fonts(k)
        self._metrics()
        side = getattr(self, "side_frame", None)
        if side is not None and side.winfo_exists():
            side.config(width=self.side_w)
        self._theme(self.prefs.get("theme"))
        self._repaint_buttons()           # uploaded button pictures, at the new size
        self._fit_tabs()                  # labels grew or shrank under the strip

    def _redraw_marks(self, key=None):
        """Draw every app mark again (or only `key`'s): after the corners
        change, and when an uploaded icon arrives or is taken away."""
        for k, entries in list(self.marks.items()):
            if key is not None and k != key:
                continue
            for canvas, size, spec in entries:
                try:
                    if canvas.winfo_exists():
                        self._draw_mark(canvas, size, spec)
                except tk.TclError:
                    pass

    def _theme(self, name):
        if name not in THEMES:
            return
        self.C = ui.palette(name, self.prefs.get("accent"))
        self.prefs.set(theme=name)
        self.configure(bg=self.C["bg"])
        self._forget()
        for widget in list(self.skin):
            try:
                widget.config(**{k: self.C[v] for k, v in self.skin[widget].items()})
            except tk.TclError:
                self.skin.pop(widget, None)
        for canvas in list(self.dot_role):
            self._set_dot(canvas, self.dot_role[canvas])
        # Drawn canvases are repainted, never reconfigured - the arc is made of
        # palette colours it plotted itself, same as the app marks and the dots.
        for canvas in list(self.arcs):
            self._paint_arc(canvas)
        for pill in self.pills:           # `_forget` above swept both lists
            pill.paint(self.C)
        for _widget, draw in self.repaints:
            try:
                draw()
            except tk.TclError:
                pass
        self.composer_paint()
        for s in self.sessions.values():
            if s.view is not None:
                self._tags(s.view)
        for app_id, view in list(self.tool_views.items()):
            if view.winfo_exists():
                self._tool_tags(view)
            else:
                del self.tool_views[app_id]
        for app_id, view in list(self.log_views.items()):
            if view.winfo_exists():
                self._log_tags(view)
            else:
                del self.log_views[app_id]
        self._build_apps()                # rows carry their own hover colours
        for sid in self.order:
            self._paint_tab(sid)
        self._apply_status()
        self._sync_bridges()
        self._menus()                     # menu colours are set at build time
        if callable(self.windows.get("prefs_paint")):
            self.windows["prefs_paint"]()

    # ------------------------------------------------------------------- menus
