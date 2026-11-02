"""Compact tool-call cards. Exact arguments and results live in the transcript."""
import tkinter as tk

from core.activity import tool_presentation, progress_presentation
from core import ui


def fit_text(text, font, width):
    text = " ".join(str(text).split())
    if font.measure(text) <= width:
        return text
    if width < font.measure("…"):
        return ""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.measure(text[:mid] + "…") <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…"


class ToolCard(tk.Canvas):
    def __init__(self, parent, name, arguments, *, via, font, small, px, skin, repaint, colours, dots, toggle):
        super().__init__(parent, highlightthickness=0, bd=0, cursor="hand2", takefocus=1)
        self.name = name
        self.title, self.context, self.icon, self.source = tool_presentation(name, arguments)
        if via:
            self.context = "Step in %s%s" % (str(via).replace("_", " "),
                                             " · " + self.context if self.context else "")
        self.font, self.small, self.px = font, small, px
        self.indent = px(16) if via else 0
        self.status, self.progress, self.expanded = "running", "", False
        self.palette = None
        self.card_width = px(580)
        self.badges, self.fraction = [], None
        self.visible_badges = []
        self.configure(width=self.card_width, height=px(44))
        skin(self, bg="bg")
        self.dots = dots(self, bg="card", role="accent")
        self.dots.bind("<ButtonRelease-1>", lambda _e: toggle())
        self.bind("<ButtonRelease-1>", lambda _e: toggle())
        self.bind("<Return>", lambda _e: (toggle(), "break")[-1])
        self.bind("<space>", lambda _e: (toggle(), "break")[-1])
        self.bind("<Enter>", lambda _e: self._hover(True))
        self.bind("<Leave>", lambda _e: self._hover(False))
        self.hot = False
        repaint(self, lambda: self.paint(colours()))
        self.paint(colours())

    def _hover(self, hot):
        self.hot = hot
        if self.palette is not None:
            self.paint(self.palette)

    def resize(self, width):
        width = max(self.px(180), min(self.px(680), int(width)))
        if width != self.card_width:
            self.card_width = width
            if self.palette is not None:
                self.paint(self.palette)

    def set_progress(self, text):
        self.progress = str(text)
        self.badges, self.fraction = progress_presentation(text)
        if self.palette is not None:
            self.paint(self.palette)

    def set_status(self, status):
        self.status = status
        # Live timing belongs to a running call. Keep only reported counters
        # after it settles; the complete last progress message remains in details.
        self.badges = [(text, role) for text, role in self.badges
                       if any(word in text for word in ("file changed", "files changed", " steps", " decision"))]
        if self.dots is not None:
            self.dots.destroy()
            self.dots = None
        if self.palette is not None:
            self.paint(self.palette)

    def set_expanded(self, expanded):
        self.expanded = expanded
        if self.palette is not None:
            self.paint(self.palette)

    def _symbol(self, x, y, ink):
        p = self.px
        if self.icon == "search":
            self.create_oval(x - p(6), y - p(6), x + p(4), y + p(4), outline=ink, width=p(1.5))
            self.create_line(x + p(3), y + p(3), x + p(8), y + p(8), fill=ink, width=p(1.5))
        elif self.icon == "folder":
            self.create_line(x - p(8), y + p(6), x - p(8), y - p(6), x - p(2), y - p(6),
                             x, y - p(3), x + p(8), y - p(3), x + p(8), y + p(6),
                             x - p(8), y + p(6), fill=ink, width=p(1.5), joinstyle="round")
        elif self.icon == "file":
            self.create_rectangle(x - p(5), y - p(8), x + p(6), y + p(8), outline=ink, width=p(1.5))
            for dy in (-2, 2, 6):
                self.create_line(x - p(2), y + p(dy), x + p(3), y + p(dy), fill=ink)
        elif self.icon == "code":
            self.create_line(x - p(7), y - p(4), x - p(3), y, x - p(7), y + p(4),
                             fill=ink, width=p(1.5), joinstyle="round")
            self.create_line(x, y + p(5), x + p(7), y + p(5), fill=ink, width=p(1.5))
        else:
            self.create_oval(x - p(6), y - p(6), x + p(6), y + p(6), outline=ink, width=p(1.5))
            self.create_line(x - p(3), y, x + p(3), y, fill=ink, width=p(1.5))

    def paint(self, palette):
        self.palette = palette
        C, p, w = palette, self.px, self.card_width
        self.delete("all")
        role = {"running": "accent", "ok": "ok", "error": "err",
                "skipped": "muted", "unknown": "warn"}.get(self.status, "warn")
        word = {"running": "Working", "ok": "Done", "error": "Failed",
                "skipped": "Not run", "unknown": "Unconfirmed"}.get(self.status, "Unconfirmed")
        pad, line = p(12), self.font.metrics("linespace")
        subline = self.small.metrics("linespace")
        header_h = max(p(42), line + 2 * pad)
        context = (self.source + " · " if self.source and self.context else "") + self.context
        context_y = header_h - p(3)
        y = header_h + (subline + p(7) if context else 0)
        layouts = []
        chip_h, chip_gap = subline + p(8), p(6)
        if self.badges:
            x, y = pad, y + p(5)
            for text, ink_role in self.badges:
                shown = fit_text(text, self.small, w - 2 * pad - p(18))
                chip_w = self.small.measure(shown) + p(18)
                if x > pad and x + chip_w > w - pad:
                    x, y = pad, y + chip_h + chip_gap
                layouts.append((x, y, chip_w, shown, ink_role))
                x += chip_w + chip_gap
            y += chip_h + pad
        else:
            y += p(3) if not context else p(6)
        if self.fraction is not None:
            y += p(8)
        h = y
        self.configure(width=w, height=h)
        self.configure(bg=C["bg"])
        edge = C["accent"] if self.hot else C["border"]
        ui.rounded(self, 0, 0, w, h, p(11), fill=edge, outline=edge)
        ui.rounded(self, 1, 1, w - 1, h - 1, p(10), fill=C["card"], outline=C["card"])
        ink = C[role]
        icon_bg = ui.blend(ink, C["card"], .91)
        icon_x, icon_y = pad + p(14), header_h / 2
        ui.rounded(self, pad, icon_y - p(14), pad + p(28), icon_y + p(14), p(7),
                   fill=icon_bg, outline=icon_bg)
        self._symbol(icon_x, icon_y, ink)
        status_w, status_h = self.small.measure(word) + p(18), subline + p(8)
        status_x = w - pad - p(22) - status_w
        status_y = (header_h - status_h) / 2
        status_bg = ui.blend(ink, C["card"], .91)
        ui.rounded(self, status_x, status_y, status_x + status_w, status_y + status_h,
                   p(7), fill=status_bg, outline=status_bg)
        self.create_text(status_x + status_w / 2, header_h / 2, text=word, font=self.small,
                         fill=ink, tags="state")
        title_x = pad + p(40)
        title_end = status_x - p(10)
        if self.dots is not None:
            dw, dh = self.dots.winfo_reqwidth(), self.dots.winfo_reqheight()
            self.dots.place(x=status_x - dw - p(8), y=(header_h - dh) / 2)
            title_end -= dw + p(8)
        self.shown_title = fit_text(self.title, self.font, max(1, title_end - title_x))
        self.create_text(title_x, header_h / 2, text=self.shown_title, font=self.font,
                         fill=C["text"], anchor="w", tags="title")
        cx, cy = w - pad - p(6), header_h / 2
        points = ((cx - p(4), cy - p(2), cx, cy + p(2), cx + p(4), cy - p(2)) if self.expanded else
                  (cx - p(2), cy - p(4), cx + p(2), cy, cx - p(2), cy + p(4)))
        self.create_line(*points, fill=C["muted"], width=p(1.5), joinstyle="round")
        if context:
            self.create_text(title_x, context_y + subline / 2, anchor="w",
                             text=fit_text(context, self.small, w - title_x - pad), font=self.small,
                             fill=C["muted"], tags="context")
        self.visible_badges = [shown for _, _, _, shown, _ in layouts]
        for x, cy, chip_w, shown, ink_role in layouts:
            tint = ui.blend(C[ink_role], C["card"], .94)
            ui.rounded(self, x, cy, x + chip_w, cy + chip_h, p(6), fill=tint, outline=tint)
            self.create_text(x + p(9), cy + chip_h / 2, anchor="w", text=shown,
                             font=self.small, fill=C[ink_role], tags="progress")
        if self.fraction is not None:
            y = h - p(8)
            self.create_line(pad, y, w - pad, y, fill=C["border"], width=p(3), capstyle="round")
            if self.fraction > 0:
                self.create_line(pad, y, pad + (w - 2 * pad) * self.fraction, y,
                                 fill=C["accent"], width=p(3), capstyle="round")
