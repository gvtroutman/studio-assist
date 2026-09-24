#!/usr/bin/env python3
"""The look: palette roles, and the primitives drawn from them.

Split out of `studio_chat` because it is the one part of the window that is
about appearance alone. Widgets are built against role names, never literal
hex - that is what lets Preferences repaint the running window instead of
rebuilding it, and a rebuild would throw away every transcript.

This module does import tkinter (`Pill` is a Canvas), unlike `studio_doctor`
and `studio_files`.
"""

import math
import tkinter as tk


# ------------------------------------------------------------------ appearance
# Widgets are built against role names, never literal hex, and every one of them
# is registered in `Chat.skin`. That is what makes Preferences able to repaint
# the running window instead of rebuilding it - a rebuild would throw away every
# transcript. Anything that puts a colour on the queue sends a role name too.
DARK = {
    "bg": "#141413", "side": "#1a1a18", "head": "#1a1a18", "card": "#232321",
    "hover": "#262623", "border": "#302f2c", "text": "#ecebe8",
    "muted": "#928d86", "faint": "#6b6862", "accent": "#d97757",
    "accent_dk": "#c26343", "accent_fg": "#16150f", "ok": "#5fb87f",
    "warn": "#e0a458", "err": "#e0685c", "sel": "#3d3b37", "code": "#d7d1c9",
    "asst": "#8fb0c9",
}

# Neutral, not warm: a near-white canvas with true white surfaces on it, and the
# orange as the one colour in the window. The earlier beige greys read as dated.
LIGHT = {
    "bg": "#f7f7f8", "side": "#ffffff", "head": "#ffffff", "card": "#ffffff",
    "hover": "#efeff1", "border": "#e4e4e7", "text": "#18181b",
    "muted": "#52525b", "faint": "#71717a", "accent": "#c2582f",
    "accent_dk": "#a44821", "accent_fg": "#ffffff", "ok": "#2f7d52",
    "warn": "#96650f", "err": "#b23b30", "sel": "#e4e4ea", "code": "#3f3f46",
    "asst": "#2c6a91",
}

THEMES = {"dark": DARK, "light": LIGHT}
THEME_NAMES = [("dark", "Dark"), ("light", "Light")]


def blend(a, b, t):
    """Hex colour `a` moved `t` of the way to `b`. Dots that breathe and bands
    that sweep need the shades between two palette roles, and a palette holds
    the ends only."""
    # Both channel lists are built eagerly. A generator expression per colour
    # reads more neatly and is wrong: it closes over the comprehension's loop
    # variable, so by the time `zip` draws from either one both yield `b`, and
    # every blend in the window silently comes out as its second colour. That
    # cost an afternoon - the dots stopped pulsing, the shimmer vanished and
    # the arc's ghost went the colour of the panel, all without an error.
    t = min(1.0, max(0.0, t))
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(
        int(round(x + (y - x) * t)) for x, y in zip(ca, cb))


def pretty_host(url):
    """100.127.17.38:1234 - the scheme and /v1 are noise in a 236px rail."""
    s = url.replace("https://", "").replace("http://", "").rstrip("/")
    return s[:-3].rstrip("/") if s.endswith("/v1") else s


def clip(s, n):
    """Truncate rather than let a label wrap mid-word."""
    return s if len(s) <= n else s[:n - 1] + "…"


def rounded(canvas, x1, y1, x2, y2, r, **kw):
    """Rounded rectangle - Tk's canvas has no primitive for it."""
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


def lifted(canvas, w, h, r, fill, under, tags):
    """A rounded surface raised off `under` by a two-pixel shadow instead of
    a ring round it. The shadow is `under` darkened, not a palette role, so it
    is the same faint step in either theme; the surface stops two pixels short
    of `h` to leave room for it, which callers' insets already clear."""
    for drop, k in ((2, 0.035), (1, 0.07)):
        edge = blend(under, "#000000", k)
        rounded(canvas, 0, drop, w, h - 2 + drop, r, fill=edge, outline=edge,
                tags=tags)
    rounded(canvas, 0, 0, w, h - 2, r, fill=fill, outline=fill, tags=tags)


# The "still working" dots: three discs that hop in a wave, after Motion's
# jumping-dots loader - each rises and falls with an ease in and out, mirrored,
# and each starts a beat after the one on its left. In frames of the one
# animation tick (70ms) rather than seconds, because that tick is all there is.
JUMP_FRAMES = 6                           # one rise, or one fall: about 0.4s, twice Motion's pace
JUMP_STAGGER = 2                          # frames between one dot and the next: about 0.14s


def jumps(frame, n=3):
    """How high each dot is at `frame`: 0 sitting on the line, 1 at the top."""
    out = []
    for i in range(n):
        t = ((frame - i * JUMP_STAGGER) % (2 * JUMP_FRAMES)) / float(JUMP_FRAMES)
        t = 2.0 - t if t > 1.0 else t     # up, then the same way back down
        out.append(0.5 - 0.5 * math.cos(math.pi * t))
    return tuple(out)


def jump_metrics(font):
    """(disc size, stride, rise) for dots sitting beside text in `font`: a
    little heavier than a full stop, so they read as a mark and not as
    punctuation, and hopping a bit over their own height - Motion hops 1.5x,
    which does not fit on a button."""
    size = max(3, int(round(font.metrics("ascent") * 0.28)))
    return size, size + max(2, int(round(size * 0.6))), int(round(size * 1.2))


def jump_width(font):
    size, stride, _rise = jump_metrics(font)
    return 2 * stride + size


def draw_jumps(canvas, x, baseline, lift, disc, font):
    """Plot the dots at `lift` (from `jumps`) with their left edge at `x`,
    resting on `baseline`. `disc` is a ready image: a canvas oval this small
    comes out as an octagon, which is why the status dots are images too."""
    size, stride, rise = jump_metrics(font)
    for i, up in enumerate(lift):
        canvas.create_image(x + i * stride + size / 2.0,
                            baseline - size / 2.0 - up * rise, image=disc)


_KEEP = object()                          # `Pill.set`: leave this one as it is


class Pill(tk.Canvas):
    """
    A rounded button. Tk's Button is a rectangle and nothing on it bends, so
    this draws its own: a smoothed polygon with the label on top. `roles` are
    palette role names; `paint(C)` is called with the palette on every theme
    switch, and `set()` covers what _apply_status used to config() on the
    Button - the text and whether it takes clicks.
    """

    def __init__(self, parent, text, command, font, roles, padx=18, pady=6, r=12,
                 anchor="center", round=False, **kw):
        tk.Canvas.__init__(self, parent, highlightthickness=0, bd=0, cursor="hand2",
                           **kw)
        self.command, self.font, self.roles = command, font, roles
        self.padx, self.pady, self.r = padx, pady, r
        self.text, self.lit, self.C = text, False, None
        self.down = False
        # While the label describes something still happening, `lift` holds the
        # jumping dots' heights and they ride after the text. `disc(colour,
        # size)` is the window's image cache; this module draws no PNGs.
        self.lift, self.disc = None, None
        # "center" sizes itself to its label, the way a button does. "w" takes
        # whatever width its packer gives it, reads from the left and wraps
        # rather than running off the end - for a row that happens to be
        # clickable, like an option in the model's question form.
        self.anchor = anchor
        # `round`: never narrower than it is tall, and ends that are half
        # circles - a disc around a one-character label, a lozenge when the
        # label grows into a word ("Stop").
        self.round = round
        if anchor == "w":
            self.bind("<Configure>", lambda ev: self.C and self.paint(self.C))
        # Press and release, not one click: a button that only reacts once the
        # work has started reads as a button that missed the press. The dip is
        # drawn the moment the mouse goes down and the command runs on release,
        # which is what every other button on the machine does - including
        # letting go somewhere else to change your mind.
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Enter>", lambda ev: self._light(True, ev))
        self.bind("<Leave>", lambda ev: self._light(False, ev))

    @property
    def state(self):
        return str(tk.Canvas.cget(self, "state")) or "normal"

    def cget(self, key):
        """Reads like the Button it replaced: `text` and `state` answer."""
        return self.text if key == "text" else tk.Canvas.cget(self, key)

    def invoke(self):
        """Also like the Button it replaced. The question form's options are
        driven by this in the tests, and it has to go through the same guard a
        real click does - a settled form must send nothing, and that is the
        whole point of greying it."""
        if self.state == "normal":
            self.command()

    def _press(self, _ev):
        if self.state == "normal":
            self.down = True
            if self.C:
                self.paint(self.C)
        return "break"

    def _release(self, ev):
        was, self.down = self.down, False
        if self.C:
            self.paint(self.C)
        # Only inside: dragging off the pill before letting go cancels, the
        # way a button is meant to.
        if was and self.state == "normal" and 0 <= ev.x < self.winfo_width() \
                and 0 <= ev.y < self.winfo_height():
            self.command()
        return "break"

    def _light(self, on, ev=None):
        """Hover, and the press that survives it. Dragging off the pill lifts
        it; dragging back on with the button still held presses it again, the
        way a real button does, so a wobble between press and release is not
        a click thrown away."""
        self.lit = on
        held = ev is not None and bool(ev.state & 0x0100)   # button 1 still down
        self.down = on and held and self.state == "normal"
        if self.C:
            self.paint(self.C)

    def set(self, text=None, state=None, lift=_KEEP):
        if text is not None:
            self.text = text
        if lift is not _KEEP:
            self.lift = lift
        if state is not None:
            self.config(state=state)      # the canvas's own option
        if self.C:
            self.paint(self.C)

    def paint(self, C):
        self.C = C
        bg, fg, active, off, off_fg = (C[r] for r in self.roles)
        self.delete("all")
        fill = off if self.state != "normal" else active if self.lit else bg
        ink = off_fg if self.state != "normal" else fg
        # Pressed, the pill sits a pixel in on every side and its label a pixel
        # down. Nothing changes colour, so it reads as the button going in
        # rather than as a second hover state.
        d = 1 if self.down and self.state == "normal" else 0
        if self.anchor == "w":
            # The width is the packer's; the height is whatever the text needs
            # once wrapped, which is only known after it is laid out - so the
            # label goes down first and the shape is drawn behind it.
            w = max(self.font.measure("n") * 4, self.winfo_width())
            label = self.create_text(self.padx + d, self.pady + d, text=self.text,
                                     font=self.font, fill=ink, anchor="nw",
                                     width=max(1, w - 2 * self.padx))
            h = (self.bbox(label)[3] - self.bbox(label)[1]) + 2 * self.pady
            self.config(height=h)
            shape = rounded(self, d, d, w - d, h - d, self.r, fill=fill,
                            outline=fill)
            self.tag_lower(shape)         # drawn second, so it has to go under
        else:
            dots = self.lift is not None and self.disc is not None
            tw = self.font.measure(self.text)
            gap = self.font.measure(" ") if dots else 0
            trail = jump_width(self.font) + gap if dots else 0
            w = tw + trail + 2 * self.padx
            h = self.font.metrics("linespace") + 2 * self.pady
            if self.round:
                # Two circles and the band between them. A smoothed polygon
                # asked for a radius of half the height undershoots it by
                # about half again, and came out a rounded square.
                w = max(w, h)
                k = h - 2 * d
                for x in (d, w - d - k):
                    self.create_oval(x, d, x + k, h - d, fill=fill, outline=fill)
                self.create_rectangle(d + k / 2.0, d, w - d - k / 2.0, h - d,
                                      fill=fill, outline=fill)
                self.config(width=w, height=h)
            else:
                self.config(width=w, height=h)
                rounded(self, d, d, w - d, h - d, self.r, fill=fill, outline=fill)
            left = (w - tw - trail) / 2.0
            self.create_text(left + tw / 2.0 + d, h / 2 + d, text=self.text,
                             font=self.font, fill=ink)
            if dots:
                size = jump_metrics(self.font)[0]
                baseline = (h - self.font.metrics("linespace")) / 2.0                     + self.font.metrics("ascent") + d
                draw_jumps(self, left + tw + gap + d, baseline, self.lift,
                           self.disc(ink, size), self.font)
        self.config(cursor="hand2" if self.state == "normal" else "arrow")
