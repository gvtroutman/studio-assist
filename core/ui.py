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
    "muted": "#928d86", "faint": "#6b6862", "accent": "#abb899",
    "accent_dk": "#939e84", "accent_fg": "#16150f", "ok": "#5fb87f",
    "warn": "#e0a458", "err": "#e0685c", "sel": "#3d3b37", "code": "#d7d1c9",
    "asst": "#8fb0c9", "strip": "#0b0b0a", "strip_hi": "#1e1e1c",
    "hilite": "#505649",
}

# `strip` is the band the tabs sit in, a step off `bg` so the selected tab -
# which is `bg`, the page's own colour - reads as part of the page below it;
# `strip_hi` is an unselected tab under the pointer. `hilite` is highlighted
# text, under the theme's own `text`: the accent faded 60% into `bg` (`palette`
# derives it again from a picked accent). `sel` is a hover grey - in Light it
# is nearly the page, and a selection drawn in it could not be seen.
# Neutral, not warm: a near-white canvas with true white surfaces on it, and the
# sage as the one colour in the window. The earlier beige greys read as dated.
# The sage is darkened here: #abb899 itself is too pale to read on white.
LIGHT = {
    "bg": "#f7f7f8", "side": "#ffffff", "head": "#ffffff", "card": "#ffffff",
    "hover": "#efeff1", "border": "#e4e4e7", "text": "#18181b",
    "muted": "#52525b", "faint": "#71717a", "accent": "#636b59",
    "accent_dk": "#565c4c", "accent_fg": "#ffffff", "ok": "#2f7d52",
    "warn": "#96650f", "err": "#b23b30", "sel": "#e4e4ea", "code": "#3f3f46",
    "asst": "#2c6a91", "strip": "#dfe3da", "strip_hi": "#ebeee7",
    "hilite": "#bcbfb8",
}

THEMES = {"dark": DARK, "light": LIGHT}
THEME_NAMES = [("dark", "Dark"), ("light", "Light")]


def is_hex(s):
    """A "#rrggbb" string - the only shape a saved accent may take."""
    return (isinstance(s, str) and len(s) == 7 and s[0] == "#"
            and all(c in "0123456789abcdefABCDEF" for c in s[1:]))


def palette(name, accent=None):
    """Theme `name` with the user's accent laid over it. The pressed shade and
    the text on the accent are derived, so one pick recolours every accent role
    and the label on a pale accent stays readable."""
    p = dict(THEMES.get(name, DARK))
    if is_hex(accent):
        accent = accent.lower()
        r, g, b = (int(accent[i:i + 2], 16) for i in (1, 3, 5))
        light = 0.299 * r + 0.587 * g + 0.114 * b > 150
        p["accent"] = accent
        p["accent_dk"] = blend(accent, "#000000", 0.14)
        p["accent_fg"] = "#16150f" if light else "#ffffff"
        p["hilite"] = blend(accent, p["bg"], 0.6)
    return p


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


def fit_chars(s, font, room):
    """The most characters of `s` whose `clip` fits `room` pixels in `font`
    (anything with Tk's `measure`); never fewer than four, a name's stub."""
    n = len(s)
    while n > 4 and font.measure(clip(s, n)) > room:
        n -= 1
    return max(n, 4)


# Preferences > Corners and Text size. Every corner `rounded` draws is scaled
# by ROUNDING, so one number reshapes the window; the text sizes scale the
# window's fonts (`Chat._text_size`). Both are sliders over these ranges, and
# a saved value outside one (a hand-edited settings file) falls back to 1.0.
ROUNDING = 1.0
ROUNDING_RANGE = (0.0, 2.0)               # square .. twice the designed corner
TEXT_RANGE = (0.8, 1.6)                   # of the size each font was made at


def in_range(value, bounds):
    """A saved slider value: a plain number inside `bounds` (True is not 1)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and bounds[0] <= value <= bounds[1])


def rounded(canvas, x1, y1, x2, y2, r, **kw):
    """Rounded rectangle - Tk's canvas has no primitive for it. The radius is
    the caller's times ROUNDING; grown past the caller's own, it stops at half
    the shape, where a smoothed polygon would otherwise fold over itself."""
    grown = r * ROUNDING
    if grown > r:
        grown = min(grown, max(r, min(x2 - x1, y2 - y1) / 2.0))
    r = grown
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    # Under a pixel of radius, smoothing only softens the corners into a
    # blur of a bevel; "Square" means square.
    return canvas.create_polygon(pts, smooth=r >= 1, **kw)


def tab_flare(r):
    """How far a browser tab's feet reach past its sides: its corner radius,
    scaled like every other corner. The canvas a tab is drawn on is this much
    wider each side than the tab itself."""
    return int(round(r * ROUNDING))


def browser_tab(canvas, x1, y1, x2, y2, r, **kw):
    """A browser tab: rounded on top, and at the bottom curving outward into
    the surface under it, so the tab and the page read as one piece. Plotted
    point by point - a smoothed polygon cannot tell a concave corner from a
    convex one. `r` is the caller's radius; the feet reach `tab_flare(r)`
    outside x1..x2, so the canvas has to leave that much room."""
    r = min(tab_flare(r), (y2 - y1) / 2.0, (x2 - x1) / 2.0)
    if r < 1:
        return canvas.create_polygon(x1, y1, x2, y1, x2, y2, x1, y2, **kw)
    steps = max(4, int(r))

    def arc(cx, cy, a0, a1):
        return [c for i in range(steps + 1)
                for c in (cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / steps)),
                          cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / steps)))]
    pts = (arc(x1 - r, y2 - r, 90, 0)           # left foot, curving up
           + arc(x1 + r, y1 + r, 180, 270)      # top left
           + arc(x2 - r, y1 + r, 270, 360)      # top right
           + arc(x2 + r, y2 - r, 180, 90))      # right foot, curving out
    return canvas.create_polygon(pts, **kw)


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
            # What the user set for this label in Preferences > Icons, asked of
            # the pill's own window (its Tk root): an icon the text's height,
            # drawn before the text and kept on the pill (Tk drops an image
            # nothing holds); whether to show "text", "both" or "icon" - icon
            # alone drops the words only when there is a picture; and the
            # words it was renamed to. `self.text` stays the program's own,
            # since that is what code compares. Asked of the root, not held on
            # the class: a class attribute pointed at whichever window was
            # made last, and kept a closed one's Tk images alive.
            root = self._root()
            show = getattr(root, "pill_show", lambda _t: "both")(self.text)
            icon = getattr(root, "pill_icon", None)
            self.image = (icon(self.text, self.font.metrics("linespace"))
                          if icon is not None and show != "text" else None)
            shown = getattr(root, "pill_rename", lambda _t: None)(self.text) or self.text
            words = shown if not (self.image and show == "icon") else ""
            tw = self.font.measure(words)
            lead = (self.image.width() + (self.font.measure(" ") if words else 0)
                    if self.image else 0)
            w = tw + trail + lead + 2 * self.padx
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
            left = (w - tw - trail - lead) / 2.0
            if self.image:
                self.create_image(left + d, h / 2 + d, image=self.image, anchor="w")
                left += lead
            self.create_text(left + tw / 2.0 + d, h / 2 + d, text=words,
                             font=self.font, fill=ink)
            if dots:
                size = jump_metrics(self.font)[0]
                baseline = (h - self.font.metrics("linespace")) / 2.0                     + self.font.metrics("ascent") + d
                draw_jumps(self, left + tw + gap + d, baseline, self.lift,
                           self.disc(ink, size), self.font)
        self.config(cursor="hand2" if self.state == "normal" else "arrow")


class Slider(tk.Canvas):
    """
    A horizontal slider whose handle is a pill. Tk's Scale draws a square
    block on a square trough and neither bends, so this draws its own: a
    thin round-capped track and a lozenge riding it. It answers what callers
    asked of the Scale it replaced - `variable`, `from`, `to`, `resolution`
    and `command` through `config`/`cget`, `get()` and `set()` - and `cget
    ("command")` is a Tcl command name, as the Scale's was. `roles` are the
    palette names of (track, handle, handle under the pointer, value text,
    disabled); `paint(C)` is called with the palette on every theme switch.

    As with the Scale, `set()` and a drag call `command` with the value as
    text, and writing the variable only moves the handle. A press off the
    handle jumps it there rather than stepping one resolution towards it.
    """

    OWN = ("variable", "from", "to", "resolution", "command", "showvalue")

    def __init__(self, parent, variable, from_, to, command=None, resolution=1.0,
                 length=100, thick=14, handle=26, track=4, showvalue=False,
                 font=None, roles=("border", "accent", "accent_dk", "muted", "faint"),
                 **kw):
        tk.Canvas.__init__(self, parent, highlightthickness=0, bd=0, width=length,
                           height=thick, cursor="hand2", **kw)
        self.var, self.lo, self.hi = variable, float(from_), float(to)
        self.res, self.showvalue, self.font = float(resolution), showvalue, font
        self.thick, self.handle, self.track, self.roles = thick, max(handle, thick), track, roles
        self.command, self.cmd_name = None, ""
        self._command(command)
        self.lit = self.dragging = False
        self.grab, self.C = 0.0, None
        self.trace = variable.trace_add("write", self._moved)
        self.bind("<Destroy>", self._gone, add="+")
        self.bind("<Configure>", lambda ev: self.C and self.paint(self.C))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Enter>", lambda ev: self._light(True))
        self.bind("<Leave>", lambda ev: self._light(False))
        self.bind("<Left>", lambda ev: self._step(-1))
        self.bind("<Right>", lambda ev: self._step(1))

    # ------------------------------------------------------------ as a Scale
    def _command(self, command):
        self.command = command
        self.cmd_name = self.register(command) if command else ""

    def configure(self, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        own = {k.rstrip("_"): kw.pop(k) for k in list(kw) if k.rstrip("_") in self.OWN}
        if not own:
            return tk.Canvas.configure(self, **kw)
        if kw:
            tk.Canvas.configure(self, **kw)
        if "variable" in own:
            self.var.trace_remove("write", self.trace)
            self.var = own["variable"]
            self.trace = self.var.trace_add("write", self._moved)
        self.lo = float(own.get("from", self.lo))
        self.hi = float(own.get("to", self.hi))
        self.res = float(own.get("resolution", self.res))
        self.showvalue = own.get("showvalue", self.showvalue)
        if "command" in own:
            self._command(own["command"])
        if self.C:
            self.paint(self.C)

    config = configure

    def cget(self, key):
        key = key.rstrip("_")
        if key == "variable":
            return str(self.var)
        if key == "command":
            return self.cmd_name
        if key in ("from", "to", "resolution", "showvalue"):
            return {"from": self.lo, "to": self.hi, "resolution": self.res,
                    "showvalue": self.showvalue}[key]
        return tk.Canvas.cget(self, key)

    __getitem__ = cget

    def get(self):
        try:
            return float(self.getvar(str(self.var)))
        except (tk.TclError, ValueError):
            return self.lo

    def set(self, value):
        """Round to the resolution, clamp to the range and, if that changed
        the value, write the variable and call `command` - as a drag does."""
        value = self._fit(value)
        if self._text(value) == self._text(self.get()):
            return
        text = self._text(value)
        self.setvar(str(self.var), text)
        if self.command is not None:
            self.command(text)

    def _fit(self, value):
        value = float(value)
        if self.res > 0:
            value = round(value / self.res) * self.res
        return min(max(value, min(self.lo, self.hi)), max(self.lo, self.hi))

    def _text(self, value):
        """The value as the Scale wrote it: as many decimals as the
        resolution has, so a whole-number slider fills an IntVar cleanly."""
        shown = "%g" % self.res if self.res > 0 else "0.01"
        places = len(shown.split(".")[1]) if "." in shown else 0
        return "%.*f" % (places, value)

    # ------------------------------------------------------------- the hand
    def _span(self):
        """(left end, usable width) of the handle's centre."""
        w = self.winfo_width() if self.winfo_width() > 1 else int(tk.Canvas.cget(self, "width"))
        return self.handle / 2.0, max(1.0, w - self.handle)

    def _x(self, value):
        left, room = self._span()
        t = 0.0 if self.hi == self.lo else (value - self.lo) / (self.hi - self.lo)
        return left + min(1.0, max(0.0, t)) * room

    def _value(self, x):
        left, room = self._span()
        return self.lo + (x - left) / room * (self.hi - self.lo)

    def _live(self):
        return str(tk.Canvas.cget(self, "state")) != "disabled"

    def _press(self, ev):
        if not self._live():
            return "break"
        self.focus_set()
        hx = self._x(self.get())
        # On the handle it keeps where it was taken; off it, it jumps there.
        self.grab = ev.x - hx if abs(ev.x - hx) <= self.handle / 2.0 else 0.0
        self.dragging = True
        self.set(self._value(ev.x - self.grab))
        self._repaint()
        return "break"

    def _drag(self, ev):
        if self.dragging:
            self.set(self._value(ev.x - self.grab))
        return "break"

    def _release(self, _ev):
        self.dragging = False
        self._repaint()
        return "break"

    def _light(self, on):
        self.lit = on
        self._repaint()

    def _step(self, sign):
        if self._live():
            step = self.res if self.res > 0 else abs(self.hi - self.lo) / 100.0
            self.set(self.get() + sign * step * (1 if self.hi >= self.lo else -1))
        return "break"

    def _moved(self, *_args):
        self._repaint()

    def _repaint(self):
        try:
            if self.C and self.winfo_exists():
                self.paint(self.C)
        except tk.TclError:
            pass

    def _gone(self, ev):
        """A variable outlives its slider (the Scene Builder rebuilds its
        panel on every pick), so the trace that names it must go with it."""
        if ev.widget is self:
            try:
                self.var.trace_remove("write", self.trace)
            except (tk.TclError, ValueError):
                pass

    def paint(self, C):
        self.C = C
        track, hand, hover, ink, off = (C[r] for r in self.roles)
        live = self._live()
        self.delete("all")
        top = self.font.metrics("linespace") + 2 if self.showvalue and self.font else 0
        self.config(height=top + self.thick)
        w = self.winfo_width() if self.winfo_width() > 1 else int(tk.Canvas.cget(self, "width"))
        mid = top + self.thick / 2.0
        left, room = self._span()
        self.create_line(left, mid, left + room, mid, width=self.track,
                         capstyle="round", fill=track)
        value = self.get()
        hx = self._x(value)
        fill = off if not live else hover if (self.lit or self.dragging) else hand
        # Two circles and the band between them, as `Pill(round=True)` draws:
        # a smoothed polygon asked for half-height corners undershoots them.
        k, x1 = self.thick, hx - self.handle / 2.0
        for x in (x1, x1 + self.handle - k):
            self.create_oval(x, top, x + k, top + k, fill=fill, outline=fill)
        self.create_rectangle(x1 + k / 2.0, top, x1 + self.handle - k / 2.0, top + k,
                              fill=fill, outline=fill)
        if top:
            text = self._text(self._fit(value))
            half = self.font.measure(text) / 2.0
            self.create_text(min(max(hx, half), w - half), top / 2.0, text=text,
                             font=self.font, fill=ink if live else off)
        self.config(cursor="hand2" if live else "arrow")
