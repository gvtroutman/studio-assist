"""A view cube for picking where the camera looks at a person from.

The user, 2026-09-29: "use angles as a preset and ask which way we want it to
look based on a cube like bambu studio has". The cube is the person: the
front face has a face drawn on it, and "right" and "left" are theirs. Each
face is cut in three both ways, so a click lands on one of 26 parts - the
middle of a face, a strip along an edge, a square at a corner - and that part
is a view (`breed.VIEW_KEYS`): the camera stands out along it. Drag turns the
cube to reach the far side; a click (a press that barely moved) picks or
drops that part.

The geometry is plain functions (`cells`, `basis`), the window `ViewCube`.
"""

import math
import tkinter as tk

import apps.image_studio.breed as sb
from core.ui import blend

INNER = 0.55        # half the width of a face's middle, the cube being -1..1
DRAG = 4            # px a press may move and still be a click
HOME = (35.0, 22.0) # the cube turned to show front, their right and top
PITCH_MAX = 80.0

LABELS = {(0, 0, -1): "BACK", (1, 0, 0): "RIGHT", (-1, 0, 0): "LEFT",
          (0, 1, 0): "TOP", (0, -1, 0): "BOTTOM"}      # front has the face


def _band(i):
    return {-1: (-1.0, -INNER), 0: (-INNER, INNER), 1: (INNER, 1.0)}[i]


def cells():
    """[(key, normal, corners)]: the 54 squares on the cube's faces, each the
    part (view key) it belongs to - 1 square for a face's middle, 2 for an
    edge, 3 for a corner."""
    out = []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for s in (1, -1):
            normal = tuple(s if a == axis else 0 for a in range(3))
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    key = [0, 0, 0]
                    key[axis], key[u], key[v] = s, i, j
                    (u0, u1), (v0, v1) = _band(i), _band(j)
                    corners = []
                    for cu, cv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                        p = [0.0, 0.0, 0.0]
                        p[axis], p[u], p[v] = float(s), cu, cv
                        corners.append(tuple(p))
                    out.append((tuple(key), normal, corners))
    return out


CELLS = cells()


def basis(yaw, pitch):
    """(toward the eye, screen right, screen up) for an eye at `yaw` degrees
    round to the person's right and `pitch` above - an eye in front (0, 0)
    sees their right on its left."""
    a, b = math.radians(yaw), math.radians(pitch)
    d = (math.sin(a) * math.cos(b), math.sin(b), math.cos(a) * math.cos(b))
    r = (-d[2], 0.0, d[0])                         # d x up
    n = math.hypot(r[0], r[2]) or 1.0
    r = (r[0] / n, 0.0, r[2] / n)
    up = (r[1] * d[2] - r[2] * d[1], r[2] * d[0] - r[0] * d[2],
          r[0] * d[1] - r[1] * d[0])               # r x d
    return d, r, up


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def facing(yaw, pitch):
    """The cells turned towards the eye, [(key, normal, corners, lit)]: a
    cube is convex, so the faces towards the eye never overlap."""
    d = basis(yaw, pitch)[0]
    return [(k, n, c, dot(n, d)) for k, n, c in CELLS if dot(n, d) > 1e-6]


class ViewCube(tk.Canvas):
    """The cube on a canvas. `chosen` is the list of picked view names, in the
    order picked; `on_change(chosen)` hears every pick and drop. `colors` is
    the host's palette, read at each draw so a theme change shows on the next."""

    def __init__(self, master, colors, size, chosen=(), on_change=None, font=None):
        super().__init__(master, width=size, height=size, bd=0, highlightthickness=0,
                         cursor="hand2")
        self.colors, self.size, self.font = colors, size, font
        self.chosen = [n for n in chosen if n in sb.VIEW_OF]
        self.on_change = on_change
        self.yaw, self.pitch = HOME
        self.hover = None
        self.press = None
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Motion>", lambda ev: self._hover(self.key_at(ev.x, ev.y)))
        self.bind("<Leave>", lambda ev: self._hover(None))
        self.draw()

    # ------------------------------------------------------------ drawing
    def _xy(self, p, r, up):
        k, c = self.size * 0.27, self.size / 2.0
        return c + k * dot(p, r), c - k * dot(p, up)

    def draw(self):
        C = self.colors() if callable(self.colors) else self.colors
        self.delete("all")
        self.config(bg=C["bg"])
        _, r, up = basis(self.yaw, self.pitch)
        picked = {sb.VIEW_OF[n] for n in self.chosen}
        line = blend(C["bg"], C["faint"], 0.45)      # shows on the dark and the light
        for key, normal, corners, lit in facing(self.yaw, self.pitch):
            if key in picked:
                fill = C["accent"]
            elif key == self.hover:
                fill = C["sel"]
            else:
                fill = blend(C["border"], C["card"], 0.35 + 0.65 * lit)
            pts = [v for p in corners for v in self._xy(p, r, up)]
            self.create_polygon(*pts, fill=fill, outline=line, width=1,
                                tags=("cell", "k%d_%d_%d" % key))
        for normal, word in LABELS.items():
            if dot(normal, basis(self.yaw, self.pitch)[0]) > 0.35:
                x, y = self._xy(normal, r, up)
                self.create_text(x, y, text=word, fill=C["muted"], font=self.font,
                                 state="disabled", tags="deco")
        if dot((0, 0, 1), basis(self.yaw, self.pitch)[0]) > 0.2:
            self._face(C, r, up)

    def _face(self, C, r, up):
        """Two eyes and a mouth on the front, so the cube reads as a head."""
        ink = C["text"]
        for ex in (-0.24, 0.24):
            x, y = self._xy((ex, 0.16, 1.0), r, up)
            e = max(2, self.size // 60)
            self.create_oval(x - e, y - e, x + e, y + e, fill=ink, outline="",
                             state="disabled", tags="deco")
        mouth = [v for t in range(7) for v in self._xy(
            (-0.22 + 0.44 * t / 6, -0.2 - 0.08 * math.sin(math.pi * t / 6), 1.0), r, up)]
        self.create_line(*mouth, fill=ink, width=max(1, self.size // 90), smooth=True,
                         state="disabled", tags="deco")

    # ------------------------------------------------------------ input
    def key_at(self, x, y):
        for item in reversed(self.find_overlapping(x, y, x, y)):
            for tag in self.gettags(item):
                if tag.startswith("k") and "_" in tag:
                    return tuple(int(v) for v in tag[1:].split("_"))
        return None

    def _hover(self, key):
        if key != self.hover:
            self.hover = key
            self.draw()

    def _press(self, ev):
        self.press = (ev.x, ev.y, self.yaw, self.pitch, False)

    def _drag(self, ev):
        if self.press is None:
            return
        x0, y0, yaw, pitch, moved = self.press
        if not moved and abs(ev.x - x0) + abs(ev.y - y0) < DRAG:
            return
        self.press = (x0, y0, yaw, pitch, True)
        self.yaw = (yaw + (ev.x - x0) * 0.6 + 180.0) % 360.0 - 180.0
        self.pitch = max(-PITCH_MAX, min(PITCH_MAX, pitch + (ev.y - y0) * 0.6))
        self.hover = None
        self.draw()

    def _release(self, ev):
        press, self.press = self.press, None
        if press is None or press[4]:
            return
        key = self.key_at(ev.x, ev.y)
        if key is not None:
            self.toggle(sb.view_name(key))

    def toggle(self, name):
        if name in self.chosen:
            self.chosen.remove(name)
        else:
            self.chosen.append(name)
        self.draw()
        if self.on_change:
            self.on_change(list(self.chosen))

    def set_chosen(self, names):
        self.chosen = [n for n in names if n in sb.VIEW_OF]
        self.draw()
        if self.on_change:
            self.on_change(list(self.chosen))

    def home(self):
        self.yaw, self.pitch = HOME
        self.draw()
