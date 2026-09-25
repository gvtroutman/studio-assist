#!/usr/bin/env python3
"""
studio_scene_ui - the Scene Builder window, over `studio_scene`.

One screen, laid out like a slicer's: the library and the scene's objects on
the left, the viewport in the middle with Move / Rotate / Scale over it, the
selected object's controls on the right, and Open / Save / Generate along the
bottom. The viewport always looks through the one camera, and the lit
rectangle in it is the frame the picture is made from; `studio_scene.png`
rasterises that same rectangle for the reference.

A collaborator of `studio_images_ui.ImageStudio`, like the character creator:
it borrows the tab's widget helpers (`frame`, `label`, `button`, `choice`,
`scrolled`, `slider`) and the window's `_entry` and palette roles, and it
generates through the tab - the scene's words go into the form's Scene field,
the frame into its Source reference, and the tab's own `generate` routes,
refuses, queues and records the job.

The mouse, in the viewport:
- left on an object: select it (a click on a hand selects the hand's
  controls) and drag it with the tool - Move slides it on its floor (Shift
  lifts it), Rotate turns it, Scale sizes it;
- left on empty floor or sky: orbit the camera;
- right (or middle): pan the camera; the wheel: in and out.
Keys, with the viewport focused: M, R, S for the tools, Delete.
"""

import copy
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import studio_scene as sc

FILETYPES = [("Scenes", "*.scene.json"), ("JSON", "*.json"), ("All files", "*.*")]
TOOLS = [("move", "Move", "m"), ("rotate", "Rotate", "r"), ("scale", "Scale", "s")]
GRID = "#9c978f"
GRID_AXIS = "#b1aca4"
SCENE_ROW = "Scene and camera"
REACH = 20                    # m either way an object can be placed


def rgb_hex(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


class SceneBuilder:
    def __init__(self, owner, path=None):
        self.owner = o = owner
        host = self.host = owner.host
        form_scene = owner.scene.get("1.0", "end").strip()
        self.scene = sc.new_scene(details=form_scene)
        self.path = None
        self.dirty = False
        self.sel = None               # an object's id; None is the scene and camera
        self.part = "body"
        self.tool = "move"
        self.drag = None
        self.vars = {}                # key -> (DoubleVar, read) for the inspector's sliders
        self.tool_pills = {}
        self.frame_rect = (0, 0, 1, 1, 1.0)      # ox, oy, w, h, scale on the canvas

        win = self.win = tk.Toplevel(host)
        win.title("Scene Builder")
        win.transient(host)
        host._skin(win, bg="bg")
        win.geometry("%dx%d" % (host._px(1240), host._px(780)))
        win.protocol("WM_DELETE_WINDOW", self.close)

        # Fixed before expanding (AGENTS.md: pack order): the foot, both
        # side columns, then the viewport takes what is left.
        foot = o.frame(win)
        foot.pack(side="bottom", fill="x", padx=o.px(12), pady=(0, o.px(12)))
        self.go = o.button(foot, "Generate", self.generate, kind="accent")
        self.go.pack(side="right")
        o.button(foot, "Save as…", self.save_as).pack(side="right", padx=(0, o.px(6)))
        o.button(foot, "Save", self.save).pack(side="right", padx=(0, o.px(6)))
        o.button(foot, "Open…", self.open).pack(side="right", padx=(0, o.px(6)))
        o.button(foot, "New", self.new, kind="ghost").pack(side="right", padx=(0, o.px(6)))
        self.model_row = o.frame(foot)
        self.model_row.pack(side="right", padx=(0, o.px(14)))
        self.msg = o.label(foot, "", "muted", host.f_small, wraplength=o.px(520))
        self.msg.pack(side="left", fill="x", expand=True)

        left = o.frame(win)
        left.config(width=o.px(190))
        left.pack_propagate(False)
        left.pack(side="left", fill="y", padx=o.px(12), pady=o.px(12))
        o.cap(left, "Library")
        for a in sc.ASSETS:
            o.button(left, "+  " + a["label"], lambda a=a["id"]: self.add(a),
                     anchor="w").pack(side="top", fill="x", pady=(0, o.px(3)))
        o.cap(left, "In the scene")
        self.lb = tk.Listbox(left, width=1, bd=0, highlightthickness=0, activestyle="none",
                             font=host.f_ui, exportselection=False)
        host._skin(self.lb, bg="card", fg="text", selectbackground="sel",
                   selectforeground="text")
        row = o.frame(left)
        row.pack(side="bottom", fill="x", pady=(o.px(6), 0))
        o.button(row, "Duplicate", self.duplicate).pack(side="left")
        o.button(row, "Delete", self.delete, kind="ghost").pack(side="left",
                                                                padx=(o.px(4), 0))
        self.lb.pack(side="top", fill="both", expand=True)
        self.lb.bind("<<ListboxSelect>>", lambda ev: self._picked())

        right_outer, self.panel = o.scrolled(win)
        right_outer.config(width=o.px(340))
        right_outer.pack_propagate(False)
        right_outer.pack(side="right", fill="y", pady=o.px(12), padx=(0, o.px(12)))

        mid = o.frame(win)
        mid.pack(side="left", fill="both", expand=True, pady=o.px(12))
        bar = o.frame(mid)
        bar.pack(side="top", fill="x", pady=(0, o.px(6)))
        for key, label, k in TOOLS:
            pill = o.button(bar, "%s  %s" % (label, k.upper()), lambda t=key: self.set_tool(t),
                            kind="accent" if key == self.tool else "quiet")
            pill.pack(side="left", padx=(0, o.px(4)))
            self.tool_pills[key] = pill
        o.button(bar, "Reset camera", self.reset_camera, kind="ghost").pack(side="right")
        o.label(mid, "Drag an object to move, rotate or size it · drag empty space to "
                "orbit · right-drag to pan · wheel to zoom", "faint", host.f_small).pack(
            side="bottom", fill="x", pady=(o.px(4), 0))
        self.canvas = tk.Canvas(mid, highlightthickness=0, bd=0, bg=rgb_hex(sc.SKY),
                                takefocus=1)
        self.canvas.pack(side="top", fill="both", expand=True)
        c = self.canvas
        c.bind("<Configure>", lambda ev: self.draw())
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._motion)
        c.bind("<ButtonRelease-1>", self._release)
        for b in ("2", "3"):
            c.bind("<ButtonPress-%s>" % b, self._pan_start)
            c.bind("<B%s-Motion>" % b, self._pan)
            c.bind("<ButtonRelease-%s>" % b, self._release)
        c.bind("<MouseWheel>", lambda ev: self._zoom(-1 if ev.delta > 0 else 1))
        c.bind("<Button-4>", lambda ev: self._zoom(-1))
        c.bind("<Button-5>", lambda ev: self._zoom(1))
        c.bind("<Key>", self._key)
        host._repaint_on_theme(c, self.draw)

        self._models()
        self._list()
        self._inspect()
        if path:
            self.open(path)
        else:
            self.status("Add a person or a prop from the library, then frame it with the "
                        "camera.", "muted")

    # ================================================================ state
    def status(self, text, role="muted"):
        self.msg.config(text=text)
        self.owner.skin(self.msg, bg="bg", fg=role)

    def obj(self, oid=None):
        oid = self.sel if oid is None else oid
        return next((x for x in self.scene["objects"] if x["id"] == oid), None)

    def changed(self, rebuild_list=False):
        """The scene changed: redraw, and say what the words will be."""
        self.dirty = True
        if rebuild_list:
            self._list()
        self.draw()
        self._words()

    def _title(self):
        name = os.path.basename(self.path) if self.path else "untitled"
        self.win.title("Scene Builder - %s%s" % (name, " *" if self.dirty else ""))

    # ============================================================== objects
    def add(self, asset_id):
        obj = sc.new_object(asset_id, self.scene["objects"])
        # Put it where the camera is looking, beside anything already there.
        t = self.scene["camera"]["target"]
        taken = len(self.scene["objects"])
        obj["position"] = [round(t[0] + (0.9 * taken if taken else 0), 2), 0.0,
                           round(t[2], 2)]
        self.scene["objects"].append(obj)
        self.select(obj["id"])
        self.changed(rebuild_list=True)
        self.status("%s added. %s" % (obj["name"], sc.ASSET[asset_id]["about"]), "muted")
        self.check()                  # says now, not at Generate, if the frame would go unused
        return obj

    def duplicate(self):
        src = self.obj()
        if src is None:
            return None
        obj = copy.deepcopy(src)
        fresh = sc.new_object(src["asset"], self.scene["objects"])
        obj["id"], obj["name"] = fresh["id"], fresh["name"]
        obj["position"] = [src["position"][0] + 0.8, src["position"][1], src["position"][2]]
        self.scene["objects"].append(obj)
        self.select(obj["id"])
        self.changed(rebuild_list=True)
        return obj

    def delete(self):
        obj = self.obj()
        if obj is None:
            return
        self.scene["objects"].remove(obj)
        self.select(None)
        self.changed(rebuild_list=True)
        self.status("%s removed." % obj["name"], "muted")

    def select(self, oid, part=None):
        if oid != self.sel or (part and part != self.part):
            self.sel = oid
            if part:
                self.part = part
            elif self.obj() is None or self.obj()["asset"] != "person":
                self.part = "body"
            self._list()
            self._inspect()
        self.draw()

    def _list(self):
        self.lb.delete(0, "end")
        self.lb.insert("end", SCENE_ROW)
        rows = [None] + [x["id"] for x in self.scene["objects"]]
        for x in self.scene["objects"]:
            self.lb.insert("end", "   " + (x["name"] or sc.ASSET[x["asset"]]["label"]))
        at = rows.index(self.sel) if self.sel in rows else 0
        self.lb.selection_clear(0, "end")
        self.lb.selection_set(at)
        self.lb.see(at)
        self._title()

    def _picked(self):
        sel = self.lb.curselection()
        if not sel:
            return
        rows = [None] + [x["id"] for x in self.scene["objects"]]
        self.select(rows[sel[0]] if sel[0] < len(rows) else None)

    # ============================================================ inspector
    def _clear(self):
        for w in self.panel.winfo_children():
            w.destroy()
        self.vars = {}
        self.words_label = None

    def _slider(self, parent, key, label, read, write, lo, hi, res=1.0):
        """A labelled slider over one number in the scene. `read` gets it,
        `write` sets it; `sync` puts the slider back after a drag moved it."""
        o = self.owner
        row = o.frame(parent)
        row.pack(side="top", fill="x")
        o.label(row, label, "muted", self.host.f_small, width=16).pack(side="left")
        var = tk.DoubleVar(value=read())

        def moved(_v):
            try:
                x = float(var.get())
            except (tk.TclError, ValueError):
                return
            if abs(x - read()) > 1e-9:
                write(x)
                self.changed()
        s = o.slider(row, var, lo, hi, command=moved)
        s.config(resolution=res)
        s.pack(side="left", fill="x", expand=True)
        self.vars[key] = (var, read)
        return s

    def sync(self):
        """Put every slider back to the scene's value (after a drag)."""
        for var, read in self.vars.values():
            try:
                if abs(var.get() - read()) > 1e-9:
                    var.set(read())
            except tk.TclError:
                pass
        self._words()

    def _text(self, parent, value, on_change, height=4):
        shell = self.owner.frame(parent, "card")
        shell.pack(side="top", fill="x")
        t = tk.Text(shell, height=height, wrap="word", bd=0, highlightthickness=0,
                    font=self.host.f_ui, padx=self.owner.px(8), pady=self.owner.px(6),
                    undo=True)
        self.owner.skin(t, bg="card", fg="text", insertbackground="accent",
                        selectbackground="sel")
        t.insert("1.0", value)
        t.pack(fill="x")
        t.bind("<KeyRelease>", lambda ev: on_change(t.get("1.0", "end").rstrip("\n")))
        return t

    def _inspect(self):
        self._clear()
        obj = self.obj()
        if obj is None:
            self._inspect_scene()
        else:
            self._inspect_object(obj)

    def _inspect_scene(self):
        o, p, s = self.owner, self.panel, self.scene
        cam = s["camera"]
        o.cap(p, "Scene details")

        def details(v):
            s["details"] = v
            self.changed()
        self.details = self._text(p, s["details"], details, height=5)
        o.label(p, "Where this is and what matters in it: the place, the light, the "
                "weather, the state of things.", "faint", self.host.f_small,
                wraplength=o.px(310)).pack(side="top", fill="x")

        o.cap(p, "Frame")
        o.choice(p, [(k, label) for k, label, _, _ in sc.FRAMES], s["frame"],
                 self._set_frame).pack(side="top", anchor="w")

        o.cap(p, "Camera")
        lens = o.frame(p)
        lens.pack(side="top", fill="x", pady=(0, o.px(4)))
        for mm in sc.LENSES:
            o.button(lens, "%dmm" % mm, lambda mm=mm: self._set_lens(mm),
                     kind="option").pack(side="left", padx=(0, o.px(4)))

        def put(key, conv=float):
            def write(x):
                cam[key] = conv(x)
            return write
        self._slider(p, "lens", "Lens (mm)", lambda: cam["lens"], put("lens"), 14, 135)
        self._slider(p, "yaw", "Orbit", lambda: cam["yaw"], put("yaw"), 0, 359)
        self._slider(p, "pitch", "Look down", lambda: cam["pitch"], put("pitch"), -30, 80)
        self._slider(p, "distance", "Distance (m)", lambda: cam["distance"],
                     put("distance"), 0.5, 25, 0.1)

        def aim(i):
            def write(x):
                cam["target"][i] = x
            return write
        self._slider(p, "aim_y", "Aim height (m)", lambda: cam["target"][1], aim(1),
                     0, 3, 0.05)

        o.cap(p, "Redraw strength")
        self._slider(p, "redraw", "Denoise", lambda: s["redraw"],
                     lambda x: s.__setitem__("redraw", x), 0.3, 0.95, 0.05)
        o.label(p, "Lower keeps the blockout's layout and shapes; higher lets the "
                "picture move further from them. 0.6 to 0.75 usually keeps the "
                "composition.", "faint", self.host.f_small,
                wraplength=o.px(310)).pack(side="top", fill="x")
        self._words_box()

    def _inspect_object(self, obj):
        o, p = self.owner, self.panel
        a = sc.ASSET[obj["asset"]]
        o.cap(p, a["label"])
        name = tk.StringVar(value=obj["name"])
        e = self.host._entry(p, name)
        e.master.pack(side="top", fill="x")

        def renamed(_ev=None):
            obj["name"] = name.get()
            self.dirty = True
            self._list()
            self._words()
        e.bind("<KeyRelease>", renamed)

        o.cap(p, "What they are doing" if a["kind"] == "person" else "Description")

        def described(v):
            obj["description"] = v
            self.dirty = True
            self._title()
            self._words()
        self._text(p, obj["description"], described)
        o.label(p, ("Their action, and what matters about them: PPE (hard hat, "
                    "hi-vis, gloves, face shield up or down), what they hold. "
                    "Sent exactly as written." if a["kind"] == "person" else
                    "What it is and its state: open or closed, on or off, full or "
                    "empty. Sent exactly as written."),
                "faint", self.host.f_small, wraplength=o.px(310)).pack(side="top",
                                                                       fill="x")

        o.cap(p, "Colour")
        sw = o.frame(p)
        sw.pack(side="top", fill="x")
        for hexc, cname in sc.COLOURS:
            chip = tk.Canvas(sw, width=o.px(20), height=o.px(20), highlightthickness=0,
                             bd=0, cursor="hand2")
            o.skin(chip, bg="bg")
            ring = "accent" if hexc.lower() == obj["colour"].lower() else "border"
            chip.create_oval(1, 1, o.px(19), o.px(19), fill=hexc,
                             outline=self.host.C[ring], width=2)
            chip.bind("<Button-1>", lambda ev, h=hexc: self._set_colour(h))
            chip.pack(side="left", padx=(0, o.px(3)))

        if a["kind"] == "person":
            self._pose_controls(obj)

        o.cap(p, "Place")
        pos, rot, scale = obj["position"], obj["rotation"], obj["scale"]

        def at(vec, i):
            def write(x):
                vec[i] = x
            return write
        self._slider(p, "x", "Left / right (m)", lambda: pos[0], at(pos, 0),
                     -REACH, REACH, 0.05)
        self._slider(p, "z", "Back / front (m)", lambda: pos[2], at(pos, 2),
                     -REACH, REACH, 0.05)
        self._slider(p, "y", "Floor height (m)", lambda: pos[1], at(pos, 1), 0, 3, 0.05)
        self._slider(p, "yaw", "Turn", lambda: rot[0], at(rot, 0), -180, 180)
        if a["kind"] == "prop":
            self._slider(p, "pitch", "Tip forward", lambda: rot[1], at(rot, 1), -90, 90)
            self._slider(p, "roll", "Tip sideways", lambda: rot[2], at(rot, 2), -90, 90)
            o.cap(p, "Size (m)")
            for i, label in enumerate(("Width", "Height", "Depth")):
                self._slider(p, "size%d" % i, label, lambda i=i: scale[i], at(scale, i),
                             0.05, 5, 0.05)
        else:
            def uniform(x):
                scale[:] = [x, x, x]
            self._slider(p, "size", "Size", lambda: scale[0], uniform, 0.5, 1.3, 0.01)
        self._words_box()

    def _pose_controls(self, obj):
        o, p = self.owner, self.panel
        pose = obj["pose"]
        o.cap(p, "Pose")
        self.pose_pill = o.choice(p, [(k, label) for k, label, _ in sc.POSES] +
                                  ([("", "Custom")] if not pose["preset"] else []),
                                  pose["preset"], self._set_pose)
        self.pose_pill.pack(side="top", anchor="w")
        tabs = o.frame(p)
        tabs.pack(side="top", fill="x", pady=(o.px(8), o.px(4)))
        for i, (part, label) in enumerate(sc.PARTS):
            o.button(tabs, label, lambda part=part: self.select(obj["id"], part),
                     kind="accent" if part == self.part else "quiet",
                     font=self.host.f_small).grid(row=i // 3, column=i % 3, sticky="ew",
                                                  padx=(0, o.px(3)), pady=(0, o.px(3)))
        for col in range(3):
            tabs.columnconfigure(col, weight=1)
        ctl = pose["controls"]
        for part, key, label, lo, hi in sc.CONTROLS:
            if part != self.part:
                continue

            def write(x, key=key):
                ctl[key] = x
                if pose["preset"]:        # moved by hand: no longer the preset
                    pose["preset"] = ""
                    self.pose_pill.set(text="Custom  ▾")
            self._slider(p, key, label, lambda key=key: ctl[key], write, lo, hi)
        o.button(p, "Reset %s" % sc.PART_NAMES[self.part].lower(), self._reset_part,
                 kind="ghost").pack(side="top", anchor="w", pady=(o.px(4), 0))
        o.label(p, "Click a hand, foot or the head in the viewport to pose that part.",
                "faint", self.host.f_small, wraplength=o.px(310)).pack(side="top",
                                                                       fill="x")

    def _words_box(self):
        o = self.owner
        o.cap(self.panel, "Words sent with the frame")
        box = o.frame(self.panel, "card")
        box.pack(side="top", fill="x", pady=(0, o.px(12)))
        self.words_label = o.label(box, "", "text", self.host.f_small, bg="card",
                                   wraplength=o.px(300))
        self.words_label.pack(side="top", fill="x", padx=o.px(8), pady=o.px(6))
        self._words()

    def _words(self):
        self._title()
        if not getattr(self, "words_label", None):
            return
        try:
            words = sc.scene_text(self.scene)
            text = words.text + ("\n\n" + "\n".join("• " + n for n in words.notes)
                                 if words.notes else "")
            self.words_label.config(text=text)
        except tk.TclError:
            pass

    # -------------------------------------------------------------- setters
    def _set_frame(self, key):
        self.scene["frame"] = key
        self.changed()

    def _set_lens(self, mm):
        self.scene["camera"]["lens"] = float(mm)
        self.sync()
        self.changed()

    def _set_colour(self, hexc):
        obj = self.obj()
        if obj is not None:
            obj["colour"] = hexc
            self._inspect()
            self.changed()

    def _set_pose(self, preset):
        obj = self.obj()
        if obj is None or preset not in sc.POSE_VALUES:
            return
        obj["pose"] = {"preset": preset, "controls": sc.pose_controls(preset)}
        self._inspect()
        self.changed()

    def _reset_part(self):
        obj = self.obj()
        if obj is None or "pose" not in obj:
            return
        rest = sc.pose_controls("standing")
        for part, key, _, _, _ in sc.CONTROLS:
            if part == self.part:
                obj["pose"]["controls"][key] = rest[key]
        obj["pose"]["preset"] = ""
        self.sync()
        self.changed()

    def set_tool(self, tool):
        self.tool = tool
        for key, pill in self.tool_pills.items():
            kind = "accent" if key == tool else "quiet"
            pill.roles = self.host.PILL_ROLES[kind]
            pill.paint(self.host.C)

    def reset_camera(self):
        self.scene["camera"] = sc.new_scene()["camera"]
        self.sync()
        self.changed()

    # ============================================================= viewport
    def _fit(self):
        c = self.canvas
        cw, ch = max(2, c.winfo_width()), max(2, c.winfo_height())
        w, h = sc.frame_size(self.scene)
        m = self.owner.px(26)
        k = max(0.01, min((cw - 2 * m) / w, (ch - 2 * m) / h))
        self.frame_rect = ((cw - w * k) / 2, (ch - h * k) / 2, w, h, k)
        return self.frame_rect

    def to_frame(self, x, y):
        ox, oy, _, _, k = self.frame_rect
        return (x - ox) / k, (y - oy) / k

    def draw(self):
        """The scene through the camera. Inside the frame this is exactly the
        reference `studio_scene.png` makes; outside it is dimmed context."""
        c = self.canvas
        try:
            c.delete("all")
        except tk.TclError:
            return
        ox, oy, w, h, k = self._fit()
        cw, ch = c.winfo_width(), c.winfo_height()
        C = self.host.C
        polys = sc.render(self.scene, w, h)
        at = lambda pts: [v for x, y in pts for v in (ox + x * k, oy + y * k)]  # noqa
        for poly in polys:
            if poly.owner is None:
                c.create_polygon(at(poly.pts), fill=rgb_hex(poly.rgb), outline="")
                for a, b, axis in sc.grid_lines(self.scene, w, h):
                    c.create_line(*at((a, b)), fill=GRID_AXIS if axis else GRID)
                continue
            fill = rgb_hex(poly.rgb)
            chosen = poly.owner == self.sel
            c.create_polygon(at(poly.pts), fill=fill,
                             outline=C["accent"] if chosen else fill,
                             tags=("o:" + poly.owner, "p:" + str(poly.part)))
        if not any(p.owner is None for p in polys):
            for a, b, axis in sc.grid_lines(self.scene, w, h):
                c.create_line(*at((a, b)), fill=GRID_AXIS if axis else GRID)
        x0, y0, x1, y1 = ox, oy, ox + w * k, oy + h * k
        for box in ((0, 0, cw, y0), (0, y1, cw, ch), (0, y0, x0, y1), (x1, y0, cw, y1)):
            c.create_rectangle(*box, fill=C["bg"], outline="", stipple="gray50")
        c.create_rectangle(x0, y0, x1, y1, outline=C["accent"], width=2)
        c.create_text(x0 + 6, y0 - 4, anchor="sw", fill=C["muted"], font=self.host.f_small,
                      text="Frame %d x %d · %dmm" % (w, h, round(self.scene["camera"]["lens"])))

    def camera(self):
        w, h = sc.frame_size(self.scene)
        return sc.Camera(self.scene["camera"], w, h)

    def hit(self, x, y):
        """-> (object id, part) under canvas point (x, y), or (None, None)."""
        for item in reversed(self.canvas.find_overlapping(x, y, x, y)):
            tags = self.canvas.gettags(item)
            oid = next((t[2:] for t in tags if t.startswith("o:")), None)
            if oid is not None:
                part = next((t[2:] for t in tags if t.startswith("p:")), "body")
                return oid, part
        return None, None

    def _press(self, ev):
        self.canvas.focus_set()
        oid, part = self.hit(ev.x, ev.y)
        cam = copy.deepcopy(self.scene["camera"])
        if oid is None:
            self.select(None)
            self.drag = {"kind": "orbit", "x": ev.x, "y": ev.y, "camera": cam}
            return
        self.select(oid, part if self.obj(oid)["asset"] == "person" else None)
        obj = self.obj()
        # Moving slides the object on a level plane through its middle, not
        # the floor: a ray through a person's chest meets the floor far
        # behind them, nearly edge on, and a few pixels became metres.
        lo, hi = sc.bounds(obj)
        mid = tuple((a + b) / 2 for a, b in zip(lo, hi))
        fx, fy = self.to_frame(ev.x, ev.y)
        grab = self.camera().on_floor(fx, fy, mid[1])
        if grab is None or sc.dot(sc.sub(grab, mid), sc.sub(grab, mid)) > 4.0:
            grab = None               # edge on: move by the screen instead
        self.drag = {"kind": self.tool, "x": ev.x, "y": ev.y, "grab": grab, "mid": mid,
                     "start": copy.deepcopy(obj), "shift": bool(ev.state & 0x0001)}

    def _motion(self, ev):
        d = self.drag
        if not d:
            return
        dx, dy = ev.x - d["x"], ev.y - d["y"]
        if d["kind"] == "orbit":
            cam = self.scene["camera"]
            cam["yaw"] = (d["camera"]["yaw"] - dx * 0.4) % 360
            cam["pitch"] = max(-30.0, min(80.0, d["camera"]["pitch"] + dy * 0.3))
            self.draw()
            return
        obj, start = self.obj(), d["start"]
        if obj is None:
            return
        if d["kind"] == "move":
            cam = self.camera()
            metres = max(0.2, sc.dot(sc.sub(d["mid"], cam.eye), cam.f)) / (
                cam.k * self.frame_rect[4])                  # per canvas pixel, there
            if d["shift"] or ev.state & 0x0001:
                obj["position"][1] = round(max(0.0, min(3.0, start["position"][1]
                                                        - dy * metres)), 3)
            else:
                fx, fy = self.to_frame(ev.x, ev.y)
                now = cam.on_floor(fx, fy, d["mid"][1]) if d["grab"] else None
                if now is not None:
                    step = (now[0] - d["grab"][0], now[2] - d["grab"][2])
                else:
                    right = sc.norm((cam.r[0], 0, cam.r[2]))
                    ahead = sc.norm((cam.f[0], 0, cam.f[2]))
                    move = sc.add(sc.mul(right, dx * metres), sc.mul(ahead, -dy * metres * 2))
                    step = (move[0], move[2])
                for i, j in ((0, 0), (2, 1)):
                    obj["position"][i] = round(max(-REACH, min(REACH, start["position"][i]
                                                               + step[j])), 3)
        elif d["kind"] == "rotate":
            yaw = start["rotation"][0] + dx * 0.6
            obj["rotation"][0] = round((yaw + 180) % 360 - 180, 1)
        elif d["kind"] == "scale":
            f = 2 ** (-dy / 150.0)
            lo, hi = (0.5, 1.3) if obj["asset"] == "person" else (0.05, 5)
            obj["scale"] = [round(max(lo, min(hi, v * f)), 3) for v in start["scale"]]
        self.dirty = True
        self.draw()

    def _release(self, _ev=None):
        if self.drag:
            self.drag = None
            self.sync()
            self._title()

    def _pan_start(self, ev):
        self.drag = {"kind": "pan", "x": ev.x, "y": ev.y,
                     "target": list(self.scene["camera"]["target"])}

    def _pan(self, ev):
        d = self.drag
        if not d or d["kind"] != "pan":
            return
        cam = self.camera()
        metres = self.scene["camera"]["distance"] / (cam.k * self.frame_rect[4])
        dx, dy = (ev.x - d["x"]) * metres, (ev.y - d["y"]) * metres
        move = sc.add(sc.mul(cam.r, -dx), sc.mul(cam.u, dy))
        t = [a + b for a, b in zip(d["target"], move)]
        t[1] = max(0.0, t[1])
        self.scene["camera"]["target"] = t
        self.dirty = True
        self.draw()

    def _zoom(self, direction):
        cam = self.scene["camera"]
        cam["distance"] = max(0.5, min(25.0, cam["distance"] * (1.1 if direction > 0 else 0.9)))
        self.sync()
        self.changed()

    def _key(self, ev):
        key = (ev.keysym or "").lower()
        for tool, _, k in TOOLS:
            if key == k:
                self.set_tool(tool)
                return "break"
        if key in ("delete", "backspace"):
            self.delete()
            return "break"
        return None

    # ================================================================ files
    def _ask_save(self):
        """True to go on (saved, or thrown away), False to stay."""
        if not self.dirty or not self.scene["objects"]:
            return True
        answer = messagebox.askyesnocancel(
            "Scene Builder", "Save the changes to this scene first?", parent=self.win)
        if answer is None:
            return False
        return self.save() if answer else True

    def new(self):
        if not self._ask_save():
            return
        self.scene = sc.new_scene()
        self.path, self.dirty = None, False
        self.select(None)
        self._list()
        self._inspect()
        self.draw()
        self.status("New scene.", "muted")

    def open(self, path=None):
        if path is None:
            if not self._ask_save():
                return False
            os.makedirs(sc.scenes_dir(), exist_ok=True)
            path = filedialog.askopenfilename(parent=self.win, title="Open a scene",
                                              initialdir=sc.scenes_dir(),
                                              filetypes=FILETYPES)
            if not path:
                return False
        try:
            scene, problems = sc.load(path)
        except (OSError, ValueError) as e:
            self.status("Could not open %s: %s" % (os.path.basename(path), e), "err")
            return False
        self.scene, self.path, self.dirty = scene, path, False
        self.sel = None
        self._list()
        self._inspect()
        self.draw()
        self.status(" ".join(problems) if problems else "Opened %s." % os.path.basename(path),
                    "warn" if problems else "ok")
        return True

    def save(self, path=None):
        path = path or self.path
        if not path:
            return self.save_as()
        try:
            sc.save(self.scene, path)
        except OSError as e:
            self.status("Could not save: %s" % e, "err")
            return False
        self.path, self.dirty = path, False
        self._title()
        self.status("Saved %s." % os.path.basename(path), "ok")
        return True

    def save_as(self):
        os.makedirs(sc.scenes_dir(), exist_ok=True)
        path = filedialog.asksaveasfilename(parent=self.win, title="Save the scene",
                                            initialdir=sc.scenes_dir(),
                                            defaultextension=".scene.json",
                                            filetypes=FILETYPES)
        return self.save(path) if path else False

    def close(self, final=False):
        """Close the window, asking first about unsaved changes. `final` is
        the Image Studio going away: save or not, but the window goes."""
        if final and self.dirty and self.scene["objects"]:
            if messagebox.askyesno("Scene Builder", "Save the changes to this scene "
                                   "before it closes?", parent=self.win):
                self.save()
        elif not final and not self._ask_save():
            return
        self.win.destroy()
        if self.owner.scene_builder is self:
            self.owner.scene_builder = None

    # ============================================================= generate
    def _models(self):
        o = self.owner
        for w in self.model_row.winfo_children():
            w.destroy()
        items = [(m["id"], m["label"]) for m in o.studio.lib.all("models")]
        o.label(self.model_row, "Model", "muted", self.host.f_small).pack(
            side="left", padx=(0, o.px(6)))
        o.choice(self.model_row, items, o.settings["model"], self._set_model).pack(
            side="left")

    def _set_model(self, mid):
        self.owner.settings["model"] = mid
        self.owner._rebuild_models()
        self.owner._recheck()
        self.check()

    def takes_source(self, model_id):
        """Whether `model_id`'s workflow, on any backend, has a source input."""
        lib = self.owner.studio.lib
        model = lib.get("models", model_id)
        if model is None:
            return False
        wids = {model["workflow"]} | {b.get("workflow") for b in
                                      (model.get("backends") or {}).values()
                                      if isinstance(b, dict) and b.get("workflow")}
        for wid in wids:
            try:
                if (self.owner.studio.workflow_loader(wid).get("references") or {}).get("source"):
                    return True
            except Exception:           # noqa - a broken template is compose()'s to name
                continue
        return False

    def check(self):
        """-> the reason Generate would not use the frame, or ''."""
        mid = self.owner.settings["model"]
        if not self.scene["objects"]:
            return "Add a person or a prop first: the frame is what the picture is made from."
        if not self.takes_source(mid):
            model = self.owner.studio.lib.get("models", mid)
            able = [m["label"] for m in self.owner.studio.lib.all("models")
                    if self.takes_source(m["id"])]
            why = ("%s's workflow takes no source picture, so the frame would not be "
                   "used." % (model["label"] if model else mid))
            why += (" Choose %s under Model." % " or ".join(able) if able else
                    " No model in the library takes one yet (Models…).")
            self.status(why, "warn")
            return why
        return ""

    def generate(self):
        """Render the frame, hand the words and the frame to the Image
        Studio's form, and generate there - its routing, refusals, queue and
        History, unchanged. The whole scene rides along in the settings."""
        why = self.check()
        if why:
            self.status(why, "err")
            return False
        try:
            ref = sc.write_reference(self.scene)
        except OSError as e:
            self.status("Could not write the frame: %s" % e, "err")
            return False
        words, ref, extra = sc.generation(self.scene, ref)
        o = self.owner
        o.scene.delete("1.0", "end")
        o.scene.insert("1.0", words.text)
        o._set_ref("source", ref)
        if self.path:
            extra["scene_file"] = self.path
        sent = o.generate(extra=extra)
        said = o.note.cget("text")
        if sent:
            self.status("Sent to the Image Studio with the frame (%s). %s"
                        % (os.path.basename(ref), " ".join(words.notes)), "ok")
        else:
            self.status(said or "Not sent.", "err")
        return sent
