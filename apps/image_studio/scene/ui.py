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
- with Look at: click a person and a ring of head poses opens round their
  head - ahead, up, down, left, right and the diagonals, as the camera sees
  them - with Camera and Point... under it; after Point, a click on a face, a
  prop, the floor or the sky is where they look, followed as they move (Esc
  closes the ring; a crowd has no heads to pose);
- left on empty floor or sky: orbit the camera;
- right (or middle): pan the camera; the wheel: in and out.
Keys, with the viewport focused: M, R, S, L for the tools, Esc, Delete. Anywhere in
the window but a text box: Ctrl+Z undo, Ctrl+Y or Ctrl+Shift+Z redo; the
History menu over the viewport jumps to any step (`studio_scene.History`).

**Floor and walls**, the second row of the list, dresses the room: a few words
for the floor or the walls, and Make sends them to the Image Studio as a text
to image job (`studio_scene.texture_settings`); the finished picture comes
back onto that surface (`ImageStudio._job_changed` -> `texture_done`).
Picture… puts a PNG from disk there instead. The viewport draws pictured
surfaces in their mean colour at once and the pictures a moment later, baked
at half size off the drag (`_bake`), since a per-pixel fill in Python is too
slow for every mouse move. Its Floor shape is the floor seen from above,
4 x 4 dots over the room's width and depth: drag one up or down, right-click
it back to 0, or start from a preset (a hill, a dip, a rise behind...). The
dots pull one smooth surface (`studio_scene.Ground`), and everything on the
floor stays standing on it.
"""

import base64
import copy
import json
import math
import os
import threading
import time
import uuid
import tkinter as tk
from tkinter import filedialog, messagebox

import apps.image_studio.imagegen as ig
import apps.image_studio.scene.scene as sc

FILETYPES = [("Scenes", "*.scene.json"), ("JSON", "*.json"), ("All files", "*.*")]
TOOLS = [("move", "Move", "m"), ("rotate", "Rotate", "r"), ("scale", "Scale", "s"),
         ("look", "Look at", "l")]
GRID = "#9c978f"
GRID_AXIS = "#b1aca4"
SCENE_ROW = "Scene and camera"
ROOM_ROW = "Floor and walls"
ROOM = "\0room"              # the room's row in the list; never an object's id
BAKE = 2                      # the viewport's pictures are baked at 1/BAKE size
REACH = 20                    # m either way an object can be placed
CAMERA_TILE = 96              # px, before the display's scale; a camera's button


def _inside(x, y, pts):
    """Even-odd: is (x, y) inside the polygon `pts`?"""
    hit = False
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            hit = not hit
    return hit


def rgb_hex(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


class SceneBuilder:
    def __init__(self, owner, path=None):
        self.owner = o = owner
        host = self.host = owner.host
        form_scene = owner.scene.get("1.0", "end").strip()
        self.scene = sc.new_scene(details=form_scene)
        self._bake_camera(owner.settings.get("camera_profile"))   # the form's Shot on card
        self.path = None
        self.dirty = False
        self.recovery_path = self._new_recovery_path()
        self.sel = None               # an object's id; None is the scene and camera
        self.part = "body"
        self.look_section = ig.LOOKS[0][0]
        self.outfit_name = ""        # the outfit preset box, kept across redraws
        self.tool = "move"
        self.aiming = None          # Look at: the person whose point the next click is
        self.ring = None            # Look at: {"oid", "x", "y"} while the ring is up
        self.drag = None
        self.vars = {}                # key -> (DoubleVar, read) for the inspector's sliders
        self.tool_pills = {}
        self.frame_rect = (0, 0, 1, 1, 1.0)      # ox, oy, w, h, scale on the canvas
        self.making = {}              # surface -> words sent, while its picture is made
        self.posing = set()           # ids of the people a photo's pose is being found for
        self.picturing = False        # a scene is being made from a picture
        self.taken = set()            # ids of the finished picture jobs already used
        self.cards = {}               # (texture, w, h) -> PhotoImage: props' pictures, drawn
        self.backdrop = (None, None)  # (key, PhotoImage): the room's pictures, baked
        self.bake_after = None
        self.shape_canvas = None      # the floor's side view, while the room is inspected
        self.shape_drag = None        # {"i": the dot, "dots": all of them} while one is dragged
        self.suggestion = None        # what Enrich offered (sc.new_suggestion), awaiting an answer
        self.enriching = False

        win = self.win = tk.Toplevel(host)
        win.title("Scene Builder")
        win.transient(host)
        host._skin(win, bg="bg")
        win.geometry("%dx%d" % (host._px(1240), host._px(780)))
        win.minsize(host._px(1000), host._px(640))
        win.protocol("WM_DELETE_WINDOW", self.close)

        # Fixed before expanding (AGENTS.md: pack order): the foot, both
        # side columns, then the viewport takes what is left.
        foot = o.frame(win)
        foot.pack(side="bottom", fill="x", padx=o.px(12), pady=(0, o.px(12)))
        self.msg = o.label(foot, "", "muted", host.f_small, wraplength=o.px(950))
        self.msg.pack(side="bottom", fill="x", pady=(o.px(6), 0))
        self.go = o.button(foot, "Generate picture", self.generate, kind="accent")
        self.go.pack(side="right")
        self.model_row = o.frame(foot)
        self.model_row.pack(side="left", padx=(0, o.px(14)))

        files = o.frame(win)
        files.pack(side="top", fill="x", padx=o.px(12), pady=(o.px(8), 0))
        self.file_pill = o.button(files, "File ▾", self._post_file_menu)
        self.file_pill.pack(side="left")
        self.picture_pill = o.button(files, "Build from photo…", self.from_picture)
        self.picture_pill.pack(side="left", padx=(o.px(6), 0))
        o.button(files, "Suggest details…", self.enrich, kind="ghost").pack(
            side="left", padx=(o.px(6), 0))

        left = o.frame(win)
        left.config(width=o.px(190))
        left.pack_propagate(False)
        left.pack(side="left", fill="y", padx=o.px(12), pady=o.px(12))
        o.cap(left, "Add to scene")
        # People one button each; shapes and props a menu each, or the list
        # of what is in the scene has no room left.
        for gid, glabel in sc.ASSET_GROUPS:
            items = [a for a in sc.ASSETS if a["group"] == gid]
            if gid == "people":
                for a in items:
                    o.button(left, "+  " + a["label"], lambda a=a["id"]: self.add(a),
                             anchor="w").pack(side="top", fill="x", pady=(0, o.px(3)))
            else:
                self._library_menu(left, glabel, items).pack(side="top", fill="x",
                                                             pady=(0, o.px(3)))
        self.stand_in_btn = o.button(left, "+  Stand-in…", self._stand_in_menu, anchor="w")
        self.stand_in_btn.pack(side="top", fill="x", pady=(o.px(2), o.px(3)))
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
        history_bar = o.frame(mid)
        history_bar.pack(side="top", fill="x", pady=(0, o.px(6)))
        o.button(history_bar, "Reset camera", self.reset_camera, kind="ghost").pack(side="right")
        small = dict(kind="quiet", font=host.f_small, padx=o.px(10))
        self.history_pill = o.button(history_bar, "History ▾", lambda: None, **small)
        self.history_pill.command = self._history_menu
        self.history_pill.pack(side="right", padx=(0, o.px(10)))
        self.redo_pill = o.button(history_bar, "Redo", self.redo, **small)
        self.redo_pill.pack(side="right", padx=(0, o.px(3)))
        self.undo_pill = o.button(history_bar, "Undo", self.undo, **small)
        self.undo_pill.pack(side="right", padx=(0, o.px(3)))
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
        # On the window, so they work wherever the focus is - except in a
        # text box, which undoes its own typing (`_undo_key`).
        for seq, fn in (("<Control-z>", self.undo), ("<Control-Z>", self.redo),
                        ("<Control-y>", self.redo), ("<Control-Y>", self.redo)):
            win.bind(seq, lambda ev, fn=fn: self._undo_key(ev, fn))

        self.history = sc.History(self.scene)
        self.remember_after = None
        self._undo_buttons()
        self._models()
        self._list()
        self._inspect()
        if path:
            self.open(path)
        else:
            self.status("Add a person or a prop from the library, then frame it with the "
                        "camera.", "muted")

    # ================================================================ state
    def _file_menu(self):
        menu = self.host._menu()
        menu.add_command(label="New scene", command=self.new)
        menu.add_command(label="Open scene…", command=self.open)
        menu.add_command(label="Recover scene…", command=self.recover)
        menu.add_separator()
        menu.add_command(label="Save scene", command=self.save)
        menu.add_command(label="Save scene as…", command=self.save_as)
        return menu

    def _post_file_menu(self):
        pill = self.file_pill
        self._file_menu().tk_popup(pill.winfo_rootx(),
                                  pill.winfo_rooty() + pill.winfo_height())

    def status(self, text, role="muted"):
        self.msg.config(text=text)
        self.owner.skin(self.msg, bg="bg", fg=role)

    def obj(self, oid=None):
        oid = self.sel if oid is None else oid
        return next((x for x in self.scene["objects"] if x["id"] == oid), None)

    def changed(self, rebuild_list=False):
        """The scene changed: redraw, say what the words will be, and make
        it an undo step once the edit settles."""
        self.dirty = True
        sc.aim_heads(self.scene)
        if rebuild_list:
            self._list()
        self.draw()
        self._words()
        self.remember_soon()

    # ============================================================== history
    # Every edit reaches `changed()` (or `remember_soon()` for the few that
    # only retitle), and a step is recorded once edits stop for REMEMBER ms:
    # a slider dragged or a word typed is one step, not one per tick. A
    # viewport drag records on release. Undo and redo record anything still
    # pending first, so the last edit is never the one skipped.
    REMEMBER = 600

    def remember_soon(self):
        if self.remember_after is not None:
            self.win.after_cancel(self.remember_after)
        self.remember_after = self.win.after(self.REMEMBER, self.remember)

    def remember(self):
        """Record the scene as it is now as an undo step, if it changed."""
        if self.remember_after is not None:
            try:
                self.win.after_cancel(self.remember_after)
            except tk.TclError:
                pass
            self.remember_after = None
        if self.drag:                 # mid-drag: the release records it
            return None
        label = self.history.record(self.scene, self.sel)
        if label:
            self.dirty = self.history.unsaved(self.scene)
            self._save_recovery()
            self._undo_buttons()
        return label

    def undo(self):
        return self._step(-1)

    def redo(self):
        return self._step(1)

    def _step(self, way):
        if self.drag:
            return False
        self.remember()
        h = self.history
        if way < 0:
            undone = h.steps[h.at][0]
            return self._restore(h.undo(), "Undid", undone)
        return self._restore(h.redo(), "Redid")

    def go_to(self, i):
        """Put the scene back (or forward) to history step `i`."""
        if self.drag:
            return False
        self.remember()
        return self._restore(self.history.go(i), "Back to")

    def _restore(self, got, verb, what=None):
        if got is None:
            if verb != "Back to":
                self.status("Nothing to %s." % ("undo" if verb == "Undid" else "redo"),
                            "muted")
            return False
        scene, sel = got
        # In place, so a pose still being found from a photo lands in it
        # (`pose_from_photo` checks it is the same scene).
        self.scene.clear()
        self.scene.update(scene)
        ids = {x["id"] for x in self.scene["objects"]}
        self.sel = sel if sel == ROOM or sel in ids else None
        obj = self.obj()
        if obj is None or obj["asset"] != "person":
            self.part = "body"
        self.dirty = self.history.unsaved(self.scene)
        self._save_recovery()
        self._undo_buttons()
        self._list()
        self._inspect()
        self.draw()
        h = self.history
        self.status("%s: %s." % (verb, what or h.steps[h.at][0]), "muted")
        return True

    def _forget(self, label="New scene"):
        """A different scene: its history starts again, with nothing unsaved."""
        if self.remember_after is not None:
            self.win.after_cancel(self.remember_after)
            self.remember_after = None
        self.history.reset(self.scene, label=label)
        self.recovery_path = self._new_recovery_path()
        self._undo_buttons()

    def _undo_buttons(self):
        h = self.history
        self.undo_pill.set(state="normal" if h.can_undo() else "disabled")
        self.redo_pill.set(state="normal" if h.can_redo() else "disabled")

    def _undo_key(self, ev, fn):
        # A text box or a name entry undoes its own typing, as it would anywhere.
        if isinstance(ev.widget, (tk.Text, tk.Entry)):
            return None
        fn()
        return "break"

    def _history_menu(self):
        """Every step, newest first: the one the scene is in is marked, the
        ones after it (still redoable) are faint. Choosing one goes there."""
        self.remember()
        o, h, pill = self.owner, self.history, self.history_pill
        menu = tk.Menu(pill, tearoff=0)
        o.skin(menu, bg="card", fg="text", activebackground="sel",
               activeforeground="text")
        for i in range(len(h.steps) - 1, -1, -1):
            label = h.steps[i][0]
            menu.add_command(label=("●  " if i == h.at else "     ") + label,
                             command=lambda i=i: self.go_to(i))
            if i > h.at:
                menu.entryconfig(menu.index("end"), foreground=self.host.C["faint"])
        menu.tk_popup(pill.winfo_rootx(), pill.winfo_rooty() + pill.winfo_height())

    def _title(self):
        name = os.path.basename(self.path) if self.path else "untitled"
        self.win.title("Scene Builder - %s%s" % (name, " *" if self.dirty else ""))

    # ============================================================== objects
    def _stand_in_menu(self):
        """A shape already named and sized as a common thing."""
        m = self.host._menu()
        for label, aid, _, scale, _ in sc.STAND_INS:
            m.add_command(label="%s  (%s, %s m)" % (
                label, sc.ASSET[aid]["label"].lower(), " × ".join("%g" % v for v in scale)),
                command=lambda a=aid, s=label: self.add(a, s))
        self.host._popup(m, self.stand_in_btn)

    def add(self, asset_id, stand_in=None):
        obj = sc.new_object(asset_id, self.scene["objects"], stand_in)
        seeded = asset_id == "person" and not sc.people(self.scene) and self._form_look(obj)
        # Put it where the camera is looking, beside anything already there.
        t = self.scene["camera"]["target"]
        taken = len(self.scene["objects"])
        obj["position"] = [round(t[0] + (0.9 * taken if taken else 0), 2), 0.0,
                           round(t[2], 2)]
        sc.stand(self.scene["room"], obj)
        self.scene["objects"].append(obj)
        self.select(obj["id"])
        self.changed(rebuild_list=True)
        self.status("%s added%s. %s" % (obj["name"], " with the form's look" if seeded else "",
                                          sc.ASSET[asset_id]["about"]), "muted")
        self.check()                  # says now, not at Generate, if the frame would go unused
        return obj

    def _library_menu(self, parent, label, items):
        """A library button that posts a menu of `items` to add."""
        o = self.owner
        pill = o.button(parent, "+  %s  ▾" % label, lambda: None, anchor="w")

        def post():
            menu = tk.Menu(pill, tearoff=0)
            o.skin(menu, bg="card", fg="text", activebackground="sel",
                   activeforeground="text")
            for a in items:
                menu.add_command(label=a["label"], command=lambda a=a["id"]: self.add(a))
            menu.tk_popup(pill.winfo_rootx(), pill.winfo_rooty() + pill.winfo_height())
        pill.command = post
        return pill

    def _form_look(self, obj):
        """The scene's first person takes the look already on the form, since
        a scene with people in it sends their looks and not the form's."""
        o = self.owner
        look = sc.clean_look(o.collect_looks())
        cid = o.settings.get("character") or ""
        if not look and not cid:
            return False
        obj["look"], obj["character"] = look, cid
        rec = o.studio.lib.get("characters", cid) if cid else None
        if rec is not None and rec["name"] not in {x["name"] for x in self.scene["objects"]}:
            obj["name"] = rec["name"]
        return True

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
        if part:
            section = "Pose" if part != "body" else "Object"
        else:
            section = "Scene" if oid is None else "Object"
        if (oid != self.sel or (part and part != self.part)
                or section != getattr(self, "inspector_section", "Object")):
            self.inspector_section = section
            self.sel = oid
            if part:
                self.part = part
            elif self.obj() is None or self.obj()["asset"] != "person":
                self.part = "body"
            self._list()
            self._inspect()
            self.panel.canvas.yview_moveto(0)
        self.draw()

    def rows(self):
        return [None, ROOM] + [x["id"] for x in self.scene["objects"]]

    def _list(self):
        self.lb.delete(0, "end")
        self.lb.insert("end", SCENE_ROW)
        self.lb.insert("end", ROOM_ROW)
        rows = self.rows()
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
        rows = self.rows()
        self.select(rows[sel[0]] if sel[0] < len(rows) else None)

    # ============================================================ inspector
    def _clear(self):
        for w in self.panel.winfo_children():
            w.destroy()
        self.vars = {}
        self.words_label = None
        self.shape_canvas = None

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
        if self.sel == ROOM:
            self._inspect_room()
        elif obj is None:
            self._inspect_scene()
        else:
            self._inspect_object(obj)

    def _inspect_scene(self):
        o, p = self.owner, self.panel
        section = getattr(self, "inspector_section", "Scene")
        tabs = o.frame(p)
        tabs.pack(side="top", fill="x", pady=(0, o.px(8)))
        for name in ("Scene", "Camera"):
            o.button(tabs, name, lambda n=name: self._inspector_tab(n),
                     kind="accent" if name == section else "quiet").pack(
                side="left", padx=(0, o.px(4)))
        if section == "Camera":
            self._camera_controls()
        else:
            self._scene_controls()
        self._words_box()

    def _camera_controls(self):
        o, p, s = self.owner, self.panel, self.scene
        cam = s["camera"]
        o.cap(p, "Frame")
        # Only the frames the camera body shoots (all of them with none set).
        fits = sc.FORMAT_FRAMES.get(cam.get("format"))
        o.choice(p, [(k, label) for k, label, _, _ in sc.FRAMES if not fits or k in fits],
                 s["frame"], self._set_frame).pack(side="top", anchor="w")

        o.cap(p, "Camera body")
        o.label(p, "The camera this scene is shot on. Its chemistry - film stock or "
                "digital colour science - is added to the words, and its native lens "
                "replaces the one below.", "faint", self.host.f_small,
                wraplength=o.px(310)).pack(side="top", fill="x")
        self._camera_profile_tiles()
        o.button(p, "Cameras…", self.owner.edit_camera_profiles, kind="ghost").pack(
            side="top", anchor="w", pady=(0, o.px(10)))

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

    def _camera_profile_tiles(self):
        """The camera library as a grid of pictures, each its own photo of
        that camera where there is one, else a blank tile with its name -
        `ImageStudio._build_style_tiles`'s fallback for a style with no
        example. A click bakes that camera's chemistry, and its native lens
        when it names one, into the scene (`_set_camera_profile`)."""
        import apps.image_studio.ui as studio_ui   # lazy: studio_ui imports this module
        o, p = self.owner, self.panel
        cams = self.owner.studio.lib.all("camera_profiles")
        grid = o.frame(p)
        grid.pack(side="top", fill="x", pady=(0, o.px(4)))
        self.camera_tiles, self.camera_photos = {}, []
        side = o.px(CAMERA_TILE)
        current = self.scene["camera"].get("profile")
        for n, cp in enumerate(cams):
            tile = tk.Frame(grid, bd=0, highlightthickness=o.px(2), cursor="hand2")
            o.skin(tile, bg="bg", highlightbackground="accent" if cp["id"] == current
                   else "border")
            tile.grid(row=n // 3, column=n % 3, padx=(0, o.px(6)), pady=(0, o.px(6)),
                      sticky="n")
            img = (studio_ui.photo_at(cp["image"], side, grid)
                   if cp["image"] and os.path.isfile(cp["image"]) else None)
            if img is not None:
                self.camera_photos.append(img)
                pic = tk.Label(tile, image=img, bd=0)
            else:
                pic = o.frame(tile, "card")
                pic.config(width=side, height=side)
                pic.pack_propagate(False)
                o.label(pic, cp["name"], "faint", self.host.f_small, bg="card",
                        wraplength=side - o.px(8)).pack(expand=True)
            pic.pack(side="top")
            name = o.label(tile, cp["name"], "text", self.host.f_small, wraplength=side)
            name.config(anchor="center", justify="center")
            name.pack(side="top", fill="x", pady=(o.px(2), o.px(2)))
            for w in (tile, pic, name, *pic.winfo_children()):
                w.bind("<Button-1>", lambda ev, cid=cp["id"]: self._set_camera_profile(cid))
            self.camera_tiles[cp["id"]] = tile
        if not cams:
            o.label(grid, "No cameras yet - add one with Cameras… below.",
                    "faint").pack(side="left")

    def _set_camera_profile(self, cid):
        """Attach a camera's chemistry (and its native lens, when it names
        one) to the scene; baked in as plain words, so a saved scene keeps
        its camera's look even if the library entry is later edited or
        removed."""
        self._bake_camera(cid)
        self._inspect()               # the tiles' highlight and the frames it offers
        self.sync()
        self.changed()                # via _words, the form's Shot on deck turns to it

    def _bake_camera(self, cid):
        """A camera's name, format, chemistry and native lens into the scene,
        and the frame to its shape - no widgets touched."""
        cam = self.scene["camera"]
        cp = self.owner.studio.lib.get("camera_profiles", cid) if cid else None
        cam["profile"] = cp["id"] if cp else ""
        cam["body"] = cp["name"] if cp and cp["id"] != "none" else ""
        cam["format"] = (cp.get("format") or "") if cp else ""
        cam["chemistry"] = cp["chemistry"] if cp else ""
        if cp and cp["lens"]:
            cam["lens"] = float(cp["lens"])
        # The viewfinder is the camera's: a square camera shoots square, a
        # 35mm one 3:2, held the way the frame already was.
        self.scene["frame"] = sc.camera_frame(cam["format"], self.scene["frame"])

    def _scene_controls(self):
        o, p, s = self.owner, self.panel, self.scene
        o.cap(p, "Scene details")

        def details(v):
            s["details"] = v
            self.changed()
        self.details = self._text(p, s["details"], details, height=5)
        o.label(p, "Where this is and what matters in it: the place, the light, the "
                "weather, the state of things.", "faint", self.host.f_small,
                wraplength=o.px(310)).pack(side="top", fill="x")
        self._enrich_box(p)

        o.cap(p, "What the picture follows")

        def put_scene(key):
            def write(x):
                s[key] = x
            return write
        self._slider(p, "pose_strength", "Pose", lambda: s["pose_strength"],
                     put_scene("pose_strength"), 0.0, 1.0, 0.05)
        self._slider(p, "depth_strength", "Layout (depth)", lambda: s["depth_strength"],
                     put_scene("depth_strength"), 0.0, 1.0, 0.05)
        self._slider(p, "frame_keep", "Grey frame kept", lambda: s["frame_keep"],
                     put_scene("frame_keep"), 0.0, sc.FRAME_KEEP_MAX, 0.05)
        self._slider(p, "face_likeness", "Face likeness", lambda: s["face_likeness"],
                     put_scene("face_likeness"), *sc.FACE_LIKENESS_RANGE, 0.05)
        self._slider(p, "head_depth", "Head shape (face)", lambda: s["head_depth"],
                     put_scene("head_depth"), *sc.HEAD_DEPTH_RANGE, 0.05)
        real = tk.BooleanVar(value=s["real_faces"])

        def flip():
            s["real_faces"] = bool(real.get())
            self.changed()
        b = tk.Checkbutton(p, text="Real faces where a photo's angle fits", variable=real,
                           command=flip, anchor="w", font=self.host.f_small, bd=0,
                           highlightthickness=0)
        o.skin(b, bg="bg", fg="text", activebackground="bg", selectcolor="card",
               activeforeground="text")
        b.pack(side="top", fill="x")
        self.vars["real_faces"] = (real, lambda: s["real_faces"])
        regional = tk.BooleanVar(value=s["regional_prompting"])

        def flip_regional():
            s["regional_prompting"] = bool(regional.get())
            self.changed()
        rb = tk.Checkbutton(p, text="Regional character prompting (experimental)",
                            variable=regional, command=flip_regional, anchor="w",
                            font=self.host.f_small, bd=0, highlightthickness=0)
        o.skin(rb, bg="bg", fg="text", activebackground="bg", selectcolor="card",
               activeforeground="text")
        rb.pack(side="top", fill="x")
        self.vars["regional_prompting"] = (regional, lambda: s["regional_prompting"])
        o.label(p, "Two or more named characters each get their own words in their own "
                "part of the picture, instead of one shared paragraph - less hair, "
                "clothing and prop bleed between them. Needs a model whose workflow "
                "supports it (Z-Image Turbo); otherwise it is ignored.",
                "faint", self.host.f_small, wraplength=o.px(310)).pack(side="top", fill="x")
        o.label(p, "Pose holds each body's joints; layout holds where everything is and "
                "how far away. Both leave how things look to the words. The grey frame "
                "is off by default: kept, the picture copies the mannequins' blocky "
                "shapes - 0.1 to 0.25 pins props and framing. 0 turns any of them off. "
                "A model with no ControlNet uses the frame alone, 0.3 kept at least. "
                "Face likeness: a face with a face picture is drawn from it in the picture "
                "itself, then redrawn close up at this strength last of all. Higher looks "
                "more like the picture; past 0.6 the head can stop matching the body. "
                "Real faces: then each person's own face is pasted over theirs from the "
                "photo of them turned most like it - only when it is close enough for "
                "the face's size; the PuLID picture is kept beside it in History.",
                "faint", self.host.f_small, wraplength=o.px(310)).pack(side="top", fill="x")
        self._words_box()

    def _enrich_box(self, p):
        """The detail Enrich offered, with its three answers, and the details
        already added, each removable."""
        o, e = self.owner, self.scene.setdefault("enrich", sc.new_enrich())
        if self.suggestion:
            sug = self.suggestion
            detail, why = sug["detail"], sug["why"]
            if sug["shape"] in sc.ASSET:
                why = ("Adds a %s%s, %s. " % (sc.ASSET[sug["shape"]]["label"].lower(),
                                              ' "%s"' % sug["name"] if sug["name"] else "",
                                              sug["where"].replace("_", " ")) + why).strip()
            else:
                why = ("Words only. " + why).strip()
            o.cap(p, "Suggested detail")
            card = o.frame(p, "card")
            card.pack(side="top", fill="x")
            o.label(card, detail, "text", self.host.f_ui, bg="card",
                    wraplength=o.px(300)).pack(side="top", fill="x", padx=o.px(8),
                                                pady=(o.px(6), 0))
            if why:
                o.label(card, why, "faint", self.host.f_small, bg="card",
                        wraplength=o.px(300)).pack(side="top", fill="x", padx=o.px(8))
            row = o.frame(card, "card")
            row.pack(side="top", fill="x", padx=o.px(8), pady=o.px(6))
            o.button(row, "Add", lambda: self._answer("add"), kind="accent",
                     bg="card").pack(side="left")
            o.button(row, "Skip", lambda: self._answer("skip"), bg="card").pack(
                side="left", padx=(o.px(4), 0))
            o.button(row, "Don't suggest again", lambda: self._answer("never"),
                     kind="ghost", bg="card").pack(side="left", padx=(o.px(4), 0))
        if e["added"]:
            o.cap(p, "Lived-in details")
            for i, detail in enumerate(e["added"]):
                row = o.frame(p, "card")
                row.pack(side="top", fill="x", pady=(0, o.px(3)))
                o.button(row, "×", lambda i=i: self._unadd(i), kind="ghost",
                         bg="card").pack(side="right", anchor="n")
                o.label(row, detail, "text", self.host.f_small, bg="card",
                        wraplength=o.px(270)).pack(side="left", fill="x", expand=True,
                                                   padx=o.px(8), pady=o.px(4))

    def _llm(self):
        """The model the tabs already use, or what the host has loaded: a new
        load on the shared card would cost more than the suggestion."""
        import core.agent as eng
        base = getattr(self.host, "host", None) or eng.DEFAULT_HOST
        llm = getattr(self.host, "llm", None)
        model = getattr(llm, "model", None)
        if not model:
            ok, loaded, ids, _, err = eng.probe_models(base)
            if not ok:
                raise RuntimeError("Cannot reach the LLM PC at %s (%s)." % (base, err))
            model = eng.pick_model(loaded, ids, getattr(self.host, "want_model", None))
            if not model:
                raise RuntimeError("The LLM PC serves no model.")
        return eng.LLM(base, model, temperature=0.9, timeout=180)

    def enrich(self):
        """Ask the LLM PC for one lived-in detail, off the UI thread."""
        if self.enriching:
            return
        self.enriching = True
        self.suggestion = None
        if self.sel is not None:
            self.select(None)
        else:
            self._inspect()
        self.status("Asking the LLM PC for one lived-in detail…", "muted")
        mine, scene, box = self.scene, copy.deepcopy(self.scene), {}

        def work():
            try:
                box["got"] = sc.suggest(scene, self._llm())
            except Exception as ex:      # the host's errors are all worth saying
                box["error"] = ex
        worker = threading.Thread(target=work, daemon=True)
        worker.start()

        def wait():
            if worker.is_alive():
                self.win.after(150, wait)
                return
            self.enriching = False
            if self.scene is not mine:       # New or Open while it thought
                return
            if "error" in box:
                self.status("Enrich: %s" % box["error"], "err")
                return
            self.suggestion = box["got"]
            self.status("One detail suggested — Add, Skip, or Don't suggest again.", "muted")
            if self.sel is None:
                self._inspect()
        self.win.after(150, wait)

    def _answer(self, answer):
        if not self.suggestion:
            return
        sug, self.suggestion = self.suggestion, None
        obj = sc.place_suggestion(self.scene, sug) if answer == "add" else None
        if obj:
            self.scene["objects"].append(obj)
            sc.enrich_answer(self.scene, sug["detail"], "placed")
            self.status("Added %s to the scene, %s. Drag it wherever it looks right."
                        % (obj["name"], ", ".join(sc.placement(self.scene, obj) or [])),
                        "ok")
            self.select(obj["id"])
            self.changed(rebuild_list=True)
            return
        sc.enrich_answer(self.scene, sug["detail"], answer)
        say = {"add": "Added to the words.", "skip": "Skipped.",
               "never": "Won't be suggested again."}[answer]
        if answer == "add" and sug["shape"] in sc.ASSET:
            say = "No free place in the frame for it, so it went into the words."
        self.status(say, "muted")
        self._inspect()
        self.changed()

    def _unadd(self, i):
        added = self.scene["enrich"]["added"]
        if 0 <= i < len(added):
            sc.enrich_answer(self.scene, added.pop(i), "skip")
            self._inspect()
            self.changed()

    def _inspect_room(self):
        o, p, room = self.owner, self.panel, self.scene["room"]

        def put(key):
            def write(x):
                if key in ("width", "depth"):
                    self._room_size(key, x)       # the floor's shape spans them
                else:
                    room[key] = x
            return write
        for key, label, _ in sc.SURFACES:
            o.cap(p, label)
            if key == "wall":
                o.choice(p, [("off", "No walls"), ("on", "Four walls")],
                         "on" if room["walls"] else "off", self._set_walls).pack(
                    side="top", anchor="w", pady=(0, o.px(4)))
                if not room["walls"]:
                    continue
                for dim, what in (("width", "Width (m)"), ("depth", "Depth (m)"),
                                  ("height", "Height (m)")):
                    self._slider(p, dim, what, lambda dim=dim: room[dim], put(dim),
                                 *sc.ROOM_LIMITS[dim], 0.25)
            self._surface(key)
            if key == "floor":
                self._floor_shape()
        o.label(p, "Write what the surface is - \"polished concrete with worn yellow "
                "safety lines\", \"whitewashed brick\" - and Make: the Image Studio "
                "makes a flat, repeating picture of it with the model chosen below, and "
                "it is laid on the surface. The words also go into the prompt.",
                "faint", self.host.f_small, wraplength=o.px(310)).pack(side="top", fill="x")
        self._words_box()

    def _surface(self, key):
        """One surface's words, its Make / Picture… / Plain, what it wears now
        and how big one copy of the picture is."""
        o, p = self.owner, self.panel
        face = self.scene["room"][key]

        def said(v):
            face["prompt"] = v
            self.dirty = True
            self._words()
            self.remember_soon()
        self._text(p, face["prompt"], said, height=2)
        row = o.frame(p)
        row.pack(side="top", fill="x", pady=(o.px(4), 0))
        o.button(row, "Make " + sc.SURFACE_NAMES[key].lower(),
                 lambda: self.make_texture(key), kind="accent").pack(side="left")
        o.button(row, "Picture…", lambda: self.choose_texture(key)).pack(
            side="left", padx=(o.px(4), 0))
        o.button(row, "Plain", lambda: self.set_texture(key, ""), kind="ghost").pack(
            side="left", padx=(o.px(4), 0))
        if key in self.making:
            now = "Making it in the Image Studio…"
        elif face["image"]:
            now = "Wearing %s" % os.path.basename(face["image"])
            if sc.texture(face["image"]) is None:
                now += " - missing or unreadable, so drawn plain"
        else:
            now = "Plain."
        o.label(p, now, "muted", self.host.f_small, wraplength=o.px(310)).pack(
            side="top", fill="x", pady=(o.px(2), o.px(2)))
        self._slider(p, key + "_size", "Pattern size (m)", lambda: face["size"],
                     lambda x: face.__setitem__("size", x), 0.25, 10, 0.25)

    # ---------------------------------------------------------- floor shape
    # The floor from above: the 4 x 4 dots (`room["grid"]`) over the room's
    # width x depth, each dragged up or down, the floor shaded by height
    # under them, front (where the camera starts) at the bottom. The camera
    # and what stands on the floor are marked, so the view can be read.
    SHAPE_SIDE, SHAPE_PAD = 300, 18        # px, before scaling
    SHAPE_HIT = 11                          # px from a dot that grabs it
    SHAPE_SNAP = 0.05                       # m a dot moves in
    SHAPE_PX_PER_M = 40                     # px dragged for a metre, before scaling
    SHAPE_CELLS = 18                        # the shading's cells each way
    SHAPE_COLOURS = ((-2.0, (47, 75, 102)), (0.0, sc.FLOOR), (2.0, (217, 201, 163)),
                     (4.0, (246, 234, 208)))

    def _floor_shape(self):
        o, p, room = self.owner, self.panel, self.scene["room"]
        o.cap(p, "Floor shape")
        row = o.frame(p)
        row.pack(side="top", fill="x", pady=(0, o.px(4)))
        o.choice(row, [(k, label) for k, label, _ in sc.FLOOR_PRESETS], "Shape like…",
                 self.floor_preset).pack(side="left")
        o.button(row, "Flatten", lambda: self.floor_preset("flat"), kind="ghost").pack(
            side="left", padx=(o.px(4), 0))
        side = o.px(self.SHAPE_SIDE)
        c = tk.Canvas(p, width=side, height=side, highlightthickness=0, bd=0,
                      cursor="sb_v_double_arrow")
        o.skin(c, bg="card")
        c.pack(side="top", anchor="w")
        c.bind("<ButtonPress-1>", self._shape_press)
        c.bind("<B1-Motion>", self._shape_motion)
        c.bind("<ButtonRelease-1>", self._shape_release)
        c.bind("<Button-3>", self._shape_reset)
        self.shape_canvas, self.shape_drag = c, None
        if room["walls"]:
            o.label(p, "The shape spans the room's floor, inside the walls.", "faint",
                    self.host.f_small).pack(side="top", anchor="w")
        else:
            for dim, what in (("width", "Area width (m)"), ("depth", "Area depth (m)")):
                self._slider(p, "area_" + dim, what, lambda dim=dim: room[dim],
                             lambda x, dim=dim: self._room_size(dim, x),
                             *sc.ROOM_LIMITS[dim], 0.25)
        o.label(p, "Drag a dot up to raise the floor round it, down to lower it; "
                "right-click one to put it back to 0. Each dot pulls a wide, smooth area, "
                "not one point. Front, at the bottom, is the side the camera starts on. "
                "Everything on the floor stays standing on it.", "faint",
                self.host.f_small, wraplength=o.px(310)).pack(side="top", fill="x")
        self._draw_shape()

    def _room_size(self, dim, x):
        """The room's width or depth: the floor's shape stretches with it,
        and what stands on the floor stays on it."""
        sc.shape_floor(self.scene, **{dim: x})

    def _shape_frame(self):
        """-> (to canvas, the dot's canvas point): the shaped area fitted
        in the square, left to right and back (top) to front (bottom)."""
        room, o = self.scene["room"], self.owner
        w, d, side = room["width"], room["depth"], o.px(self.SHAPE_SIDE)
        k = (side - 2 * o.px(self.SHAPE_PAD)) / max(w, d)
        x0, y0 = (side - w * k) / 2, (side - d * k) / 2
        xy = lambda x, z: (x0 + (x + w / 2) * k, y0 + (z + d / 2) * k)  # noqa: E731
        dot_xy = lambda r, c: xy(-w / 2 + w * c / (sc.GRID - 1),  # noqa: E731
                                 -d / 2 + d * r / (sc.GRID - 1))
        return xy, dot_xy

    def _shape_grid(self):
        """The dots' heights, rows back to front: a flat floor's are 0."""
        grid = self.scene["room"].get("grid") or []
        return [list(r) for r in grid] or [[0.0] * sc.GRID for _ in range(sc.GRID)]

    def _height_colour(self, h):
        stops = self.SHAPE_COLOURS
        if h <= stops[0][0]:
            return rgb_hex(stops[0][1])
        for (ha, ca), (hb, cb) in zip(stops, stops[1:]):
            if h <= hb:
                t = (h - ha) / (hb - ha)
                return rgb_hex(tuple(int(round(u + (v - u) * t)) for u, v in zip(ca, cb)))
        return rgb_hex(stops[-1][1])

    def _draw_shape(self):
        c = self.shape_canvas
        if c is None:
            return
        try:
            c.delete("all")
        except tk.TclError:
            self.shape_canvas = None
            return
        o, C, small = self.owner, self.host.C, self.host.f_small
        room = self.scene["room"]
        side = o.px(self.SHAPE_SIDE)
        xy, dot_xy = self._shape_frame()
        g = sc.Ground(room)
        w, d, n = room["width"], room["depth"], self.SHAPE_CELLS
        for j in range(n):                            # the floor, shaded by height
            for i in range(n):
                xa, za = -w / 2 + w * i / n, -d / 2 + d * j / n
                (ax, ay), (bx, by) = xy(xa, za), xy(xa + w / n, za + d / n)
                c.create_rectangle(ax, ay, bx + 1, by + 1, outline="",
                                   fill=self._height_colour(g.at(xa + w / n / 2,
                                                                 za + d / n / 2)))
        (ax, ay), (bx, by) = xy(-w / 2, -d / 2), xy(w / 2, d / 2)
        c.create_rectangle(ax, ay, bx, by, outline=C["border"])
        c.create_text(side / 2, ay - 2, text="back", anchor="s", fill=C["faint"], font=small)
        c.create_text(side / 2, by + 2, text="front", anchor="n", fill=C["faint"],
                      font=small)
        for obj in self.scene["objects"]:             # what stands on it
            x, y = xy(obj["position"][0], obj["position"][2])
            if 0 <= x <= side and 0 <= y <= side:
                chosen = obj["id"] == self.sel
                c.create_rectangle(x - 3, y - 3, x + 3, y + 3, outline=C["card"],
                                   fill=C["accent"] if chosen else C["text"])
        grid, r = self._shape_grid(), o.px(6)
        held = self.shape_drag and (self.shape_drag["r"], self.shape_drag["c"])
        for row in range(sc.GRID):
            for col in range(sc.GRID):
                x, y = dot_xy(row, col)
                big = r + 2 if held == (row, col) else r
                c.create_oval(x - big, y - big, x + big, y + big, fill=C["accent"],
                              outline=C["card"], width=2)
                v = grid[row][col]
                if v:
                    c.create_text(x + r + 2, y - r, text="%+.2g" % v, anchor="sw",
                                  fill=C["text"], font=small)
        cam = sc.Camera(self.scene["camera"], 1, 1)   # last, on top: the camera, looking
        x, y = xy(cam.eye[0], cam.eye[2])
        x, y = max(6, min(side - 6, x)), max(6, min(side - 6, y))
        f = sc.norm((cam.f[0], 0.0, cam.f[2])) if abs(cam.f[1]) < 0.999 else (0.0, 0.0, -1.0)
        s = o.px(8)
        tip = (x + f[0] * s, y + f[2] * s)
        wing = (-f[2] * s * 0.6, f[0] * s * 0.6)
        c.create_polygon(tip[0], tip[1], x - f[0] * s * 0.5 + wing[0],
                         y - f[2] * s * 0.5 + wing[1], x - f[0] * s * 0.5 - wing[0],
                         y - f[2] * s * 0.5 - wing[1], fill=C["text"], outline=C["card"])

    def _shape_hit(self, x, y):
        """The (row, column) of the dot under (x, y) on the editor, or None."""
        _, dot_xy = self._shape_frame()
        best = None
        for row in range(sc.GRID):
            for col in range(sc.GRID):
                dx, dy = dot_xy(row, col)
                d = math.hypot(x - dx, y - dy)
                if d <= self.owner.px(self.SHAPE_HIT) and (best is None or d < best[0]):
                    best = (d, row, col)
        return None if best is None else best[1:]

    def _shape_press(self, ev):
        hit = self._shape_hit(ev.x, ev.y)
        if hit is None:
            return
        r, c = hit
        self.shape_drag = {"r": r, "c": c, "y": ev.y, "grid": self._shape_grid()}
        self.shape_drag["start"] = self.shape_drag["grid"][r][c]
        self._draw_shape()

    def _shape_motion(self, ev):
        d = self.shape_drag
        if not d:
            return
        lo, hi = sc.GRID_HEIGHT
        v = d["start"] + (d["y"] - ev.y) / float(self.owner.px(self.SHAPE_PX_PER_M))
        v = max(lo, min(hi, round(round(v / self.SHAPE_SNAP) * self.SHAPE_SNAP, 2)))
        if d["grid"][d["r"]][d["c"]] != v:
            d["grid"][d["r"]][d["c"]] = v
            self._shape_set(d["grid"])

    def _shape_release(self, _ev=None):
        if self.shape_drag:
            self.shape_drag = None
            self._draw_shape()
            self.remember()

    def _shape_reset(self, ev):
        hit = self._shape_hit(ev.x, ev.y)
        if hit is None:
            return
        grid = self._shape_grid()
        grid[hit[0]][hit[1]] = 0.0
        self._shape_set(grid)
        self.remember()

    def floor_preset(self, key):
        self._shape_set(sc.FLOOR_PRESET[key])
        self.remember()
        label = dict((k, v) for k, v, _ in sc.FLOOR_PRESETS)[key]
        self.status("The floor is flat again." if key == "flat" else
                    "Floor: %s. Drag the dots to change it." % label.lower())

    def _shape_set(self, grid):
        """The floor takes these heights, and everything on it stays on it."""
        before = self.scene["room"].get("grid") or []
        sc.shape_floor(self.scene, grid)
        if self.scene["room"]["grid"] != before:
            self.changed()                # redraws the view, this one with it
        else:
            self._draw_shape()

    def _inspect_object(self, obj):
        o, p = self.owner, self.panel
        a = sc.ASSET[obj["asset"]]
        if a["kind"] == "person":
            section = getattr(self, "inspector_section", "Object")
            tabs = o.frame(p)
            tabs.pack(side="top", fill="x", pady=(0, o.px(8)))
            for name in ("Object", "Pose", "Look"):
                o.button(tabs, name, lambda n=name: self._inspector_tab(n),
                         kind="accent" if name == section else "quiet").pack(
                    side="left", padx=(0, o.px(4)))
            if section == "Pose":
                self._pose_controls(obj)
                self._words_box()
                return
            if section == "Look":
                self._look_controls(obj)
                self._face_controls(obj)
                self._words_box()
                return
        o.cap(p, a["label"] if a["kind"] == "person" else "%s - a stand-in" % a["label"])
        name = tk.StringVar(value=obj["name"])
        e = self.host._entry(p, name)
        e.master.pack(side="top", fill="x")

        def renamed(_ev=None):
            obj["name"] = name.get()
            self.dirty = True
            self._list()
            self._words()
            self.remember_soon()
        e.bind("<KeyRelease>", renamed)

        o.cap(p, {"person": "What they are doing",
                  "crowd": "Who they are and what they are doing"}.get(a["kind"],
                                                                       "Description"))

        def described(v):
            obj["description"] = v
            self.dirty = True
            self._title()
            self._words()
            self.remember_soon()
        self._text(p, obj["description"], described)
        if a["kind"] == "crowd":
            self._crowd_controls(obj)

        if a["kind"] == "prop":
            self._picture_controls(obj)
        pictured = bool(sc.texture(obj.get("picture")))

        if a["kind"] != "crowd" and not pictured:          # a crowd's colours are its people's own
            o.cap(p, "Colour (skin, and whatever is not worn)" if a["kind"] == "person"
                  else "Colour")
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

        o.cap(p, "Place")
        pos, rot, scale = obj["position"], obj["rotation"], obj["scale"]

        def at(vec, i):
            def write(x):
                vec[i] = x
            return write
        room = self.scene["room"]

        def along(i):                 # on a shaped floor, it rides it
            def write(x):
                lift = sc.above_floor(room, obj)
                pos[i] = x
                sc.stand(room, obj, lift)
            return write
        self._slider(p, "x", "Left / right (m)", lambda: pos[0], along(0),
                     -REACH, REACH, 0.05)
        self._slider(p, "z", "Back / front (m)", lambda: pos[2], along(2),
                     -REACH, REACH, 0.05)
        self._slider(p, "y", "Above the floor (m)",
                     lambda: round(sc.above_floor(room, obj), 3),
                     lambda x: sc.stand(room, obj, x), 0, 3, 0.05)
        if not pictured:
            self._slider(p, "yaw", "Turn", lambda: rot[0], at(rot, 0), -180, 180)
        if pictured:
            def tall(x):                  # the picture keeps its shape
                scale[0], scale[1] = round(scale[0] * x / scale[1], 3), x
            o.cap(p, "Size (m)")
            self._slider(p, "size1", "Height", lambda: scale[1], tall, 0.05, 5, 0.05)
        elif a["kind"] == "prop":
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

    def _inspector_tab(self, name):
        self.inspector_section = name
        self._inspect()
        self.panel.canvas.yview_moveto(0)

    def _face_controls(self, obj):
        """The face this person is drawn with, last of all: a picture of
        theirs, else their character's face photos or identity's, else none (the face is
        redrawn from the words)."""
        o, p = self.owner, self.panel
        o.cap(p, "Face")
        lib = o.studio.lib
        path, source = sc.face_picture(obj, {c["id"]: c for c in lib.all("characters")},
                                       {d["id"]: d for d in lib.all("identities")})
        row = o.frame(p)
        row.pack(side="top", fill="x")
        thumb = self._face_thumb(path)
        if thumb is not None:
            tk.Label(row, image=thumb, bd=0).pack(side="left", padx=(0, o.px(6)))
            self.face_thumb = thumb
        o.label(row, os.path.basename(path) if path else "None: drawn from the words",
                "muted" if path else "faint", self.host.f_small).pack(side="left")
        o.button(row, "Clear", lambda: self._set_face(""), kind="ghost").pack(side="right")
        o.button(row, "Choose…", self.choose_face, kind="quiet").pack(side="right")
        o.label(p, ("From %s. " % source if path else "") +
                "Their face is drawn from it in the picture itself, then refined close up "
                "at the end. Use a clear, front-on picture of this one person: in a picture "
                "of two, the bigger face is taken. Say glasses and skin in the look. FLUX "
                "only.", "faint", self.host.f_small,
                wraplength=o.px(310)).pack(side="top", fill="x")

    def _face_thumb(self, path, side=48):
        """A small PhotoImage of `path`, or None when Tk cannot read it
        (Tk reads PNG and GIF; a JPEG is shown by name alone)."""
        if not path or not path.lower().endswith((".png", ".gif")):
            return None
        try:
            img = tk.PhotoImage(master=self.win, file=path)
        except tk.TclError:
            return None
        k = max(1, -(-max(img.width(), img.height()) // side))
        return img.subsample(k, k)

    def choose_face(self):
        obj = self.obj()
        if obj is None or obj["asset"] != "person":
            return
        path = filedialog.askopenfilename(
            parent=self.win, title="A picture of %s's face" % obj["name"],
            filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")])
        if path:
            try:
                path = self.owner.studio.lib.keep_reference(path, "scene faces")
            except OSError as e:
                self.status("Could not keep a copy of %s: %s" % (os.path.basename(path), e),
                            "err")
                return
            self._set_face(path)

    def _set_face(self, path):
        obj = self.obj()
        if obj is None or obj["asset"] != "person":
            return
        obj["face"] = path
        self._inspect()
        self.changed()

    def _look_controls(self, obj):
        """Who this person is, as the Image Studio's character creator says
        it: a character copied on, then the look's sections one at a time -
        the same slots, picks and sliders as the form, kept on this person."""
        o, p = self.owner, self.panel
        o.cap(p, "Look")
        chars = [("", "No character")] + [(c["id"], c["name"])
                                          for c in o.studio.lib.all("characters")]
        if obj["character"] and obj["character"] not in dict(chars):
            chars.append((obj["character"], "%s (not in the library)" % obj["character"]))
        row = o.frame(p)
        row.pack(side="top", fill="x")
        o.choice(row, chars, obj["character"], self._set_character).pack(side="left")
        o.button(row, "Clear look", self._clear_look, kind="ghost").pack(side="right")
        tabs = o.frame(p)
        tabs.pack(side="top", fill="x", pady=(o.px(6), o.px(2)))
        for i, (name, _) in enumerate(ig.LOOKS):
            o.button(tabs, name, lambda n=name: self._look_tab(n),
                     kind="accent" if name == self.look_section else "quiet",
                     font=self.host.f_small, padx=o.px(6), pady=o.px(2)).grid(
                row=i // 3, column=i % 3, sticky="ew", padx=(0, o.px(3)), pady=(0, o.px(3)))
        for col in range(3):
            tabs.columnconfigure(col, weight=1)
        look = obj["look"]
        section = dict(ig.LOOKS)[self.look_section]
        text = {k: tk.StringVar(value=look.get(k, "")) for k, *_ in section}
        steps = {k: tk.IntVar(value=int(look.get(k, 0))) for k in ig.SLIDER_KEYS}
        relight = [None]

        def changed():
            for k, var in text.items():
                v = var.get().strip()
                if v:
                    look[k] = v
                else:
                    look.pop(k, None)
            if self.look_section == ig.SLIDER_SECTION:
                for k, var in steps.items():
                    if int(var.get()):
                        look[k] = int(var.get())
                    else:
                        look.pop(k, None)
            if relight[0]:
                relight[0]()
            # The body and the clothes are the mannequin's shape and colours.
            self.changed()
        if self.look_section == ig.SLIDER_SECTION:
            o.slider_rows(p, steps, changed)
        relight[0] = o.look_rows(p, section, text, changed)
        if self.look_section in ("Clothes", "Accessories"):
            self._outfit_controls(obj)
        self.look_vars, self.look_changed = text, changed

    def _crowd_controls(self, obj):
        """How many, over how much floor, facing which way, doing what, and
        dressed from which outfit presets; Shuffle deals a new crowd."""
        o, p, c = self.owner, self.panel, obj["crowd"]
        o.cap(p, "Crowd")

        def put(key, conv=float):
            def write(x):
                c[key] = conv(x)
            return write
        self._slider(p, "crowd_count", "People", lambda: c["count"],
                     put("count", lambda x: int(round(x))), *sc.CROWD_LIMITS["count"])
        self._slider(p, "crowd_width", "Spread wide (m)", lambda: c["width"], put("width"),
                     *sc.CROWD_LIMITS["width"], 0.25)
        self._slider(p, "crowd_depth", "Spread deep (m)", lambda: c["depth"], put("depth"),
                     *sc.CROWD_LIMITS["depth"], 0.25)

        def pick(key):
            def picked(v):
                c[key] = v
                self.changed()
            return picked
        for key, label, items in (("facing", "Facing", sc.CROWD_FACING),
                                  ("activity", "Doing", sc.CROWD_ACTIVITY)):
            row = o.frame(p)
            row.pack(side="top", fill="x", pady=(o.px(4), 0))
            o.label(row, label, "muted", self.host.f_small, width=16).pack(side="left")
            o.choice(row, items, c[key], pick(key)).pack(side="left")
        row = o.frame(p)
        row.pack(side="top", fill="x", pady=(o.px(4), 0))
        o.label(row, "Dressed in", "muted", self.host.f_small, width=16).pack(side="left")
        presets = self.owner.studio.lib.all("outfits")
        items = [("", "Everyday clothes")]
        if presets:
            items += [("*", "Every outfit preset, mixed")] + [
                (r["id"], r["name"]) for r in presets]
        now = c.get("dressed", "")
        if now and now not in dict(items):
            items.append((now, now + " (as it was)"))
        o.choice(row, items, now, lambda v: self._dress_crowd(obj, v)).pack(side="left")
        row = o.frame(p)
        row.pack(side="top", fill="x", pady=(o.px(6), 0))
        o.button(row, "Shuffle the people", lambda: self._shuffle_crowd(obj)).pack(
            side="left")
        o.label(p, "Each person is different: height, build, skin, clothes, pose. "
                "Shuffle deals a new set; the words say only the crowd.", "faint",
                self.host.f_small, wraplength=o.px(310)).pack(side="top", fill="x")

    def _dress_crowd(self, obj, choice):
        """Dress a crowd from outfit presets: copies of their looks, as a
        character's look is copied onto a person."""
        presets = self.owner.studio.lib.all("outfits")
        if choice == "*":
            wear = presets
        elif choice:
            wear = [r for r in presets if r["id"] == choice]
            if not wear:
                return                      # "as it was": keep what it wears
        else:
            wear = []
        obj["crowd"]["wear"] = [dict(r["looks"]) for r in wear]
        obj["crowd"]["dressed"] = choice
        self.changed()

    def _shuffle_crowd(self, obj):
        import random
        obj["crowd"]["seed"] = random.randrange(1, 2 ** 31)
        self.changed()

    def _look_tab(self, name):
        self.inspector_section = "Look"
        self.look_section = name
        self._inspect()

    def _set_character(self, cid):
        """Copy a character's look onto the selected person, as the form
        does: a copy, not a link, so the scene file holds the whole look.
        A person still under the library's default name takes the
        character's."""
        obj = self.obj()
        if obj is None or obj["asset"] != "person":
            return
        rec = self.owner.studio.lib.get("characters", cid) if cid else None
        obj["character"] = cid if rec is not None else ""
        if rec is not None:
            obj["look"] = sc.character_look(rec, obj["look"])
            # Their identity's head shape comes with them, when it has one.
            ident = self._identity_of(obj)
            head = sc.mq.clean_head((ident or {}).get("head"))
            if head:
                obj["head"] = head
            default = sc.ASSET["person"]["name"]
            if obj["name"] == default or obj["name"].startswith(default + " "):
                taken = {x["name"] for x in self.scene["objects"] if x is not obj}
                if rec["name"] not in taken:
                    obj["name"] = rec["name"]
        self._list()
        self._inspect()
        self.changed()

    def _outfit_controls(self, obj):
        """Clothes presets, kept in the Image Studio's library (`outfits`):
        put one on, or save what this person wears under a name. Putting
        one on replaces the Clothes and Accessories slots, as a copy."""
        o, p = self.owner, self.panel
        lib = o.studio.lib
        o.cap(p, "Outfit presets")
        row = o.frame(p)
        row.pack(side="top", fill="x")
        presets = [(r["id"], r["name"]) for r in lib.all("outfits")]
        o.choice(row, presets or [("", "No presets yet")], "",
                 self._put_on_outfit).pack(side="left")
        o.label(row, "puts on its clothes and accessories", "faint",
                self.host.f_small).pack(side="left", padx=(o.px(6), 0))
        row = o.frame(p)
        row.pack(side="top", fill="x", pady=(o.px(4), 0))
        name = tk.StringVar(value=self.outfit_name)
        e = self.host._entry(row, name)
        e.master.pack(side="left", fill="x", expand=True)

        def typed(_ev=None):
            self.outfit_name = name.get()
        e.bind("<KeyRelease>", typed)
        o.button(row, "Delete", lambda: self._delete_outfit(name.get()),
                 kind="ghost", font=self.host.f_small).pack(side="right",
                                                            padx=(o.px(4), 0))
        o.button(row, "Save outfit", lambda: self._save_outfit(name.get()),
                 font=self.host.f_small).pack(side="right", padx=(o.px(4), 0))

    def _put_on_outfit(self, oid):
        obj = self.obj()
        rec = self.owner.studio.lib.get("outfits", oid) if oid else None
        if obj is None or obj["asset"] != "person" or rec is None:
            return
        obj["look"] = sc.wear_outfit(rec, obj["look"])
        self.outfit_name = rec["name"]
        self._inspect()
        self.changed()

    def _save_outfit(self, name):
        """What the selected person wears, as a preset called `name`; a
        preset of that name already is replaced, after asking."""
        obj, name = self.obj(), name.strip()
        if obj is None or obj["asset"] != "person":
            return
        if not name:
            self.status("Name the outfit first, in the box beside Save outfit.", "warn")
            return
        looks = sc.outfit_looks(obj["look"])
        if not looks:
            self.status("This person wears nothing in the Clothes or Accessories "
                        "slots yet, so there is no outfit to save.", "warn")
            return
        lib = self.owner.studio.lib
        records = [dict(r) for r in lib.all("outfits")]
        same = next((r for r in records if r["name"].lower() == name.lower()), None)
        if same is not None:
            if not messagebox.askyesno("Scene Builder", "Replace the outfit preset "
                                       "\"%s\"?" % same["name"], parent=self.win):
                return
            same["looks"] = looks
        else:
            records.append({"name": name, "looks": looks})
        try:
            lib.save("outfits", records)
        except OSError as e:
            self.status("Could not save the outfit: %s" % e, "err")
            return
        self.outfit_name = name
        self.status("Saved the outfit \"%s\"." % name, "ok")
        self._inspect()

    def _delete_outfit(self, name):
        lib = self.owner.studio.lib
        name = name.strip()
        rec = next((r for r in lib.all("outfits") if r["name"].lower() == name.lower()),
                   None)
        if rec is None:
            self.status("No outfit preset is called \"%s\". Type the name of one "
                        "to delete it." % name if name else
                        "Type the name of the outfit preset to delete.", "warn")
            return
        if not messagebox.askyesno("Scene Builder", "Delete the outfit preset \"%s\"? "
                                   "People already wearing it keep their clothes."
                                   % rec["name"], parent=self.win):
            return
        try:
            lib.save("outfits", [r for r in lib.all("outfits") if r is not rec])
        except OSError as e:
            self.status("Could not delete the outfit: %s" % e, "err")
            return
        self.outfit_name = ""
        self.status("Deleted the outfit \"%s\"." % rec["name"], "ok")
        self._inspect()

    def _clear_look(self):
        obj = self.obj()
        if obj is None or obj["asset"] != "person":
            return
        obj["look"], obj["character"] = {}, ""
        self._inspect()
        self.changed()

    def _picture_controls(self, obj):
        """A prop's picture: Make paints it from its name and description,
        Picture… takes a PNG, Shape goes back to the grey box."""
        o, p = self.owner, self.panel
        o.cap(p, "Looks like")
        row = o.frame(p)
        row.pack(side="top", fill="x")
        o.button(row, "Make picture", lambda: self.make_picture(obj["id"]),
                 kind="accent").pack(side="left")
        o.button(row, "Picture…", lambda: self.choose_picture(obj["id"])).pack(
            side="left", padx=(o.px(4), 0))
        o.button(row, "Shape", lambda: self.set_picture(obj["id"], ""), kind="ghost").pack(
            side="left", padx=(o.px(4), 0))
        if "obj:" + obj["id"] in self.making:
            now = "Making it in the Image Studio…"
        elif obj.get("picture"):
            now = "A picture: %s" % os.path.basename(obj["picture"])
            if sc.texture(obj["picture"]) is None:
                now += " - missing or unreadable, so drawn as its shape"
        else:
            now = "Its shape, in its colour."
        o.label(p, now, "muted", self.host.f_small, wraplength=o.px(310)).pack(
            side="top", fill="x", pady=(o.px(2), 0))
        o.label(p, "Make paints it from its name and description in the Image Studio, on "
                "white, cuts it out and stands it here in place of the shape, turned to "
                "the camera. It keeps its height; its width follows the picture.",
                "faint", self.host.f_small, wraplength=o.px(310)).pack(side="top", fill="x")

    def _pose_controls(self, obj):
        o, p = self.owner, self.panel
        pose = obj["pose"]
        o.cap(p, "Pose")
        row = o.frame(p)
        row.pack(side="top", fill="x")
        self.pose_pill = o.choice(row, [(k, label) for k, label, _ in sc.POSES] +
                                  ([("", "Custom")] if not pose["preset"] else []),
                                  pose["preset"], self._set_pose)
        self.pose_pill.pack(side="left")
        busy = obj["id"] in self.posing
        photo = o.button(row, "Finding the pose…" if busy else "From a photo…",
                         lambda: self.pose_from_photo(obj["id"]), kind="ghost",
                         font=self.host.f_small)
        photo.pack(side="left", padx=(o.px(6), 0))
        if busy:
            photo.set(state="disabled")
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
                if key in ("head_turn", "head_nod") and obj.pop("look_at", None):
                    self.status("%s no longer looks at a point." % obj["name"])
                if pose["preset"]:        # moved by hand: no longer the preset
                    pose["preset"] = ""
                    self.pose_pill.set(text="Custom  ▾")
            self._slider(p, key, label, lambda key=key: ctl[key], write, lo, hi)
        o.button(p, "Reset %s" % sc.PART_NAMES[self.part].lower(), self._reset_part,
                 kind="ghost").pack(side="top", anchor="w", pady=(o.px(4), 0))
        if self.part == "head":
            eyes = o.frame(p)
            eyes.pack(side="top", fill="x", pady=(o.px(4), 0))
            o.button(eyes, "Look at camera", self.look_at_camera,
                     kind="ghost").pack(side="left")
            if obj.get("look_at"):
                o.button(eyes, "Stop looking", self.stop_looking,
                         kind="ghost").pack(side="left", padx=(o.px(6), 0))
            o.label(p, "Or press L and click where they should look.",
                    "faint", self.host.f_small, wraplength=o.px(310)).pack(side="top",
                                                                           fill="x")
            self._head_shape(obj)
        o.label(p, "Click a hand, foot or the head in the viewport to pose that part.",
                "faint", self.host.f_small, wraplength=o.px(310)).pack(side="top",
                                                                       fill="x")

    def _head_shape(self, obj):
        """The head's proportions (`mq.HEAD_SHAPE`), the middle of each its
        own; and, when their character has an identity, its head to load
        or this one to save to it, so every scene of theirs inherits it."""
        o, p = self.owner, self.panel
        o.cap(p, "Head shape")
        for key, label, _, _ in sc.mq.HEAD_SHAPE:
            def read(key=key):
                return (obj.get("head") or {}).get(key, 0.5)

            def write(x, key=key):
                head = dict(obj.get("head") or {}, **{key: x})
                head = sc.mq.clean_head(head)
                if head:
                    obj["head"] = head
                else:
                    obj.pop("head", None)
            self._slider(p, key, label, read, write, 0.0, 1.0, 0.01)
        row = o.frame(p)
        row.pack(side="top", fill="x", pady=(o.px(4), 0))
        o.button(row, "Reset shape", self._reset_head, kind="ghost").pack(side="left")
        ident = self._identity_of(obj)
        if ident is not None:
            if ident.get("head"):
                o.button(row, "Load %s's" % ident["name"], self._load_head,
                         kind="ghost").pack(side="left", padx=(o.px(6), 0))
            o.button(row, "Save to %s" % ident["name"], self._save_head,
                     kind="ghost").pack(side="left", padx=(o.px(6), 0))

    def _identity_of(self, obj):
        """The identity record behind a person's character, else None."""
        lib = self.owner.studio.lib
        rec = lib.get("characters", obj.get("character")) if obj.get("character") else None
        iid = (rec or {}).get("identity")
        return lib.get("identities", iid) if iid else None

    def _reset_head(self):
        obj = self.obj()
        if obj is not None and obj.pop("head", None) is not None:
            self.sync()
            self.changed()

    def _load_head(self):
        obj = self.obj()
        ident = self._identity_of(obj) if obj else None
        if ident is None:
            return
        head = sc.mq.clean_head(ident.get("head"))
        if head:
            obj["head"] = head
        else:
            obj.pop("head", None)
        self.sync()
        self.changed()
        self.status("%s has %s's head shape." % (obj["name"], ident["name"]))

    def _save_head(self):
        """This person's head shape onto their identity, in the library."""
        obj = self.obj()
        ident = self._identity_of(obj) if obj else None
        if ident is None:
            return
        lib = self.owner.studio.lib
        records = [dict(r, head=obj.get("head") or {}) if r["id"] == ident["id"] else r
                   for r in lib.all("identities")]
        try:
            lib.save("identities", records)
        except OSError as e:
            self.status("Could not save: %s" % e, "err")
            return
        self.status("Saved the head shape to %s; every scene of theirs can load it."
                    % ident["name"], "ok")
        self._inspect()

    def _words_box(self):
        o = self.owner
        o.cap(self.panel, "Words sent with the viewfinder")
        box = o.frame(self.panel, "card")
        box.pack(side="top", fill="x", pady=(0, o.px(12)))
        self.words_label = o.label(box, "", "text", self.host.f_small, bg="card",
                                   wraplength=o.px(300))
        self.words_label.pack(side="top", fill="x", padx=o.px(8), pady=o.px(6))
        self._words()

    def _words(self):
        self._title()
        self.owner.show_shot_on()
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
    def _set_walls(self, on):
        self.scene["room"]["walls"] = on == "on"
        self._inspect()
        self.changed()
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
        self.inspector_section = "Pose"
        obj["pose"] = {"preset": preset, "controls": sc.pose_controls(preset)}
        self._inspect()
        self.changed()

    def pose_from_photo(self, oid, path=None):
        """Pose a person as the most prominent person in a photo: a backend's
        pose finder (`Studio.find_poses`) reads the photo's points and
        `studio_scene.fit_pose` fits the controls and the way they face to
        them, both off the UI thread; the result lands through `_posed`."""
        obj = self.obj(oid)
        if obj is None or "pose" not in obj or oid in self.posing:
            return False
        if path is None:
            path = filedialog.askopenfilename(
                parent=self.win, title="A photo of the pose for %s" % obj["name"],
                filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp *.bmp"),
                           ("All files", "*.*")])
        if not path:
            return False
        shape = sc.body_shape(obj.get("look"))
        box = {}

        def work():
            try:
                data, b = self.owner.studio.find_poses(path)
                folk = sc.photo_people(data)
                if not folk:
                    raise ValueError("%s found no one in %s." % (b["name"],
                                                                os.path.basename(path)))
                box["fit"] = sc.fit_pose(folk[0]["points"], shape, folk[0]["box"])
                box["count"] = len(folk)
            except (ig.ComfyError, OSError, ValueError) as e:
                box["error"] = e
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        self.posing.add(oid)
        if self.sel == oid:
            self._inspect()
        self.status("Finding the pose in %s…" % os.path.basename(path), "muted")

        scene = self.scene

        def wait():
            if worker.is_alive():
                self.win.after(100, wait)
                return
            self.posing.discard(oid)
            if self.scene is scene:       # not another scene opened meanwhile
                self._posed(oid, path, box)
        self.win.after(100, wait)
        return True

    def _posed(self, oid, path, box):
        obj = self.obj(oid)
        if obj is None:                   # deleted meanwhile
            return
        name = os.path.basename(path)
        if "error" in box:
            if self.sel == oid:
                self._inspect()
            self.status("Could not pose from %s: %s" % (name, box["error"]), "err")
            return
        fit = box["fit"]
        obj["pose"] = {"preset": "", "controls": fit.controls}
        # The photo's camera is the scene's: facing it, then turned as they were.
        w, h = sc.frame_size(self.scene)
        eye = sc.Camera(self.scene["camera"], w, h).eye
        x, _, z = obj["position"]
        toward = math.degrees(math.atan2(eye[0] - x, eye[2] - z))
        obj["rotation"][0] = round((toward + fit.yaw + 180) % 360 - 180, 1)
        if self.sel == oid:
            self._inspect()
        self.changed()
        said = ["Posed %s from %s" % (obj["name"], name)]
        if box["count"] > 1:
            said.append("the most prominent of %d people" % box["count"])
        if fit.unseen:
            said.append("the photo does not show %s, so %s at rest"
                        % (" or ".join(fit.unseen), "it is" if len(fit.unseen) == 1
                                                    else "they are"))
        self.status("; ".join(said) + "." + (
            " The fit is rough: adjust it by hand." if fit.rough else
            " A flat photo cannot say how far a limb reaches towards the camera; "
            "adjust it by hand if it looks off."), "err" if fit.rough else "ok")

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
        self.aiming = self.ring = None
        if tool == "look":
            self.status("Click a person to choose where they look.")
        self.draw()
        for key, pill in self.tool_pills.items():
            kind = "accent" if key == tool else "quiet"
            pill.roles = self.host.PILL_ROLES[kind]
            pill.paint(self.host.C)

    def reset_camera(self):
        """Back to the starting view; the camera body stays the same."""
        body = self.scene["camera"].get("profile")
        self.scene["camera"] = sc.new_scene()["camera"]
        self._bake_camera(body)
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
        room = [p for p in polys if p.owner is None]
        # The room flat first, everywhere (outside the frame too), shadows
        # as their flat stand-in colours; then, in the frame, its pictures
        # once baked for this view, shadows multiplied in.
        for poly in room:
            c.create_polygon(at(poly.pts), fill=rgb_hex(poly.rgb), outline="")
        if any(p.tex for p in room):
            img = self._backdrop(w, h, k, room)
            if img is not None:
                c.create_image(ox, oy, image=img, anchor="nw")
        for a, b, axis in sc.grid_lines(self.scene, w, h):
            c.create_line(*at((a, b)), fill=GRID_AXIS if axis else GRID)
        # The chosen object is outlined face by face - but a crowd, whose
        # faces are many small people, gets a box round it instead, or the
        # outline is all there is to see of them.
        sel = self.obj()
        crowd = sel is not None and sel["asset"] == "crowd"
        around = []
        cards, self.cards = self.cards, {}
        for poly in polys:
            if poly.owner is None:
                continue
            if isinstance(poly.tex, sc.CutMap):
                self._card(poly, at(poly.pts), cards)
                continue
            fill = rgb_hex(poly.rgb)
            chosen = poly.owner == self.sel
            c.create_polygon(at(poly.pts), fill=fill,
                             outline=C["accent"] if chosen and not crowd else fill,
                             tags=("o:" + poly.owner, "p:" + str(poly.part)))
            if chosen and crowd:
                around += at(poly.pts)
        if around:
            c.create_rectangle(min(around[0::2]) - 4, min(around[1::2]) - 4,
                               max(around[0::2]) + 4, max(around[1::2]) + 4,
                               outline=C["accent"], dash=(4, 3), width=2)
        # Where the chosen person looks: a dashed line from their eyes to a dot.
        if sel is not None and sel.get("look_at"):
            cam = sc.Camera(self.scene["camera"], w, h)
            a, b = cam.project(sc.eye_point(sel)), cam.project(sel["look_at"])
            if a and b:
                c.create_line(*at([a[:2], b[:2]]), fill=C["accent"], dash=(4, 3), width=2)
            if b:
                (bx, by), r = at([b[:2]]), 5
                c.create_oval(bx - r, by - r, bx + r, by + r, fill=C["accent"],
                              outline=C["bg"], width=2)
        x0, y0, x1, y1 = ox, oy, ox + w * k, oy + h * k
        for box in ((0, 0, cw, y0), (0, y1, cw, ch), (0, y0, x0, y1), (x1, y0, cw, y1)):
            c.create_rectangle(*box, fill=C["bg"], outline="", stipple="gray50")
        c.create_rectangle(x0, y0, x1, y1, outline=C["accent"], width=2)
        # The lit rectangle is the viewfinder: what is in it is what the
        # picture is shown (the maps) and told (the words), nothing outside.
        c.create_text(x0 + 6, y0 - 4, anchor="sw", fill=C["muted"], font=self.host.f_small,
                      text="Viewfinder · %s · %dmm · %d x %d" % (
                          self.camera_body() or "no camera set",
                          round(self.scene["camera"]["lens"]), w, h))
        if self.ring and self.obj(self.ring["oid"]) is not None:
            self._draw_ring()
        self._draw_shape()

    def _card(self, poly, flat, old):
        """A prop's picture on the canvas: the picture scaled to the card's
        box on screen (a canvas polygon cannot wear one). Scaled pictures are
        kept between draws, so a drag rescales only when the size changes."""
        c = self.canvas
        xs, ys = flat[0::2], flat[1::2]
        x0, y0 = min(xs), min(ys)
        bw = max(1, min(1600, int(round(max(xs) - x0))))
        bh = max(1, min(1600, int(round(max(ys) - y0))))
        tex = poly.tex.tex
        key = (id(tex), bw, bh)
        img = old.get(key) or self.cards.get(key)
        if img is None:
            rgba = bytearray(bw * bh * 4)
            for y in range(bh):
                j = min(tex.h - 1, y * tex.h // bh) * tex.w
                for x in range(bw):
                    i = j + min(tex.w - 1, x * tex.w // bw)
                    o = (y * bw + x) * 4
                    rgba[o:o + 3] = tex.px[i]
                    rgba[o + 3] = tex.alpha[i]
            data = sc.studio_icons.png(bytes(rgba), bw, bh)
            img = tk.PhotoImage(master=self.win, data=base64.b64encode(data).decode("ascii"))
        self.cards[key] = img
        tags = ("o:" + poly.owner, "p:body")
        c.create_image(x0, y0, image=img, anchor="nw", tags=tags)
        if poly.owner == self.sel:
            c.create_rectangle(x0, y0, x0 + bw, y0 + bh, outline=self.host.C["accent"],
                               tags=tags)

    def _backdrop_key(self, w, h, k, room_polys=None):
        """-> (the view and the room's pictures, the shadows on them)."""
        room = self.scene["room"]
        stamps = []
        for key, _, _ in sc.SURFACES:
            try:
                stamps.append(os.path.getmtime(room[key]["image"]))
            except OSError:
                stamps.append(None)
        if room_polys is None:
            room_polys = [p for p in sc.render(self.scene, w, h) if p.owner is None]
        shadows = [[round(v, 1) for pt in p.pts for v in pt] for p in room_polys if p.dim]
        return (json.dumps([w, h, round(k, 5), self.scene["camera"], room, stamps],
                           sort_keys=True), json.dumps(shadows))

    def _backdrop(self, w, h, k, room_polys):
        """The room's pictures for this exact view, or None while they bake
        (the flat colours stand in). A new view asks for a bake a moment after
        the last change, so a drag is never held up by one. When only the
        shadows moved - an object dragged over a pictured floor - the last
        bake stays up meanwhile, so the floor does not flash plain."""
        key = self._backdrop_key(w, h, k, room_polys)
        if self.backdrop[0] == key:
            return self.backdrop[1]
        if self.bake_after is not None:
            self.win.after_cancel(self.bake_after)
        self.bake_after = self.win.after(150, self._bake)
        if self.backdrop[0] is not None and self.backdrop[0][0] == key[0]:
            return self.backdrop[1]
        return None

    def _bake(self):
        self.bake_after = None
        try:
            ox, oy, w, h, k = self._fit()
            bw, bh = max(8, int(round(w * k / BAKE))), max(8, int(round(h * k / BAKE)))
            data = sc.backdrop_png(self.scene, bw, bh)
            img = tk.PhotoImage(master=self.win, data=base64.b64encode(data).decode("ascii"))
            if BAKE > 1:
                img = img.zoom(BAKE)
        except tk.TclError:
            return
        self.backdrop = (self._backdrop_key(w, h, k), img)
        self.draw()

    def camera(self):
        w, h = sc.frame_size(self.scene)
        return sc.Camera(self.scene["camera"], w, h)

    def camera_body(self):
        """The name of the camera this scene is shot on, or '' for none. A
        scene saved before the name was kept finds it by its profile id."""
        cam = self.scene["camera"]
        if cam.get("body"):
            return cam["body"]
        pid = cam.get("profile")
        cp = self.owner.studio.lib.get("camera_profiles", pid) if pid and pid != "none" else None
        return cp["name"] if cp else ""

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
        # Look at: a click on a person opens the ring of head poses round
        # their head; its Point item makes the next click where they look.
        if self.tool == "look" and self._ring_press(ev.x, ev.y, oid):
            return
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

    RING = 62                # px, the ring's radius
    RING_ITEM = 17           # px, each item's

    def _ring_items(self):
        """-> [(key, label, x, y)] for the open ring: the head poses round
        it, and Camera and Point under it."""
        r = self.ring
        items = []
        for key, label, dx, dy in sc.HEAD_POSES:
            f = self.RING * (0.71 if dx and dy else 1)
            items.append((key, label, r["x"] + dx * f, r["y"] + dy * f))
        below = r["y"] + self.RING + 2.4 * self.RING_ITEM
        items += [("camera", "Camera", r["x"] - 40, below),
                  ("point", "Point\u2026", r["x"] + 40, below)]
        return items

    def _ring_press(self, x, y, oid):
        """A click with Look at; True when it was used up here."""
        if self.aiming:
            aiming, self.aiming = self.obj(self.aiming), None
            if aiming is not None:
                self._look_at(aiming, x, y, oid)
                return True
        if self.ring:
            obj = self.obj(self.ring["oid"])
            hit = next((k for k, _, ix, iy in self._ring_items()
                        if abs(x - ix) <= self.RING_ITEM * (2 if k in ("camera", "point")
                                                            else 1) + 3
                        and abs(y - iy) <= self.RING_ITEM + 3), None)
            self.ring = None
            if obj is None or not hit:
                self.draw()
            elif hit == "camera":
                self.look_at_camera()
            elif hit == "point":
                self.aiming = obj["id"]
                self.status("Click where %s should look - a face, a thing, the floor."
                            % obj["name"])
                self.draw()
            else:
                sc.head_pose(self.scene, obj, hit)
                label = next(lb for k, lb, _, _ in sc.HEAD_POSES if k == hit)
                self.status("%s looks %s." % (obj["name"], label.lower()))
                self.sync()
                self.changed()
            return True
        if oid is None:
            return False
        obj = self.obj(oid)
        if obj["asset"] == "crowd":
            self.status("A crowd's heads can't be posed - only a person's.")
            return True
        if obj["asset"] != "person":
            return False
        self.select(oid, "head")
        w, h = sc.frame_size(self.scene)
        sk, k, shift = sc.rigs(obj)[0]
        head = sc.Camera(self.scene["camera"], w, h).project(
            sc.add(sc.mul(sk["head"][0], k), shift))
        ox, oy, _, _, fk = self.frame_rect
        cx, cy = (ox + head[0] * fk, oy + head[1] * fk) if head else (x, y)
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        m = self.RING + self.RING_ITEM + 4           # the whole ring on the canvas
        self.ring = {"oid": oid, "x": max(m, min(cw - m, cx)),
                     "y": max(m, min(ch - m - 2.4 * self.RING_ITEM, cy))}
        self.status("Choose where %s looks." % obj["name"])
        self.draw()
        return True

    def _draw_ring(self):
        c, C, r = self.canvas, self.host.C, self.ring
        c.create_oval(r["x"] - self.RING, r["y"] - self.RING, r["x"] + self.RING,
                      r["y"] + self.RING, outline=C["accent"], width=2, dash=(3, 3))
        arrows = {"ahead": "\u25cf", "up": "\u2191", "up_right": "\u2197",
                  "right": "\u2192", "down_right": "\u2198", "down": "\u2193",
                  "down_left": "\u2199", "left": "\u2190", "up_left": "\u2196"}
        for key, label, x, y in self._ring_items():
            wide = key in ("camera", "point")
            rx, ry = self.RING_ITEM * (2 if wide else 1), self.RING_ITEM
            box = c.create_rectangle if wide else c.create_oval
            box(x - rx, y - ry, x + rx, y + ry, fill=C["bg"], outline=C["accent"],
                width=2, tags=("ring", "ring:" + key))
            c.create_text(x, y, text=arrows.get(key, label), fill=C["accent"],
                          font=self.host.f_small, tags=("ring", "ring:" + key))

    def _look_at(self, obj, x, y, oid):
        """Point `obj`'s eyes at what is under canvas (x, y): the face
        clicked, at its depth; else the floor; else far off in the sky."""
        cam = self.camera()
        fx, fy = self.to_frame(x, y)
        ray = cam.ray(fx, fy)
        point = None
        if oid is not None:
            under = [p for p in sc.render(self.scene, *sc.frame_size(self.scene))
                     if p.owner is not None and _inside(fx, fy, p.pts)]
            if under:
                depth = min(p.depth for p in under)
                point = sc.add(cam.eye, sc.mul(ray, depth / max(1e-6, sc.dot(ray, cam.f))))
        if point is None:
            point = sc.floor_point(cam, self.scene["room"], fx, fy)
            if point is None or sc.dot(sc.sub(point, cam.eye), sc.sub(point, cam.eye)) > 900:
                point = sc.add(cam.eye, sc.mul(ray, 30.0))
        obj["look_at"] = [round(v, 3) for v in point]
        sc.aim_head(obj)
        what = self.obj(oid)["name"] if oid is not None and oid != obj["id"] else None
        self.status("%s looks at %s." % (obj["name"], what or "that point"))
        self.sync()
        self.changed()

    def look_at_camera(self):
        obj = self.obj()
        if obj is None or obj["asset"] != "person":
            return
        obj["look_at"] = [round(v, 3) for v in self.camera().eye]
        self.sync()
        self.changed()

    def stop_looking(self):
        obj = self.obj()
        if obj is None or not obj.pop("look_at", None):
            return
        self.sync()
        self.changed()

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
            room = self.scene["room"]
            lift = sc.above_floor(room, start)
            if d["shift"] or ev.state & 0x0001:
                sc.stand(room, obj, max(0.0, min(3.0, lift - dy * metres)))
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
                sc.stand(room, obj, lift)
        elif d["kind"] == "rotate":
            yaw = start["rotation"][0] + dx * 0.6
            obj["rotation"][0] = round((yaw + 180) % 360 - 180, 1)
        elif d["kind"] == "scale":
            f = 2 ** (-dy / 150.0)
            lo, hi = (0.5, 1.3) if obj["asset"] in ("person", "crowd") else (0.05, 5)
            obj["scale"] = [round(max(lo, min(hi, v * f)), 3) for v in start["scale"]]
        self.dirty = True
        sc.aim_heads(self.scene)
        self.draw()

    def _release(self, _ev=None):
        if self.drag:
            self.drag = None
            self.sync()
            self._title()
            self.remember()

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
        if key == "escape":
            if self.ring or self.aiming:
                self.ring = self.aiming = None
                self.draw()
                return "break"
            self.select(None)
            return "break"
        if key in ("delete", "backspace"):
            self.delete()
            return "break"
        return None

    # ================================================================ files
    RECOVERY_KEEP = 20                # the newest recovery copies kept; older ones go

    def _new_recovery_path(self):
        folder = os.path.join(self.owner.studio.lib.root, "scene-recovery")
        try:
            old = sorted((os.path.join(folder, n) for n in os.listdir(folder)
                          if n.endswith(".scene.json")), key=os.path.getmtime, reverse=True)
            for path in old[self.RECOVERY_KEEP:]:
                os.remove(path)
        except OSError:
            pass                      # none yet, or one in use: pruning can wait
        return os.path.join(folder, time.strftime("Scene-%Y%m%d-%H%M%S-")
                            + uuid.uuid4().hex[:8] + ".scene.json")

    def _save_recovery(self):
        if not self.dirty or not self.has_content():
            return                    # nothing unsaved to recover
        try:
            sc.save(self.scene, self.recovery_path)
        except OSError as error:
            self.status("Could not save the recovery copy: %s. Save the scene to a writable folder."
                        % error, "warn")

    def recover(self):
        folder = os.path.join(self.owner.studio.lib.root, "scene-recovery")
        if not os.path.isdir(folder):
            return self.status("No recovery copies yet.", "muted")
        path = filedialog.askopenfilename(parent=self.win, title="Recover a scene",
                                          initialdir=folder, filetypes=FILETYPES)
        if path and self.open(path):
            self.path = None
            self.dirty = True
            self._title()
            self.status("Recovery copy opened. Save as a scene to keep working on it.", "ok")

    def _ask_save(self):
        """True to go on (saved, or thrown away), False to stay."""
        if not self.dirty or not self.has_content():
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
        self._bake_camera(self.owner.settings.get("camera_profile"))
        self.making, self.suggestion = {}, None
        self.path, self.dirty = None, False
        self._forget()
        self.select(None)
        self._list()
        self._inspect()
        self.draw()
        self.status("New scene.", "muted")

    def from_picture(self, path=None):
        """A new scene made from a photo (`studio_scene.picture_scene`): the
        pose finder's people, each posed and stood where they are, and the
        vision model's words for the setting and each person when it can
        look. Both run off the UI thread; the scene lands through `_pictured`."""
        if self.picturing:
            return False
        if path is None:
            if not self._ask_save():
                return False
            path = filedialog.askopenfilename(
                parent=self.win, title="A picture to make the scene from",
                filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp *.bmp"),
                           ("All files", "*.*")])
        if not path:
            return False
        vision = getattr(self.host, "vision", None)
        box = {}

        def work():
            try:
                data, _ = self.owner.studio.find_poses(path)
                if vision is not None:
                    folk = sc.photo_people(data)[:sc.PICTURE_PEOPLE]
                    try:
                        w, h = data.get("width") or 0, data.get("height") or 0
                        box["read"] = sc.picture_answer(vision.ask(
                            path, sc.picture_question(folk, w, h), 2000), w, h)
                    except Exception as e:          # the words are extra; the scene is not
                        box["unread"] = "the vision model could not read it (%s)" % e
                else:
                    box["unread"] = "no vision model is connected"
                box["scene"], box["notes"] = sc.picture_scene(data, box.get("read"))
            except (ig.ComfyError, OSError, ValueError) as e:
                box["error"] = e
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        self.picturing = True
        self.picture_pill.set(state="disabled")
        self.status("Making a scene from %s: finding the people%s…" % (
            os.path.basename(path), " and reading the picture" if vision else ""), "muted")

        def wait():
            if not self.win.winfo_exists():
                return
            if worker.is_alive():
                self.win.after(100, wait)
                return
            self.picturing = False
            self.picture_pill.set(state="normal")
            self._pictured(path, box)
        self.win.after(100, wait)
        return True

    def _pictured(self, path, box):
        name = os.path.basename(path)
        if "error" in box:
            self.status("Could not make a scene from %s: %s" % (name, box["error"]), "err")
            return
        self.scene, self.path = box["scene"], None
        self.making = {}
        self.sel, self.inspector_section = None, "Scene"
        self._forget("From %s" % name)
        self.dirty = True
        self._title()
        self._list()
        self._inspect()
        self.draw()
        n = len(sc.people(self.scene))
        said = ["Made a scene from %s: %d %s posed and placed as in it"
                % (name, n, "person" if n == 1 else "people")]
        if "unread" in box:
            said.append("the words are left to you, as %s" % box["unread"])
        # The floor and walls the photo showed are made at once, as a Make
        # on each would; one that is not sent is said, and Make is still there.
        room = self.scene["room"]
        made = [sc.SURFACE_NAMES[k].lower() for k, _, _ in sc.SURFACES
                if room[k]["prompt"].strip() and (k != "wall" or room["walls"])
                and self.make_texture(k)]
        if made:
            said.append("making the %s in the Image Studio" % " and ".join(made))
        elif room["floor"]["prompt"].strip():
            said.append("the floor was not sent to the Image Studio (%s)"
                        % (self.owner.note.cget("text") or "see the form"))
        self.status("; ".join(said) + ". " + " ".join(box["notes"]) + (
            " The lens and camera tilt are guessed; a flat picture cannot say them."),
            "warn" if box["notes"] or "unread" in box else "ok")

    def open(self, path=None):
        if not self._ask_save():
            return False
        if path is None:
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
        self.sel, self.inspector_section = None, "Scene"
        self.making, self.suggestion = {}, None
        self._forget("Opened %s" % os.path.basename(path))
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
        self.history.mark_saved(self.scene)
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

    def close(self, final=False, confirmed=False):
        """False vetoes closing, including when the enclosing tab/app closes."""
        self._save_recovery()
        if not confirmed and not self._ask_save():
            return False
        if self.remember_after is not None:
            self.win.after_cancel(self.remember_after)
            self.remember_after = None
        self.win.destroy()
        if self.owner.scene_builder is self:
            self.owner.scene_builder = None
            self.owner.show_shot_on()
        return True

    def has_content(self):
        """Anything worth saving or generating from: an object, or a room
        that is more than the plain floor."""
        return (bool(self.scene["objects"]) or bool(self.scene.get("details", "").strip())
                or self.scene["room"] != sc.new_room())

    # ============================================================ textures
    def make_texture(self, key):
        """Send the surface's words to the Image Studio as text to image.
        The picture comes back through `texture_done`."""
        face = self.scene["room"][key]
        name = sc.SURFACE_NAMES[key].lower()
        if not face["prompt"].strip():
            self.status("Write what the %s is first, then Make." % name, "err")
            return False
        if key == "wall" and not self.scene["room"]["walls"]:
            self.scene["room"]["walls"] = True
            self.changed()
        o = self.owner
        base = sc.texture_settings(self.scene, key, o.settings["model"],
                                   o.settings.get("backend") or "auto")
        if not o.generate(base=base):
            self.status(o.note.cget("text") or "Not sent.", "err")
            return False
        self.making[key] = face["prompt"]
        self._inspect()
        self.status("Making the %s in the Image Studio; it goes on the %s when it is done."
                    % (name, name), "muted")
        return True

    def texture_done(self, job):
        """From the Image Studio, on the UI thread: a surface's job finished.
        The picture is read and shrunk off the UI thread (a full-size PNG takes
        a few seconds in pure Python), then put on the surface."""
        key = job.settings.get("scene_texture")
        if job.settings.get("scene_picture"):
            key = "obj:" + job.settings["scene_picture"]
        if key not in self.making or job.id in self.taken:
            return                     # a job for a scene no longer open, or seen
        self.taken.add(job.id)
        name = self._named(key)
        if job.status != "complete" or not job.outputs:
            self.making.pop(key, None)
            self._inspect()
            self.status("The %s could not be made: %s" % (name, job.detail or job.status),
                        "err")
            return
        self._import(key, job.outputs[0])

    def choose_texture(self, key):
        path = filedialog.askopenfilename(parent=self.win, title="A picture for the %s"
                                          % sc.SURFACE_NAMES[key].lower(),
                                          filetypes=[("PNG pictures", "*.png"),
                                                     ("All files", "*.*")])
        if path:
            if key == "wall":
                self.scene["room"]["walls"] = True
            self.making[key] = ""
            self._import(key, path)

    def _named(self, key):
        if key.startswith("obj:"):
            obj = self.obj(key[4:])
            return "picture of %s" % (obj["name"] if obj else "the prop")
        return sc.SURFACE_NAMES.get(key, key).lower()

    def _import(self, key, src):
        box = {}

        def work():
            try:
                if key.startswith("obj:"):
                    box["path"], box["aspect"] = sc.import_cutout(src)
                else:
                    box["path"] = sc.import_texture(src)
            except (OSError, ValueError) as e:
                box["error"] = e
        worker = threading.Thread(target=work, daemon=True)
        worker.start()

        def wait():
            if worker.is_alive():
                self.win.after(100, wait)
                return
            if key not in self.making:
                return
            self.making.pop(key, None)
            if "error" in box:
                self._inspect()
                self.status("Could not use %s: %s" % (os.path.basename(src), box["error"]),
                            "err")
            elif key.startswith("obj:"):
                self.set_picture(key[4:], box["path"], box["aspect"])
            else:
                self.set_texture(key, box["path"])
        self.win.after(100, wait)

    def make_picture(self, oid):
        """Paint the prop from its name and description in the Image Studio;
        the picture comes back through `texture_done`, cut out."""
        obj = self.obj(oid)
        if obj is None or "picture" not in obj:
            return False
        o = self.owner
        base = sc.picture_settings(obj, o.settings["model"],
                                   o.settings.get("backend") or "auto")
        if not o.generate(base=base):
            self.status(o.note.cget("text") or "Not sent.", "err")
            return False
        self.making["obj:" + oid] = obj["description"]
        self._inspect()
        self.status("Making a picture of %s in the Image Studio; it stands in for the shape "
                    "when it is done." % obj["name"], "muted")
        if not obj["description"].strip():
            self.status("Making a picture of %s from its name alone - a description (what "
                        "it is, its state) gives the picture more to go on." % obj["name"],
                        "warn")
        return True

    def choose_picture(self, oid):
        path = filedialog.askopenfilename(parent=self.win, title="A picture of it",
                                          filetypes=[("PNG pictures", "*.png"),
                                                     ("All files", "*.*")])
        if path and self.obj(oid) is not None:
            self.making["obj:" + oid] = ""
            self._import("obj:" + oid, path)

    def set_picture(self, oid, path, aspect=None):
        """Stand `path` in for the prop's shape ('' goes back to the shape).
        The prop keeps its height and takes the picture's width."""
        self.making.pop("obj:" + oid, None)
        obj = self.obj(oid)
        if obj is None:
            return
        obj["picture"] = path
        if path and aspect:
            obj["scale"][0] = round(max(0.05, min(20.0, obj["scale"][1] * aspect)), 3)
        if self.sel == oid:
            self._inspect()
        self.changed()
        self.status("%s now looks like %s." % (obj["name"], os.path.basename(path)) if path
                    else "%s is its shape again." % obj["name"], "ok")

    def set_texture(self, key, path):
        self.making.pop(key, None)
        self.scene["room"][key]["image"] = path
        if self.sel == ROOM:
            self._inspect()
        self.changed()
        name = sc.SURFACE_NAMES[key]
        self.status("%s now wears %s." % (name, os.path.basename(path)) if path else
                    "%s is plain again." % name, "ok")

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

    def takes(self, model_id):
        """The scene's reference kinds (`sc.MAP_KINDS`) `model_id`'s workflows,
        on any backend, have an input for: pose and composition are its
        ControlNet, source image to image."""
        lib = self.owner.studio.lib
        model = lib.get("models", model_id)
        if model is None:
            return set()
        wids = {model["workflow"]} | {b.get("workflow") for b in
                                      (model.get("backends") or {}).values()
                                      if isinstance(b, dict) and b.get("workflow")}
        out = set()
        for wid in wids:
            try:
                workflow = self.owner.studio.workflow_loader(wid)
                refs = workflow.get("references") or {}
            except Exception:           # noqa - a broken template is compose()'s to name
                continue
            out |= {k for k in sc.MAP_KINDS if refs.get(k)}
            if workflow.get("multi_identity"):
                out.add("face_positions")
        return out

    def check(self):
        """-> the reason Generate would not use the frame, or ''."""
        mid = self.owner.settings["model"]
        if not self.has_content():
            return ("Add a person, a prop, walls or a floor first: the frame is what the "
                    "picture is made from.")
        if not self.takes(mid):
            model = self.owner.studio.lib.get("models", mid)
            able = [m["label"] for m in self.owner.studio.lib.all("models")
                    if self.takes(m["id"])]
            why = ("%s's workflow takes no pose, depth or source picture, so the scene "
                   "would not be used." % (model["label"] if model else mid))
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
        o = self.owner
        try:
            maps, notes = sc.scene_maps(self.scene, self.takes(o.settings["model"]))
        except OSError as e:
            self.status("Could not write the scene's pictures: %s" % e, "err")
            return False
        chars = {c["id"]: c for c in o.studio.lib.all("characters")}
        idents = {d["id"]: d for d in o.studio.lib.all("identities")}
        words, extra = sc.generation(self.scene, maps, chars, idents)
        # The scene's maps take the form's pose, composition and source
        # slots for this job; its other references (a face, a style) stay.
        refs = {k: x for k, x in (o.collect().get("references") or {}).items()
                if k not in sc.MAP_KINDS}
        refs.update(extra["references"])
        extra["references"] = refs
        # A scene character's face is its identity's LoRA: add it to the
        # form's ticked identities for this job, at its own strength.
        idents = o.collect()["identities"]
        have = {d["id"] for d in idents}
        for iid in extra.pop("scene_identities", []):
            ident = o.studio.lib.get("identities", iid)
            if ident and iid not in have:
                idents.append({"id": iid, "strength": ident["strength"]})
        extra["identities"] = idents
        o.scene.delete("1.0", "end")
        o.scene.insert("1.0", words.text)
        if self.path:
            extra["scene_file"] = self.path
        sent = o.generate(extra=extra)
        said = o.note.cget("text")
        if sent:
            sent_as = {"pose": "pose map", "composition": "depth map", "source": "frame"}
            self.status("Sent to the Image Studio with the %s (%s). %s" % (
                ", ".join(sent_as[k] for k in sc.MAP_KINDS if k in maps) or
                ("face positions and words" if "face_positions" in self.takes(o.settings["model"])
                 else "words alone"),
                ", ".join(os.path.basename(maps[k]) for k in sc.MAP_KINDS if k in maps),
                " ".join(words.notes + notes)), "ok")
        else:
            self.status(said or "Not sent.", "err")
        return sent
