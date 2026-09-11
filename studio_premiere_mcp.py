#!/usr/bin/env python3
"""
studio_premiere_mcp - an MCP stdio bridge to the Premiere Pro on this machine.

Premiere registers no COM automation, so the road in is a CEP panel
(`premiere_panel/`, installed with `--install-panel`) that runs a loopback HTTP
server inside Premiere and evaluates the ExtendScript this bridge posts to it.
Every tool here is a short ExtendScript run through `studio_cep.CepHost`. If
Premiere is closed, or the panel is not open, a call says so and what to do;
nothing here can start the app.

Sequences go by name; clips on a timeline by `clip_id` and project items by
`item_id` - Premiere's own stable `nodeId`s - never by index: indices shift on
every insert and names repeat. `ppro_get_sequence` and `ppro_get_project` are
where the ids come from. Time is SECONDS everywhere; tracks count from 1 (V1,
A1); Motion's Position is normalised 0..1 across the frame; Scale and Opacity
are percent.

Stdlib only. The protocol is studio_mcp's; this file is the tools.

    python studio_premiere_mcp.py --list-tools
    python studio_premiere_mcp.py --check
    python studio_premiere_mcp.py --install-panel      # with Premiere closed
"""

import glob
import json
import os
import sys
import tempfile
import time

import studio_cep as cep
import studio_com as com
import studio_mcp
from studio_cep import CepError
from studio_com import b, i, n, obj, s

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URL = "http://127.0.0.1:%s" % (os.environ.get("STUDIO_PREMIERE_PORT") or "7787")
PREMIERE_URL = os.environ.get("PREMIERE_URL", DEFAULT_URL).rstrip("/")
PANEL_ID = "org.methodandform.studio-premiere"
PROCESS_NAMES = ("Adobe Premiere Pro (Beta).exe", "Adobe Premiere Pro.exe")
HOST = cep.CepHost(PREMIERE_URL, "Premiere Pro", PROCESS_NAMES[0], PANEL_ID,
                   os.path.join(HERE, "premiere_panel"))
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "studio_premiere_previews")
# Where Premiere and Media Encoder keep the .epr export presets ppro_export takes.
PRESET_GLOBS = [
    r"C:\Program Files\Adobe\Adobe Premiere Pro*\MediaIO\systempresets\*\*.epr",
    r"C:\Program Files\Adobe\Adobe Media Encoder*\MediaIO\systempresets\*\*.epr",
    os.path.join(os.path.expanduser("~"), "Documents", "Adobe", "Adobe Media Encoder",
                 "*", "Presets", "*.epr"),
]
# Sequence presets (.sqpreset) for ppro_new_sequence: app.project.createNewSequence
# opens the New Sequence dialog in Premiere 27 and hangs the panel behind it, so
# sequences are made from a preset through QE, which is silent, and the settings
# are then overridden. The HD 1080p family is the base when no preset is named.
SEQ_PRESET_GLOBS = [
    r"C:\Program Files\Adobe\Adobe Premiere Pro*\Settings\SequencePresets\**\*.sqpreset",
    os.path.join(os.path.expanduser("~"), "Documents", "Adobe", "Premiere Pro*", "Profile-*",
                 "Settings", "Custom", "*.sqpreset"),
]
BASE_PRESET_FPS = (23.976, 25, 29.97, 50, 59.94)
EXPORT_TIMEOUT = 3600
TICKS = 254016000000            # Premiere's ticks per second


def running():
    """Either Premiere - the Beta this studio runs, or a release build."""
    return any(com.process_running(p) for p in PROCESS_NAMES)


HOST.running = running

# ExtendScript helpers every tool body can use. Time objects never reach the
# serializer: `__sec` reads them as rounded seconds and `__time` makes one.
HELPERS = r"""
var __TPS = 254016000000;
function __proj() {
  if (!app.isDocumentOpen() || !app.project) __fail("No project is open in Premiere Pro. Open one with ppro_open_project or make one with ppro_new_project.");
  return app.project;
}
function __sec(t) {
  try {
    if (t === null || t === undefined) return null;
    if (typeof t === "object" && t.seconds !== undefined) return __round(t.seconds);
    if (typeof t === "string") return __round(Number(t) / __TPS);
    return __round(Number(t));
  } catch (e) { return null; }
}
function __time(sec) { var t = new Time(); t.seconds = Number(sec); return t; }
function __fps(sq) { return Math.round(__TPS / Number(sq.timebase) * 1000) / 1000; }
function __tc(sq, sec) {
  var fps = __fps(sq), f = Math.round(sec * fps), r = Math.round(fps);
  var ff = f % r, ss = Math.floor(f / r) % 60, mm = Math.floor(f / r / 60) % 60, hh = Math.floor(f / r / 3600);
  function p(x) { return (x < 10 ? "0" : "") + x; }
  return p(hh) + ":" + p(mm) + ":" + p(ss) + ":" + p(ff);
}
function __seq(name) {
  var p = __proj();
  if (name) {
    for (var i = 0; i < p.sequences.numSequences; i++) {
      var q = p.sequences[i]; if (q.name === name || q.sequenceID === name) return q;
    }
    __fail("No sequence named " + name + ". ppro_get_project lists the sequences.");
  }
  if (!p.activeSequence) __fail("No sequence is open in the timeline. Name one, or make it active with ppro_set_sequence.");
  return p.activeSequence;
}
function __active(sq) { try { return !!(app.project.activeSequence && app.project.activeSequence.sequenceID === sq.sequenceID); } catch (e) { return false; } }
function __seqinfo(sq) {
  var o = {name: sq.name, sequence_id: sq.sequenceID, width: sq.frameSizeHorizontal, height: sq.frameSizeVertical,
           fps: __fps(sq), duration: __sec(sq.end), video_tracks: sq.videoTracks.numTracks,
           audio_tracks: sq.audioTracks.numTracks, active: __active(sq)};
  try { o.playhead = __sec(sq.getPlayerPosition()); } catch (e) {}
  try { o.in_point = __sec(sq.getInPointAsTime()); o.out_point = __sec(sq.getOutPointAsTime()); } catch (e) {}
  if (o.in_point < 0) o.in_point = null; if (o.out_point < 0) o.out_point = null;   // unset is a negative sentinel
  return o;
}
function __tracks(sq, kind) { return kind === "audio" ? sq.audioTracks : sq.videoTracks; }
function __track(sq, kind, index) {
  var ts = __tracks(sq, kind);
  if (index < 1 || index > ts.numTracks) __fail("No " + kind + " track " + index + " in " + sq.name + ": it has " + ts.numTracks + ". Tracks count from 1.");
  return ts[index - 1];
}
function __clipinfo(c, kind, ti, ci) {
  var o = {clip_id: c.nodeId, name: c.name, track: (kind === "audio" ? "A" : "V") + (ti + 1), kind: kind,
           track_index: ti + 1, index: ci, start: __sec(c.start), end: __sec(c.end),
           in_point: __sec(c.inPoint), out_point: __sec(c.outPoint), duration: __sec(c.duration)};
  try { o.disabled = !!c.disabled; } catch (e) {}
  try { if (c.projectItem) { o.item_id = c.projectItem.nodeId; o.media_path = c.projectItem.getMediaPath(); } } catch (e) {}
  return o;
}
function __clips(sq, kind) {
  var out = [], ts = __tracks(sq, kind);
  for (var t = 0; t < ts.numTracks; t++) for (var c = 0; c < ts[t].clips.numItems; c++) out.push(__clipinfo(ts[t].clips[c], kind, t, c));
  return out;
}
function __clip(sq, id) {
  var kinds = ["video", "audio"];
  for (var k = 0; k < 2; k++) {
    var ts = __tracks(sq, kinds[k]);
    for (var t = 0; t < ts.numTracks; t++) for (var c = 0; c < ts[t].clips.numItems; c++) {
      var it = ts[t].clips[c]; if (it.nodeId === id) return {clip: it, kind: kinds[k], ti: t, ci: c};
    }
  }
  __fail("No clip with clip_id " + id + " in sequence " + sq.name + ". ppro_get_sequence lists the clips and their ids.");
}
function __trackinfo(tr, kind, ti) {
  var o = {track: (kind === "audio" ? "A" : "V") + (ti + 1), kind: kind, index: ti + 1, name: tr.name, clips: tr.clips.numItems};
  try { o.muted = !!tr.isMuted(); } catch (e) {}
  try { o.locked = !!tr.isLocked(); } catch (e) {}
  return o;
}
function __markers(sq) {
  var out = [], ms = sq.markers;
  if (!ms || !ms.numMarkers) return out;
  var m = ms.getFirstMarker();
  while (m && out.length < 500) {
    var o = {name: m.name, start: __sec(m.start), end: __sec(m.end), comments: m.comments, type: m.type};
    try { o.color = m.getColorByIndex(); } catch (e) {}
    out.push(o);
    m = ms.getNextMarker(m);
  }
  return out;
}
function __kind(it) {
  try { if (it.type === 2) return "bin"; if (it.type === 3) return "root"; if (it.isSequence()) return "sequence"; } catch (e) {}
  return "clip";
}
function __iteminfo(it, depth) {
  var kind = __kind(it), o = {item_id: it.nodeId, name: it.name, kind: kind, depth: depth || 0};
  if (kind === "clip") {
    try { o.media_path = it.getMediaPath(); } catch (e) {}
    try { o.in_point = __sec(it.getInPoint()); o.out_point = __sec(it.getOutPoint()); } catch (e) {}
    try { o.offline = !!it.isOffline(); } catch (e) {}
  }
  if (kind === "bin" || kind === "root") o.children = it.children.numItems;
  try { o.color_label = it.getColorLabel(); } catch (e) {}
  return o;
}
function __walkitems(bin, depth, out, max) {
  for (var i = 0; i < bin.children.numItems && out.length < max; i++) {
    var it = bin.children[i]; out.push(__iteminfo(it, depth));
    if (__kind(it) === "bin") __walkitems(it, depth + 1, out, max);
  }
  return out;
}
function __finditem(bin, id) {
  for (var i = 0; i < bin.children.numItems; i++) {
    var it = bin.children[i]; if (it.nodeId === id) return it;
    if (__kind(it) === "bin") { var f = __finditem(it, id); if (f) return f; }
  }
  return null;
}
function __item(id) {
  var root = __proj().rootItem;
  if (!id || id === "root") return root;
  var it = __finditem(root, id);
  if (!it) __fail("No project item with item_id " + id + ". ppro_get_project lists the items and their ids.");
  return it;
}
function __bin(id) {
  var it = __item(id); var k = __kind(it);
  if (k !== "bin" && k !== "root") __fail("item_id " + id + " is a " + k + ", not a bin.");
  return it;
}
function __qe() { app.enableQE(); if (typeof qe === "undefined" || !qe.project) __fail("Premiere's QE scripting layer is not available in this build."); return qe; }
function __qeseq(sq) {
  var q = __qe(); if (!__active(sq)) { app.project.activeSequence = sq; }
  var qs = q.project.getActiveSequence(); if (!qs) __fail("QE could not find the active sequence.");
  return qs;
}
function __qeclip(sq, found) {
  var qs = __qeseq(sq);
  var qt = found.kind === "audio" ? qs.getAudioTrackAt(found.ti) : qs.getVideoTrackAt(found.ti);
  var want = found.clip.start.ticks, name = found.clip.name;
  for (var i = 0; i < qt.numItems; i++) {
    var qi = qt.getItemAt(i);
    try { if (qi.type === "Empty") continue; } catch (e) {}
    try { if (qi.name === name && (!qi.start || String(qi.start.ticks) === String(want))) return qi; } catch (e) {}
  }
  __fail("Could not match clip " + name + " on the QE side; try ppro_run_jsx for this one.");
}
function __component(c, name) {
  var want = String(name).toLowerCase();
  for (var i = 0; i < c.components.numItems; i++) if (String(c.components[i].displayName).toLowerCase() === want) return c.components[i];
  var names = []; for (var j = 0; j < c.components.numItems; j++) names.push(c.components[j].displayName);
  __fail("Clip " + c.name + " has no effect named " + name + ". It has: " + names.join(", "));
}
function __shown(name) { return !!name && !/^_ /.test(String(name)); }   // Premiere's internal parameters
function __param(comp, name) {
  var want = String(name).toLowerCase();
  for (var i = 0; i < comp.properties.numItems; i++) if (String(comp.properties[i].displayName).toLowerCase() === want) return comp.properties[i];
  var names = []; for (var j = 0; j < comp.properties.numItems; j++) if (__shown(comp.properties[j].displayName)) names.push(comp.properties[j].displayName);
  __fail("Effect " + comp.displayName + " has no property named " + name + ". It has: " + names.join(", "));
}
"""


def J(v):
    """A Python value as an ExtendScript literal."""
    return json.dumps(v)


def fs(path):
    """A path as Premiere wants it: native, through File.fsName. QE refuses a forward-slash string."""
    return "new File(%s).fsName" % com.js_path(path)


def run(body, timeout=cep.DEFAULT_TIMEOUT):
    return HOST.run(HELPERS + body, timeout=timeout)


def result(text, images=(), error=False, structured=None):
    blocks = [studio_mcp.image_block(d, "image/png") for d in images]
    return studio_mcp.result(text, blocks, error=error, structured=structured)


def seq_arg(a):
    return J(a.get("sequence") or "")


def tc(sec, fps):
    """Seconds as hh:mm:ss:ff for a caption; the model keeps working in seconds."""
    f = int(round(sec * fps))
    r = max(1, int(round(fps)))
    return "%02d:%02d:%02d:%02d" % (f // r // 3600, f // r // 60 % 60, f // r % 60, f % r)


def describe_clip(c):
    return "%s %r (clip_id %s) %s-%ss" % (c["track"], c["name"], c["clip_id"], c["start"], c["end"])


def describe_item(it):
    extra = ""
    if it["kind"] == "clip" and it.get("media_path"):
        extra = " " + it["media_path"]
    elif it["kind"] in ("bin", "root"):
        extra = " (%d item(s))" % it.get("children", 0)
    return "%s %r (item_id %s)%s" % (it["kind"], it["name"], it["item_id"], extra)


# ------------------------------------------------------------------ discover

def t_status(a):
    if not running():
        return result("Premiere Pro is not running. Start it with the Start button; the bridge "
                      "panel opens with it.", structured={"running": False, "reachable": False})
    try:
        ping = HOST.ping()
    except CepError as e:
        return result(str(e), structured={"running": True, "reachable": False})
    info = run("""
      var o = {running: true, reachable: true, version: app.version, project: null};
      if (app.isDocumentOpen() && app.project) {
        var p = app.project;
        o.project = {name: p.name, path: p.path, sequences: p.sequences.numSequences,
                     active_sequence: p.activeSequence ? p.activeSequence.name : null};
      }
      return o;""")
    info["panel"] = ping
    if not info["project"]:
        text = "Premiere Pro %s is running with no project open." % info["version"]
    else:
        p = info["project"]
        text = "Premiere Pro %s: project %r (%s), %d sequence(s), active is %r." % (
            info["version"], p["name"], p["path"], p["sequences"], p["active_sequence"])
    return result(text, structured=info)


def t_get_project(a):
    info = run("""
      var p = __proj(); var o = {name: p.name, path: p.path, sequences: [], items: []};
      for (var i = 0; i < p.sequences.numSequences; i++) o.sequences.push(__seqinfo(p.sequences[i]));
      o.items = __walkitems(p.rootItem, 0, [], %d);
      return o;""" % a.get("max_items", 300))
    lines = ["Project %r at %s" % (info["name"], info["path"])]
    lines.append("Sequences (%d):" % len(info["sequences"]))
    for q in info["sequences"]:
        lines.append("  - %r%s  %sx%s @ %sfps, %ss, V%d/A%d" % (
            q["name"], " (active)" if q["active"] else "", q["width"], q["height"], q["fps"],
            q["duration"], q["video_tracks"], q["audio_tracks"]))
    lines.append("Project panel (%d item(s)):" % len(info["items"]))
    for it in info["items"]:
        lines.append("  " * it["depth"] + "  - " + describe_item(it) +
                     (" [offline]" if it.get("offline") else ""))
    return result("\n".join(lines), structured=info)


def t_get_sequence(a):
    info = run("""
      var sq = __seq(%s); var o = __seqinfo(sq); o.tracks = []; o.clips = [];
      for (var t = 0; t < sq.videoTracks.numTracks; t++) o.tracks.push(__trackinfo(sq.videoTracks[t], "video", t));
      for (var t = 0; t < sq.audioTracks.numTracks; t++) o.tracks.push(__trackinfo(sq.audioTracks[t], "audio", t));
      o.clips = __clips(sq, "video").concat(__clips(sq, "audio"));
      try { o.markers = __markers(sq); } catch (e) { o.markers = []; }
      return o;""" % seq_arg(a))
    lines = ["Sequence %r  %sx%s @ %sfps, %ss long, playhead at %ss (%s)%s" % (
        info["name"], info["width"], info["height"], info["fps"], info["duration"],
        info.get("playhead"), tc(info.get("playhead") or 0, info["fps"]),
        "" if info["active"] else " [not the active sequence]")]
    for tr in info["tracks"]:
        flags = "".join(f for f, on in ((" muted", tr.get("muted")), (" locked", tr.get("locked"))) if on)
        lines.append("  %s %r: %d clip(s)%s" % (tr["track"], tr["name"], tr["clips"], flags))
    for c in info["clips"]:
        lines.append("    - " + describe_clip(c) + (" [disabled]" if c.get("disabled") else "") +
                     (" source %s-%ss" % (c["in_point"], c["out_point"])))
    for m in info.get("markers", []):
        lines.append("  marker %r at %ss%s" % (m["name"], m["start"],
                                              " (%s)" % m["comments"] if m.get("comments") else ""))
    return result("\n".join(lines), structured=info)


def t_get_clip(a):
    info = run("""
      var sq = __seq(%s); var f = __clip(sq, %s); var c = f.clip; var o = __clipinfo(c, f.kind, f.ti, f.ci);
      try { o.speed = c.getSpeed(); } catch (e) {}
      o.effects = [];
      for (var i = 0; i < c.components.numItems; i++) {
        var cp = c.components[i]; var e = {name: cp.displayName, properties: []};
        try { e.match_name = cp.matchName; } catch (x) {}
        for (var j = 0; j < cp.properties.numItems; j++) {
          var pp = cp.properties[j]; if (!__shown(pp.displayName)) continue;
          var pr = {name: pp.displayName};
          try { pr.value = pp.getValue(); } catch (x) { pr.value = null; }
          try { pr.keyframed = !!pp.isTimeVarying(); } catch (x) {}
          e.properties.push(pr);
        }
        o.effects.push(e);
      }
      return o;""" % (seq_arg(a), J(a["clip_id"])))
    speed = info.get("speed")
    lines = [describe_clip(info) + ", source %s-%ss%s" % (
        info["in_point"], info["out_point"],
        ", speed %sx" % speed if speed not in (None, 1) else "")]
    if info.get("media_path"):
        lines.append("  media: %s (item_id %s)" % (info["media_path"], info.get("item_id")))
    for e in info["effects"]:
        lines.append("  effect %r:" % e["name"])
        for p in e["properties"]:
            lines.append("    %s = %s%s" % (p["name"], json.dumps(p["value"]),
                                            " (keyframed)" if p.get("keyframed") else ""))
    return result("\n".join(lines), structured=info)


def t_screenshot(a):
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    base = os.path.join(PREVIEW_DIR, "frame_%d" % int(time.time() * 1000))
    path = base + ".png"                    # QE appends the extension itself
    info = run("""
      var sq = __seq(%s); var t = %s; if (t === null) t = __sec(sq.getPlayerPosition());
      var qs = __qeseq(sq);
      var ok = qs.exportFramePNG(__tc(sq, t), %s);
      if (ok === false) __fail("Premiere refused to export the frame at " + __tc(sq, t));
      return {sequence: sq.name, time: t, timecode: __tc(sq, t), width: sq.frameSizeHorizontal, height: sq.frameSizeVertical};"""
               % (seq_arg(a), J(a.get("time")), fs(base)))
    if not com.wait_for_file(path, timeout=60):
        raise CepError("Premiere reported the frame exported but %s never appeared." % path)
    with open(path, "rb") as f:
        data = f.read()
    return result("Frame of %r at %ss (%s), %sx%s, saved to %s" % (
        info["sequence"], info["time"], info["timecode"], info["width"], info["height"], path),
        images=[data])


def preset_tag(path):
    """The four-character format code Premiere folders its presets by ('H264', 'MooV')."""
    folder = os.path.basename(os.path.dirname(path))
    code = folder.split("_")[-1]
    try:
        return bytes.fromhex(code).decode("ascii").strip() if len(code) == 8 else folder
    except ValueError:
        return folder


def list_presets(search=""):
    seen, out = set(), []
    want = (search or "").lower()
    for pattern in PRESET_GLOBS:
        for p in sorted(glob.glob(pattern)):
            name = os.path.splitext(os.path.basename(p))[0]
            tag = preset_tag(p)
            if want and want not in name.lower() and want not in tag.lower():
                continue
            key = (tag, name)
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": name, "format": tag, "path": p})
    return out


def t_list_presets(a):
    presets = list_presets(a.get("search", ""))
    limit = a.get("limit", 60)
    lines = ["%s: %s  %s" % (p["format"], p["name"], p["path"]) for p in presets[:limit]]
    if len(presets) > limit:
        lines.append("... and %d more; narrow with search" % (len(presets) - limit))
    if not lines:
        lines = ["No export presets match %r." % a.get("search", "")]
    return result("\n".join(lines), structured={"presets": presets[:limit], "total": len(presets)})


# -------------------------------------------------------------------- create

def t_open_project(a):
    path = a["path"]
    if not os.path.isfile(path):
        raise CepError("No file at %s" % path)
    info = run("""
      var ok = app.openDocument(%s, true, true, true);
      if (!ok) __fail("Premiere did not open " + %s + "; it may be asking about the file.");
      var p = __proj(); return {name: p.name, path: p.path, sequences: p.sequences.numSequences,
                                active_sequence: p.activeSequence ? p.activeSequence.name : null};"""
               % (fs(path), J(path)), timeout=300)
    return result("Opened project %r with %d sequence(s); active is %r." % (
        info["name"], info["sequences"], info["active_sequence"]), structured=info)


def t_new_project(a):
    path = a["path"]
    if not path.lower().endswith(".prproj"):
        path += ".prproj"
    if os.path.exists(path):
        raise CepError("%s exists; open it with ppro_open_project or choose another path" % path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    info = run("""
      var ok = app.newProject(%s);
      if (!ok) __fail("Premiere did not create the project.");
      var p = __proj(); return {name: p.name, path: p.path};""" % fs(path), timeout=300)
    return result("Made project %r at %s. It has no sequences yet - ppro_new_sequence makes one."
                  % (info["name"], info["path"]), structured=info)


def t_import_files(a):
    paths = a["paths"]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise CepError("No file at %s" % ", ".join(missing))
    info = run("""
      var p = __proj(); var bin = __bin(%s); var files = %s;
      var ok = p.importFiles(files, true, bin, %s);
      if (!ok) __fail("Premiere refused the import; a file may be an unsupported format.");
      var out = [];
      for (var i = 0; i < files.length; i++) {
        var hits = p.rootItem.findItemsMatchingMediaPath(files[i], 1);
        if (hits && hits.length) out.push(__iteminfo(hits[0], 0)); else out.push({name: files[i], item_id: null, kind: "clip"});
      }
      return out;""" % (J(a.get("bin") or ""), J([os.path.abspath(p) for p in paths]),
                        J(bool(a.get("image_sequence", False)))), timeout=600)
    lines = ["Imported %d item(s):" % len(info)] + ["  - " + describe_item(it) for it in info]
    return result("\n".join(lines), structured={"items": info})


def t_create_bin(a):
    info = run("""
      var parent = __bin(%s); var b = parent.createBin(%s);
      if (!b) __fail("Premiere did not create the bin.");
      return __iteminfo(b, 0);""" % (J(a.get("parent") or ""), J(a["name"])))
    return result("Made " + describe_item(info), structured=info)


def list_sequence_presets(search=""):
    want = (search or "").lower()
    out = []
    for pattern in SEQ_PRESET_GLOBS:
        for path in sorted(glob.glob(pattern, recursive=True)):
            name = os.path.splitext(os.path.basename(path))[0]
            if not want or want in name.lower() or want in path.lower():
                out.append({"name": name, "path": path})
    return out


def sequence_preset(a):
    """The .sqpreset a new sequence starts from: the one named, else HD 1080p at the nearest fps."""
    want = a.get("preset")
    if want:
        if os.path.isfile(want):
            return want
        hits = [p for p in list_sequence_presets(want) if p["name"].lower() == want.lower()]             or list_sequence_presets(want)
        if len(hits) == 1:
            return hits[0]["path"]
        if hits:
            raise CepError("%d sequence presets match %r: %s. Give the full .sqpreset path."
                           % (len(hits), want, ", ".join(h["name"] for h in hits[:8])))
        raise CepError("No sequence preset named %r under Premiere's Settings/SequencePresets." % want)
    fps = a.get("fps", 25)
    base = min(BASE_PRESET_FPS, key=lambda f: abs(f - fps))
    hits = [p for p in list_sequence_presets("HD 1080p") if p["name"] == "HD 1080p %s fps" % base]
    if not hits:
        raise CepError("Premiere's HD 1080p sequence presets were not found; name a preset (.sqpreset path).")
    return hits[0]["path"]


def t_new_sequence(a):
    settings = []
    if "width" in a or "height" in a:
        settings.append("if (%s !== null) st.videoFrameWidth = %s; if (%s !== null) st.videoFrameHeight = %s;"
                        % (J(a.get("width")), J(a.get("width")), J(a.get("height")), J(a.get("height"))))
    if "fps" in a:
        settings.append("var fr = new Time(); fr.ticks = String(Math.round(__TPS / %s)); st.videoFrameRate = fr;"
                        % a["fps"])
    apply = ""
    if settings:
        apply = ("try { var st = sq.getSettings(); %s sq.setSettings(st); } "
                 "catch (e) { note = 'Premiere kept the preset settings: ' + e.message; }" % " ".join(settings))
    items = a.get("item_ids") or []
    if items:
        make = """
          var its = []; for (var i = 0; i < %s.length; i++) its.push(__item(%s[i]));
          var sq = p.createNewSequenceFromClips(%s, its, __bin(%s));
          if (!sq) sq = p.activeSequence;""" % (J(items), J(items), J(a["name"]), J(a.get("bin") or ""))
    else:
        # QE wants the preset as a native path, and only takes it from File.fsName.
        make = """
          var ok = __qe().project.newSequence(%s, %s);
          if (!ok) __fail("Premiere refused to make the sequence from the preset " + %s);
          var sq = p.activeSequence;""" % (J(a["name"]), fs(sequence_preset(a)),
                                          J(os.path.basename(sequence_preset(a))))
    info = run("""
      var p = __proj(); var note = ""; %s
      if (!sq || sq.name !== %s) __fail("Premiere did not make the sequence.");
      %s
      var o = __seqinfo(sq); o.note = note; return o;""" % (make, J(a["name"]), apply))
    return result("Made sequence %r, %sx%s @ %sfps; it is the active sequence.%s" % (
        info["name"], info["width"], info["height"], info["fps"],
        " " + info["note"] if info.get("note") else ""), structured=info)


def t_add_to_sequence(a):
    mode = a.get("mode", "insert")
    body = """
      var sq = __seq(%s); var it = __item(%s); if (__kind(it) === "bin") __fail("item_id " + it.nodeId + " is a bin; add a clip or sequence item.");
      var t = %s; if (t === null) t = __sec(sq.getPlayerPosition());
      var v = %d, au = %d;
      var before = {}; var all = __clips(sq, "video").concat(__clips(sq, "audio")); for (var i = 0; i < all.length; i++) before[all[i].clip_id] = 1;
      var ok = false, err = "";
      try { ok = sq.%s(it, __time(t), v - 1, au - 1); } catch (e1) {
        try { ok = sq.%s(it, __time(t).ticks, v - 1, au - 1); } catch (e2) { err = e2.message; } }
      if (ok === false) __fail("Premiere refused to " + %s + " the clip" + (err ? ": " + err : "") + ". Is the target track locked?");
      var added = []; all = __clips(sq, "video").concat(__clips(sq, "audio"));
      for (var j = 0; j < all.length; j++) if (!before[all[j].clip_id]) added.push(all[j]);
      return {sequence: sq.name, time: t, added: added};""" % (
        seq_arg(a), J(a["item_id"]), J(a.get("time")), a.get("video_track", 1), a.get("audio_track", 1),
        "insertClip" if mode == "insert" else "overwriteClip",
        "insertClip" if mode == "insert" else "overwriteClip", J(mode))
    info = run(body)
    if not info["added"]:
        return result("Premiere reported the %s at %ss but no new clip appeared in %r; call "
                      "ppro_get_sequence to see the timeline." % (mode, info["time"], info["sequence"]),
                      structured=info)
    lines = ["%s at %ss in %r:" % ("Inserted" if mode == "insert" else "Overwrote", info["time"], info["sequence"])]
    lines += ["  - " + describe_clip(c) for c in info["added"]]
    return result("\n".join(lines), structured=info)


def t_add_marker(a):
    info = run("""
      var sq = __seq(%s); var t = %s; var m = sq.markers.createMarker(t);
      if (!m) __fail("Premiere did not create the marker.");
      %s %s %s %s
      return {name: m.name, start: __sec(m.start), end: __sec(m.end), comments: m.comments, sequence: sq.name};""" % (
        seq_arg(a), a["time"],
        "m.name = %s;" % J(a["name"]) if a.get("name") else "",
        "m.comments = %s;" % J(a["comments"]) if a.get("comments") else "",
        "m.end = t + %s;" % a["duration"] if a.get("duration") else "",   # a number; a Time is refused
        "try { m.setColorByIndex(%d); } catch (e) {}" % a["color"] if "color" in a else ""))
    return result("Added marker %r at %ss in %r." % (info["name"], info["start"], info["sequence"]),
                  structured=info)


# ---------------------------------------------------------------------- edit

def t_set_clip(a):
    sets = []
    if "name" in a:
        sets.append("c.name = %s;" % J(a["name"]))
    if "disabled" in a:
        sets.append("c.disabled = %s;" % J(a["disabled"]))
    if "start" in a:
        sets.append("c.start = __time(%s);" % a["start"])
    if "end" in a:
        sets.append("c.end = __time(%s);" % a["end"])
    if "in_point" in a:
        sets.append("c.inPoint = __time(%s);" % a["in_point"])
    if "out_point" in a:
        sets.append("c.outPoint = __time(%s);" % a["out_point"])
    if not sets:
        raise CepError("nothing to set: give at least one of name, disabled, start, end, in_point, out_point")
    info = run("""
      var sq = __seq(%s); var f = __clip(sq, %s); var c = f.clip; %s
      return __clipinfo(c, f.kind, f.ti, f.ci);""" % (seq_arg(a), J(a["clip_id"]), " ".join(sets)))
    return result("Updated " + describe_clip(info) + ", source %s-%ss." % (info["in_point"], info["out_point"]),
                  structured=info)


def t_remove_clip(a):
    info = run("""
      var sq = __seq(%s); var f = __clip(sq, %s); var o = __clipinfo(f.clip, f.kind, f.ti, f.ci);
      f.clip.remove(%s, true);
      return {removed: o, remaining: __clips(sq, "video").length + __clips(sq, "audio").length};"""
               % (seq_arg(a), J(a["clip_id"]), J(bool(a.get("ripple", False)))))
    return result("Removed %s%s. %d clip(s) remain." % (
        describe_clip(info["removed"]), " and closed the gap" if a.get("ripple") else ", leaving a gap",
        info["remaining"]), structured=info)


def t_set_clip_property(a):
    if "time" in a:
        write = """
          var t = __time(%s);
          if (!pp.isTimeVarying()) pp.setTimeVarying(true);
          pp.addKey(t); pp.setValueAtKey(t, v, true);""" % a["time"]
    else:
        write = "if (pp.isTimeVarying()) pp.setTimeVarying(false); pp.setValue(v, true);"
    info = run("""
      var sq = __seq(%s); var f = __clip(sq, %s); var c = f.clip;
      var comp = __component(c, %s); var pp = __param(comp, %s); var v = %s;
      %s
      var now = null; try { now = %s; } catch (e) {}
      return {clip: __clipinfo(c, f.kind, f.ti, f.ci), effect: comp.displayName, property: pp.displayName,
              value: now, keyframed: !!pp.isTimeVarying()};""" % (
        seq_arg(a), J(a["clip_id"]), J(a["effect"]), J(a["property"]), J(a["value"]), write,
        "pp.getValueAtKey(t)" if "time" in a else "pp.getValue()"))
    return result("Set %s > %s on %s to %s%s." % (
        info["effect"], info["property"], describe_clip(info["clip"]), json.dumps(info["value"]),
        " as a keyframe at %ss" % a["time"] if "time" in a else ""), structured=info)


def t_add_effect(a):
    info = run("""
      var sq = __seq(%s); var f = __clip(sq, %s); var q = __qe();
      var fx = f.kind === "audio" ? q.project.getAudioEffectByName(%s) : q.project.getVideoEffectByName(%s);
      if (!fx) __fail("Premiere has no " + f.kind + " effect named " + %s + ". Use the name shown in the Effects panel.");
      var qi = __qeclip(sq, f);
      var ok = f.kind === "audio" ? qi.addAudioEffect(fx) : qi.addVideoEffect(fx);
      if (ok === false) __fail("Premiere refused to add the effect.");
      var names = []; for (var i = 0; i < f.clip.components.numItems; i++) names.push(f.clip.components[i].displayName);
      return {clip: __clipinfo(f.clip, f.kind, f.ti, f.ci), effects: names};""" % (
        seq_arg(a), J(a["clip_id"]), J(a["effect"]), J(a["effect"]), J(a["effect"])))
    return result("Added %r to %s. Its effects are now: %s. ppro_get_clip shows the properties." % (
        a["effect"], describe_clip(info["clip"]), ", ".join(info["effects"])), structured=info)


def t_razor(a):
    info = run("""
      var sq = __seq(%s); var t = %s; if (t === null) t = __sec(sq.getPlayerPosition());
      var qs = __qeseq(sq); var cut = []; var which = %s;
      if (which === "all" || which === "video") for (var v = 0; v < sq.videoTracks.numTracks; v++) { try { qs.getVideoTrackAt(v).razor(__tc(sq, t)); cut.push("V" + (v + 1)); } catch (e) {} }
      if (which === "all" || which === "audio") for (var au = 0; au < sq.audioTracks.numTracks; au++) { try { qs.getAudioTrackAt(au).razor(__tc(sq, t)); cut.push("A" + (au + 1)); } catch (e) {} }
      return {sequence: sq.name, time: t, timecode: __tc(sq, t), tracks: cut,
              clips: __clips(sq, "video").concat(__clips(sq, "audio"))};""" % (
        seq_arg(a), J(a.get("time")), J(a.get("tracks", "all"))))
    return result("Cut %s at %ss (%s) in %r. Clip ids changed on the cut tracks - the timeline now:\n%s" % (
        ", ".join(info["tracks"]) or "nothing", info["time"], info["timecode"], info["sequence"],
        "\n".join("  - " + describe_clip(c) for c in info["clips"])), structured=info)


def t_set_track(a):
    sets = []
    if "muted" in a:
        sets.append("tr.setMute(%s ? 1 : 0);" % J(a["muted"]))
    if "locked" in a:
        sets.append("tr.setLocked(%s ? 1 : 0);" % J(a["locked"]))
    if "name" in a:
        sets.append("tr.name = %s;" % J(a["name"]))
    if not sets:
        raise CepError("nothing to set: give at least one of muted, locked, name")
    info = run("""
      var sq = __seq(%s); var tr = __track(sq, %s, %d); %s
      return __trackinfo(tr, %s, %d);""" % (seq_arg(a), J(a["kind"]), a["track_index"], " ".join(sets),
                                            J(a["kind"]), a["track_index"] - 1))
    return result("Track %s %r: %d clip(s)%s%s." % (
        info["track"], info["name"], info["clips"], ", muted" if info.get("muted") else "",
        ", locked" if info.get("locked") else ""), structured=info)


def t_set_sequence(a):
    sets = []
    if a.get("active"):
        sets.append("try { app.project.activeSequence = sq; } catch (e) { app.project.openSequence(sq.sequenceID); }")
    if "name" in a:
        sets.append("sq.name = %s;" % J(a["name"]))
    if "playhead" in a:
        sets.append("sq.setPlayerPosition(__time(%s).ticks);" % a["playhead"])
    if "in_point" in a:
        sets.append("sq.setInPoint(%s);" % a["in_point"])
    if "out_point" in a:
        sets.append("sq.setOutPoint(%s);" % a["out_point"])
    if not sets:
        raise CepError("nothing to set: give at least one of active, name, playhead, in_point, out_point")
    info = run("var sq = __seq(%s); %s return __seqinfo(sq);" % (seq_arg(a), " ".join(sets)))
    return result("Sequence %r%s: playhead %ss, in %s, out %s." % (
        info["name"], " (active)" if info["active"] else "", info.get("playhead"),
        info.get("in_point"), info.get("out_point")), structured=info)


def t_set_item(a):
    sets = []
    if "name" in a:
        sets.append("if (k === 'bin') it.renameBin(%s); else it.name = %s;" % (J(a["name"]), J(a["name"])))
    if "color_label" in a:
        sets.append("it.setColorLabel(%d);" % a["color_label"])
    if "in_point" in a:
        sets.append("it.setInPoint(%s, 4);" % a["in_point"])
    if "out_point" in a:
        sets.append("it.setOutPoint(%s, 4);" % a["out_point"])
    if not sets:
        raise CepError("nothing to set: give at least one of name, color_label, in_point, out_point")
    info = run("""
      var it = __item(%s); var k = __kind(it); if (k === "root") __fail("The project root cannot be changed.");
      %s return __iteminfo(it, 0);""" % (J(a["item_id"]), " ".join(sets)))
    return result("Updated " + describe_item(info) + (
        ", source %s-%ss" % (info["in_point"], info["out_point"]) if info.get("in_point") is not None else ""),
        structured=info)


def t_delete_bin(a):
    info = run("""
      var it = __bin(%s); if (__kind(it) === "root") __fail("The project root cannot be deleted.");
      var o = __iteminfo(it, 0); it.deleteBin(); return o;""" % J(a["item_id"]))
    return result("Deleted %s and everything in it." % describe_item(info), structured=info)


# --------------------------------------------------------------------- files

def t_save(a):
    info = run("var p = __proj(); p.save(); return {name: p.name, path: p.path};", timeout=600)
    return result("Saved %r to %s." % (info["name"], info["path"]), structured=info)


def t_save_as(a):
    path = a["path"]
    if not path.lower().endswith(".prproj"):
        path += ".prproj"
    if os.path.exists(path) and not a.get("overwrite", False):
        raise CepError("%s exists; pass overwrite=true to replace it" % path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    info = run("var p = __proj(); p.saveAs(%s); return {name: p.name, path: p.path};" % fs(path),
               timeout=600)
    return result("Saved as %s; the open project is now %r." % (info["path"], info["name"]), structured=info)


def t_export(a):
    preset = a["preset"]
    if not os.path.isfile(preset):
        hits = [p for p in list_presets(preset) if p["name"].lower() == preset.lower()] or list_presets(preset)
        if len(hits) == 1:
            preset = hits[0]["path"]
        elif hits:
            raise CepError("%d presets match %r: %s. Give the full path from ppro_list_presets."
                           % (len(hits), preset, ", ".join("%s (%s)" % (h["name"], h["format"]) for h in hits[:8])))
        else:
            raise CepError("No export preset named %r; ppro_list_presets shows what is installed." % preset)
    out = a["output_path"]
    if os.path.exists(out) and not a.get("overwrite", False):
        raise CepError("%s exists; pass overwrite=true to replace it" % out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    work = {"entire": 0, "in_out": 1, "work_area": 2}[a.get("range", "entire")]
    if a.get("queue"):
        info = run("""
          var sq = __seq(%s); app.encoder.launchEncoder();
          var job = app.encoder.encodeSequence(sq, %s, %s, %d, 1);
          app.encoder.startBatch();
          return {sequence: sq.name, job: job, output: %s};""" % (
            seq_arg(a), fs(out), fs(preset), work, J(out)), timeout=300)
        return result("Queued %r in Media Encoder as job %s, rendering to %s. It runs in the "
                      "background; the file appears when it finishes." % (
                          info["sequence"], info["job"], info["output"]), structured=info)
    info = run("""
      var sq = __seq(%s); var started = new Date().getTime();
      var r = sq.exportAsMediaDirect(%s, %s, %d);
      return {sequence: sq.name, output: %s, result: String(r), seconds: Math.round((new Date().getTime() - started) / 1000)};""" % (
        seq_arg(a), fs(out), fs(preset), work, J(out)), timeout=EXPORT_TIMEOUT)
    if not com.wait_for_file(out, timeout=30):
        raise CepError("Premiere reported %r for the export but %s did not appear." % (info["result"], out))
    info["size"] = os.path.getsize(out)
    return result("Exported %r to %s (%s bytes) in %ss." % (
        info["sequence"], out, info["size"], info["seconds"]), structured=info)


def t_close_project(a):
    info = run("""
      var p = __proj(); var nm = p.name;
      var ok = p.closeDocument(%s, 0);
      return {name: nm, closed: ok !== false};""" % ("1" if a.get("save") else "0"), timeout=300)
    return result("Closed project %r%s." % (info["name"], " after saving" if a.get("save") else " without saving"),
                  structured=info)


# -------------------------------------------------------------------- script

def t_run_jsx(a):
    value = run(a["code"], timeout=a.get("timeout", cep.DEFAULT_TIMEOUT))
    text = json.dumps(value, indent=1) if not isinstance(value, str) else value
    return result(text if text else "(no return value)")


# --------------------------------------------------------------------- table

SEQ = s("Sequence name, as ppro_get_project shows it. Default: the active sequence.")
CLIP = s("The clip's clip_id from ppro_get_sequence.")
ITEM = s("The project item's item_id from ppro_get_project.")
SEC = "Seconds from the start of the sequence."

TOOLS = [
    ("ppro_status", t_status,
     "Whether Premiere Pro is running and its bridge panel is answering, the version, and "
     "the open project. Call this first in a session and whenever a call reports it could "
     "not reach Premiere; its text says exactly what to do.",
     obj({})),
    ("ppro_get_project", t_get_project,
     "The open project: every sequence (name, size, fps, duration, tracks, which is active) "
     "and the project panel as a tree of bins and clips with their item_ids and media paths. "
     "item_ids are what ppro_add_to_sequence and ppro_set_item want.",
     obj({"max_items": i("Cap on project items listed. Default 300.", minimum=10, maximum=5000)})),
    ("ppro_get_sequence", t_get_sequence,
     "One sequence in full: its settings, playhead, in/out, every track (muted, locked), "
     "every clip on every track with its clip_id, start, end, source in/out in seconds, "
     "and the markers. clip_ids are what every clip tool wants; call this after anything "
     "that adds, removes or cuts clips.",
     obj({"sequence": SEQ})),
    ("ppro_get_clip", t_get_clip,
     "One clip in detail: timing, speed, media, and every effect on it with each "
     "property's current value - Motion (Position, Scale, Rotation, Anchor Point), "
     "Opacity, and any effect added. Property names here are what ppro_set_clip_property takes.",
     obj({"clip_id": CLIP, "sequence": SEQ}, ["clip_id"])),
    ("ppro_screenshot", t_screenshot,
     "A PNG of one frame of the sequence as it renders now, at a time in seconds (default "
     "the playhead). Returns the image so it can be shown, and the path it was saved to. "
     "Makes the sequence active if it was not.",
     obj({"time": n(SEC + " Default: the playhead."), "sequence": SEQ})),
    ("ppro_list_presets", t_list_presets,
     "The export presets installed here (H.264, ProRes, PNG sequence...), each with its "
     "format and the .epr path ppro_export takes. Search narrows by name or format.",
     obj({"search": s("Case-insensitive text to match, e.g. 'H264', 'Match Source', 'ProRes'."),
          "limit": i("Most presets to list. Default 60.", minimum=1, maximum=1000)})),
    ("ppro_open_project", t_open_project,
     "Open a .prproj from a path on this workstation. It becomes the open project.",
     obj({"path": s("Full Windows path of the .prproj file.")}, ["path"])),
    ("ppro_new_project", t_new_project,
     "Make a new, empty project at a path and open it. It has no sequence until ppro_new_sequence.",
     obj({"path": s("Full Windows path for the .prproj; the extension is added if missing.")}, ["path"])),
    ("ppro_import_files", t_import_files,
     "Import media files (video, audio, images, a ComfyUI picture...) into the project "
     "panel and return their item_ids. Importing puts nothing on a timeline; "
     "ppro_add_to_sequence does that.",
     obj({"paths": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                    "description": "Full Windows paths of the files."},
          "bin": s("item_id of the bin to import into. Default: the project root."),
          "image_sequence": b("Import numbered stills as one image sequence. Default false.")},
         ["paths"])),
    ("ppro_create_bin", t_create_bin,
     "Make a bin (folder) in the project panel.",
     obj({"name": s("Bin name."), "parent": s("item_id of the parent bin. Default: the project root.")},
         ["name"])),
    ("ppro_new_sequence", t_new_sequence,
     "Make a new sequence and make it active. With item_ids it takes its settings from the "
     "first clip and puts them all on the timeline. Otherwise it starts from a sequence "
     "preset - Premiere's HD 1080p at the nearest fps (default 25), or the one named - and "
     "width, height and fps then override it, so any frame size works.",
     obj({"name": s("Sequence name."),
          "preset": s("A sequence preset name ('UHD (4K) 2160p 25 fps', 'Social Media Portrait 9x16 30 fps') "
                      "or .sqpreset path. Default: HD 1080p at the nearest fps."),
          "width": i("Frame width in pixels.", minimum=16, maximum=16384),
          "height": i("Frame height in pixels.", minimum=16, maximum=16384),
          "fps": n("Frames per second, e.g. 23.976, 25, 29.97, 30, 60.", minimum=1, maximum=240),
          "item_ids": {"type": "array", "items": {"type": "string"},
                       "description": "Project item_ids to build the sequence from, in order."},
          "bin": s("item_id of the bin to make it in. Default: the project root.")},
         ["name"])),
    ("ppro_add_to_sequence", t_add_to_sequence,
     "Put a project item on the timeline at a time in seconds (default the playhead). "
     "insert pushes later clips along; overwrite replaces what is there. A clip with "
     "audio lands on both the video and the audio track given. Returns the new clip_ids.",
     obj({"item_id": ITEM, "time": n(SEC + " Default: the playhead."),
          "video_track": i("Video track, counting from 1 (V1). Default 1.", minimum=1, maximum=99),
          "audio_track": i("Audio track, counting from 1 (A1). Default 1.", minimum=1, maximum=99),
          "mode": s("insert or overwrite. Default insert.", enum=["insert", "overwrite"]),
          "sequence": SEQ},
         ["item_id"])),
    ("ppro_add_marker", t_add_marker,
     "Add a sequence marker at a time in seconds, with an optional name, comment, duration "
     "and colour index (0..7 in Premiere's marker colour order).",
     obj({"time": n(SEC), "name": s("Marker name."), "comments": s("Comment text."),
          "duration": n("Seconds; 0 for a point marker.", minimum=0),
          "color": i("0..7.", minimum=0, maximum=7), "sequence": SEQ},
         ["time"])),
    ("ppro_set_clip", t_set_clip,
     "Change a clip on the timeline: rename, enable/disable, move (start), trim its "
     "timeline end, or change which part of the source it shows (in_point/out_point, in "
     "seconds of the source). Only the given fields change.",
     obj({"clip_id": CLIP, "name": s("New name."), "disabled": b("true hides the clip in the output."),
          "start": n("New timeline start in seconds; the clip moves, keeping its length."),
          "end": n("New timeline end in seconds; trims or extends the tail."),
          "in_point": n("Source in point, seconds into the media."),
          "out_point": n("Source out point, seconds into the media."), "sequence": SEQ},
         ["clip_id"])),
    ("ppro_remove_clip", t_remove_clip,
     "Remove a clip from the timeline - leaving a gap, or with ripple=true closing it. "
     "Ask the user before removing anything they did not ask to remove.",
     obj({"clip_id": CLIP, "ripple": b("Close the gap. Default false."), "sequence": SEQ}, ["clip_id"])),
    ("ppro_set_clip_property", t_set_clip_property,
     "Set one property of one effect on a clip, by the names ppro_get_clip shows: effect "
     "'Motion' properties Position ([x, y] normalised 0..1 across the frame, centre is "
     "[0.5, 0.5]), Scale (percent), Rotation (degrees); effect 'Opacity' property Opacity "
     "(0..100); or any property of an added effect. With time, the value is written as a "
     "keyframe at that second - use two or more for movement; without, the property is set "
     "flat and its keyframes are removed.",
     obj({"clip_id": CLIP, "effect": s("Effect name as ppro_get_clip shows it, e.g. Motion."),
          "property": s("Property name as ppro_get_clip shows it, e.g. Scale."),
          "value": {"description": "A number, or [x, y] for Position.",
                    "anyOf": [{"type": "number"}, {"type": "boolean"},
                              {"type": "array", "items": {"type": "number"}}]},
          "time": n("Keyframe time, seconds from the sequence start."), "sequence": SEQ},
         ["clip_id", "effect", "property", "value"])),
    ("ppro_add_effect", t_add_effect,
     "Add a video or audio effect to a clip by its Effects-panel name: 'Gaussian Blur', "
     "'Lumetri Color', 'Black & White', 'Crop'... Then ppro_get_clip shows its properties "
     "and ppro_set_clip_property sets them.",
     obj({"clip_id": CLIP, "effect": s("The effect's display name."), "sequence": SEQ},
         ["clip_id", "effect"])),
    ("ppro_razor", t_razor,
     "Cut every clip crossing a time in seconds (default the playhead) on all tracks, or "
     "only the video or audio ones. Clip ids on the cut tracks change: re-read them.",
     obj({"time": n(SEC + " Default: the playhead."),
          "tracks": s("Which tracks. Default all.", enum=["all", "video", "audio"]), "sequence": SEQ})),
    ("ppro_set_track", t_set_track,
     "Mute, lock or rename one track.",
     obj({"kind": s("video or audio.", enum=["video", "audio"]),
          "track_index": i("Track number counting from 1.", minimum=1, maximum=99),
          "muted": b("Mute or unmute."), "locked": b("Lock or unlock."), "name": s("New name."),
          "sequence": SEQ},
         ["kind", "track_index"])),
    ("ppro_set_sequence", t_set_sequence,
     "Make a sequence active in the timeline, rename it, move the playhead, or set its "
     "in and out points (seconds).",
     obj({"sequence": SEQ, "active": b("Open this sequence in the timeline."), "name": s("New name."),
          "playhead": n(SEC), "in_point": n(SEC), "out_point": n(SEC)})),
    ("ppro_set_item", t_set_item,
     "Change a project item: rename a clip or bin, set its colour label (0..15), or set the "
     "source in/out points (seconds into the media) that ppro_add_to_sequence will use.",
     obj({"item_id": ITEM, "name": s("New name."),
          "color_label": i("0..15 in Premiere's label colour order.", minimum=0, maximum=15),
          "in_point": n("Seconds into the media."), "out_point": n("Seconds into the media.")},
         ["item_id"])),
    ("ppro_delete_bin", t_delete_bin,
     "Delete a bin and everything in it from the project panel. Clips already on a timeline "
     "go offline. Ask before deleting anything the user did not ask to remove.",
     obj({"item_id": ITEM}, ["item_id"])),
    ("ppro_save", t_save,
     "Save the project to its own file.",
     obj({})),
    ("ppro_save_as", t_save_as,
     "Save the project under a new path; the open project becomes that file. Refuses to "
     "overwrite unless told to.",
     obj({"path": s("Full Windows path; .prproj is added if missing."),
          "overwrite": b("Default false.")}, ["path"])),
    ("ppro_export", t_export,
     "Render a sequence to a file with an export preset - a preset name or .epr path from "
     "ppro_list_presets. By default Premiere renders it here and the call waits (minutes "
     "for a long sequence); queue=true hands it to Media Encoder and returns at once. "
     "Refuses to overwrite unless told to.",
     obj({"preset": s("Preset name (e.g. 'Match Source - High bitrate') or full .epr path."),
          "output_path": s("Full Windows path for the rendered file, with its extension."),
          "range": s("What to render. Default entire.", enum=["entire", "in_out", "work_area"]),
          "queue": b("Send to Media Encoder instead of rendering here. Default false."),
          "overwrite": b("Default false."), "sequence": SEQ},
         ["preset", "output_path"])),
    ("ppro_close_project", t_close_project,
     "Close the open project, discarding changes unless save=true. Ask before discarding work.",
     obj({"save": b("Save first. Default false.")})),
    ("ppro_run_jsx", t_run_jsx,
     "Run ExtendScript inside Premiere Pro for anything no other tool covers. The code is "
     "the body of a function: `return` a value (string, number, array or plain object) to "
     "see it. `app.project`, `app.project.activeSequence`, and QE after `app.enableQE()` "
     "are available; Time objects should be returned as `.seconds`. Address clips and "
     "items by nodeId inside it as well.",
     obj({"code": s("ExtendScript source."),
          "timeout": i("Seconds to allow. Default 120.", minimum=5, maximum=3600)},
         ["code"])),
]

READ_ONLY = {"ppro_status", "ppro_get_project", "ppro_get_sequence", "ppro_get_clip",
             "ppro_screenshot", "ppro_list_presets"}

HINTS = {
    "ppro_open_project": {"destructive": False, "idempotent": True},
    "ppro_new_project": {"destructive": False},
    "ppro_import_files": {"destructive": False, "idempotent": True},
    "ppro_create_bin": {"destructive": False},
    "ppro_new_sequence": {"destructive": False},
    "ppro_add_to_sequence": {"destructive": False},
    "ppro_add_marker": {"destructive": False},
    "ppro_add_effect": {"destructive": False},
    "ppro_set_clip": {"idempotent": True},
    "ppro_set_clip_property": {"idempotent": True},
    "ppro_set_track": {"idempotent": True},
    "ppro_set_sequence": {"idempotent": True},
    "ppro_set_item": {"idempotent": True},
    "ppro_save": {"idempotent": True},
    "ppro_save_as": {"idempotent": True},
    "ppro_export": {"idempotent": True},
    "ppro_remove_clip": {"destructive": True},
    "ppro_delete_bin": {"destructive": True},
    "ppro_close_project": {"destructive": True},
}

SERVER = studio_mcp.Server(
    "studio-premiere-mcp", "1.0",
    studio_mcp.tools_from_table(TOOLS, read_only=READ_ONLY, **HINTS),
    errors=(CepError, KeyError, TypeError, ValueError, OSError),
    instructions="Premiere Pro on this workstation, driven through a bridge panel that runs "
                 "its scripting engine. Start with ppro_status, then ppro_get_project for "
                 "item_ids and ppro_get_sequence for clip_ids. Seconds everywhere; tracks "
                 "count from 1.")


def tool_list():
    return [t.spec() for t in SERVER.tools]


def call_tool(name, arguments):
    """Call a tool from Python: every refusal is a result, never an exception."""
    try:
        return SERVER.call_tool(name, arguments)
    except studio_mcp.JSONRPCError as e:
        return result(e.message, error=True)


def serve(inp=None, out=None):
    SERVER.serve(inp, out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--install-panel" in argv:
        if running():
            print("Quit Premiere Pro first; CEP reads the extensions folder at startup.")
            return 1
        dest = HOST.install_panel()
        print("Installed the bridge panel at %s" % dest)
        missing = cep.debug_mode_missing()
        if missing:
            print("CEP will not load an unsigned panel until PlayerDebugMode is set. Run, for each:")
            for v in missing:
                print(r"    reg add HKCU\Software\Adobe\CSXS.%s /v PlayerDebugMode /t REG_SZ /d 1" % v)
        print("Start Premiere Pro; the panel is under Window > Extensions > Studio Assistant Bridge "
              "and listens on %s." % PREMIERE_URL)
        return 0
    return studio_mcp.main(SERVER, argv)


if __name__ == "__main__":
    sys.exit(main())
