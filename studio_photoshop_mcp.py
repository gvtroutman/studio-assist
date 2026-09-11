#!/usr/bin/env python3
"""
studio_photoshop_mcp - an MCP stdio bridge to the Photoshop on this machine.

Photoshop needs no plugin: on Windows it registers `Photoshop.Application`, a
COM server whose `DoJavaScript` runs ExtendScript inside the live app. Every
tool here is a short ExtendScript run through `studio_com.ComHost`, which keeps
one PowerShell worker holding the COM handle. If Photoshop is closed, the
first call starts it - COM does that on attach.

Layers are addressed by `layer_id` - Photoshop's own stable integer `id` -
never by index or name: names repeat and indices shift on every insert.
`ps_get_document` is where the ids come from. All geometry is pixels with the
origin top-left; opacity is 0..100; colours are "#RRGGBB".

Stdlib only. The protocol is studio_mcp's; this file is the tools.

    python studio_photoshop_mcp.py --list-tools
    python studio_photoshop_mcp.py --check
"""

import json
import os
import sys
import tempfile
import time

import studio_com as com
import studio_mcp
from studio_com import ComError, b, i, n, obj, s

PROGID = os.environ.get("PHOTOSHOP_PROGID", "Photoshop.Application")
HOST = com.ComHost(PROGID, "Photoshop", "Photoshop.exe")
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "studio_photoshop_previews")

# Ruler and type units pinned to pixels and every dialog suppressed for the
# duration of a call, then put back the way the user had them.
SETUP = """
  var __ru = app.preferences.rulerUnits, __tu = app.preferences.typeUnits, __dd = app.displayDialogs;
  app.preferences.rulerUnits = Units.PIXELS; app.preferences.typeUnits = TypeUnits.PIXELS;
  app.displayDialogs = DialogModes.NO;"""
TEARDOWN = """
  app.preferences.rulerUnits = __ru; app.preferences.typeUnits = __tu; app.displayDialogs = __dd;"""

# ExtendScript helpers every tool body can use. `__doc` resolves the document a
# call is about (active unless named); `__layer` resolves a layer id or fails
# with the sentence the model needs.
HELPERS = r"""
function __doc(name) {
  if (app.documents.length === 0) __fail("No document is open in Photoshop. Open one with ps_open or make one with ps_new_document.");
  if (name) {
    for (var i = 0; i < app.documents.length; i++) if (app.documents[i].name === name) return app.documents[i];
    __fail("No open document is named " + name + ". ps_list_documents shows what is open.");
  }
  return app.activeDocument;
}
function __enum(v, prefix) { return String(v).replace(prefix + ".", "").toLowerCase(); }
function __bounds(l) { var bb = l.bounds; return [__px(bb[0]), __px(bb[1]), __px(bb[2]), __px(bb[3])]; }
function __kind(l) {
  if (l.typename === "LayerSet") return "group";
  try { return __enum(l.kind, "LayerKind").replace("normal", "pixel"); } catch (e) { return "pixel"; }
}
function __info(l, depth) {
  var o = {layer_id: l.id, name: l.name, kind: __kind(l), visible: l.visible,
           opacity: __round(l.opacity), blend_mode: __enum(l.blendMode, "BlendMode"),
           locked: l.allLocked, depth: depth || 0};
  if (l.typename !== "LayerSet") {
    try { o.bounds = __bounds(l); } catch (e) {}
    if (o.kind === "text") { try { o.text = l.textItem.contents; } catch (e) {} }
  }
  return o;
}
function __walk(c, depth, out) {
  for (var i = 0; i < c.layers.length; i++) {
    var l = c.layers[i]; out.push(__info(l, depth));
    if (l.typename === "LayerSet") __walk(l, depth + 1, out);
  }
  return out;
}
function __find(c, id) {
  for (var i = 0; i < c.layers.length; i++) {
    var l = c.layers[i]; if (l.id === id) return l;
    if (l.typename === "LayerSet") { var f = __find(l, id); if (f) return f; }
  }
  return null;
}
function __layer(doc, id) {
  var l = __find(doc, id);
  if (!l) __fail("No layer with layer_id " + id + " in " + doc.name + ". ps_get_document lists the layers and their ids.");
  return l;
}
function __color(hex) { var c = new SolidColor(); c.rgb.hexValue = String(hex).replace("#", ""); return c; }
function __docinfo(d) {
  return {name: d.name, path: (function () { try { return d.fullName.fsName; } catch (e) { return null; } })(),
          width: __px(d.width), height: __px(d.height), resolution: __round(d.resolution),
          mode: __enum(d.mode, "DocumentMode"), bits: __enum(d.bitsPerChannel, "BitsPerChannelType").replace("bpc", ""),
          saved: d.saved, active: d === app.activeDocument, layer_count: d.layers.length};
}
function __setBlend(l, name) {
  var key = String(name).toUpperCase().replace(/[ \-]/g, "");
  var map = {NORMAL: "NORMAL", DISSOLVE: "DISSOLVE", DARKEN: "DARKEN", MULTIPLY: "MULTIPLY", COLORBURN: "COLORBURN",
             LINEARBURN: "LINEARBURN", LIGHTEN: "LIGHTEN", SCREEN: "SCREEN", COLORDODGE: "COLORDODGE",
             LINEARDODGE: "LINEARDODGE", OVERLAY: "OVERLAY", SOFTLIGHT: "SOFTLIGHT", HARDLIGHT: "HARDLIGHT",
             VIVIDLIGHT: "VIVIDLIGHT", LINEARLIGHT: "LINEARLIGHT", PINLIGHT: "PINLIGHT", HARDMIX: "HARDMIX",
             DIFFERENCE: "DIFFERENCE", EXCLUSION: "EXCLUSION", HUE: "HUE", SATURATION: "SATURATION",
             COLOR: "COLORBLEND", LUMINOSITY: "LUMINOSITY", PASSTHROUGH: "PASSTHROUGH"};
  if (!map[key]) __fail("Unknown blend mode " + name);
  l.blendMode = BlendMode[map[key]];
}
"""

BLEND_MODES = ["normal", "dissolve", "darken", "multiply", "color burn", "linear burn",
               "lighten", "screen", "color dodge", "linear dodge", "overlay", "soft light",
               "hard light", "vivid light", "linear light", "pin light", "hard mix",
               "difference", "exclusion", "hue", "saturation", "color", "luminosity",
               "pass through"]
FORMATS = ["psd", "png", "jpg", "tif"]
ANCHORS = {"center": "MIDDLECENTER", "top left": "TOPLEFT", "top": "TOPCENTER",
           "top right": "TOPRIGHT", "left": "MIDDLELEFT", "right": "MIDDLERIGHT",
           "bottom left": "BOTTOMLEFT", "bottom": "BOTTOMCENTER", "bottom right": "BOTTOMRIGHT"}


def J(v):
    """A Python value as an ExtendScript literal."""
    return json.dumps(v)


def run(body, timeout=com.DEFAULT_TIMEOUT):
    return HOST.run(HELPERS + body, timeout=timeout, setup=SETUP, teardown=TEARDOWN)


def result(text, images=(), error=False, structured=None):
    blocks = [studio_mcp.image_block(d, "image/png") for d in images]
    return studio_mcp.result(text, blocks, error=error, structured=structured)


def doc_arg(a):
    return J(a.get("document") or "")


def describe_layer(info):
    extra = ""
    if info.get("bounds"):
        bb = info["bounds"]
        extra = " at [%s, %s] size %sx%s" % (bb[0], bb[1], round(bb[2] - bb[0], 2),
                                              round(bb[3] - bb[1], 2))
    return "%s layer %r (layer_id %s)%s" % (info["kind"], info["name"], info["layer_id"], extra)


# ------------------------------------------------------------------ discover

def t_status(a):
    if not HOST.running():
        return result("Photoshop is not running. Any other tool starts it (the first call "
                      "then takes a while), or use the Start button.",
                      structured={"running": False, "documents": []})
    info = run("""
      var docs = []; for (var i = 0; i < app.documents.length; i++) docs.push(__docinfo(app.documents[i]));
      return {running: true, version: app.version, documents: docs,
              active: app.documents.length ? app.activeDocument.name : null};""")
    if not info["documents"]:
        text = "Photoshop %s is running with no document open." % info["version"]
    else:
        text = "Photoshop %s: %d document(s) open, active is %r." % (
            info["version"], len(info["documents"]), info["active"])
    return result(text, structured=info)


def t_list_documents(a):
    docs = run("""
      var docs = []; for (var i = 0; i < app.documents.length; i++) docs.push(__docinfo(app.documents[i]));
      return docs;""")
    if not docs:
        return result("No document is open.", structured={"documents": []})
    lines = ["%s%s  %sx%s px, %s, %s%s" % (d["name"], " (active)" if d["active"] else "",
                                            d["width"], d["height"], d["mode"],
                                            d["path"] or "unsaved",
                                            "" if d["saved"] else ", unsaved changes")
             for d in docs]
    return result("\n".join(lines), structured={"documents": docs})


def t_get_document(a):
    info = run("""
      var d = __doc(%s); var o = __docinfo(d);
      o.layers = __walk(d, 0, []);
      try { o.active_layer_id = d.activeLayer.id; } catch (e) { o.active_layer_id = null; }
      return o;""" % doc_arg(a))
    lines = ["%s  %sx%s px @ %s ppi, %s %s-bit, %d top-level layer(s)" % (
        info["name"], info["width"], info["height"], info["resolution"], info["mode"],
        info["bits"], info["layer_count"])]
    for l in info["layers"]:
        lines.append("  " * l["depth"] + "- " + describe_layer(l) +
                     ("" if l["visible"] else " [hidden]") +
                     (" opacity %s" % l["opacity"] if l["opacity"] != 100 else "") +
                     (" %s" % l["blend_mode"] if l["blend_mode"] not in ("normal", "passthrough") else "") +
                     (" text=%r" % l["text"][:60] if l.get("text") else ""))
    return result("\n".join(lines), structured=info)


def t_get_layer(a):
    info = run("""
      var d = __doc(%s); var l = __layer(d, %d); var o = __info(l, 0);
      if (o.kind === "text") {
        var t = l.textItem;
        o.text = t.contents;
        try { o.font = t.font; } catch (e) {}
        try { o.font_size = __px(t.size); } catch (e) {}
        try { o.color = "#" + t.color.rgb.hexValue; } catch (e) {}
        try { o.position = [__px(t.position[0]), __px(t.position[1])]; } catch (e) {}
      }
      if (l.typename === "LayerSet") o.children = __walk(l, 1, []);
      return o;""" % (doc_arg(a), a["layer_id"]))
    return result(json.dumps(info, indent=1), structured=info)


def t_screenshot(a):
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    path = os.path.join(PREVIEW_DIR, "preview_%d.png" % int(time.time() * 1000))
    max_size = a.get("max_size", 1024)
    info = run("""
      var d = __doc(%s); var keep = app.activeDocument;
      var dup = d.duplicate("studio_preview", true);
      try {
        try { dup.flatten(); } catch (e) {}
        if (dup.mode !== DocumentMode.RGB) dup.changeMode(ChangeMode.RGB);
        dup.bitsPerChannel = BitsPerChannelType.EIGHT;
        var w = __px(dup.width), h = __px(dup.height), m = %d;
        if (w > m || h > m) { var f = m / Math.max(w, h); dup.resizeImage(UnitValue(Math.round(w * f), "px"), UnitValue(Math.round(h * f), "px"), null, ResampleMethod.BICUBICSHARPER); }
        var o = new PNGSaveOptions(); o.compression = 6; o.interlaced = false;
        dup.saveAs(new File(%s), o, true, Extension.LOWERCASE);
        var out = {width: __px(dup.width), height: __px(dup.height), source: d.name};
      } finally { dup.close(SaveOptions.DONOTSAVECHANGES); app.activeDocument = keep; }
      return out;""" % (doc_arg(a), max_size, com.js_path(path)))
    com.wait_for_file(path)
    with open(path, "rb") as f:
        data = f.read()
    return result("Flattened preview of %s at %sx%s, saved to %s" % (
        info["source"], info["width"], info["height"], path), images=[data])


# -------------------------------------------------------------------- create

def t_new_document(a):
    mode = {"rgb": "RGB", "grayscale": "GRAYSCALE", "cmyk": "CMYK"}[a.get("mode", "rgb")]
    fill = {"white": "WHITE", "transparent": "TRANSPARENT", "background": "BACKGROUNDCOLOR"}[
        a.get("fill", "white")]
    info = run("""
      var d = app.documents.add(UnitValue(%d, "px"), UnitValue(%d, "px"), %s, %s, NewDocumentMode.%s, DocumentFill.%s);
      return __docinfo(d);""" % (a["width"], a["height"], a.get("resolution", 72),
                                 J(a.get("name", "Untitled")), mode, fill))
    return result("Made %r, %sx%s px at %s ppi, %s. It is the active document." % (
        info["name"], info["width"], info["height"], info["resolution"], info["mode"]),
        structured=info)


def t_open(a):
    path = a["path"]
    if not os.path.isfile(path):
        raise ComError("No file at %s" % path)
    info = run("var d = app.open(new File(%s)); var o = __docinfo(d); o.layers = __walk(d, 0, []); return o;"
               % com.js_path(path))
    return result("Opened %s: %sx%s px, %s, %d layer(s). It is the active document." % (
        info["name"], info["width"], info["height"], info["mode"], info["layer_count"]),
        structured=info)


def t_add_text_layer(a):
    body = """
      var d = __doc(%s); var l = d.artLayers.add(); l.kind = LayerKind.TEXT;
      var t = l.textItem; t.kind = TextType.POINTTEXT;
      t.contents = %s; t.size = UnitValue(%s, "px"); t.color = __color(%s);
      t.position = [UnitValue(%s, "px"), UnitValue(%s, "px")];
      %s
      l.name = %s;
      return __info(l, 0);""" % (
        doc_arg(a), J(a["text"]), a.get("size", 48), J(a.get("color", "#000000")),
        a.get("x", 50), a.get("y", 100),
        ("try { t.font = %s; } catch (e) { __fail('Photoshop has no font named ' + %s + ' (use the PostScript name, e.g. ArialMT)'); }"
         % (J(a["font"]), J(a["font"]))) if a.get("font") else "",
        J(a.get("name") or a["text"][:40]))
    info = run(body)
    return result("Added " + describe_layer(info) + ". The position is the text baseline's "
                  "left end.", structured=info)


def t_add_fill_layer(a):
    r = a.get("bounds")
    select = ("d.selection.select([[%s,%s],[%s,%s],[%s,%s],[%s,%s]]);"
              % (r[0], r[1], r[2], r[1], r[2], r[3], r[0], r[3])) if r else "d.selection.selectAll();"
    info = run("""
      var d = __doc(%s); var l = d.artLayers.add(); l.name = %s;
      %s
      d.selection.fill(__color(%s)); d.selection.deselect();
      l.opacity = %s;
      return __info(l, 0);""" % (doc_arg(a), J(a.get("name", "Fill")), select,
                                 J(a["color"]), a.get("opacity", 100)))
    return result("Added " + describe_layer(info) + " filled with %s." % a["color"], structured=info)


def t_place_file(a):
    path = a["path"]
    if not os.path.isfile(path):
        raise ComError("No file at %s" % path)
    info = run("""
      var d = __doc(%s);
      var desc = new ActionDescriptor();
      desc.putPath(charIDToTypeID("null"), new File(%s));
      desc.putEnumerated(charIDToTypeID("FTcs"), charIDToTypeID("QCSt"), charIDToTypeID("Qcsa"));
      executeAction(charIDToTypeID("Plc "), desc, DialogModes.NO);
      var l = d.activeLayer; %s
      return __info(l, 0);""" % (doc_arg(a), com.js_path(path),
                                 "l.name = %s;" % J(a["name"]) if a.get("name") else ""))
    return result("Placed %s as a smart object: %s" % (os.path.basename(path), describe_layer(info)),
                  structured=info)


# ---------------------------------------------------------------------- edit

def t_set_layer(a):
    sets = []
    if "name" in a:
        sets.append("l.name = %s;" % J(a["name"]))
    if "visible" in a:
        sets.append("l.visible = %s;" % J(a["visible"]))
    if "opacity" in a:
        sets.append("l.opacity = %s;" % a["opacity"])
    if "blend_mode" in a:
        sets.append("__setBlend(l, %s);" % J(a["blend_mode"]))
    if "locked" in a:
        sets.append("l.allLocked = %s;" % J(a["locked"]))
    text_sets = []
    if "text" in a:
        text_sets.append("t.contents = %s;" % J(a["text"]))
    if "font_size" in a:
        text_sets.append('t.size = UnitValue(%s, "px");' % a["font_size"])
    if "color" in a:
        text_sets.append("t.color = __color(%s);" % J(a["color"]))
    if "font" in a:
        text_sets.append("t.font = %s;" % J(a["font"]))
    if text_sets:
        sets.append('if (__kind(l) !== "text") __fail("layer_id " + l.id + " is not a text layer; '
                    'text, font, font_size and color only apply to text layers");'
                    " var t = l.textItem; " + " ".join(text_sets))
    if not sets:
        raise ComError("nothing to set: give at least one of name, visible, opacity, "
                       "blend_mode, locked, text, font, font_size, color")
    info = run("var d = __doc(%s); var l = __layer(d, %d); %s return __info(l, 0);"
               % (doc_arg(a), a["layer_id"], " ".join(sets)))
    return result("Updated " + describe_layer(info), structured=info)


def t_move_layer(a):
    if "x" in a or "y" in a:
        move = """
          var bb = __bounds(l);
          var dx = (%s === null ? 0 : %s - bb[0]), dy = (%s === null ? 0 : %s - bb[1]);
          l.translate(dx, dy);""" % (J(a.get("x")), J(a.get("x")), J(a.get("y")), J(a.get("y")))
    elif "dx" in a or "dy" in a:
        move = "l.translate(%s, %s);" % (a.get("dx", 0), a.get("dy", 0))
    else:
        raise ComError("give x/y (new top-left) or dx/dy (offset)")
    info = run("var d = __doc(%s); var l = __layer(d, %d); %s return __info(l, 0);"
               % (doc_arg(a), a["layer_id"], move))
    return result("Moved " + describe_layer(info), structured=info)


def t_reorder_layer(a):
    where = a["position"]
    if where in ("above", "below") and "relative_to" not in a:
        raise ComError("position %r needs relative_to (a layer_id)" % where)
    op = {"top": "l.move(d, ElementPlacement.PLACEATBEGINNING);",
          "bottom": "l.move(d, ElementPlacement.PLACEATEND);",
          "above": "l.move(__layer(d, %s), ElementPlacement.PLACEBEFORE);" % a.get("relative_to"),
          "below": "l.move(__layer(d, %s), ElementPlacement.PLACEAFTER);" % a.get("relative_to")}[where]
    info = run("var d = __doc(%s); var l = __layer(d, %d); %s return {layer: __info(l, 0), layers: __walk(d, 0, [])};"
               % (doc_arg(a), a["layer_id"], op))
    order = ", ".join("%s(%s)" % (l["name"], l["layer_id"]) for l in info["layers"])
    return result("Moved %s to %s. Order, top first: %s" % (describe_layer(info["layer"]), where, order),
                  structured=info)


def t_duplicate_layer(a):
    info = run("""
      var d = __doc(%s); var l = __layer(d, %d); var c = l.duplicate(); %s
      return __info(c, 0);""" % (doc_arg(a), a["layer_id"],
                                 "c.name = %s;" % J(a["name"]) if a.get("name") else ""))
    return result("Duplicated into " + describe_layer(info), structured=info)


def t_delete_layer(a):
    info = run("""
      var d = __doc(%s); var l = __layer(d, %d); var o = __info(l, 0); l.remove();
      return {deleted: o, layers: __walk(d, 0, [])};""" % (doc_arg(a), a["layer_id"]))
    return result("Deleted %s. %d layer(s) remain." % (describe_layer(info["deleted"]),
                                                       len(info["layers"])), structured=info)


def t_adjust_layer(a):
    ops = []
    if "brightness" in a or "contrast" in a:
        ops.append("l.adjustBrightnessContrast(%s, %s);" % (a.get("brightness", 0), a.get("contrast", 0)))
    if "hue" in a or "saturation" in a or "lightness" in a:
        ops.append("(function(){ var desc = new ActionDescriptor(); desc.putBoolean(charIDToTypeID('Clrz'), false);"
                   " var list = new ActionList(); var adj = new ActionDescriptor();"
                   " adj.putInteger(charIDToTypeID('H   '), %s); adj.putInteger(charIDToTypeID('Strt'), %s);"
                   " adj.putInteger(charIDToTypeID('Lght'), %s); list.putObject(charIDToTypeID('Hst2'), adj);"
                   " desc.putList(charIDToTypeID('Adjs'), list); d.activeLayer = l;"
                   " executeAction(charIDToTypeID('HStr'), desc, DialogModes.NO); })();"
                   % (a.get("hue", 0), a.get("saturation", 0), a.get("lightness", 0)))
    if a.get("desaturate"):
        ops.append("l.desaturate();")
    if a.get("invert"):
        ops.append("l.invert();")
    if a.get("auto_levels"):
        ops.append("l.autoLevels();")
    if a.get("auto_contrast"):
        ops.append("l.autoContrast();")
    if a.get("gaussian_blur"):
        ops.append("l.applyGaussianBlur(%s);" % a["gaussian_blur"])
    if a.get("sharpen"):
        ops.append("l.applySharpen();")
    if not ops:
        raise ComError("nothing to adjust: give at least one adjustment")
    info = run("""
      var d = __doc(%s); var l = __layer(d, %d);
      if (l.typename === "LayerSet") __fail("layer_id " + l.id + " is a group; adjust a layer inside it");
      if (__kind(l) !== "pixel") { try { l.rasterize(RasterizeType.ENTIRELAYER); } catch (e) { __fail("layer " + l.name + " is a " + __kind(l) + " layer and could not be rasterized for adjustment: " + e.message); } }
      %s return __info(l, 0);""" % (doc_arg(a), a["layer_id"], " ".join(ops)))
    return result("Adjusted " + describe_layer(info), structured=info)


def t_resize_image(a):
    if "width" not in a and "height" not in a:
        raise ComError("give width and/or height")
    info = run("""
      var d = __doc(%s); var w = %s, h = %s, cw = __px(d.width), ch = __px(d.height);
      if (w === null) w = Math.round(cw * h / ch); if (h === null) h = Math.round(ch * w / cw);
      d.resizeImage(UnitValue(w, "px"), UnitValue(h, "px"), %s, %s);
      return __docinfo(d);""" % (doc_arg(a), J(a.get("width")), J(a.get("height")),
                                 J(a.get("resolution")),
                                 "ResampleMethod.BICUBIC" if a.get("resample", True) else "ResampleMethod.NONE"))
    return result("%s is now %sx%s px at %s ppi." % (info["name"], info["width"], info["height"],
                                                     info["resolution"]), structured=info)


def t_resize_canvas(a):
    info = run("""
      var d = __doc(%s); var w = %s, h = %s;
      d.resizeCanvas(UnitValue(w === null ? __px(d.width) : w, "px"), UnitValue(h === null ? __px(d.height) : h, "px"), AnchorPosition.%s);
      return __docinfo(d);""" % (doc_arg(a), J(a.get("width")), J(a.get("height")),
                                 ANCHORS[a.get("anchor", "center")]))
    return result("%s canvas is now %sx%s px." % (info["name"], info["width"], info["height"]),
                  structured=info)


def t_crop(a):
    r = a["bounds"]
    info = run("var d = __doc(%s); d.crop([%s, %s, %s, %s]); return __docinfo(d);"
               % (doc_arg(a), r[0], r[1], r[2], r[3]))
    return result("Cropped %s to %sx%s px." % (info["name"], info["width"], info["height"]),
                  structured=info)


# --------------------------------------------------------------------- files

def t_save(a):
    info = run("""
      var d = __doc(%s);
      var has = false; try { d.fullName; has = true; } catch (e) {}
      if (!has) __fail(d.name + " has never been saved; use ps_save_as with a path.");
      d.save(); return __docinfo(d);""" % doc_arg(a))
    return result("Saved %s to %s." % (info["name"], info["path"]), structured=info)


def t_save_as(a):
    fmt = a["format"]
    path = a["path"]
    if not path.lower().endswith("." + fmt) and not (fmt == "jpg" and path.lower().endswith(".jpeg")) \
            and not (fmt == "tif" and path.lower().endswith(".tiff")):
        path = path + "." + fmt
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path) and not a.get("overwrite", False):
        raise ComError("%s exists; pass overwrite=true to replace it" % path)
    opts = {"psd": "var o = new PhotoshopSaveOptions(); o.layers = true; o.embedColorProfile = true;",
            "png": "var o = new PNGSaveOptions(); o.compression = 6;",
            "jpg": "var o = new JPEGSaveOptions(); o.quality = %d; o.embedColorProfile = true;" % a.get("quality", 10),
            "tif": "var o = new TiffSaveOptions(); o.layers = true; o.imageCompression = TIFFEncoding.TIFFLZW;"}[fmt]
    info = run("""
      var d = __doc(%s); %s
      d.saveAs(new File(%s), o, %s, Extension.LOWERCASE);
      return __docinfo(d);""" % (doc_arg(a), opts, com.js_path(path), J(a.get("as_copy", True))))
    com.wait_for_file(path)
    return result("Saved %s as %s%s." % (info["name"], path,
                                         " (a copy; the open document is unchanged)" if a.get("as_copy", True) else ""),
                  structured=info)


def t_close_document(a):
    name = run("""
      var d = __doc(%s); var nm = d.name;
      d.close(%s ? SaveOptions.SAVECHANGES : SaveOptions.DONOTSAVECHANGES); return nm;"""
               % (doc_arg(a), J(a.get("save", False))))
    return result("Closed %s%s." % (name, " after saving" if a.get("save") else " without saving"))


# -------------------------------------------------------------------- script

def t_run_jsx(a):
    value = run(a["code"], timeout=a.get("timeout", com.DEFAULT_TIMEOUT))
    text = json.dumps(value, indent=1) if not isinstance(value, str) else value
    return result(text if text else "(no return value)")


# --------------------------------------------------------------------- table

DOC = s("Open document name, as ps_list_documents shows it. Default: the active document.")
LAYER = i("The layer's layer_id from ps_get_document.")

TOOLS = [
    ("ps_status", t_status,
     "Whether Photoshop is running, its version, and the open documents. Call this "
     "first in a session and whenever a call reports it could not reach Photoshop.",
     obj({})),
    ("ps_list_documents", t_list_documents,
     "Every open document: name, size, mode, file path and whether it has unsaved changes.",
     obj({})),
    ("ps_get_document", t_get_document,
     "One document with its full layer tree: for each layer the layer_id, name, kind "
     "(pixel, text, smartobject, group, adjustment...), visibility, opacity, blend mode "
     "and pixel bounds [left, top, right, bottom]. Layer ids are what every edit tool wants.",
     obj({"document": DOC})),
    ("ps_get_layer", t_get_layer,
     "One layer in detail - for a text layer the contents, font, size, colour and position; "
     "for a group its children.",
     obj({"layer_id": LAYER, "document": DOC}, ["layer_id"])),
    ("ps_screenshot", t_screenshot,
     "A flattened PNG of the document as it looks now, scaled to fit max_size. Returns the "
     "image so it can be shown and the path it was saved to. The document is not changed.",
     obj({"document": DOC,
          "max_size": i("Longest side in pixels. Default 1024.", minimum=64, maximum=4096)})),
    ("ps_new_document", t_new_document,
     "Make a new document and make it active.",
     obj({"name": s("Document name. Default Untitled."),
          "width": i("Pixels.", minimum=1, maximum=30000),
          "height": i("Pixels.", minimum=1, maximum=30000),
          "resolution": n("Pixels per inch. Default 72.", minimum=1, maximum=3000),
          "mode": s("Colour mode. Default rgb.", enum=["rgb", "grayscale", "cmyk"]),
          "fill": s("Background fill. Default white.", enum=["white", "transparent", "background"])},
         ["width", "height"])),
    ("ps_open", t_open,
     "Open an image or PSD from a path on this workstation; it becomes the active document.",
     obj({"path": s("Full Windows path of the file.")}, ["path"])),
    ("ps_add_text_layer", t_add_text_layer,
     "Add a point-text layer. x,y is the left end of the first baseline in pixels from "
     "the top-left; size is pixels; color is #RRGGBB.",
     obj({"text": s("The text."), "x": n("Baseline start, pixels from the left. Default 50."),
          "y": n("Baseline, pixels from the top. Default 100."),
          "size": n("Font size in pixels. Default 48.", minimum=1, maximum=5000),
          "color": s("#RRGGBB. Default #000000.", pattern=com.HEX),
          "font": s("PostScript font name, e.g. ArialMT or Helvetica-Bold. Default: Photoshop's current font."),
          "name": s("Layer name. Default: the text."), "document": DOC},
         ["text"])),
    ("ps_add_fill_layer", t_add_fill_layer,
     "Add a pixel layer filled with one colour - the whole canvas, or just a rectangle.",
     obj({"color": s("#RRGGBB.", pattern=com.HEX),
          "bounds": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                     "description": "[left, top, right, bottom] in pixels. Omit to fill the whole canvas."},
          "opacity": n("0..100. Default 100.", minimum=0, maximum=100),
          "name": s("Layer name. Default Fill."), "document": DOC},
         ["color"])),
    ("ps_place_file", t_place_file,
     "Place an image file (PNG, JPEG, PSD, AI, SVG...) into the document as a new smart "
     "object layer, centred and fitted. Use this to bring in a picture from ComfyUI or elsewhere.",
     obj({"path": s("Full Windows path of the file."), "name": s("Layer name. Default: the file name."),
          "document": DOC}, ["path"])),
    ("ps_set_layer", t_set_layer,
     "Change a layer's name, visibility, opacity, blend mode or lock - and for a text "
     "layer its text, font, font_size (pixels) or color (#RRGGBB). Only the given fields change.",
     obj({"layer_id": LAYER, "name": s("New name."), "visible": b("Show or hide."),
          "opacity": n("0..100.", minimum=0, maximum=100),
          "blend_mode": s("Blend mode.", enum=BLEND_MODES), "locked": b("Lock or unlock everything."),
          "text": s("New contents (text layers)."), "font": s("PostScript font name (text layers)."),
          "font_size": n("Pixels (text layers).", minimum=1, maximum=5000),
          "color": s("#RRGGBB (text layers).", pattern=com.HEX), "document": DOC},
         ["layer_id"])),
    ("ps_move_layer", t_move_layer,
     "Move a layer's pixels: to a new top-left x,y, or by an offset dx,dy. Pixels, origin top-left.",
     obj({"layer_id": LAYER, "x": n("New left edge."), "y": n("New top edge."),
          "dx": n("Offset right (negative left)."), "dy": n("Offset down (negative up)."),
          "document": DOC}, ["layer_id"])),
    ("ps_reorder_layer", t_reorder_layer,
     "Change stacking: to the top or bottom of the document, or directly above or below "
     "another layer (relative_to).",
     obj({"layer_id": LAYER, "position": s("Where.", enum=["top", "bottom", "above", "below"]),
          "relative_to": i("The other layer's layer_id, for above/below."), "document": DOC},
         ["layer_id", "position"])),
    ("ps_duplicate_layer", t_duplicate_layer,
     "Duplicate a layer (or group) directly above itself.",
     obj({"layer_id": LAYER, "name": s("Name for the copy."), "document": DOC}, ["layer_id"])),
    ("ps_delete_layer", t_delete_layer,
     "Delete a layer or group. Ask the user before deleting anything they did not ask to remove.",
     obj({"layer_id": LAYER, "document": DOC}, ["layer_id"])),
    ("ps_adjust_layer", t_adjust_layer,
     "Apply image adjustments to a layer's pixels, destructively, in this order: "
     "brightness/contrast, hue/saturation/lightness, desaturate, invert, auto levels, auto "
     "contrast, gaussian blur, sharpen. A text or smart object layer is rasterized first.",
     obj({"layer_id": LAYER,
          "brightness": i("-150..150", minimum=-150, maximum=150),
          "contrast": i("-50..100", minimum=-50, maximum=100),
          "hue": i("-180..180 degrees", minimum=-180, maximum=180),
          "saturation": i("-100..100", minimum=-100, maximum=100),
          "lightness": i("-100..100", minimum=-100, maximum=100),
          "desaturate": b("Remove all colour."), "invert": b("Invert."),
          "auto_levels": b("Auto levels."), "auto_contrast": b("Auto contrast."),
          "gaussian_blur": n("Radius in pixels.", minimum=0.1, maximum=250),
          "sharpen": b("One pass of Sharpen."), "document": DOC},
         ["layer_id"])),
    ("ps_resize_image", t_resize_image,
     "Resample the whole image to a new pixel size. Give one dimension to keep the aspect ratio.",
     obj({"width": i("Pixels.", minimum=1, maximum=30000), "height": i("Pixels.", minimum=1, maximum=30000),
          "resolution": n("New ppi, or omit to keep."), "resample": b("Default true. false changes only the ppi metadata."),
          "document": DOC})),
    ("ps_resize_canvas", t_resize_canvas,
     "Change the canvas size without scaling the pixels, anchored where given.",
     obj({"width": i("Pixels. Omit to keep."), "height": i("Pixels. Omit to keep."),
          "anchor": s("Where the existing pixels stay. Default center.", enum=sorted(ANCHORS)),
          "document": DOC})),
    ("ps_crop", t_crop,
     "Crop the document to a pixel rectangle.",
     obj({"bounds": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                     "description": "[left, top, right, bottom] in pixels."}, "document": DOC},
         ["bounds"])),
    ("ps_save", t_save,
     "Save a document to its own file. Fails for a document that has never been saved - use ps_save_as.",
     obj({"document": DOC})),
    ("ps_save_as", t_save_as,
     "Save to a path in a format. Default is a copy, leaving the open document as it was; "
     "as_copy=false makes the file the document's own. Refuses to overwrite unless told to.",
     obj({"path": s("Full Windows path; the extension is added if missing."),
          "format": s("File format.", enum=FORMATS),
          "quality": i("JPEG quality 0..12. Default 10.", minimum=0, maximum=12),
          "as_copy": b("Default true."), "overwrite": b("Default false."), "document": DOC},
         ["path", "format"])),
    ("ps_close_document", t_close_document,
     "Close a document, discarding changes unless save=true. Ask before discarding work.",
     obj({"document": DOC, "save": b("Save first. Default false.")})),
    ("ps_run_jsx", t_run_jsx,
     "Run ExtendScript inside Photoshop for anything no other tool covers. The code is the "
     "body of a function: `return` a value (string, number, array or plain object) to see "
     "it. `app`, `app.activeDocument`, ActionDescriptor and executeAction are all available; "
     "ruler units are pixels and dialogs are suppressed while it runs.",
     obj({"code": s("ExtendScript source."),
          "timeout": i("Seconds to allow. Default 120.", minimum=5, maximum=1800)},
         ["code"])),
]

READ_ONLY = {"ps_status", "ps_list_documents", "ps_get_document", "ps_get_layer", "ps_screenshot"}

HINTS = {
    "ps_new_document": {"destructive": False},
    "ps_open": {"destructive": False, "idempotent": True},
    "ps_add_text_layer": {"destructive": False},
    "ps_add_fill_layer": {"destructive": False},
    "ps_place_file": {"destructive": False},
    "ps_duplicate_layer": {"destructive": False},
    "ps_set_layer": {"idempotent": True},
    "ps_reorder_layer": {"destructive": False, "idempotent": True},
    "ps_save": {"idempotent": True},
    "ps_save_as": {"idempotent": True},
    "ps_delete_layer": {"destructive": True},
    "ps_close_document": {"destructive": True},
}

SERVER = studio_mcp.Server(
    "studio-photoshop-mcp", "1.0",
    studio_mcp.tools_from_table(TOOLS, read_only=READ_ONLY, **HINTS),
    errors=(ComError, KeyError, TypeError, ValueError, OSError),
    instructions="Photoshop on this workstation, driven through its own scripting engine. "
                 "Start with ps_status, then ps_get_document for layer ids. Pixels, "
                 "origin top-left, opacity 0..100, colours #RRGGBB.")


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


if __name__ == "__main__":
    sys.exit(studio_mcp.main(SERVER))
