#!/usr/bin/env python3
"""
studio_illustrator_mcp - an MCP stdio bridge to the Illustrator on this machine.

Like Photoshop, Illustrator on Windows registers a COM server
(`Illustrator.Application`) with `DoJavaScript`, so the bridge is ExtendScript
run through `studio_com.ComHost` and nothing has to be installed in the app.

Page items are addressed by `uuid` - Illustrator's own stable string id on
every item - never by index or name. `ai_list_items` is where the uuids come
from. Layers and artboards are addressed by name.

Coordinates as the model sees them are the artboard's: x to the right and y
DOWN from the top-left corner of the artboard the item is on, in points (a
point is a pixel at 72 ppi, Illustrator's native unit). Illustrator itself
counts y upward; every helper here flips it, so the bridge speaks the same
language as Photoshop and After Effects and a small model never has to
remember which app is upside down.

    python studio_illustrator_mcp.py --list-tools
    python studio_illustrator_mcp.py --check
"""

import json
import os
import sys
import tempfile
import time

import studio_com as com
import studio_mcp
from studio_com import ComError, b, i, n, obj, s

PROGID = os.environ.get("ILLUSTRATOR_PROGID", "Illustrator.Application")
HOST = com.ComHost(PROGID, "Illustrator", "Illustrator.exe")
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "studio_illustrator_previews")

# Artboard coordinates for the duration of a call - positions relative to the
# active artboard's top-left rather than the document's page - then restored.
# Illustrator has no dialog switch like Photoshop's; `userInteractionLevel`
# is the nearest thing.
SETUP = """
  var __cs = app.coordinateSystem, __ui = app.userInteractionLevel;
  app.coordinateSystem = CoordinateSystem.ARTBOARDCOORDINATESYSTEM;
  app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;"""
TEARDOWN = """
  app.coordinateSystem = __cs; app.userInteractionLevel = __ui;"""

HELPERS = r"""
function __doc(name) {
  if (app.documents.length === 0) __fail("No document is open in Illustrator. Open one with ai_open or make one with ai_new_document.");
  if (name) {
    for (var i = 0; i < app.documents.length; i++) if (app.documents[i].name === name) return app.documents[i];
    __fail("No open document is named " + name + ". ai_list_documents shows what is open.");
  }
  return app.activeDocument;
}
function __enum(v, prefix) { return String(v).replace(prefix + ".", "").toLowerCase(); }
function __hex2(n) { var h = Math.round(n).toString(16); return h.length < 2 ? "0" + h : h; }
function __hex(c) {
  if (!c) return null;
  var t = c.typename;
  if (t === "RGBColor") return "#" + __hex2(c.red) + __hex2(c.green) + __hex2(c.blue);
  if (t === "GrayColor") { var g = Math.round(255 * (100 - c.gray) / 100); return "#" + __hex2(g) + __hex2(g) + __hex2(g); }
  if (t === "CMYKColor") {
    var r = 255 * (1 - c.cyan / 100) * (1 - c.black / 100), g2 = 255 * (1 - c.magenta / 100) * (1 - c.black / 100), b2 = 255 * (1 - c.yellow / 100) * (1 - c.black / 100);
    return "#" + __hex2(r) + __hex2(g2) + __hex2(b2);
  }
  if (t === "NoColor") return null;
  if (t === "SpotColor") return "spot:" + c.spot.name;
  if (t === "GradientColor") return "gradient";
  if (t === "PatternColor") return "pattern";
  return String(t);
}
function __color(hex) {
  var h = String(hex).replace("#", ""); var c = new RGBColor();
  c.red = parseInt(h.substr(0, 2), 16); c.green = parseInt(h.substr(2, 2), 16); c.blue = parseInt(h.substr(4, 2), 16);
  return c;
}
function __bounds(it) { var g = it.geometricBounds; return [__round(g[0]), __round(-g[1]), __round(g[2]), __round(-g[3])]; }
function __type(it) {
  var t = it.typename;
  return {PathItem: "path", CompoundPathItem: "compound_path", TextFrame: "text", GroupItem: "group",
          PlacedItem: "placed", RasterItem: "raster", SymbolItem: "symbol", MeshItem: "mesh",
          PluginItem: "plugin", GraphItem: "graph", LegacyTextItem: "legacy_text", NonNativeItem: "non_native"}[t] || t;
}
function __info(it) {
  var o = {uuid: String(it.uuid), type: __type(it), name: it.name || "", layer: it.layer.name,
           hidden: it.hidden, locked: it.locked, opacity: __round(it.opacity)};
  try { o.bounds = __bounds(it); } catch (e) {}
  if (it.typename === "PathItem" || it.typename === "CompoundPathItem") {
    var p = it.typename === "PathItem" ? it : (it.pathItems.length ? it.pathItems[0] : null);
    if (p) {
      o.fill = p.filled ? __hex(p.fillColor) : null;
      o.stroke = p.stroked ? __hex(p.strokeColor) : null;
      o.stroke_width = p.stroked ? __round(p.strokeWidth) : 0;
    }
  } else if (it.typename === "TextFrame") {
    o.text = it.contents.length > 120 ? it.contents.substr(0, 120) + "..." : it.contents;
    try { var ca = it.textRange.characterAttributes; o.font_size = __round(ca.size); o.fill = __hex(ca.fillColor); o.font = ca.textFont.name; } catch (e) {}
  } else if (it.typename === "GroupItem") {
    o.items = it.pageItems.length;
  } else if (it.typename === "PlacedItem" || it.typename === "RasterItem") {
    try { o.file = it.file.fsName; } catch (e) {}
  }
  return o;
}
function __item(doc, uuid) {
  var it = null;
  try { it = doc.getPageItemFromUuid(String(uuid)); } catch (e) { it = null; }
  if (!it) {
    for (var i = 0; i < doc.pageItems.length; i++) if (String(doc.pageItems[i].uuid) === String(uuid)) { it = doc.pageItems[i]; break; }
  }
  if (!it) __fail("No item with uuid " + uuid + " in " + doc.name + ". ai_list_items lists the items and their uuids.");
  return it;
}
function __layer(doc, name) {
  if (!name) return doc.activeLayer;
  for (var i = 0; i < doc.layers.length; i++) if (doc.layers[i].name === name) return doc.layers[i];
  __fail("No layer named " + name + " in " + doc.name + ". ai_get_document lists the layers.");
}
function __target(doc, name) {
  var l = __layer(doc, name);
  if (l.locked) __fail("Layer " + l.name + " is locked; unlock it with ai_set_layer or name another layer.");
  if (!l.visible) __fail("Layer " + l.name + " is hidden; show it with ai_set_layer or name another layer.");
  return l;
}
function __artboards(doc) {
  var out = [];
  for (var i = 0; i < doc.artboards.length; i++) {
    var r = doc.artboards[i].artboardRect;
    out.push({index: i, name: doc.artboards[i].name, width: __round(r[2] - r[0]), height: __round(r[1] - r[3]),
              active: i === doc.artboards.getActiveArtboardIndex()});
  }
  return out;
}
function __artboard(doc, name) {
  if (name === "" || name === null || name === undefined) return doc.artboards.getActiveArtboardIndex();
  for (var i = 0; i < doc.artboards.length; i++) if (doc.artboards[i].name === name) return i;
  __fail("No artboard named " + name + ". ai_get_document lists the artboards.");
}
function __path(d) {
  try { var f = new File(d.fullName.fsName); return f.exists ? d.fullName.fsName : null; } catch (e) { return null; }
}
function __docinfo(d) {
  var ab = __artboards(d);
  return {name: d.name, path: __path(d), color_mode: __enum(d.documentColorSpace, "DocumentColorSpace"), saved: d.saved,
          active: d === app.activeDocument, artboards: ab, layer_count: d.layers.length, item_count: d.pageItems.length};
}
function __layers(doc) {
  var out = [];
  for (var i = 0; i < doc.layers.length; i++) {
    var l = doc.layers[i];
    out.push({name: l.name, visible: l.visible, locked: l.locked, items: l.pageItems.length,
              sublayers: l.layers.length, active: l === doc.activeLayer});
  }
  return out;
}
function __font(name) {
  try { return app.textFonts.getByName(name); }
  catch (e) { __fail("Illustrator has no font named " + name + " (use the PostScript name, e.g. ArialMT or Helvetica-Bold)"); }
}
function __applyStyle(it, fill, stroke, strokeWidth) {
  var targets = [];
  if (it.typename === "PathItem") targets = [it];
  else if (it.typename === "CompoundPathItem") { for (var i = 0; i < it.pathItems.length; i++) targets.push(it.pathItems[i]); }
  else if (it.typename === "TextFrame") {
    var ca = it.textRange.characterAttributes;
    if (fill !== undefined) { if (fill === "none") ca.fillColor = new NoColor(); else ca.fillColor = __color(fill); }
    if (stroke !== undefined) { if (stroke === "none") ca.strokeColor = new NoColor(); else ca.strokeColor = __color(stroke); }
    if (strokeWidth !== undefined) ca.strokeWeight = strokeWidth;
    return;
  }
  else if (it.typename === "GroupItem") { for (var j = 0; j < it.pathItems.length; j++) targets.push(it.pathItems[j]); }
  else __fail("fill and stroke do not apply to a " + __type(it) + " item");
  for (var k = 0; k < targets.length; k++) {
    var p = targets[k];
    if (fill !== undefined) { if (fill === "none") p.filled = false; else { p.filled = true; p.fillColor = __color(fill); } }
    if (stroke !== undefined) { if (stroke === "none") p.stroked = false; else { p.stroked = true; p.strokeColor = __color(stroke); } }
    if (strokeWidth !== undefined) { p.strokeWidth = strokeWidth; if (strokeWidth > 0 && stroke === undefined) p.stroked = true; }
  }
}
"""

COLOR = r"^(#?[0-9a-fA-F]{6}|none)$"
SHAPES = ["rectangle", "rounded_rectangle", "ellipse", "line", "polygon", "star"]
FORMATS = ["ai", "pdf", "svg", "png", "jpg"]


def J(v):
    return json.dumps(v)


def run(body, timeout=com.DEFAULT_TIMEOUT):
    return HOST.run(HELPERS + body, timeout=timeout, setup=SETUP, teardown=TEARDOWN)


def result(text, images=(), error=False, structured=None):
    blocks = [studio_mcp.image_block(d, "image/png") for d in images]
    return studio_mcp.result(text, blocks, error=error, structured=structured)


def doc_arg(a):
    return J(a.get("document") or "")


def describe(it):
    bb = it.get("bounds")
    where = (" at [%s, %s] size %sx%s" % (bb[0], bb[1], round(bb[2] - bb[0], 2), round(bb[3] - bb[1], 2))
             if bb else "")
    name = " %r" % it["name"] if it.get("name") else ""
    return "%s%s (uuid %s) on layer %r%s" % (it["type"], name, it["uuid"], it["layer"], where)


def style_js(a):
    """The fill/stroke/stroke_width arguments as an __applyStyle call, or ''."""
    if not any(k in a for k in ("fill", "stroke", "stroke_width")):
        return ""
    return "__applyStyle(it, %s, %s, %s);" % (
        J(a["fill"]) if "fill" in a else "undefined",
        J(a["stroke"]) if "stroke" in a else "undefined",
        J(a["stroke_width"]) if "stroke_width" in a else "undefined")


# ------------------------------------------------------------------ discover

def t_status(a):
    if not HOST.running():
        return result("Illustrator is not running. Any other tool starts it (the first call "
                      "then takes a while), or use the Start button.",
                      structured={"running": False, "documents": []})
    info = run("""
      var docs = []; for (var i = 0; i < app.documents.length; i++) docs.push(__docinfo(app.documents[i]));
      return {running: true, version: app.version, documents: docs,
              active: app.documents.length ? app.activeDocument.name : null};""")
    if not info["documents"]:
        text = "Illustrator %s is running with no document open." % info["version"]
    else:
        text = "Illustrator %s: %d document(s) open, active is %r." % (
            info["version"], len(info["documents"]), info["active"])
    return result(text, structured=info)


def t_list_documents(a):
    docs = run("""
      var docs = []; for (var i = 0; i < app.documents.length; i++) docs.push(__docinfo(app.documents[i]));
      return docs;""")
    if not docs:
        return result("No document is open.", structured={"documents": []})
    lines = ["%s%s  %d artboard(s), %s, %d item(s), %s%s" % (
        d["name"], " (active)" if d["active"] else "", len(d["artboards"]), d["color_mode"],
        d["item_count"], d["path"] or "unsaved", "" if d["saved"] else ", unsaved changes") for d in docs]
    return result("\n".join(lines), structured={"documents": docs})


def t_get_document(a):
    info = run("var d = __doc(%s); var o = __docinfo(d); o.layers = __layers(d); return o;" % doc_arg(a))
    lines = ["%s  %s, %d item(s), %s" % (info["name"], info["color_mode"], info["item_count"],
                                          info["path"] or "unsaved")]
    lines.append("artboards:")
    for ab in info["artboards"]:
        lines.append("  - %r %sx%s pt%s" % (ab["name"], ab["width"], ab["height"],
                                            " (active)" if ab["active"] else ""))
    lines.append("layers, top first:")
    for l in info["layers"]:
        lines.append("  - %r %d item(s)%s%s%s" % (l["name"], l["items"],
                                                  "" if l["visible"] else " [hidden]",
                                                  " [locked]" if l["locked"] else "",
                                                  " (active)" if l["active"] else ""))
    return result("\n".join(lines), structured=info)


def t_list_items(a):
    limit = a.get("limit", 60)
    info = run("""
      var d = __doc(%s); var src = %s ? __layer(d, %s).pageItems : d.pageItems;
      var out = [], total = src.length, lim = %d;
      for (var i = 0; i < total && out.length < lim; i++) {
        var it = src[i]; if (%s && it.parent.typename === "GroupItem") continue;
        out.push(__info(it));
      }
      return {items: out, total: total};""" % (doc_arg(a), J(bool(a.get("layer"))), J(a.get("layer", "")),
                                               limit, J(not a.get("include_nested", False))))
    if not info["items"]:
        return result("No items%s." % (" on layer %r" % a["layer"] if a.get("layer") else ""),
                      structured=info)
    lines = [describe(it) + ("" if not it.get("hidden") else " [hidden]") +
             ("" if not it.get("locked") else " [locked]") +
             (" fill %s" % it["fill"] if it.get("fill") else "") +
             (" stroke %s %spt" % (it["stroke"], it.get("stroke_width")) if it.get("stroke") else "") +
             (" text=%r" % it["text"] if it.get("text") else "")
             for it in info["items"]]
    head = "%d item(s), top of the stack first%s:" % (
        len(info["items"]), " (of %d)" % info["total"] if info["total"] > len(info["items"]) else "")
    return result("\n".join([head] + lines), structured=info)


def t_get_item(a):
    info = run("""
      var d = __doc(%s); var it = __item(d, %s); var o = __info(it);
      if (it.typename === "TextFrame") { o.text = it.contents; o.kind = __enum(it.kind, "TextType"); }
      if (it.typename === "GroupItem") { o.children = []; for (var i = 0; i < it.pageItems.length; i++) o.children.push(__info(it.pageItems[i])); }
      if (it.typename === "PathItem") { o.closed = it.closed; o.points = it.pathPoints.length; }
      return o;""" % (doc_arg(a), J(a["uuid"])))
    return result(json.dumps(info, indent=1), structured=info)


def t_screenshot(a):
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    path = os.path.join(PREVIEW_DIR, "preview_%d.png" % int(time.time() * 1000))
    info = run("""
      var d = __doc(%s); var idx = __artboard(d, %s); var was = d.artboards.getActiveArtboardIndex();
      d.artboards.setActiveArtboardIndex(idx);
      var r = d.artboards[idx].artboardRect, w = r[2] - r[0], h = r[1] - r[3];
      var scale = Math.min(1, %d / Math.max(w, h)) * 100;
      var o = new ExportOptionsPNG24(); o.artBoardClipping = true; o.antiAliasing = true;
      o.transparency = false; o.horizontalScale = scale; o.verticalScale = scale;
      try { d.exportFile(new File(%s), ExportType.PNG24, o); } finally { d.artboards.setActiveArtboardIndex(was); }
      return {artboard: d.artboards[idx].name, width: Math.round(w * scale / 100), height: Math.round(h * scale / 100), source: d.name};"""
               % (doc_arg(a), J(a.get("artboard", "")), a.get("max_size", 1024), com.js_path(path)))
    com.wait_for_file(path)
    with open(path, "rb") as f:
        data = f.read()
    return result("Preview of artboard %r in %s at %sx%s, saved to %s" % (
        info["artboard"], info["source"], info["width"], info["height"], path), images=[data])


# -------------------------------------------------------------------- create

def t_new_document(a):
    mode = "DocumentColorSpace.%s" % a.get("color_mode", "rgb").upper()
    info = run("""
      var d = app.documents.add(%s, %s, %s, %d);
      %s
      return __docinfo(d);""" % (mode, a["width"], a["height"], a.get("artboards", 1),
                                 "" if not a.get("name") else "try { d.name = %s; } catch (e) {}" % J(a["name"])))
    return result("Made %r: %d artboard(s) of %sx%s pt, %s. It is the active document; save it "
                  "with ai_save_as." % (info["name"], len(info["artboards"]), a["width"],
                                        a["height"], info["color_mode"]), structured=info)


def t_open(a):
    path = studio_mcp.local_path(a["path"])
    if not os.path.isfile(path):
        raise ComError("No file at %s" % path)
    info = run("var d = app.open(new File(%s)); var o = __docinfo(d); o.layers = __layers(d); return o;"
               % com.js_path(path))
    return result("Opened %s: %d artboard(s), %d layer(s), %d item(s). It is the active document."
                  % (info["name"], len(info["artboards"]), info["layer_count"], info["item_count"]),
                  structured=info)


def t_add_text(a):
    info = run("""
      var d = __doc(%s); var layer = __target(d, %s);
      var t = layer.textFrames.add(); t.contents = %s;
      var ca = t.textRange.characterAttributes; ca.size = %s; ca.fillColor = __color(%s);
      %s
      t.position = [%s, -(%s)];
      %s
      return __info(t);""" % (doc_arg(a), J(a.get("layer", "")), J(a["text"]), a.get("size", 24),
                              J(a.get("color", "#000000")),
                              "ca.textFont = __font(%s);" % J(a["font"]) if a.get("font") else "",
                              a.get("x", 50), a.get("y", 50),
                              "t.name = %s;" % J(a["name"]) if a.get("name") else ""))
    return result("Added " + describe(info) + ". x,y was the top-left of the text.", structured=info)


def t_add_shape(a):
    kind = a["kind"]
    x, y, w, h = a.get("x", 0), a.get("y", 0), a["width"], a["height"]
    if kind == "rectangle":
        make = "var it = layer.pathItems.rectangle(-(%s), %s, %s, %s);" % (y, x, w, h)
    elif kind == "rounded_rectangle":
        r = a.get("corner_radius", 12)
        make = "var it = layer.pathItems.roundedRectangle(-(%s), %s, %s, %s, %s, %s);" % (y, x, w, h, r, r)
    elif kind == "ellipse":
        make = "var it = layer.pathItems.ellipse(-(%s), %s, %s, %s);" % (y, x, w, h)
    elif kind == "line":
        make = ("var it = layer.pathItems.add(); it.setEntirePath([[%s, -(%s)], [%s, -(%s)]]); "
                "it.filled = false; it.stroked = true;" % (x, y, x + w, y + h))
    elif kind == "polygon":
        make = ("var it = layer.pathItems.polygon(%s + %s / 2, -(%s) - %s / 2, Math.min(%s, %s) / 2, %d);"
                % (x, w, y, h, w, h, a.get("sides", 6)))
    else:
        make = ("var it = layer.pathItems.star(%s + %s / 2, -(%s) - %s / 2, Math.min(%s, %s) / 2, "
                "Math.min(%s, %s) / 4, %d);" % (x, w, y, h, w, h, w, h, a.get("points", 5)))
    style = {}
    if kind == "line":
        style = {"stroke": a.get("stroke", "#000000"), "stroke_width": a.get("stroke_width", 1)}
    else:
        style = {"fill": a.get("fill", "#808080"), "stroke": a.get("stroke", "none")}
        if "stroke_width" in a:
            style["stroke_width"] = a["stroke_width"]
    info = run("""
      var d = __doc(%s); var layer = __target(d, %s); %s
      %s %s
      return __info(it);""" % (doc_arg(a), J(a.get("layer", "")), make, style_js(style),
                               "it.name = %s;" % J(a["name"]) if a.get("name") else ""))
    return result("Added " + describe(info), structured=info)


def t_place_file(a):
    path = studio_mcp.local_path(a["path"])
    if not os.path.isfile(path):
        raise ComError("No file at %s" % path)
    info = run("""
      var d = __doc(%s); var layer = __target(d, %s);
      var it = layer.placedItems.add(); it.file = new File(%s);
      %s
      it.position = [%s, -(%s)];
      %s
      return __info(it);""" % (doc_arg(a), J(a.get("layer", "")), com.js_path(path),
                               ("var gb = it.geometricBounds; var s = %s / (gb[2] - gb[0]); it.resize(s * 100, s * 100);"
                                % a["width"]) if a.get("width") else "",
                               a.get("x", 0), a.get("y", 0),
                               "it.name = %s;" % J(a["name"]) if a.get("name") else ""))
    return result("Placed %s as %s" % (os.path.basename(path), describe(info)), structured=info)


def t_add_layer(a):
    info = run("""
      var d = __doc(%s); var l = d.layers.add(); l.name = %s; %s
      return __layers(d);""" % (doc_arg(a), J(a["name"]),
                                "d.activeLayer = l;" if a.get("activate", True) else ""))
    return result("Added layer %r%s. Layers, top first: %s" % (
        a["name"], " and made it active" if a.get("activate", True) else "",
        ", ".join(l["name"] for l in info)), structured={"layers": info})


def t_add_artboard(a):
    info = run("""
      var d = __doc(%s); var base = d.artboards[d.artboards.length - 1].artboardRect;
      var was = d.artboards.getActiveArtboardIndex(); var x = %s, y = %s;
      app.coordinateSystem = CoordinateSystem.DOCUMENTCOORDINATESYSTEM;
      if (x === null) x = base[2] + 50; if (y === null) y = base[1]; else y = -y;
      var ab = d.artboards.add([x, y, x + %s, y - %s]); ab.name = %s;
      if (!%s) d.artboards.setActiveArtboardIndex(was);
      return __artboards(d);""" % (doc_arg(a), J(a.get("x")), J(a.get("y")), a["width"], a["height"],
                                   J(a.get("name", "Artboard")), J(bool(a.get("activate", False)))))
    return result("Added artboard %r%s. Artboards: %s" % (
        a.get("name", "Artboard"), " and made it active" if a.get("activate") else "",
        ", ".join("%r %sx%s%s" % (ab["name"], ab["width"], ab["height"], " (active)" if ab["active"] else "")
                  for ab in info)), structured={"artboards": info})


def t_activate_artboard(a):
    info = run("""
      var d = __doc(%s); d.artboards.setActiveArtboardIndex(__artboard(d, %s)); return __artboards(d);"""
               % (doc_arg(a), J(a["artboard"])))
    return result("Artboard %r is now active: positions are measured from its top-left and it is what "
                  "ai_screenshot and exports show by default." % a["artboard"],
                  structured={"artboards": info})


# ---------------------------------------------------------------------- edit

def t_set_item(a):
    sets = []
    if "name" in a:
        sets.append("it.name = %s;" % J(a["name"]))
    if "hidden" in a:
        sets.append("it.hidden = %s;" % J(a["hidden"]))
    if "locked" in a:
        sets.append("it.locked = %s;" % J(a["locked"]))
    if "opacity" in a:
        sets.append("it.opacity = %s;" % a["opacity"])
    sets.append(style_js(a))
    if "x" in a or "y" in a:
        sets.append("var p = it.position; it.position = [%s === null ? p[0] : %s, %s === null ? p[1] : -(%s)];"
                    % (J(a.get("x")), J(a.get("x")), J(a.get("y")), J(a.get("y"))))
    if "width" in a or "height" in a:
        sets.append("""var g = it.geometricBounds, cw = g[2] - g[0], ch = -(g[3] - g[1]);
          var nw = %s, nh = %s; if (nw === null) nw = cw * nh / ch; if (nh === null) nh = ch * nw / cw;
          it.resize(nw / cw * 100, nh / ch * 100, true, true, true, true, 100, Transformation.TOPLEFT);"""
                    % (J(a.get("width")), J(a.get("height"))))
    text_sets = []
    if "text" in a:
        text_sets.append("it.contents = %s;" % J(a["text"]))
    if "font_size" in a:
        text_sets.append("it.textRange.characterAttributes.size = %s;" % a["font_size"])
    if "font" in a:
        text_sets.append("it.textRange.characterAttributes.textFont = __font(%s);" % J(a["font"]))
    if text_sets:
        sets.append('if (it.typename !== "TextFrame") __fail("uuid " + it.uuid + " is not a text item; text, '
                    'font and font_size only apply to text"); ' + " ".join(text_sets))
    sets = [x for x in sets if x]
    if not sets:
        raise ComError("nothing to set: give at least one of name, hidden, locked, opacity, fill, "
                       "stroke, stroke_width, x, y, width, height, text, font, font_size")
    info = run("var d = __doc(%s); var it = __item(d, %s); %s return __info(it);"
               % (doc_arg(a), J(a["uuid"]), " ".join(sets)))
    return result("Updated " + describe(info), structured=info)


def t_transform_item(a):
    ops = []
    if "dx" in a or "dy" in a:
        ops.append("it.translate(%s, -(%s));" % (a.get("dx", 0), a.get("dy", 0)))
    if "scale" in a:
        ops.append("it.resize(%s, %s, true, true, true, true, %s, Transformation.CENTER);"
                   % (a["scale"], a["scale"], a["scale"]))
    if "rotate" in a:
        ops.append("it.rotate(%s, true, true, true, true, Transformation.CENTER);" % (-a["rotate"]))
    if not ops:
        raise ComError("give dx/dy, scale or rotate")
    info = run("var d = __doc(%s); var it = __item(d, %s); %s return __info(it);"
               % (doc_arg(a), J(a["uuid"]), " ".join(ops)))
    return result("Transformed " + describe(info), structured=info)


def t_reorder_item(a):
    where = a["position"]
    if where in ("above", "below") and not a.get("relative_to"):
        raise ComError("position %r needs relative_to (a uuid)" % where)
    op = {"top": "it.zOrder(ZOrderMethod.BRINGTOFRONT);",
          "bottom": "it.zOrder(ZOrderMethod.SENDTOBACK);",
          "up": "it.zOrder(ZOrderMethod.BRINGFORWARD);",
          "down": "it.zOrder(ZOrderMethod.SENDBACKWARD);",
          "above": "it.move(__item(d, %s), ElementPlacement.PLACEBEFORE);" % J(a.get("relative_to")),
          "below": "it.move(__item(d, %s), ElementPlacement.PLACEAFTER);" % J(a.get("relative_to"))}[where]
    info = run("var d = __doc(%s); var it = __item(d, %s); %s return __info(it);"
               % (doc_arg(a), J(a["uuid"]), op))
    return result("Moved %s %s in the stack" % (describe(info), where), structured=info)


def t_duplicate_item(a):
    info = run("""
      var d = __doc(%s); var it = __item(d, %s); var c = it.duplicate(); %s %s
      return __info(c);""" % (doc_arg(a), J(a["uuid"]),
                              "c.translate(%s, -(%s));" % (a.get("dx", 0), a.get("dy", 0)) if ("dx" in a or "dy" in a) else "",
                              "c.name = %s;" % J(a["name"]) if a.get("name") else ""))
    return result("Duplicated into " + describe(info), structured=info)


def t_delete_item(a):
    info = run("""
      var d = __doc(%s); var it = __item(d, %s); var o = __info(it); it.remove();
      return {deleted: o, remaining: d.pageItems.length};""" % (doc_arg(a), J(a["uuid"])))
    return result("Deleted %s. %d item(s) remain." % (describe(info["deleted"]), info["remaining"]),
                  structured=info)


def t_set_layer(a):
    sets = []
    if "visible" in a:
        sets.append("l.visible = %s;" % J(a["visible"]))
    if "locked" in a:
        sets.append("l.locked = %s;" % J(a["locked"]))
    if "new_name" in a:
        sets.append("l.name = %s;" % J(a["new_name"]))
    if a.get("activate"):
        sets.append("d.activeLayer = l;")
    if not sets:
        raise ComError("nothing to set: give visible, locked, new_name or activate")
    info = run("var d = __doc(%s); var l = __layer(d, %s); %s return __layers(d);"
               % (doc_arg(a), J(a["layer"]), " ".join(sets)))
    return result("Updated layer %r. Layers, top first: %s" % (
        a.get("new_name", a["layer"]),
        ", ".join("%s%s%s" % (l["name"], "" if l["visible"] else " [hidden]", " [locked]" if l["locked"] else "")
                  for l in info)), structured={"layers": info})


# --------------------------------------------------------------------- files

def t_save(a):
    info = run("""
      var d = __doc(%s);
      if (!__path(d)) __fail(d.name + " has never been saved; use ai_save_as with a path.");
      d.save(); return __docinfo(d);""" % doc_arg(a))
    return result("Saved %s to %s." % (info["name"], info["path"]), structured=info)


def t_save_as(a):
    fmt = a["format"]
    path = a["path"]
    ext = {"jpg": ("jpg", "jpeg")}.get(fmt, (fmt,))
    if not path.lower().endswith(tuple("." + e for e in ext)):
        path = path + "." + fmt
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path) and not a.get("overwrite", False):
        raise ComError("%s exists; pass overwrite=true to replace it" % path)
    scale = a.get("scale", 100)
    if fmt == "ai":
        js = "var o = new IllustratorSaveOptions(); o.embedICCProfile = true; d.saveAs(f, o);"
    elif fmt == "pdf":
        js = "var o = new PDFSaveOptions(); o.preserveEditability = true; d.saveAs(f, o);"
    elif fmt == "svg":
        js = ("var o = new ExportOptionsSVG(); o.embedRasterImages = true; o.coordinatePrecision = 3; "
              "d.exportFile(f, ExportType.SVG, o);")
    elif fmt == "png":
        js = ("var o = new ExportOptionsPNG24(); o.artBoardClipping = true; o.antiAliasing = true; "
              "o.transparency = %s; o.horizontalScale = %s; o.verticalScale = %s; "
              "d.exportFile(f, ExportType.PNG24, o);" % (J(a.get("transparent", True)), scale, scale))
    else:
        js = ("var o = new ExportOptionsJPEG(); o.artBoardClipping = true; o.antiAliasing = true; "
              "o.qualitySetting = %d; o.horizontalScale = %s; o.verticalScale = %s; "
              "d.exportFile(f, ExportType.JPEG, o);" % (a.get("quality", 80), scale, scale))
    info = run("""
      var d = __doc(%s); var f = new File(%s);
      var idx = __artboard(d, %s); var was = d.artboards.getActiveArtboardIndex();
      d.artboards.setActiveArtboardIndex(idx);
      try { %s } finally { d.artboards.setActiveArtboardIndex(was); }
      var o2 = __docinfo(d); o2.artboard = d.artboards[idx].name; return o2;"""
               % (doc_arg(a), com.js_path(path), J(a.get("artboard", "")), js))
    com.wait_for_file(path)
    note = {"ai": " (the document is now that file)", "pdf": " (the document is now that file)"}.get(
        fmt, " (artboard %r; the open document is unchanged)" % info["artboard"])
    return result("Saved %s as %s%s." % (info["name"], path, note), structured=info)


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

DOC = s("Open document name, as ai_list_documents shows it. Default: the active document.")
UUID = s("The item's uuid from ai_list_items.")
LAYER = s("Layer name. Default: the active layer.")
FILL = s("#RRGGBB, or none.", pattern=COLOR)
STROKE = s("#RRGGBB, or none.", pattern=COLOR)

TOOLS = [
    ("ai_status", t_status,
     "Whether Illustrator is running, its version, and the open documents. Call this first "
     "in a session and whenever a call reports it could not reach Illustrator.",
     obj({})),
    ("ai_list_documents", t_list_documents,
     "Every open document: name, artboards, colour mode, item count, file path, unsaved changes.",
     obj({})),
    ("ai_get_document", t_get_document,
     "One document's artboards (name, size, which is active) and layers (name, visibility, "
     "lock, item count, which is active). Items are listed by ai_list_items.",
     obj({"document": DOC})),
    ("ai_list_items", t_list_items,
     "The page items in a document or on one layer, top of the stack first: uuid, type "
     "(path, text, group, placed, raster...), name, layer, bounds [left, top, right, bottom] "
     "in points from the ACTIVE artboard's top-left, fill, stroke and text. Items inside groups are "
     "left out unless include_nested is true. uuids are what every edit tool wants.",
     obj({"document": DOC, "layer": s("Only this layer's items."),
          "include_nested": b("Also list items inside groups. Default false."),
          "limit": i("Most items to return. Default 60.", minimum=1, maximum=500)})),
    ("ai_get_item", t_get_item,
     "One item in detail: full text for a text frame, children for a group, point count for a path.",
     obj({"uuid": UUID, "document": DOC}, ["uuid"])),
    ("ai_screenshot", t_screenshot,
     "A PNG of one artboard as it looks now, scaled to fit max_size. Returns the image so it "
     "can be shown and the path it was saved to. The document is not changed.",
     obj({"document": DOC, "artboard": s("Artboard name. Default: the active artboard."),
          "max_size": i("Longest side in pixels. Default 1024.", minimum=64, maximum=4096)})),
    ("ai_new_document", t_new_document,
     "Make a new document with one or more artboards of the given size and make it active.",
     obj({"width": n("Artboard width in points.", minimum=1, maximum=16383),
          "height": n("Artboard height in points.", minimum=1, maximum=16383),
          "color_mode": s("Default rgb.", enum=["rgb", "cmyk"]),
          "artboards": i("How many artboards. Default 1.", minimum=1, maximum=100),
          "name": s("A name for the window title; the file gets its name at ai_save_as.")},
         ["width", "height"])),
    ("ai_open", t_open,
     "Open an .ai, .pdf, .svg, .eps or image file from a path on this workstation; it becomes "
     "the active document.",
     obj({"path": s("Full Windows path of the file.")}, ["path"])),
    ("ai_add_text", t_add_text,
     "Add a point-text item. x,y is the top-left of the text in points from the artboard's "
     "top-left; size is points; color is #RRGGBB.",
     obj({"text": s("The text."), "x": n("Left edge. Default 50."), "y": n("Top edge. Default 50."),
          "size": n("Font size in points. Default 24.", minimum=1, maximum=1296),
          "color": s("#RRGGBB. Default #000000.", pattern=com.HEX),
          "font": s("PostScript font name, e.g. ArialMT or Helvetica-Bold. Default: Illustrator's current font."),
          "name": s("Item name."), "layer": LAYER, "document": DOC},
         ["text"])),
    ("ai_add_shape", t_add_shape,
     "Add a shape at x,y (top-left, points from the artboard's top-left) of width x height: a "
     "rectangle, rounded_rectangle, ellipse, line (from x,y to x+width,y+height), polygon or star "
     "(fitted inside the box). Default is a grey fill and no stroke; a line is stroked black.",
     obj({"kind": s("Shape.", enum=SHAPES), "x": n("Left edge. Default 0."), "y": n("Top edge. Default 0."),
          "width": n("Points.", minimum=0), "height": n("Points.", minimum=0),
          "fill": FILL, "stroke": STROKE, "stroke_width": n("Points.", minimum=0, maximum=1000),
          "corner_radius": n("rounded_rectangle only. Default 12."),
          "sides": i("polygon only. Default 6.", minimum=3, maximum=100),
          "points": i("star only. Default 5.", minimum=3, maximum=100),
          "name": s("Item name."), "layer": LAYER, "document": DOC},
         ["kind", "width", "height"])),
    ("ai_place_file", t_place_file,
     "Place an image or PDF/AI file as a linked item at x,y, optionally scaled to a width. Use "
     "this to bring in a picture from ComfyUI or Photoshop.",
     obj({"path": s("Full Windows path of the file."), "x": n("Left edge. Default 0."),
          "y": n("Top edge. Default 0."), "width": n("Scale to this width in points, keeping aspect."),
          "name": s("Item name."), "layer": LAYER, "document": DOC},
         ["path"])),
    ("ai_add_layer", t_add_layer,
     "Add a top-level layer above the others and, by default, make it the active layer new items go on.",
     obj({"name": s("Layer name."), "activate": b("Default true."), "document": DOC}, ["name"])),
    ("ai_add_artboard", t_add_artboard,
     "Add an artboard. With no x,y it goes to the right of the last one.",
     obj({"width": n("Points.", minimum=1, maximum=16383), "height": n("Points.", minimum=1, maximum=16383),
          "x": n("Left edge in document points."), "y": n("Top edge in document points, y down."),
          "name": s("Artboard name. Default Artboard."),
          "activate": b("Make it the active artboard. Default false: the active one stays."),
          "document": DOC},
         ["width", "height"])),
    ("ai_activate_artboard", t_activate_artboard,
     "Make an artboard the active one. Item positions are measured from the ACTIVE artboard's "
     "top-left, and it is what ai_screenshot and exports show by default.",
     obj({"artboard": s("Artboard name, from ai_get_document."), "document": DOC}, ["artboard"])),
    ("ai_set_item", t_set_item,
     "Change an item: name, hidden, locked, opacity (0..100), fill, stroke, stroke_width, "
     "position x,y (new top-left), width/height (one keeps the aspect), and for text its "
     "text, font or font_size. Only the given fields change.",
     obj({"uuid": UUID, "name": s("New name."), "hidden": b("Hide or show."), "locked": b("Lock or unlock."),
          "opacity": n("0..100.", minimum=0, maximum=100), "fill": FILL, "stroke": STROKE,
          "stroke_width": n("Points.", minimum=0, maximum=1000),
          "x": n("New left edge."), "y": n("New top edge."),
          "width": n("New width in points.", minimum=0.01), "height": n("New height in points.", minimum=0.01),
          "text": s("New contents (text items)."), "font": s("PostScript font name (text items)."),
          "font_size": n("Points (text items).", minimum=1, maximum=1296), "document": DOC},
         ["uuid"])),
    ("ai_transform_item", t_transform_item,
     "Move an item by dx,dy points (y down), scale it by a percentage about its centre, or "
     "rotate it clockwise by degrees - in that order.",
     obj({"uuid": UUID, "dx": n("Points right."), "dy": n("Points down."),
          "scale": n("Percent; 100 is unchanged.", minimum=0.1, maximum=10000),
          "rotate": n("Degrees clockwise."), "document": DOC},
         ["uuid"])),
    ("ai_reorder_item", t_reorder_item,
     "Change stacking within the item's layer: top, bottom, up or down one step, or directly "
     "above or below another item (relative_to).",
     obj({"uuid": UUID, "position": s("Where.", enum=["top", "bottom", "up", "down", "above", "below"]),
          "relative_to": s("The other item's uuid, for above/below."), "document": DOC},
         ["uuid", "position"])),
    ("ai_duplicate_item", t_duplicate_item,
     "Duplicate an item in place, or offset by dx,dy points.",
     obj({"uuid": UUID, "dx": n("Points right."), "dy": n("Points down."), "name": s("Name for the copy."),
          "document": DOC}, ["uuid"])),
    ("ai_delete_item", t_delete_item,
     "Delete an item. Ask the user before deleting anything they did not ask to remove.",
     obj({"uuid": UUID, "document": DOC}, ["uuid"])),
    ("ai_set_layer", t_set_layer,
     "Show, hide, lock, unlock, rename or activate a layer.",
     obj({"layer": s("Layer name."), "visible": b("Show or hide."), "locked": b("Lock or unlock."),
          "new_name": s("Rename to this."), "activate": b("Make it the active layer."), "document": DOC},
         ["layer"])),
    ("ai_save", t_save,
     "Save a document to its own .ai file. Fails for a document that has never been saved - use ai_save_as.",
     obj({"document": DOC})),
    ("ai_save_as", t_save_as,
     "Save as .ai or .pdf (the document becomes that file), or export one artboard as .svg, "
     ".png or .jpg (the open document is unchanged). Refuses to overwrite unless told to.",
     obj({"path": s("Full Windows path; the extension is added if missing."),
          "format": s("File format.", enum=FORMATS),
          "artboard": s("Artboard to export (png, jpg, svg). Default: the active one."),
          "scale": n("Export scale percent for png/jpg. Default 100.", minimum=1, maximum=1000),
          "transparent": b("png: transparent background. Default true."),
          "quality": i("jpg quality 0..100. Default 80.", minimum=0, maximum=100),
          "overwrite": b("Default false."), "document": DOC},
         ["path", "format"])),
    ("ai_close_document", t_close_document,
     "Close a document, discarding changes unless save=true. Ask before discarding work.",
     obj({"document": DOC, "save": b("Save first. Default false.")})),
    ("ai_run_jsx", t_run_jsx,
     "Run ExtendScript inside Illustrator for anything no other tool covers. The code is the "
     "body of a function: `return` a value (string, number, array or plain object) to see it. "
     "`app` and `app.activeDocument` are available; the coordinate system is the artboard's "
     "(Illustrator's own y-up convention applies inside raw script) and alerts are suppressed.",
     obj({"code": s("ExtendScript source."),
          "timeout": i("Seconds to allow. Default 120.", minimum=5, maximum=1800)},
         ["code"])),
]

READ_ONLY = {"ai_status", "ai_list_documents", "ai_get_document", "ai_list_items", "ai_get_item",
             "ai_screenshot"}

HINTS = {
    "ai_new_document": {"destructive": False},
    "ai_open": {"destructive": False, "idempotent": True},
    "ai_add_text": {"destructive": False},
    "ai_add_shape": {"destructive": False},
    "ai_place_file": {"destructive": False},
    "ai_add_layer": {"destructive": False},
    "ai_add_artboard": {"destructive": False},
    "ai_activate_artboard": {"destructive": False, "idempotent": True},
    "ai_duplicate_item": {"destructive": False},
    "ai_set_item": {"idempotent": True},
    "ai_set_layer": {"idempotent": True},
    "ai_reorder_item": {"destructive": False, "idempotent": True},
    "ai_save": {"idempotent": True},
    "ai_save_as": {"idempotent": True},
    "ai_delete_item": {"destructive": True},
    "ai_close_document": {"destructive": True},
}

SERVER = studio_mcp.Server(
    "studio-illustrator-mcp", "1.0",
    studio_mcp.tools_from_table(TOOLS, read_only=READ_ONLY, **HINTS),
    errors=(ComError, KeyError, TypeError, ValueError, OSError),
    instructions="Illustrator on this workstation, driven through its own scripting engine. "
                 "Start with ai_status, then ai_list_items for uuids. Points, origin at the "
                 "artboard's top-left with y down, opacity 0..100, colours #RRGGBB or none.")


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
