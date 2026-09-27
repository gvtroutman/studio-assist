"""ComfyUI's own node editor inside the Image Studio: the graph a picture was
made with, loaded into ComfyUI's page, in a browser window held in the tab the
way the Milanote tab holds Milanote.

The window is `studio_milanote.Browser` pointed at a backend's ComfyUI: a
Chrome (or Edge) `--app` window on a profile of its own
(`%LOCALAPPDATA%\\StudioAssistant\\comfyui-browser`), re-parented into a frame
and sized with it. A graph goes in over DevTools: ComfyUI's page has `app`,
whose `loadApiJson(graph, name)` takes the API-format graph Image Studio
queued and opens it as a workflow tab of its own, the nodes laid out in
columns by their links. Nothing is queued by loading; Run in the page is the
user's, and what it makes lands in ComfyUI's output folder, not in History.

A record's graphs are its steps (`graph_steps`): the picture, the face pass,
the real-face paste, each labelled pass (`Job.passes`: eyes, hands, glasses,
Fix a spot's spots, the Visual Critic's redraws), and a Try On's garments.

Stdlib only, like the Milanote module it builds on.
"""

import json
import os
import time
import urllib.parse

import studio_milanote as milanote

READY_WAIT = 30.0         # seconds for ComfyUI's page to have its `app`
# Ready is more than `app`: the page then reopens the workflow tabs of its
# last session, a moment later (0.2 s here), and that replaced a graph loaded
# in between. The title names a workflow once they are open.
READY = ("!!(window.app && app.graph && typeof app.loadApiJson === 'function' "
         "&& app.vueAppReady && document.title !== 'ComfyUI')")
# Loading leaves the view where it was, often on empty canvas, so the view is
# fitted to the nodes. Not by the Fit View command: it animates by frames,
# which a window out of sight does not draw.
LOAD = ("app.loadApiJson(%s, %s).then(() => new Promise(r => setTimeout(r, 300)))"
        ".then(() => { const ns = app.graph.nodes || app.graph._nodes;"
        " if (ns.length && app.canvas.ds.fitToBounds) {"
        "  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;"
        "  for (const n of ns) { x0 = Math.min(x0, n.pos[0]); y0 = Math.min(y0, n.pos[1] - 30);"
        "   x1 = Math.max(x1, n.pos[0] + n.size[0]); y1 = Math.max(y1, n.pos[1] + n.size[1]); }"
        "  x0 -= (x1 - x0) * 0.12;"       # ComfyUI's toolbar covers the canvas's left edge
        "  app.canvas.ds.fitToBounds([x0, y0, x1 - x0, y1 - y0], {zoom: 0.9});"
        "  app.canvas.setDirty(true, true); }"
        " return ns.length; })")


def profile_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.environ.get("STUDIO_COMFY_PROFILE") or os.path.join(
        base, "StudioAssistant", "comfyui-browser")


def origin(url):
    """scheme://host:port of `url`, so 127.0.0.1:8188/ and 127.0.0.1:8188/api
    are the same ComfyUI."""
    u = urllib.parse.urlsplit(url or "")
    return "%s://%s" % (u.scheme or "http", (u.netloc or u.path).lower().rstrip("/"))


def graph_steps(rec):
    """The graphs a record (or a job's fields in a dict of the same keys) was
    made with, in the order they ran -> [(label, graph)]."""
    steps = []
    dress = (rec.get("dress") or {}).get("graphs") or []
    for n, g in enumerate(dress):
        steps.append(("Try on %d" % (n + 1) if len(dress) > 1 else "Try on", g))
    if rec.get("graph") and not any(g is rec["graph"] or g == rec["graph"] for g in dress):
        steps.append(("Picture", rec["graph"]))
    if rec.get("face_graph"):
        steps.append(("Face pass", rec["face_graph"]))
    if rec.get("paste_graph"):
        steps.append(("Real-face paste", rec["paste_graph"]))
    for p in rec.get("passes") or []:
        if p.get("graph"):
            steps.append((p.get("label") or "Pass", p["graph"]))
    return [(label, g) for label, g in steps if isinstance(g, dict) and g]


def job_fields(job):
    """A running or failed job's graphs, in `graph_steps`' keys."""
    return {"graph": job.graph, "face_graph": job.face_graph,
            "paste_graph": job.paste_graph, "passes": job.passes, "dress": job.dress}


class ComfyBrowser(milanote.Browser):
    """A backend's ComfyUI in a window of our own. Blocking methods run off
    the UI thread; `embed`, `fit`, `focus` and `release` on it."""

    name = "ComfyUI"

    def __init__(self, url, exe=None, profile=None):
        super().__init__(exe=exe, profile=profile or profile_dir(), url=url)

    def target(self, port):
        """The page showing this ComfyUI, else the first page."""
        found = milanote.pages(port)
        for t in found:
            if origin(t.get("url")) == origin(self.url):
                return t
        if found:
            return found[0]
        raise milanote.DevToolsError("the ComfyUI window has no page open")

    def show(self, graph, title, url=None):
        """Open `graph` in the ComfyUI at `url` (this one when None) as a
        workflow tab called `title`. -> how many nodes it has. Blocks."""
        dt = self.page()
        try:
            if url and origin(url) != origin(self.url):
                self.url = url
                dt.call("Page.navigate", {"url": url})
                time.sleep(0.5)           # the old page's `app` is still there a moment
            self._wait_ready(dt)
            # Twice at most: a count that is not the graph's means something
            # (the session's tabs coming back) drew over it.
            for _ in range(2):
                got = dt.call("Runtime.evaluate", {
                    "expression": LOAD % (json.dumps(graph), json.dumps(title)),
                    "awaitPromise": True, "returnByValue": True})
                if got.get("exceptionDetails"):
                    detail = got["exceptionDetails"]
                    text = ((detail.get("exception") or {}).get("description")
                            or detail.get("text") or "no reason given")
                    raise RuntimeError("ComfyUI would not open the graph: %s"
                                       % text.splitlines()[0])
                count = (got.get("result") or {}).get("value")
                if count == len(graph):
                    break
        finally:
            dt.close()
        return count

    def _wait_ready(self, dt, wait=READY_WAIT):
        deadline = time.monotonic() + wait
        while True:
            try:
                got = dt.call("Runtime.evaluate", {"expression": READY,
                                                   "returnByValue": True})
                if (got.get("result") or {}).get("value") is True:
                    return
            except milanote.DevToolsError:
                pass                      # mid-navigation: the context went
            if time.monotonic() >= deadline:
                raise RuntimeError("ComfyUI at %s did not finish loading in %d seconds. "
                                   "Is it running?" % (origin(self.url), wait))
            time.sleep(0.3)
