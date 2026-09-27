#!/usr/bin/env python3
"""
studio_appinfo - what a tab's model is told about the app it drives.

A tab's prompt teaches the bridge, not the app: the model did not know which
app it was in beyond a tool prefix, which release is installed, or what the
app is for. This module is that profile - the app's name and tab, the release
on this PC, and a short overview from Wikipedia when there is a page - cached
on disk beside the settings as appinfo/<app>.json.

`load` is instant (the cache); `refresh` reads the release and, at most once a
month, the overview, and belongs on a worker thread. Everything is
best-effort: no network, no page or an unreadable exe costs a line, never the
tab. Stdlib only.
"""

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

WIKI_MAX_AGE = 30 * 86400      # the overview changes slowly; the release is read every boot
EXTRACT_CHARS = 700
TIMEOUT = 8
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# The Wikipedia page per app id; None when there is none worth reading. An app
# not listed (a hand-entered bridge) tries "<name> (software)", then its name,
# and keeps only a page that reads as software.
WIKI = {
    "after-effects": "Adobe After Effects",
    "resolve": "DaVinci Resolve",
    "comfyui": "ComfyUI",
    "photoshop": "Adobe Photoshop",
    "illustrator": "Adobe Illustrator",
    "premiere": "Adobe Premiere Pro",
    "opencode": None,
}
SOFTWARE = re.compile(r"\b(software|application|program|editor|suite|tool|framework|"
                      r"interface|app)\b", re.I)


def cache_path(app_id, base):
    return os.path.join(base, "appinfo", app_id + ".json")


def load(app_id, base):
    """The cached profile, or {}."""
    try:
        with open(cache_path(app_id, base), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save(app_id, base, data):
    path = cache_path(app_id, base)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except Exception:
        pass


# ------------------------------------------------------------------ release

def file_version(path):
    """The product version in a Windows exe's version resource, "25.5.0.12",
    or None. ctypes over version.dll - no pywin32."""
    if not path or os.name != "nt" or not os.path.exists(path):
        return None
    try:
        import ctypes
        from ctypes import wintypes
        ver = ctypes.WinDLL("version")
        size = ver.GetFileVersionInfoSizeW(path, None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not ver.GetFileVersionInfoW(path, 0, size, buf):
            return None
        ptr, n = ctypes.c_void_p(), wintypes.UINT()
        if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(n)) or not n.value:
            return None
        # VS_FIXEDFILEINFO: signature, struc version, file MS/LS, product MS/LS, ...
        fields = ctypes.cast(ptr, ctypes.POINTER(wintypes.DWORD * 13)).contents
        ms, ls = fields[4], fields[5]
        parts = [ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF]
        while len(parts) > 2 and parts[-1] == 0:
            parts.pop()
        return ".".join(str(p) for p in parts) if any(parts) else None
    except Exception:
        return None


def command_version(exe):
    """`<exe> --version`'s first line, for a program with no version resource."""
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             timeout=TIMEOUT,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        line = (out.stdout or out.stderr or "").strip().splitlines()
        return line[0].strip()[:60] if line else None
    except Exception:
        return None


def release(app):
    """The installed release as a phrase, or None when nothing here says."""
    exe = None
    try:
        exe = app.exe()
    except Exception:
        pass
    if exe is None and getattr(app, "id", "") == "opencode":
        try:
            import studio_agent
            exe = studio_agent.opencode_exe()
        except Exception:
            exe = None
        return command_version(exe) if exe else None
    if not exe or not os.path.exists(exe):
        return None
    number = file_version(exe)
    beta = "beta" in exe.lower()
    year = re.search(r"\b(20\d\d)\b", exe)
    label = " ".join(x for x in (year.group(1) if year else "", "Beta" if beta else "") if x)
    if number and label:
        return "%s (%s)" % (number, label)
    return number or label or None


# ----------------------------------------------------------------- overview

def fetch_summary(title, opener=None):
    """Wikipedia's summary of one page: (extract, url) or None."""
    url = SUMMARY_URL + urllib.parse.quote(title.replace(" ", "_"), safe="")
    req = urllib.request.Request(url, headers={
        "User-Agent": "StudioAssist/1.0 (local desktop app)", "Accept": "application/json"})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    if data.get("type") != "standard":
        return None                       # disambiguation, missing
    extract = " ".join((data.get("extract") or "").split())
    if not extract:
        return None
    page = ((data.get("content_urls") or {}).get("desktop") or {}).get("page") or url
    return extract, page


def overview(app, opener=None):
    """(extract, url) for the app, or None when it has no page."""
    if app.id in WIKI:
        title = WIKI[app.id]
        return fetch_summary(title, opener) if title else None
    for title in ("%s (software)" % app.name, app.name):
        found = fetch_summary(title, opener)
        if found and SOFTWARE.search(found[0][:300]):
            return found
    return None


def refresh(app, base, opener=None, now=None):
    """Read the release, and the overview when the cached one is old; save and
    return the profile."""
    now = time.time() if now is None else now
    data = load(app.id, base)
    data["name"] = app.name
    data["tab"] = getattr(app, "tab", app.name)
    data["release"] = release(app)
    if "wiki_checked" not in data or now - float(data["wiki_checked"]) > WIKI_MAX_AGE:
        found = overview(app, opener)
        data["wiki_checked"] = now
        # A failed fetch keeps the overview it had rather than forgetting it.
        if found:
            data["wiki"], data["wiki_url"] = found
    save(app.id, base, data)
    return data


def render(data):
    """The profile as the prompt's section body, or ""."""
    if not data or not data.get("name"):
        return ""
    lines = ["- You are in Studio Assist's %s tab, driving %s through its bridge."
             % (data.get("tab") or data["name"], data["name"])]
    if data.get("release"):
        lines.append("- Installed release on this PC: %s. Prefer features and scripting "
                     "calls that exist in it." % data["release"])
    wiki = data.get("wiki") or ""
    if wiki:
        if len(wiki) > EXTRACT_CHARS:
            wiki = wiki[:EXTRACT_CHARS].rsplit(" ", 1)[0] + "…"
        lines.append("- What it is (Wikipedia, %s): %s" % (data.get("wiki_url", ""), wiki))
    return "\n".join(lines)
