"""
app_update - the release a sidebar row shows, and how that app is updated.

A rail row used to say "2026, Beta" or "installed": which folders exist, not
what is in them. It now says the release in the program itself ("26.5.0",
"27.1.0 Beta + 2026" when another install sits beside the one the row runs),
read from the .exe's version resource (`appinfo.file_version`) - fast enough
for `detect_apps` to do at start-up. Two have no .exe to read: OpenCode answers
`--version` and ComfyUI, on the LLM PC, answers /system_stats; `slow_release`
asks them and belongs on a worker thread.

Updating is each app's own business, so `plan` only says where to send the
user: Creative Cloud for Adobe's, winget for Blender, `opencode upgrade`, and
the download page for the rest. Nothing is updated behind the user's back, and
a console started here is not a `procs` child: an installer must not die with
this window. Like ComfyUI's Start, it is held in the Terminal tab.

Stdlib only.
"""

import json
import os
import re
import shutil
import subprocess
import urllib.request

import core.appinfo as appinfo

CREATIVE_CLOUD = r"C:\Program Files\Adobe\Adobe Creative Cloud\ACC\Creative Cloud.exe"
CREATIVE_CLOUD_WEB = "https://creativecloud.adobe.com/apps/all/desktop"
# The names agent_bridges.PRODUCTS lists - everything Creative Cloud installs.
ADOBE = frozenset(("After Effects", "Premiere Pro", "Photoshop", "Illustrator",
                   "Audition", "Media Encoder", "Acrobat"))
# Apps winget knows by id; checked with `winget list` on this PC.
WINGET = {"Blender": "BlenderFoundation.Blender"}
PAGES = {
    "Blender": "https://www.blender.org/download/",
    "DaVinci Resolve": "https://www.blackmagicdesign.com/support/family/davinci-resolve-and-fusion",
    "ComfyUI": "https://github.com/comfyanonymous/ComfyUI/releases",
}
TIMEOUT = 4


def short(version):
    """ "26.5.0.89" -> "26.5.0": the build number is noise on a rail row."""
    if not version:
        return ""
    return ".".join(version.split(".")[:3])


def release_line(exe, labels=()):
    """The row's second line for the program at `exe`, or "" when it has no
    version to read. `labels` are the installs detection found side by side
    ("2026", "Beta"); the ones that are not this exe's are named after it."""
    number = short(appinfo.file_version(exe))
    if not number:
        return ""
    year = re.search(r"\b(20\d\d)\b", exe)
    mine = "Beta" if "beta" in exe.lower() else (year.group(1) if year else None)
    text = number + (" Beta" if mine == "Beta" else "")
    others = [l for l in labels if l != mine]
    if others and mine is not None:
        text += " + " + ", ".join(others)
    return text


def slow_release(row):
    """The release of an app with no .exe to read, or "" - a subprocess or a
    request, so never on the UI thread."""
    if row.get("id") == "opencode":
        import core.agent as studio_agent
        exe = studio_agent.opencode_exe()
        found = appinfo.command_version(exe) if exe else None
        m = re.search(r"\d+(\.\d+)+", found or "")
        return m.group(0) if m else ""
    if row.get("id") == "comfyui":
        import core.agent as studio_agent
        try:
            with urllib.request.urlopen(studio_agent.COMFYUI_URL + "/system_stats",
                                        timeout=TIMEOUT) as r:
                stats = json.loads(r.read().decode("utf-8"))
            return str((stats.get("system") or {}).get("comfyui_version") or "")
        except Exception:
            return ""
    return ""


def read(row):
    """`row`'s release as it stands now - after an update, say. Its .exe's
    when it has one (a hand-entered bridge's says nothing about the app),
    otherwise by asking. Slow at worst: a worker thread's."""
    if row.get("exe") and row.get("version") not in ("bridge", "server"):
        labels = [l for l in (row.get("version") or "").split(", ") if l]
        return release_line(row["exe"], labels)
    return slow_release(row)


def plan(row):
    """How `row`'s app is updated: {"label", "tip", "open"} for a program or
    page to open, or {"label", "tip", "run"} for a console command. None for
    an app this module knows no updater for (a bridge entered by hand)."""
    name = row["name"]
    if name in ADOBE:
        cc = CREATIVE_CLOUD if os.path.isfile(CREATIVE_CLOUD) else CREATIVE_CLOUD_WEB
        return {"label": "Update in Creative Cloud...", "open": cc,
                "tip": "Update %s in Creative Cloud" % name}
    if name in WINGET and shutil.which("winget"):
        return {"label": "Update with winget...",
                "run": ["cmd", "/k", "winget", "upgrade", "--id", WINGET[name], "--exact"],
                "tip": "Update %s with winget (in the Terminal tab)" % name}
    if row.get("id") == "opencode":
        import core.agent as studio_agent
        exe = studio_agent.opencode_exe()
        if exe:
            return {"label": "Update OpenCode...", "run": ["cmd", "/k", exe, "upgrade"],
                    "tip": "Run opencode upgrade (in the Terminal tab)"}
    if name in PAGES:
        where = " - update it on the %s" % "LLM PC" if row.get("remote") else ""
        return {"label": "Download the latest %s..." % name, "open": PAGES[name],
                "tip": "Open %s's download page%s" % (name, where)}
    return None


def run(step):
    """Carry out a `plan`. Raises OSError when it cannot start."""
    if "open" in step:
        os.startfile(step["open"])
        return
    subprocess.Popen(step["run"], close_fds=True,
                     creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
