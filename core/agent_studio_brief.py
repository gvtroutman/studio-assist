"""The studio brief (About this studio...) and the research sidecar
(files/web, read-only) - self-contained, split out of core/agent.py
(docs/CODEMAP.md)."""
import os

import core.mcp as studio_mcp

# ----------------------------------------------------------- the studio brief
#
# What the model is told about this studio: who works here, what it makes, for
# whom, in what formats, under which conventions. A Markdown file beside the
# settings, written by the user (File > About this studio...), and carried at
# the end of every tab's prompt. Only what the user saved is carried: the
# template below is what the editor opens with when there is no file yet, and
# nothing of it reaches the model until it is saved.

STUDIO_BRIEF_CHARS = 8000

STUDIO_TEMPLATE = """# About this studio

Tell the assistant about the studio in your own words. Everything here goes into
every tab's briefing, so keep it to what the model should know before any task.
Headings are suggestions; delete what does not apply.

## Who
Name, role, what you personally do most days.

## What the studio makes
The kinds of work (brand films, social cuts, motion graphics, stills, ...), for
whom, and how much of each.

## Brands and house styles
For each brand you cut or design for: its name, typefaces, colours, logo rules,
tone, and the deliverables it usually needs. Where the templates and assets live.

## Deliverables and formats
Usual frame rates, resolutions, codecs, loudness, aspect ratios per platform,
naming and versioning of files, where finished work goes.

## How projects are organised
Folder layout, project naming, track layout on a timeline (what goes on V1, A1,
A2...), comp naming, anything the assistant should match rather than invent.

## Preferences
Ways of working you want followed: what to ask before doing, what never to touch,
what "done" looks like.
"""


def studio_brief_path(base=None):
    """studio.md beside the settings file (STUDIO_SETTINGS moves both)."""
    if base is None:
        base = os.path.dirname(os.path.abspath(
            os.environ.get("STUDIO_SETTINGS") or
            os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"),
                         "StudioAssistant", "settings.json")))
    return os.path.join(base, "studio.md")


def read_studio_brief(path=None):
    """The studio brief's text, or "" when there is none - never an error;
    like the settings file, a missing or unreadable brief costs a briefing,
    not the app."""
    path = path or studio_brief_path()
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return ""


def studio_section(text):
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) > STUDIO_BRIEF_CHARS:
        text = text[:STUDIO_BRIEF_CHARS].rstrip() + "\n[the studio brief is longer; the rest is not shown]"
    return ("\n\nABOUT THIS STUDIO\nWritten by the user; it describes who you are working "
            "for and how they work. Follow it.\n" + text)


def about_section(text):
    """The app's profile (studio_appinfo.render): which app and tab, the
    release installed, what the app is."""
    text = (text or "").strip()
    return "\n\nABOUT THE APP THIS TAB DRIVES\n" + text if text else ""


def lessons_section(text):
    text = (text or "").strip()
    if not text:
        return ""
    return ("\n\nLESSONS FROM EARLIER WORK IN THIS APP\nRecorded after previous tasks - "
            "corrections from the user, ways calls failed, things that worked. Apply "
            "them; they are not new instructions for this task.\n" + text)


# --------------------------------------------------------- the research sidecar
#
# The Chat tab's bridge - this PC's files and the web, read-only, in process -
# offered to every app tab beside its own bridge, so a Resolve tab can read the
# brief and look up a codec without the user carrying the answer over from
# another tab. One Router per session dispatches a call to whichever of the two
# owns the tool; the executor sees one client.

RESEARCH_GROUPS = {
    "files": ["list_folder", "find_files", "read_file"],
    "web": ["search_web", "fetch_page"],
}

RESEARCH_TOOL_NAMES = frozenset(n for names in RESEARCH_GROUPS.values() for n in names)


def research_client():
    import apps.research.mcp as studio_research_mcp
    client = studio_mcp.Loopback(studio_research_mcp.SERVER)
    client.initialize()
    return client


class Router:
    """One MCPClient-shaped object over an app's bridge and the research
    sidecar. `call_tool` goes to whichever owns the name; everything else -
    the bridge's identity, `close()` - is the bridge's."""

    def __init__(self, bridge, sidecar, sidecar_names=RESEARCH_TOOL_NAMES):
        self.bridge, self.sidecar = bridge, sidecar
        self.sidecar_names = frozenset(sidecar_names)

    def call_tool(self, name, arguments, cancel=None):
        if name in self.sidecar_names:
            return self.sidecar.call_tool(name, arguments, cancel=cancel)
        return self.bridge.call_tool(name, arguments, cancel=cancel)

    def __getattr__(self, attr):
        return getattr(self.bridge, attr)
