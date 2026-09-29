"""Bridges the user enters by hand (BridgeSpec) and detecting what is
installed on this machine. Split out of core/agent.py (docs/CODEMAP.md);
mutates APPS/TABS in place, imported from core.agent by reference so
add_bridge()/remove_bridge() stay visible there."""
import os
import re
import shlex

from core.agent import (
    AppSpec, APPS, APPS_BY_ID, TABS, TABS_BY_ID, BRIDGE_PROMPT, newest_match,
)



# --------------------------------------------------- bridges entered by hand

def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "bridge"


def split_command(line):
    """A command line as the user typed it -> (command, args). Windows quoting."""
    parts = shlex.split(line, posix=False)
    return parts[0].strip('"'), [p.strip('"') for p in parts[1:]]


class BridgeSpec(AppSpec):
    """
    An app the user connected by hand: any MCP stdio bridge, installed by them,
    entered as a command line. Nothing is known about it until it starts, so the
    registry entry is filled in two steps - what the user typed now, and what the
    bridge answers at boot (`learn()`): its tools, grouped by name prefix, and its
    own `instructions`, which become the second half of the prompt.

    `custom` is what the GUI asks before offering to edit or forget an entry.
    An entry is data (`record()`), kept in the settings file and rebuilt by
    `load_bridges()` when the window opens.
    """

    custom = True

    def __init__(self, name, command, args=(), exe="", probe="", note="", id=None,
                 code="", fg="#E6E6E6", bg="#4A4A4A"):
        words = [w for w in re.split(r"\W+", name) if w]
        initials = (words[0][0] + (words[1][0] if len(words) > 1 else words[0][1:2])
                    if words else "Mc").title()
        AppSpec.__init__(
            self, id=id or slug(name), name=name, tab=name, code=code or initials,
            fg=fg, bg=bg, exe_globs=[exe] if exe else [], probe=probe,
            command=command, args=list(args), bridge_label=_label(command, args),
            groups={}, default_groups=[], system_prompt="", examples=[
                "What can you do with %s?" % name,
                "What is open in %s right now?" % name,
            ], launch_note=note or ("Start %s yourself, with whatever panel or plugin "
                                    "its bridge needs." % name))
        self.exe_path = exe
        self.instructions = ""
        self.learned = False

    @property
    def remote(self):
        return False                      # an exe-less entry still runs here

    def installed(self):
        return True                       # the user said so by entering it

    def running(self):
        # No probe means nothing to check: the bridge answering is the evidence.
        return True if not self.probe else AppSpec.running(self)

    def launch(self):
        if not self.exe_globs:
            raise RuntimeError("%s has no program path to start. %s"
                               % (self.name, self.launch_note))
        AppSpec.launch(self)

    def learn(self, tools, instructions=""):
        """Fill in what only the running bridge knows: tools and its briefing."""
        names = [t["name"] for t in tools]
        self.groups = group_by_prefix(names)
        self.default_groups = list(self.groups)
        self.instructions = (instructions or "").strip()
        self.learned = True

    @property
    def system_prompt(self):
        brief = ("\nWHAT THE BRIDGE SAYS ABOUT ITSELF\n" + self.instructions + "\n"
                 if self.instructions else "")
        return BRIDGE_PROMPT % {"name": self.name, "instructions": brief}

    @system_prompt.setter
    def system_prompt(self, value):
        pass                              # AppSpec.__init__ assigns; the property derives

    def record(self):
        return {"id": self.id, "name": self.name, "command": self.command,
                "args": list(self.args), "exe": self.exe_path, "probe": self.probe,
                "note": self.launch_note if self.exe_path else ""}


def _label(command, args):
    line = " ".join([os.path.basename(command)] + list(args))
    return line if len(line) <= 28 else line[:27] + "…"


def group_by_prefix(names):
    """Tool groups a bridge never declared, from the names it did.

    `ppro_timeline_add`, `ppro_timeline_list` -> group "ppro_timeline"? No -
    one level: everything before the first underscore, when that makes at least
    two groups with something in them; otherwise one group, "all". The
    capabilities dialog then lets a thousand-tool bridge be narrowed to the
    families a session needs, which the request budget may require.
    """
    if not names:
        return {}
    buckets = {}
    for n in names:
        head = n.split("_", 1)[0] if "_" in n else "other"
        buckets.setdefault(head, []).append(n)
    if len(buckets) < 2 or any(len(v) < 2 for v in buckets.values()):
        return {"all": list(names)}
    return buckets


def add_bridge(spec):
    """Put a hand-entered bridge in the registry - APPS, the tab list and the
    sidebar's drivable map - replacing an earlier entry with the same id."""
    remove_bridge(spec.id)
    APPS.append(spec)
    APPS_BY_ID[spec.id] = spec
    TABS.insert(len(TABS) - 1, spec)      # chat stays last
    TABS_BY_ID[spec.id] = spec
    _rederive_drivable()
    return spec


def _rederive_drivable():
    """DRIVABLE stays derived from APPS - and a bridge written here keeps its
    row: a hand-entered bridge with the same app name gets a tab but does not
    take the sidebar row over."""
    DRIVABLE.clear()
    for a in APPS:
        if not a.custom:
            DRIVABLE[a.name] = a.id
    for a in APPS:
        if a.custom:
            DRIVABLE.setdefault(a.name, a.id)


def remove_bridge(app_id):
    spec = APPS_BY_ID.get(app_id)
    if spec is None or not spec.custom:
        return None
    APPS.remove(spec)
    del APPS_BY_ID[app_id]
    TABS.remove(spec)
    del TABS_BY_ID[app_id]
    _rederive_drivable()
    return spec


def bridge_from_record(rec):
    """A BridgeSpec from a settings record, or None for one that cannot be."""
    if not isinstance(rec, dict):
        return None
    name, command = rec.get("name"), rec.get("command")
    if not (isinstance(name, str) and name.strip() and isinstance(command, str) and command.strip()):
        return None
    args = rec.get("args") or []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        args = []
    want = rec.get("id") if isinstance(rec.get("id"), str) else slug(name)
    if want in APPS_BY_ID and not APPS_BY_ID[want].custom:
        want += "-bridge"                 # never shadow a bridge written here
    return BridgeSpec(name.strip(), command.strip(), args, exe=str(rec.get("exe") or ""),
                      probe=str(rec.get("probe") or ""), note=str(rec.get("note") or ""),
                      id=want)


def load_bridges(records):
    """Register every valid record; return the specs. Bad records are skipped,
    never fatal - this is read from a file the user may have edited."""
    out = []
    for rec in records or []:
        spec = bridge_from_record(rec)
        if spec is not None:
            out.append(add_bridge(spec))
    return out


def custom_bridges():
    return [a for a in APPS if a.custom]


def get_app(app_id):
    try:
        return TABS_BY_ID[app_id]
    except KeyError:
        raise KeyError("unknown app %r; pick from %s"
                       % (app_id, ", ".join(TABS_BY_ID)))


def installed_apps():
    """Registry apps whose executable is actually on this machine."""
    return [a for a in APPS if a.installed()]


# ------------------------------------------------------- what is on this machine

ADOBE_DIR = r"C:\Program Files\Adobe"

# Each row carries where the product's own executable sits under its install
# folder: the UI reads the app's real icon straight out of that PE file
# (core/icons.py), and falls back to the two-letter badge when it cannot.
# The globs are loose because a Beta install renames the exe after itself.
PRODUCTS = [
    ("After Effects", "Ae", "After Effects", "#9999FF", "#00005B",
     r"Support Files\AfterFX.exe"),
    ("Premiere Pro", "Pr", "Premiere Pro", "#EA77FF", "#2A0634",
     r"Adobe Premiere Pro*.exe"),
    ("Photoshop", "Ps", "Photoshop", "#31A8FF", "#001E36",
     r"Photoshop.exe"),
    ("Illustrator", "Ai", "Illustrator", "#FF9A00", "#330000",
     r"Support Files\Contents\Windows\Illustrator.exe"),
    ("Audition", "Au", "Audition", "#00E4BB", "#00312E",
     r"Adobe Audition*.exe"),
    ("Media Encoder", "Me", "Media Encoder", "#9999FF", "#1D1D2E",
     r"Adobe Media Encoder*.exe"),
    ("Acrobat", "Ac", "Acrobat", "#FF5252", "#3B0000",
     r"Acrobat\Acrobat.exe"),
]

OTHER_APPS = [
    (r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Resolve.exe",
     "Dv", "DaVinci Resolve", "#F5A623", "#2B2B2B"),
]

# Derived, never hand-maintained: an app is drivable exactly when the registry
# has a bridge for it, so the sidebar cannot claim more than the agent can do.
DRIVABLE = {a.name: a.id for a in APPS}


def detect_apps():
    """
    Installed creative apps, newest label first, then the remote ones. Pure
    filesystem, no Windows registry; `remote` says which group a row belongs
    to in the sidebar - this PC, or the LLM PC.
    """
    try:
        entries = os.listdir(ADOBE_DIR)
    except OSError:
        entries = []
    found = []
    for match, code, name, fg, bg, exe_glob in PRODUCTS:
        hits = [e for e in entries if match.lower() in e.lower()]
        if not hits:
            continue
        years, beta = set(), False
        for e in hits:
            if "beta" in e.lower():
                beta = True
                continue
            m = re.search(r"(20\d\d)", e)
            if m:
                years.add(m.group(1))
        label = ", ".join(sorted(years, reverse=True))
        if beta:
            label = (label + ", Beta") if label else "Beta"
        exe = newest_match([os.path.join(ADOBE_DIR, e, exe_glob) for e in hits])
        found.append({"code": code, "name": name, "version": label, "fg": fg,
                      "bg": bg, "id": DRIVABLE.get(name), "exe": exe,
                      "drivable": name in DRIVABLE, "remote": False})
    for path, code, name, fg, bg in OTHER_APPS:
        if os.path.exists(path):
            found.append({"code": code, "name": name, "version": "", "fg": fg,
                          "bg": bg, "id": DRIVABLE.get(name), "exe": path,
                          "drivable": name in DRIVABLE, "remote": False})
    # A served app is on this machine but has no window .exe: the registry is
    # the only evidence, and the row says "server" where a year would go. No
    # exe means no icon to read - the badge stays.
    for a in APPS:
        if a.served:
            found.append({"code": a.code, "name": a.name, "version": "server",
                          "fg": a.fg, "bg": a.bg, "id": a.id, "exe": None,
                          "drivable": True, "remote": False})
    # A bridge the user entered by hand for something not detected above -
    # Blender, a DAW, a bridge with no app behind it. One whose name matches a
    # detected row (Premiere Pro, say) has already made that row drivable.
    named = {f["name"] for f in found}
    for a in APPS:
        if a.custom and a.name not in named:
            found.append({"code": a.code, "name": a.name, "version": "bridge",
                          "fg": a.fg, "bg": a.bg, "id": a.id, "exe": a.exe(),
                          "drivable": True, "remote": False})
    # Remote apps last: nothing on this disk to find, so the registry is the
    # only evidence they exist.
    for a in APPS:
        if a.remote:
            found.append({"code": a.code, "name": a.name, "version": "", "fg": a.fg,
                          "bg": a.bg, "id": a.id, "exe": None, "drivable": True,
                          "remote": True})
    return found

