"""Display facts about a turn's tools, independent of Tk and inference."""
import json
import re

MAX_ITEMS = 80
DETAIL_CHARS = 2000
STATES = {"running": "Running", "ok": "Finished", "error": "Failed",
          "skipped": "Not run", "unknown": "Outcome unknown"}

TOOL_TITLES = {
    "opencode_status": ("Check OpenCode", "code"),
    "opencode_wait": ("Follow task progress", "code"),
    "opencode_list_sessions": ("Browse saved sessions", "folder"),
    "opencode_new_session": ("Start a coding session", "code"),
    "opencode_ask": ("Work on your request", "code"),
    "opencode_get_session": ("Read the coding session", "search"),
    "opencode_changes": ("Review code changes", "code"),
    "opencode_list_files": ("Browse project files", "folder"),
    "opencode_read_file": ("Read a project file", "file"),
    "opencode_search_files": ("Search project files", "search"),
    "opencode_abort": ("Stop the coding task", "code"),
    "read_file": ("Read a file", "file"),
    "list_folder": ("Browse a folder", "folder"),
    "find_files": ("Find files", "search"),
    "search_web": ("Search the web", "search"),
    "fetch_page": ("Read a web page", "search"),
    "run_jsx": ("Run an app script", "code"),
    "get_comp": ("Inspect the composition", "search"),
    "create_comp": ("Create a composition", "tool"),
    "set_text": ("Update text", "file"),
    "ps_run_jsx": ("Run a Photoshop script", "code"),
    "ai_run_jsx": ("Run an Illustrator script", "code"),
    "ppro_run_jsx": ("Run a Premiere script", "code"),
    "bash": ("Run a command", "code"),
    "edit": ("Edit a file", "file"),
    "read": ("Read a file", "file"),
    "glob": ("Find project files", "search"),
    "grep": ("Search project files", "search"),
}


def tool_presentation(name, arguments=None):
    """Readable UI labels; the real name and arguments stay in the details."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            arguments = {}
    args = arguments if isinstance(arguments, dict) else {}
    title, icon = TOOL_TITLES.get(name, ("", "tool"))
    if not title:
        words = args.get("action") or name
        for prefix in ("opencode_", "comfy_", "ps_", "ai_", "ppro_", "studio_"):
            if str(words).startswith(prefix):
                words = str(words)[len(prefix):]
                break
        title = " ".join(str(words).replace("_", " ").split()).capitalize()
    source = next((label for prefix, label in (("opencode_", "OpenCode"), ("comfy_", "ComfyUI"),
                  ("ps_", "Photoshop"), ("ai_", "Illustrator"), ("ppro_", "Premiere"))
                   if name.startswith(prefix)), "")
    context = next((args[k] for k in ("path", "filePath", "filepath", "query", "url", "command", "prompt", "title")
                    if isinstance(args.get(k), str) and args[k].strip()), "")
    context = " ".join(context.split())[:240]
    return title, context, icon, source


def progress_presentation(text):
    """Split the bridge's reported facts into short badges; never infer a percent."""
    badges, fraction = [], None
    for piece in str(text).split(";"):
        piece = piece.strip()
        match = re.fullmatch(r"(\d+) file\(s\) changed", piece)
        steps = re.fullmatch(r"(\d+)/(\d+) to-dos \((\d+)%\)", piece)
        decisions = re.fullmatch(r"(\d+) step\(s\) decided", piece)
        activity = re.fullmatch(r"last activity (.+) ago", piece)
        stalled = re.match(r"nothing for (.+?) - may be stuck", piece)
        if match:
            n = int(match[1])
            badges.append(("%d file%s changed" % (n, "" if n == 1 else "s"), "muted"))
        elif steps:
            done, total = int(steps[1]), int(steps[2])
            badges.append(("%d of %d steps" % (done, total), "muted"))
            if total > 0 and 0 <= done <= total:
                fraction = done / total
        elif decisions:
            n = int(decisions[1])
            badges.append(("%d decision%s" % (n, "" if n == 1 else "s"), "muted"))
        elif activity:
            badges.append(("Active now" if activity[1] == "0s" else "Last activity %s ago" % activity[1], "ok"))
        elif stalled:
            badges.append(("No activity for " + stalled[1], "warn"))
        elif piece and not piece.startswith("OpenCode is "):
            badges.append((piece, "muted"))
    return badges, fraction


def activity_heading(model):
    if model.phase.startswith("Preparing "):
        raw = model.phase[len("Preparing "):].rstrip("…")
        return "Preparing: " + tool_presentation(raw)[0]
    if model.phase.startswith("Running "):
        item = next((i for i in reversed(model.items) if i["status"] == "running"), None)
        if item:
            return tool_presentation(item["name"], item["arguments"])[0] + "…"
    return model.phase


def detail(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text[:DETAIL_CHARS] + ("\n… [clipped in the activity view]" if len(text) > DETAIL_CHARS else "")


def bridge_event(value):
    """Validate a bridge's display-only extension; it never authorizes a call."""
    if (not isinstance(value, dict) or not isinstance(value.get("id"), str)
            or not value["id"] or len(value["id"]) > 240 or not isinstance(value.get("name"), str)
            or not value["name"] or value.get("status") not in STATES):
        return None
    out = {"id": value["id"], "name": value["name"][:160], "status": value["status"],
           "text": detail(value.get("text", ""))}
    if "arguments" in value:
        out["arguments"] = detail(value["arguments"])
    return out


def progress_event(params):
    meta = params.get("_meta") if isinstance(params, dict) else None
    return bridge_event(meta.get("studio/toolActivity")) if isinstance(meta, dict) else None


class Activity:
    def __init__(self):
        self.begin()
        self.active = False

    def begin(self):
        self.items = []
        self.count = 0
        self.failures = 0
        self.omitted = 0
        self.phase = "Thinking…"
        self.active = True
        self.interrupted = False

    def _add(self, ident, name, arguments, via=""):
        if len(self.items) >= MAX_ITEMS:
            old = next((i for i in self.items if i["status"] != "running"), None)
            if old is None:
                self.omitted += 1
                self.count += 1
                return None
            self.items.remove(old)
            self.omitted += 1
        self.count += 1
        item = {"id": ident, "name": str(name)[:160], "arguments": detail(arguments),
                "status": "running", "text": "", "progress": "", "via": str(via or "")}
        self.items.append(item)
        return item

    def _working(self):
        running = next((i for i in reversed(self.items) if i["status"] == "running"), None)
        self.phase = ("Running %s…" % running["name"] if running else "Thinking…")
        if running and running["progress"]:
            self.phase += " " + running["progress"]

    def handle(self, kind, payload=None):
        """True when visible facts changed. Preparations never count as executed tools."""
        before = (self.phase, self.active)
        changed = False
        if kind == "begin":
            self.begin()
            return True
        if kind == "tool_preparing":
            self.phase = "Preparing %s…" % str(payload.get("name") or "tool")[:160]
            self.active = True
        elif kind == "tool":
            self._add("local:%d" % self.count, payload["name"], payload.get("arguments", {}), payload.get("via"))
            self.active = True
            self._working()
            changed = True
        elif kind == "tool_result":
            item = next((i for i in reversed(self.items) if i["name"] == payload["name"]
                         and i["status"] == "running" and i["id"].startswith("local:")), None)
            if item:
                item["status"] = payload.get("status", "ok")
                if item["status"] == "error":
                    self.failures += 1
                item["text"] = detail(payload.get("text", ""))
                item["progress"] = ""
                changed = True
            self._working()
        elif kind == "bridge_activity":
            ident = "bridge:" + str(payload["id"])
            item = next((i for i in self.items if i["id"] == ident), None)
            if item is None:
                item = self._add(ident, payload["name"], payload.get("arguments", {}), "OpenCode")
            if item:
                if item["status"] != payload["status"]:
                    self.failures += int(payload["status"] == "error") - int(item["status"] == "error")
                item.update(status=payload["status"], text=detail(payload.get("text", "")))
                if "arguments" in payload:
                    item["arguments"] = detail(payload["arguments"])
                changed = True
            self.active = True
            self._working()
        elif kind == "progress":
            item = next((i for i in reversed(self.items) if i["status"] == "running"), None)
            if item:
                text = str(payload)[:200]
                changed = item["progress"] != text
                item["progress"] = text
                self._working()
            elif self.active:
                self.phase = str(payload)[:200]
        elif kind in ("stream_start", "token"):
            self.active = True
            self.phase = "Writing reply…"
        elif kind == "stream_end" and self.active:
            self._working()
        elif kind in ("ask", "elicit"):
            self.phase = "Waiting for your answer"
        elif kind == "elicit_done":
            self._working()
        elif kind == "status" and self.active:
            self.phase = str(payload[0])
        elif kind in ("error", "idle"):
            unknown = False
            for item in self.items:
                if item["status"] == "running":
                    item["status"] = "unknown"
                    item["text"] = "No result received; inspect before repeating this operation."
                    unknown = True
            self.interrupted = self.interrupted or kind == "error" or bool(payload if kind == "idle" else False) or unknown
            self.phase = "Interrupted" if self.interrupted else "Reply finished"
            if self.failures and not self.interrupted:
                self.phase += " · %d failed tool call%s" % (self.failures, "" if self.failures == 1 else "s")
            self.active = False
            changed = True
        return changed or before != (self.phase, self.active)

    def as_text(self):
        lines = ["%d earlier tool call(s) omitted from this view." % self.omitted] if self.omitted else []
        for item in self.items:
            lines.append("%s — %s%s" % (STATES.get(item["status"], "Outcome unknown"), item["name"],
                         " (via %s)" % item["via"] if item["via"] else ""))
            lines.append("Arguments: " + item["arguments"])
            if item["progress"]:
                lines.append("Progress: " + item["progress"])
            if item["text"]:
                lines.append("Result: " + item["text"])
            lines.append("")
        return "\n".join(lines) or "No tools called in this turn."
