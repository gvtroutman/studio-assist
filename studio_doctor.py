#!/usr/bin/env python3
"""Where the app keeps its things, and whether they are in working order.

Split out of `studio_chat` so `--doctor` runs when the window will not.
A Tkinter that cannot open a display, a Python missing tkinter altogether,
a settings file Tk choked on - those are exactly the moments someone needs
to be told what is wrong, and importing the GUI to ask would fail for the
same reason. **Nothing in this module may import tkinter.**

It is also the one place that knows the layout of `%APPDATA%\\StudioAssistant`,
so the GUI's Diagnostics window and the console's `--doctor` report the same
facts from the same code rather than two drifting copies.
"""

import json
import os
import time

import studio_agent as eng

ERROR_LOG = "studio_assistant_error.log"
LOG_MAX_BYTES = 512 * 1024        # then it rolls over to a single `.1`


# ------------------------------------------------------------------- where
def settings_path():
    return os.environ.get("STUDIO_SETTINGS") or os.path.join(
        os.environ.get("APPDATA") or os.path.expanduser("~"),
        "StudioAssistant", "settings.json")


def data_dir():
    """Everything the app keeps lives beside the settings file, so pointing
    STUDIO_SETTINGS somewhere else moves the whole lot - which is how the
    tests keep out of the user's own running window."""
    return os.path.dirname(os.path.abspath(settings_path()))


def error_log_path():
    return os.path.join(data_dir(), ERROR_LOG)


def tasks_dir():
    return os.path.join(data_dir(), "tasks")


def log_error(text):
    """The app's one forensic record, and the only one it has: the shortcut
    starts it with `pythonw.exe`, which has no stderr for a traceback to go
    to. Everything that writes here goes through this function so the
    rollover is in one place - a tool that runs for months should not grow a
    log without end, and one generation back is as far as anyone has needed
    to look. Best-effort: a locked or read-only profile costs the entry,
    never the app. -> the path written, or None."""
    try:
        path = error_log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if os.path.getsize(path) > LOG_MAX_BYTES:
                os.replace(path, path + ".1")
        except OSError:
            pass              # no log yet, or one we cannot roll; append anyway
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n---- %s ----\n%s" % (time.strftime("%Y-%m-%d %H:%M:%S"), text))
        return path
    except OSError:
        return None


# ------------------------------------------------------------------ report
# A row is (label, value, role). The role is a palette name the GUI can paint
# with - "ok", "warn", "err", "muted" - and the console turns into a mark, so
# both renderings rank the same facts the same way.
MARKS = {"ok": "+", "warn": "!", "err": "x", "muted": " "}


def _writable(path):
    """Whether a file we may not have written yet could be written. Asking
    the directory is the only honest test: `os.access` on Windows reports the
    read-only attribute, not the ACL that actually decides."""
    parent = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(parent, exist_ok=True)
        probe = os.path.join(parent, ".write-probe")
        with open(probe, "w") as f:
            f.write("")
        os.unlink(probe)
        return True
    except OSError:
        return False


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _kb(n):
    return "%.0f KB" % (n / 1024.0) if n is not None else "-"


def python_rows():
    import sys
    rows = [("Python", "%s.%s.%s" % sys.version_info[:3],
             "ok" if sys.version_info[:2] >= (3, 8) else "err"),
            ("Interpreter", sys.executable or "?", "muted")]
    try:
        import tkinter
        rows.append(("Tkinter", "Tcl/Tk %s" % tkinter.TkVersion, "ok"))
    except Exception as e:
        rows.append(("Tkinter", "missing - the window cannot open (%s)" % e, "err"))
    return rows


def storage_rows():
    settings, log, tasks = settings_path(), error_log_path(), tasks_dir()
    writable = _writable(settings)
    rows = [("Settings", settings, "ok" if writable else "err")]
    if not writable:
        rows.append(("", "not writable - preferences will not survive a restart", "err"))
    size = _size(log)
    rows.append(("Error log", "%s (%s)" % (log, _kb(size) if size is not None
                                           else "nothing logged yet"), "muted"))
    if os.path.exists(log + ".1"):
        rows.append(("", "one rolled-over generation kept beside it", "muted"))
    count = 0
    for root, _dirs, files in os.walk(tasks):
        count += sum(1 for f in files if f.endswith(".json"))
    rows.append(("Saved tasks", "%d in %s" % (count, tasks), "muted"))
    return rows


def host_rows(host, model=None, timeout=5):
    """The inference host, and the fact that started all of this: the window
    the model is actually loaded with. LM Studio loads at 8,192 by default and
    a tab's briefing plus tools is most or all of that, which is what makes a
    model loop on one tool or get cut off mid-answer."""
    rows = [("Host", host, "muted")]
    ok, loaded, ids, vision, err = eng.probe_models(host, timeout=timeout)
    if not ok or not ids:
        rows.append(("Reachable", "no - %s" % (err or "the host serves no models"),
                     "err" if not ok else "warn"))
        # Which of the two it is decides what to go and do, and only the
        # tailnet ping can tell them apart: a firewall drops a port with no
        # listener rather than refusing it, so both read as "timed out".
        alive = eng.host_alive(host)
        if alive is True:
            rows.append(("Diagnosis", "the PC answers but LM Studio's server does "
                                      "not - start the server on it", "warn"))
        elif alive is False:
            rows.append(("Diagnosis", "the PC itself does not answer - asleep, off, "
                                      "or off the tailnet", "err"))
        return rows
    rows.append(("Reachable", "yes, %d model%s served, %d in VRAM"
                 % (len(ids), "" if len(ids) == 1 else "s", len(loaded)), "ok"))
    chosen = model or eng.pick_model(
        loaded, ids, eng.env_default("STUDIO_MODEL", "AE_AGENT_MODEL"))
    rows.append(("Model", chosen or "none chosen", "ok" if chosen else "warn"))
    rows.append(("Vision model", ", ".join(vision) or "none served - tabs cannot "
                                                     "look at their own work",
                 "ok" if vision else "warn"))
    if chosen:
        loaded, maximum = eng.context_window(host, chosen)
        if isinstance(loaded, int):
            room = "ok" if loaded > 8192 else "warn"
            rows.append(("Context window", "%s loaded%s" % (
                f"{loaded:,}", " of %s maximum" % f"{maximum:,}"
                if isinstance(maximum, int) else ""), room))
            if loaded <= 8192:
                rows.append(("", "8,192 is LM Studio's just-in-time default and is "
                                 "under some tabs' briefing alone", "warn"))
        else:
            rows.append(("Context window", "the host does not say", "muted"))
    return rows


def tab_rows(sessions):
    """One row per open tab: what its fixed prefix costs against the window
    behind it. `sessions` is an iterable of objects carrying `app`,
    `prefix_tokens` and `window`; nothing here reaches the network."""
    rows = []
    for s in sessions or []:
        used, window = getattr(s, "prefix_tokens", None), getattr(s, "window", None)
        if not isinstance(used, int):
            rows.append((s.app.name, "not measured yet - the tab has not warmed up",
                         "muted"))
            continue
        if isinstance(window, int) and window > 0:
            left = window - used
            rows.append((s.app.name, "%s of %s used by the briefing and tools, "
                                     "%s left for the conversation"
                         % (f"{used:,}", f"{window:,}", f"{left:,}"),
                         "ok" if left >= eng.ROOM else "err"))
            if left < eng.ROOM:
                rows.append(("", "under %s of room: this tab will lose its own tool "
                                 "results to truncation" % f"{eng.ROOM:,}", "err"))
        else:
            rows.append((s.app.name, "%s used by the briefing and tools" % f"{used:,}",
                         "muted"))
    return rows or [("", "no tab has warmed up yet", "muted")]


def recent_errors(limit=5):
    """The last few entries of the log, newest first - enough to recognise a
    failure without opening the file."""
    path = error_log_path()
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            body = f.read()
    except OSError:
        return [("", "no errors logged", "ok")]
    entries = [e.strip() for e in body.split("\n---- ") if e.strip()]
    if not entries:
        return [("", "no errors logged", "ok")]
    rows = []
    for entry in reversed(entries[-limit:]):
        when, _, rest = entry.partition(" ----\n")
        last = (rest.strip().rsplit("\n", 1) or [""])[-1]
        rows.append((when.strip(), last.strip()[:160] or "(empty)", "err"))
    return rows


def report(host=None, model=None, sessions=None):
    """-> [(section title, [rows])]. The GUI paints it; `as_text` prints it."""
    host = host or eng.env_default("STUDIO_HOST", "AE_AGENT_HOST",
                                   fallback=eng.DEFAULT_HOST)
    sections = [("This PC", python_rows()),
                ("Where things are kept", storage_rows()),
                ("Inference host", host_rows(host, model))]
    if sessions is not None:
        sections.append(("Open tabs", tab_rows(sessions)))
    sections.append(("Recent errors", recent_errors()))
    return sections


def as_text(sections):
    out = []
    for title, rows in sections:
        out.append(title)
        out.append("-" * len(title))
        for label, value, role in rows:
            out.append("  %s %-16s %s" % (MARKS.get(role, " "), label, value))
        out.append("")
    return "\n".join(out)


def worst(sections):
    """The rank of the unhappiest row, for an exit code: 0 fine, 1 worth a
    look, 2 something is broken."""
    rank = {"err": 2, "warn": 1}
    return max([rank.get(r[2], 0) for _t, rows in sections for r in rows] or [0])


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    sections = report()
    print(as_text(sections))
    return worst(sections)


if __name__ == "__main__":
    raise SystemExit(main())
