#!/usr/bin/env python3
"""Keep this folder in step with GitHub.

    python studio_update.py              # one pass: fetch, and fast-forward if behind
    python studio_update.py --install    # check every 5 minutes (a Windows scheduled task)
    python studio_update.py --install --every 15
    python studio_update.py --uninstall  # stop checking

GitHub cannot call into a workstation behind a router, so "pull whenever
GitHub changes" is a poll: a scheduled task runs this under `pyw` (no console
window flashing up every few minutes), it fetches, and it fast-forwards the
checked-out branch when the remote is ahead.

It only ever fast-forwards. Local commits the remote does not have, or an
edit to a file the update would overwrite, make git refuse - and this leaves
it refused and says so in the log rather than merging, stashing or
resetting anyone's work. A pass with nothing to do writes nothing, so the
log is a list of updates and of reasons one did not happen.

A window already open keeps running the code it started with; the update
takes effect the next time the app is opened.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "studio_update.log")
LOG_MAX_BYTES = 256 * 1024
TASK = "Studio Assist auto-update"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# A scheduled task has no terminal to type a password into. Without these, a
# credential prompt would hang the pass forever, invisibly, under pyw.
GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}


def log(msg):
    try:
        if os.path.exists(LOG) and os.path.getsize(LOG) > LOG_MAX_BYTES:
            os.replace(LOG, LOG + ".1")
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except OSError:
        pass
    if sys.stdout is not None:          # None under pyw
        print(msg)


def find_git():
    found = shutil.which("git")
    if found:
        return found
    # A task started at logon can have a thinner PATH than a shell.
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base:
            p = os.path.join(base, "Git", "cmd", "git.exe")
            if os.path.isfile(p):
                return p
    return None


def git(gitexe, *args, timeout=120):
    r = subprocess.run([gitexe, "-C", HERE, *args], capture_output=True, text=True,
                       timeout=timeout, creationflags=NO_WINDOW,
                       env={**os.environ, **GIT_ENV})
    return r.returncode, (r.stdout + r.stderr).strip()


def check(fetch=True):
    """What GitHub has that this folder does not, without changing anything.
    A dict: `behind` and `ahead` (commit counts), `upstream`, `commits` (the
    new commits' one-line subjects, newest first) and `problem` (why it could
    not tell, else ""). The app's Update button asks this."""
    st = {"behind": 0, "ahead": 0, "upstream": "", "commits": [], "problem": ""}
    gitexe = find_git()
    if not gitexe:
        st["problem"] = "git was not found - install Git for Windows to get updates."
        return st
    st["git"] = gitexe
    code, _ = git(gitexe, "rev-parse", "--is-inside-work-tree")
    if code:
        st["problem"] = f"{HERE} is not a git checkout - nothing to update."
        return st
    code, upstream = git(gitexe, "rev-parse", "--abbrev-ref", "@{upstream}")
    if code:
        st["problem"] = "The checked-out branch tracks no remote branch - nothing to pull from."
        return st
    st["upstream"] = upstream
    if fetch:
        remote = upstream.split("/", 1)[0]
        code, out = git(gitexe, "fetch", "--quiet", remote)
        if code:
            st["problem"] = f"Fetching from {remote} failed: {out}"
            return st
    code, counts = git(gitexe, "rev-list", "--left-right", "--count", "HEAD...@{upstream}")
    if code:
        st["problem"] = f"Could not compare with {upstream}: {counts}"
        return st
    st["ahead"], st["behind"] = (int(n) for n in counts.split())
    if st["behind"]:
        _, out = git(gitexe, "log", "--format=%s", "HEAD..@{upstream}")
        st["commits"] = [line for line in out.splitlines() if line.strip()]
    return st


def pull():
    """One pass: fetch, and fast-forward if behind. Returns (changed, what
    happened in a sentence). Everything but "already up to date" is logged."""
    st = check()
    if st["problem"]:
        log(st["problem"])
        return False, st["problem"]
    upstream, ahead, behind = st["upstream"], st["ahead"], st["behind"]
    if not behind:
        return False, f"Already up to date with {upstream}."
    gitexe = st["git"]
    code, before = git(gitexe, "rev-parse", "--short", "HEAD")
    code, out = git(gitexe, "merge", "--ff-only", "@{upstream}")
    if code:
        why = (f"this PC has {ahead} commit(s) {upstream} does not" if ahead
               else "local edits would be overwritten")
        msg = f"{upstream} is {behind} commit(s) ahead, but not updating: {why}."
        log(f"{msg}\n    {out}")
        return False, msg
    _, after = git(gitexe, "rev-parse", "--short", "HEAD")
    msg = (f"Updated {before} -> {after} from {upstream} ({behind} commit(s)). "
           "Reopen Studio Assist to use it.")
    log(msg)
    return True, msg


def update():
    """One pass. Returns True if the folder changed."""
    return pull()[0]


# -------------------------------------------------------------- scheduling
def launcher():
    """What the task runs this under: `pyw` if there is one (System32, survives
    a Python upgrade - see AGENTS.md on launchers), else the pythonw beside
    the Python running this."""
    for name in ("pyw", "pythonw"):
        found = shutil.which(name)
        if found:
            return found
    beside = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    return beside if os.path.isfile(beside) else sys.executable


def install(every):
    if os.name != "nt":
        sys.exit("--install registers a Windows scheduled task. Elsewhere, use cron:\n"
                 f"  */{every} * * * * {sys.executable} {os.path.abspath(__file__)}")
    command = f'"{launcher()}" "{os.path.abspath(__file__)}"'
    r = subprocess.run(["schtasks", "/create", "/f", "/tn", TASK, "/sc", "minute",
                        "/mo", str(every), "/tr", command],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"Could not create the scheduled task:\n{r.stdout}{r.stderr}")
    print(f"Checking GitHub every {every} minute(s) as the task '{TASK}'.\n"
          f"Updates are logged to {LOG}.")
    update()


def uninstall():
    r = subprocess.run(["schtasks", "/delete", "/f", "/tn", TASK],
                       capture_output=True, text=True)
    print("Stopped checking for updates." if r.returncode == 0
          else f"No task to remove:\n{r.stdout}{r.stderr}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--install", action="store_true", help="check on a schedule")
    p.add_argument("--uninstall", action="store_true", help="stop checking")
    p.add_argument("--every", type=int, default=5, metavar="MIN",
                   help="minutes between checks with --install (default 5)")
    a = p.parse_args(argv)
    if a.install:
        install(max(1, a.every))
    elif a.uninstall:
        uninstall()
    else:
        try:
            update()
        except Exception as e:          # under pyw nobody sees a traceback
            log(f"Update check failed: {e!r}")


if __name__ == "__main__":
    main()
