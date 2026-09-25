#!/usr/bin/env python3
"""Keep this folder in step with GitHub.

    python studio_update.py              # one pass: fetch, and fast-forward if behind
    python studio_update.py --install    # check every 5 minutes (a Windows scheduled task)
    python studio_update.py --install --every 15
    python studio_update.py --uninstall  # stop checking

GitHub cannot call into a workstation behind a router, so "pull whenever
GitHub changes" is a poll: a scheduled task runs this under `pyw` (no console
window flashing up every few minutes), it fetches, and it fast-forwards
`main` (BRANCH) when the remote is ahead.

It follows `main`, not whatever happens to be checked out. Work lands on a
feature branch, is merged, and that branch never moves again - a folder
left on one stopped updating for good. So a folder on another branch is
moved onto `main`, but only when nothing is lost: no edits to tracked files,
and every commit on it already in `main`. Otherwise it stays put and the
log says why. The old branch is left where it is.

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
# The branch the app is released on. Merged feature branches never move again.
BRANCH = os.environ.get("STUDIO_UPDATE_BRANCH", "main")
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


def follow(gitexe, target):
    """Put the folder on BRANCH, tracking `target`, if that loses nothing.
    Returns (ok, changed)."""
    code, here = git(gitexe, "symbolic-ref", "--quiet", "--short", "HEAD")
    here = here if code == 0 else "a detached HEAD"
    if here == BRANCH:
        code, out = git(gitexe, "branch", "--quiet", "--set-upstream-to", target)
        if code:
            log(f"Could not make {BRANCH} track {target}: {out}")
        return code == 0, False
    code, dirty = git(gitexe, "status", "--porcelain", "--untracked-files=no")
    if code or dirty:
        log(f"This folder is on {here}, not {BRANCH}, and has local edits - staying "
            f"there, so updates from {target} are not reaching it.")
        return False, False
    code, _ = git(gitexe, "merge-base", "--is-ancestor", "HEAD", target)
    if code:
        _, n = git(gitexe, "rev-list", "--count", f"{target}..HEAD")
        log(f"This folder is on {here}, which has {n} commit(s) {target} does not - "
            f"staying there, so updates from {target} are not reaching it.")
        return False, False
    code, _ = git(gitexe, "rev-parse", "--verify", "--quiet", "refs/heads/" + BRANCH)
    if code == 0:
        code, _ = git(gitexe, "merge-base", "--is-ancestor", BRANCH, target)
        if code:
            log(f"The local {BRANCH} has commits {target} does not - staying on {here}.")
            return False, False
        code, out = git(gitexe, "checkout", "--quiet", BRANCH)
        if code == 0:
            code, out = git(gitexe, "branch", "--quiet", "--set-upstream-to", target)
    else:
        code, out = git(gitexe, "checkout", "--quiet", "--track", "-b", BRANCH, target)
    if code:
        log(f"Could not move from {here} to {BRANCH}: {out}")
        return False, False
    log(f"Moved from {here} to {BRANCH}, which is what updates now. Everything on "
        f"{here} is already in {BRANCH}; the branch is still there.")
    return True, True


def update():
    """One pass. Returns True if the folder changed."""
    gitexe = find_git()
    if not gitexe:
        log("git was not found - install Git for Windows to get updates.")
        return False
    code, _ = git(gitexe, "rev-parse", "--is-inside-work-tree")
    if code:
        log(f"{HERE} is not a git checkout - nothing to update.")
        return False
    code, upstream = git(gitexe, "rev-parse", "--abbrev-ref", "@{upstream}")
    remote = upstream.split("/", 1)[0] if code == 0 else "origin"
    code, out = git(gitexe, "fetch", "--quiet", remote)
    if code:
        log(f"Fetching from {remote} failed: {out}")
        return False
    target = f"{remote}/{BRANCH}"
    code, _ = git(gitexe, "rev-parse", "--verify", "--quiet", target)
    if code:
        log(f"{remote} has no {BRANCH} branch - nothing to pull from.")
        return False
    ok, moved = follow(gitexe, target)
    if not ok:
        return False
    code, counts = git(gitexe, "rev-list", "--left-right", "--count", f"HEAD...{target}")
    if code:
        log(f"Could not compare with {target}: {counts}")
        return moved
    ahead, behind = (int(n) for n in counts.split())
    if not behind:
        return moved
    code, before = git(gitexe, "rev-parse", "--short", "HEAD")
    code, out = git(gitexe, "merge", "--ff-only", target)
    if code:
        why = (f"this PC has {ahead} commit(s) {target} does not" if ahead
               else "local edits would be overwritten")
        log(f"{target} is {behind} commit(s) ahead, but not updating: {why}.\n    {out}")
        return moved
    _, after = git(gitexe, "rev-parse", "--short", "HEAD")
    log(f"Updated {before} -> {after} from {target} ({behind} commit(s)). "
        "Reopen Studio Assist to use it.")
    return True


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
