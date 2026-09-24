#!/usr/bin/env python3
"""
studio_cep - the road into an Adobe app that has no COM automation.

Premiere Pro registers nothing a `New-Object -ComObject` could hold, so the
way in is the one After Effects' bridge uses: a CEP panel inside the app that
runs a loopback HTTP server. Ours (`premiere_panel/`) is deliberately dumb -
`POST /run {"script": ...}` evaluates the ExtendScript and answers with the
string it returned - so every tool body, helper and the JSON serializer stay
in Python, where the tests can read them. `studio_com.script()` wraps each
body exactly as it does for Photoshop: prelude, error folding, setup and
teardown; only the transport differs.

Two consequences worth knowing before changing anything here:

- **A call cannot start the app.** COM starts a closed Photoshop on attach;
  a panel only exists while Premiere runs and the panel is open. So `run()`
  never waits for a launch, and an unreachable panel is explained by
  `explain_unreachable()`, which looks at what is actually missing - the
  process, the installed panel, CEP's debug flag - and says the one thing to
  do next.
- **The panel is installed by copying a folder.** `install_panel()` puts
  `premiere_panel/` under `%APPDATA%\\Adobe\\CEP\\extensions`; the panel is
  unsigned, so CEP loads it only with `PlayerDebugMode` set for its CSXS
  version, which `debug_mode_missing()` reports and never sets.
"""

import json
import os
import shutil
import socket
import urllib.error
import urllib.request

import studio_com as com

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TIMEOUT = com.DEFAULT_TIMEOUT
# CSXS versions a current Premiere loads panels through; the unsigned panel
# needs PlayerDebugMode under at least the one the running app uses.
CSXS_VERSIONS = ("11", "12")


class CepError(Exception):
    """The app, the panel, or the road to it said no. Reaches the model as an isError result."""


def extensions_dir():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "Adobe", "CEP", "extensions")


def debug_mode_missing(versions=CSXS_VERSIONS):
    """CSXS versions whose PlayerDebugMode is not 1 - the flag an unsigned panel needs."""
    try:
        import winreg
    except ImportError:
        return []
    missing = []
    for v in versions:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Adobe\CSXS." + v) as k:
                value, _ = winreg.QueryValueEx(k, "PlayerDebugMode")
                if str(value) != "1":
                    missing.append(v)
        except OSError:
            missing.append(v)
    return missing


class CepHost:
    """One app's bridge panel, reached over loopback HTTP."""

    def __init__(self, url, name, process_name, panel_id, panel_src):
        self.url = url.rstrip("/")
        self.name = name                    # "Premiere Pro", for messages
        self.process_name = process_name    # for the running check
        self.panel_id = panel_id            # the folder name under CEP/extensions
        self.panel_src = panel_src          # the folder in this checkout to install from

    # ------------------------------------------------------------ the panel

    def panel_dir(self):
        return os.path.join(extensions_dir(), self.panel_id)

    def panel_installed(self):
        return os.path.isfile(os.path.join(self.panel_dir(), "CSXS", "manifest.xml"))

    def install_panel(self):
        """Copy the panel into CEP's extensions folder, replacing an older copy."""
        src = self.panel_src
        if not os.path.isfile(os.path.join(src, "CSXS", "manifest.xml")):
            raise CepError("no panel to install at %s" % src)
        dest = self.panel_dir()
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return dest

    def running(self):
        return com.process_running(self.process_name)

    def ping(self, timeout=3):
        """The panel's own status line, or CepError."""
        try:
            with urllib.request.urlopen(self.url + "/", timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            raise CepError(self.explain_unreachable(e))

    def explain_unreachable(self, err=None):
        """Why nothing answered - the one thing to do next, not a list of guesses."""
        if not self.running():
            return ("%s is not running. Start it (the Start button in this window does), "
                    "wait for it to finish loading, then try again." % self.name)
        if not self.panel_installed():
            return ("%s is running but its bridge panel is not installed. Quit %s, run "
                    "`python studio_premiere_mcp.py --install-panel`, and open %s again."
                    % (self.name, self.name, self.name))
        missing = debug_mode_missing()
        if missing:
            return ("%s is running and the panel is installed, but CEP will not load an "
                    "unsigned panel until PlayerDebugMode is set: for each of CSXS.%s, run\n"
                    "    reg add HKCU\\Software\\Adobe\\CSXS.<n> /v PlayerDebugMode /t REG_SZ /d 1\n"
                    "then restart %s." % (self.name, ", CSXS.".join(missing), self.name))
        return ("%s is running but its bridge panel is not answering at %s (%s). Open it "
                "once from Window > Extensions > Studio Assist Bridge; it starts with "
                "%s after that." % (self.name, self.url, err or "no reply", self.name))

    # ------------------------------------------------------------- calls

    def run(self, body, timeout=DEFAULT_TIMEOUT, setup="", teardown=""):
        """Run `body` inside the app and return its decoded result.

        The body is the inside of a function: `return` what the tool wants,
        as anything `__J` can serialize. A thrown error in the script comes
        back as CepError, and so does a panel that cannot be reached or that
        stays silent past `timeout` - a modal dialog inside the app does that.
        """
        js = com.script(body, setup, teardown)
        req = urllib.request.Request(
            self.url + "/run", data=json.dumps({"script": js}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                answer = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise CepError("the %s panel answered HTTP %d: %s" % (self.name, e.code, detail))
        except (socket.timeout, TimeoutError):
            raise CepError("%s did not answer within %d s. A dialog may be open in %s - "
                           "dismiss it and try again." % (self.name, timeout, self.name))
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise CepError("%s did not answer within %d s. A dialog may be open in %s - "
                               "dismiss it and try again." % (self.name, timeout, self.name))
            raise CepError(self.explain_unreachable(e.reason))
        except OSError as e:
            raise CepError(self.explain_unreachable(e))
        if not isinstance(answer, dict) or "result" not in answer:
            raise CepError("the %s panel returned something unexpected: %r" % (self.name, answer))
        return self.decode(answer["result"])

    def decode(self, raw):
        raw = (raw or "").strip()
        if not raw:
            return None
        # CEP's one error message for a script that did not parse or threw
        # outside our wrapper. Our wrapper folds every runtime error, so this
        # is a syntax error in the body - which only ppro_run_jsx can produce.
        if raw == "EvalScript error.":
            raise CepError("%s could not run the script (EvalScript error): usually a syntax "
                           "error in the ExtendScript. Check the code and try again." % self.name)
        try:
            value = json.loads(raw)
        except ValueError:
            raise CepError("%s returned something that is not JSON: %s" % (self.name, raw[:300]))
        if isinstance(value, dict) and "__error" in value:
            where = " (line %s)" % value["line"] if value.get("line") else ""
            raise CepError("%s%s" % (value["__error"], where))
        return value
