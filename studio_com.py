#!/usr/bin/env python3
"""
studio_com - the road into an Adobe app that registers COM automation.

Photoshop and Illustrator on Windows both register an out-of-process COM
server (`Photoshop.Application`, `Illustrator.Application`) whose one method
worth having is `DoJavaScript`: it runs ExtendScript inside the live app and
returns the last expression as a string. That is the whole bridge - no CEP
panel, no UXP plugin, nothing to install in the app.

Python's stdlib has no COM client and this project takes no dependencies, so
the COM call is made by a PowerShell worker this module keeps alive: one
`powershell.exe -Sta` per app, holding the COM object, reading one request per
line on stdin (the script file to run, the file to write the answer to) and
answering `ok` or `err <why>`. A dead worker - the app quit, a dialog held it
past the timeout and it was killed - is started again on the next call.

Everything a bridge sends is wrapped by `script()`: ruler units pinned to
pixels, dialogs suppressed, a JSON serializer ExtendScript lacks, and every
exception folded into `{"__error": ...}` so a bad call is a sentence rather
than a debugger window inside the user's Photoshop.
"""

import base64
import json
import os
import subprocess
import tempfile
import threading
import time

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DEFAULT_TIMEOUT = 120                 # seconds one script may hold the app
LAUNCH_GRACE = 90                     # a cold Photoshop attach can take this long


class ComError(Exception):
    """The app, or the road to it, said no. Reaches the model as an isError result."""


# One request per line: "<script path>\t<answer path>". The worker attaches
# to the COM server lazily and drops the handle on any failure, so an app that
# was closed and reopened is picked up again without restarting anything.
WORKER = r"""
[Console]::InputEncoding = [Text.Encoding]::UTF8
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$progid = '%(progid)s'
$app = $null
[Console]::Out.WriteLine('ready')
while ($true) {
  $line = [Console]::In.ReadLine()
  if ($null -eq $line) { break }
  $parts = $line.Split("`t")
  try {
    if ($null -eq $app) { $app = New-Object -ComObject $progid }
    $code = [IO.File]::ReadAllText($parts[0], [Text.Encoding]::UTF8)
    $res = $app.DoJavaScript($code, $null, 1)
    if ($null -eq $res) { $res = '' }
    [IO.File]::WriteAllText($parts[1], [string]$res, (New-Object Text.UTF8Encoding $false))
    [Console]::Out.WriteLine('ok')
  } catch {
    $app = $null
    $msg = $_.Exception.Message
    if ($_.Exception.InnerException) { $msg = $msg + ' ' + $_.Exception.InnerException.Message }
    [Console]::Out.WriteLine('err ' + ($msg -replace "[\r\n]+", ' '))
  }
}
"""

# ExtendScript is ES3: no JSON, no Array.isArray, no trim. This prelude is
# what every script the bridges send runs inside. `__J` is the serializer;
# `__px` reads a UnitValue in pixels; the wrapper pins ruler units and
# dialogs for the duration and puts them back.
PRELUDE = r"""
function __J(v) {
  var t = typeof v;
  if (v === null || v === undefined || t === "function") return "null";
  if (t === "number") return isFinite(v) ? String(v) : "null";
  if (t === "boolean") return v ? "true" : "false";
  if (t === "string") {
    var s = v.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\r/g, "\\r")
             .replace(/\n/g, "\\n").replace(/\t/g, "\\t");
    s = s.replace(/[\x00-\x1f]/g, function (c) {
      var h = c.charCodeAt(0).toString(16); return "\\u" + ("0000" + h).slice(-4); });
    return '"' + s + '"';
  }
  if (v instanceof Array) {
    var a = []; for (var i = 0; i < v.length; i++) a.push(__J(v[i]));
    return "[" + a.join(",") + "]";
  }
  if (t === "object") {
    if (v instanceof UnitValue) return __J(Math.round(v.as("px") * 100) / 100);
    var o = [];
    for (var k in v) {
      if (!v.hasOwnProperty(k)) continue;
      var x = v[k]; if (typeof x === "function") continue;
      o.push(__J(k) + ":" + __J(x));
    }
    return "{" + o.join(",") + "}";
  }
  return __J(String(v));
}
function __px(u) { return (u instanceof UnitValue) ? Math.round(u.as("px") * 100) / 100 : u; }
function __round(n) { return Math.round(n * 100) / 100; }
function __fail(msg) { throw new Error(msg); }
"""

WRAPPER = r"""
(function () {
  var __out;
  %(setup)s
  try {
    var __r = (function () {
%(body)s
    })();
    __out = __J(__r === undefined ? null : __r);
  } catch (__e) {
    __out = __J({__error: String(__e.message || __e), line: __e.line || null});
  }
  %(teardown)s
  return __out;
})();
"""


def script(body, setup="", teardown=""):
    """The full ExtendScript for one call: prelude, guards, the tool's body."""
    return PRELUDE + WRAPPER % {"body": body, "setup": setup, "teardown": teardown}


def process_running(image_name, timeout=8):
    """tasklist says whether an .exe is up - without touching COM, which would start it.

    CSV output, because the table view cuts image names at 25 characters:
    "Adobe Premiere Pro (Beta).exe" comes back without its ".exe" and no
    longer matches.
    """
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq %s" % image_name, "/NH", "/FO", "CSV"],
                             capture_output=True, text=True, timeout=timeout,
                             creationflags=NO_WINDOW).stdout
    except Exception:
        return False
    return image_name.lower() in out.lower()


class ComHost:
    """One app's COM server, reached through a PowerShell worker kept alive."""

    def __init__(self, progid, name, process_name):
        self.progid = progid
        self.name = name                    # "Photoshop", for messages
        self.process_name = process_name    # "Photoshop.exe", for the running check
        self.proc = None
        self.lock = threading.Lock()
        self.dir = tempfile.mkdtemp(prefix="studio_com_")
        self.js_path = os.path.join(self.dir, "call.jsx")
        self.out_path = os.path.join(self.dir, "answer.txt")

    def running(self):
        return process_running(self.process_name)

    # ------------------------------------------------------------ worker

    def _start(self):
        code = (WORKER % {"progid": self.progid}).encode("utf-16-le")
        self.proc = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Sta",
             "-ExecutionPolicy", "Bypass",
             "-EncodedCommand", base64.b64encode(code).decode("ascii")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, creationflags=NO_WINDOW)
        line = self._readline(30)
        if line != "ready":
            self._kill()
            raise ComError("the PowerShell worker for %s did not start (%r)"
                           % (self.name, line))

    def _kill(self):
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self._release()

    def _release(self):
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except Exception:
                pass
        self.proc = None

    def _readline(self, timeout):
        """One line from the worker, or None when it stays silent past `timeout`."""
        box = []

        def read():
            try:
                box.append(self.proc.stdout.readline())
            except Exception:
                box.append("")
        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return None
        return (box[0] or "").rstrip("\r\n") if box else ""

    def close(self):
        with self.lock:
            if self.proc is not None:
                try:
                    self.proc.stdin.close()
                    self.proc.wait(timeout=3)
                    self._release()
                except Exception:
                    self._kill()

    # ------------------------------------------------------------- calls

    def run(self, body, timeout=DEFAULT_TIMEOUT, setup="", teardown=""):
        """Run `body` inside the app and return its decoded result.

        The body is the inside of a function: `return` what the tool wants,
        as anything `__J` can serialize. A thrown error, in the script or on
        the COM side, comes back as ComError. A first call while the app is
        closed starts it - COM does that on attach - so the timeout allows
        for a cold launch.
        """
        js = script(body, setup, teardown)
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self.proc = None
                self._start()
            with open(self.js_path, "w", encoding="utf-8") as f:
                f.write(js)
            try:
                os.remove(self.out_path)
            except OSError:
                pass
            budget = timeout + (0 if self.running() else LAUNCH_GRACE)
            try:
                self.proc.stdin.write("%s\t%s\n" % (self.js_path, self.out_path))
                self.proc.stdin.flush()
            except Exception as e:
                self._kill()
                raise ComError("lost the worker for %s: %s" % (self.name, e))
            line = self._readline(budget)
            if line is None:
                self._kill()
                raise ComError("%s did not answer within %d s. A dialog may be open in "
                               "%s - dismiss it and try again." % (self.name, budget, self.name))
            if line == "":
                self._kill()
                raise ComError("the worker for %s exited during the call" % self.name)
            if line.startswith("err"):
                raise ComError(self._explain(line[3:].strip()))
            try:
                with open(self.out_path, encoding="utf-8") as f:
                    raw = f.read()
            except OSError as e:
                raise ComError("no answer file from %s: %s" % (self.name, e))
        return self._decode(raw)

    def _explain(self, msg):
        low = msg.lower()
        if "80080005" in low or "server execution failed" in low:
            return ("%s could not be started through COM (server execution failed). "
                    "Start %s yourself, wait for it to finish loading, then try again."
                    % (self.name, self.name))
        if "80010001" in low or "rejected" in low or "busy" in low or "8001010a" in low:
            return ("%s is busy - a dialog is open or a long operation is running. "
                    "Finish it in %s and try again." % (self.name, self.name))
        if "invalid class string" in low or "80040154" in low:
            return ("%s's COM automation is not registered on this machine (%s). Repair "
                    "or reinstall %s, or point %s_PROGID at the right ProgID."
                    % (self.name, self.progid, self.name, self.name.upper()))
        return "%s refused the call: %s" % (self.name, msg)

    def _decode(self, raw):
        raw = raw.strip()
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except ValueError:
            raise ComError("%s returned something that is not JSON: %s"
                           % (self.name, raw[:300]))
        if isinstance(value, dict) and "__error" in value:
            where = " (line %s)" % value["line"] if value.get("line") else ""
            raise ComError("%s%s" % (value["__error"], where))
        return value


# ------------------------------------------------------- shared tool shapes

def obj(props, required=()):
    o = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        o["required"] = list(required)
    return o


def s(desc, **kw):
    d = {"type": "string", "description": desc}
    d.update(kw)
    return d


def i(desc, **kw):
    d = {"type": "integer", "description": desc}
    d.update(kw)
    return d


def n(desc, **kw):
    d = {"type": "number", "description": desc}
    d.update(kw)
    return d


def b(desc):
    return {"type": "boolean", "description": desc}


HEX = r"^#?[0-9a-fA-F]{6}$"


def rgb(hex_color):
    """'#RRGGBB' -> (r, g, b) ints, or ComError."""
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        raise ComError("colour must be #RRGGBB, got %r" % hex_color)
    try:
        return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))
    except ValueError:
        raise ComError("colour must be #RRGGBB, got %r" % hex_color)


def js_str(value):
    """A Python string as an ExtendScript literal."""
    return json.dumps(str(value))


def js_path(path):
    """A Windows path as an ExtendScript File() argument (forward slashes)."""
    return json.dumps(os.path.abspath(path).replace("\\", "/"))


def wait_for_file(path, timeout=30):
    """Some exports return before the file is closed; wait for it to settle."""
    deadline = time.monotonic() + timeout
    last = -1
    while time.monotonic() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        if size > 0 and size == last:
            return True
        last = size
        time.sleep(0.2)
    return os.path.exists(path)
