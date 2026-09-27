"""Console windows opened outside the app, held in its Terminal tab.

A shell started by hand - ComfyUI's `Start ComfyUI (Image Studio).cmd`, a
`cmd` or `powershell` from the Start menu - is a classic console window on the
desktop. This module finds those windows, hides them, and reads what they show,
so the Terminal tab can mirror each one and type into it. The GUI half is
`studio_terminals_ui.py`.

It is a mirror, not the real window re-parented into the tab the way the
Milanote tab holds Chrome. A cross-process child window dies with its parent:
had the app crashed with ComfyUI's console inside it, the console would have
gone with it, and ComfyUI with the console. A hidden window survives anything
that happens to this app. The ledger (`HELD_FILE`) remembers what was hidden, so
a copy started after a crash finds those windows again, and a normal quit shows
every one of them on the desktop again.

- **Only visibility is touched.** Nothing here ends, signals or re-parents a
  process it did not start. Hiding is `ShowWindow(SW_HIDE)`, which also takes
  the window off the taskbar; giving it back is `SW_SHOW`.
- **Only classic console windows** (`ConsoleWindowClass`, drawn by conhost).
  A shell inside Windows Terminal is a tab of the user's own terminal app and is
  left alone: its console has no window of its own to hide.
- **Reading needs a process of its own.** `AttachConsole` joins one console at
  a time, and a process attached to a console is ended when that console
  closes. So the reading is done by this file run as `--reader`, a contained
  child of the app that attaches, reads the active screen buffer, and detaches
  again for every request. When a console closes while it is attached and it is
  ended, the app starts another.

Stdlib only (`ctypes`). Off Windows there are no console windows to find.
"""

import json
import os
import subprocess
import sys
import threading
import time

WINDOWS = sys.platform == "win32"
CLASS = "ConsoleWindowClass"
HELD_FILE = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                         "StudioAssistant", "held-consoles.json")
MAX_LINES = 3000              # of a console's screen buffer, counted up from its cursor
READ_TIMEOUT = 10             # seconds one request to the reader may take
INTERRUPT_SETTLE_S = 0.5      # attached after a Ctrl+C, until the console has sent it

if WINDOWS:
    import ctypes
    from ctypes import wintypes

    _u32 = ctypes.WinDLL("user32", use_last_error=True)
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    _u32.EnumWindows.argtypes = [_EnumProc, wintypes.LPARAM]
    _u32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _u32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _u32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _u32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _u32.IsWindow.argtypes = [wintypes.HWND]
    _u32.IsWindowVisible.argtypes = [wintypes.HWND]
    _u32.ShowWindowAsync.argtypes = [wintypes.HWND, ctypes.c_int]
    _u32.SetForegroundWindow.argtypes = [wintypes.HWND]

    SW_HIDE, SW_SHOW = 0, 5


# ------------------------------------------------------------------ windows
def _text(hwnd):
    n = _u32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    _u32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _pid(hwnd):
    pid = wintypes.DWORD()
    _u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def windows():
    """Every top-level console window: [{"hwnd", "pid", "title", "visible"}].

    `pid` is the console's first client - `cmd.exe` for a .cmd - not conhost:
    Windows answers GetWindowThreadProcessId for a console window that way."""
    if not WINDOWS:
        return []
    found = []

    def each(hwnd, _):
        name = ctypes.create_unicode_buffer(64)
        _u32.GetClassNameW(hwnd, name, 64)
        if name.value == CLASS:
            found.append({"hwnd": int(hwnd), "pid": _pid(hwnd), "title": _text(hwnd),
                          "visible": bool(_u32.IsWindowVisible(hwnd))})
        return True

    _u32.EnumWindows(_EnumProc(each), 0)
    return found


def exists(hwnd):
    return bool(WINDOWS and _u32.IsWindow(hwnd))


def title(hwnd):
    return _text(hwnd) if exists(hwnd) else ""


def hide(hwnd):
    """Async: a console that is not answering must not hang the caller."""
    if exists(hwnd):
        _u32.ShowWindowAsync(hwnd, SW_HIDE)


def show(hwnd, front=False):
    if exists(hwnd):
        _u32.ShowWindowAsync(hwnd, SW_SHOW)
        if front:
            _u32.SetForegroundWindow(hwnd)


# ------------------------------------------------------------------- ledger
class Ledger:
    """The windows this app has hidden, on disk, so none is lost to a crash."""

    def __init__(self, path=HELD_FILE):
        self.path = path
        self.lock = threading.Lock()

    def load(self):
        """{hwnd: pid} of the windows still hidden by a copy of this app."""
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return {int(h): int(p) for h, p in data.items()}
        except (OSError, ValueError, TypeError, AttributeError):
            return {}

    def save(self, held):
        with self.lock:
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({str(h): p for h, p in held.items()}, f)
                os.replace(tmp, self.path)
            except OSError:
                pass


class Holder:
    """Which console windows are hidden and held, kept in step with the
    ledger. No Tk: the app's watcher thread calls `sweep()` about once a
    second and the Terminal tab reads `held`.

    A window given back (`release`) is not taken again while it lives: the
    user asked for it on the desktop. Nor is anything whose client is this
    app or one of its children, and nor is a window that was hidden before
    the app looked - that is some other program's business (Logitech and Epic
    keep hidden consoles), unless the ledger says a copy of this app hid it."""

    def __init__(self, ledger=None, list_windows=None, hide_fn=None, show_fn=None,
                 exists_fn=None):
        self.ledger = ledger or Ledger()
        self._windows = list_windows or windows
        self._hide, self._show = hide_fn or hide, show_fn or show
        self._exists = exists_fn or exists
        self.lock = threading.Lock()
        self.held = {}                   # hwnd -> {"hwnd", "pid", "title"}, in arrival order
        self.released = set()
        self.stopped = False             # the app is quitting: take nothing more

    def own_pids(self):
        pids = {os.getpid()}
        try:
            import studio_procs as procs
            pids.update(c.pid for c in procs.live())
        except Exception:
            pass
        return pids

    def recover(self):
        """At startup: take back what a copy of this app hid and never showed
        again - it crashed, or was killed. -> the windows taken."""
        was = self.ledger.load()
        got = []
        with self.lock:
            for info in self._windows():
                if was.get(info["hwnd"]) == info["pid"] and info["hwnd"] not in self.held:
                    self.held[info["hwnd"]] = info
                    got.append(info)
            if was:
                self._save()                 # what has closed since is dropped
        return got

    def sweep(self):
        """Hide every console window newly on the desktop, and forget the held
        ones that have closed. -> (taken, gone): window infos, hwnds."""
        own = self.own_pids()
        taken, gone = [], []
        with self.lock:
            if self.stopped:
                return taken, gone
            seen = set()
            for info in self._windows():
                hwnd = info["hwnd"]
                seen.add(hwnd)
                if hwnd in self.held:
                    self.held[hwnd]["title"] = info["title"]
                    continue
                if not info["visible"] or hwnd in self.released or info["pid"] in own:
                    continue
                self._hide(hwnd)
                self.held[hwnd] = info
                taken.append(info)
            for hwnd in list(self.held):
                if hwnd not in seen and not self._exists(hwnd):
                    del self.held[hwnd]
                    gone.append(hwnd)
            self.released &= seen
            if taken or gone:
                self._save()
        return taken, gone

    def release(self, hwnd, front=False):
        """Back on the desktop, and not taken again while it lives."""
        with self.lock:
            self.held.pop(hwnd, None)
            self.released.add(hwnd)
            self._save()
        self._show(hwnd, front)

    def release_all(self):
        """Every held window back on the desktop: at quit, and when holding is
        turned off. They are not taken again by this copy of the app."""
        with self.lock:
            hwnds = list(self.held)
            self.held.clear()
            self.released.update(hwnds)
            if hwnds:
                self._save()
        for hwnd in hwnds:
            self._show(hwnd)
        return hwnds

    def snapshot(self):
        """{hwnd: info} as it is now, for another thread to walk."""
        with self.lock:
            return {h: dict(i) for h, i in self.held.items()}

    def stop(self):
        """At quit: give everything back, and take nothing after it - a sweep
        running beside the quit must not hide a window nobody will show."""
        with self.lock:
            self.stopped = True
        return self.release_all()

    def _save(self):
        self.ledger.save({h: i["pid"] for h, i in self.held.items()})


# ------------------------------------------------------------------- reader
if WINDOWS:
    class _COORD(ctypes.Structure):
        _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

    class _SMALL_RECT(ctypes.Structure):
        _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short),
                    ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]

    class _CSBI(ctypes.Structure):
        _fields_ = [("dwSize", _COORD), ("dwCursorPosition", _COORD),
                    ("wAttributes", wintypes.WORD), ("srWindow", _SMALL_RECT),
                    ("dwMaximumWindowSize", _COORD)]

    class _KEY_EVENT(ctypes.Structure):
        _fields_ = [("bKeyDown", wintypes.BOOL), ("wRepeatCount", wintypes.WORD),
                    ("wVirtualKeyCode", wintypes.WORD), ("wVirtualScanCode", wintypes.WORD),
                    ("UnicodeChar", wintypes.WCHAR), ("dwControlKeyState", wintypes.DWORD)]

    class _EVENT(ctypes.Union):
        _fields_ = [("KeyEvent", _KEY_EVENT), ("_pad", ctypes.c_byte * 16)]

    class _INPUT_RECORD(ctypes.Structure):
        _fields_ = [("EventType", wintypes.WORD), ("Event", _EVENT)]

    class _PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

    _k32.AttachConsole.argtypes = [wintypes.DWORD]
    _k32.GetConsoleWindow.restype = wintypes.HWND
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                 ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                 wintypes.HANDLE]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.GetConsoleScreenBufferInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_CSBI)]
    _k32.ReadConsoleOutputCharacterW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR,
                                                 wintypes.DWORD, _COORD,
                                                 ctypes.POINTER(wintypes.DWORD)]
    _k32.WriteConsoleInputW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_INPUT_RECORD),
                                        wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    _k32.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _k32.SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, wintypes.BOOL]
    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    _k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]

    _GENERIC_RW = 0x80000000 | 0x40000000
    _SHARE_RW = 0x1 | 0x2
    _OPEN_EXISTING = 3
    _INVALID = wintypes.HANDLE(-1).value
    _KEY_EVENT_TYPE = 0x0001
    _CTRL_C = 0
    _TH32CS_SNAPPROCESS = 0x2


def _children(pid):
    """Processes whose parent is `pid`: a console outlives the shell that
    opened it while something it started is still attached."""
    snap = _k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snap or snap == _INVALID:
        return []
    out = []
    try:
        e = _PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(e)
        ok = _k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            if e.th32ParentProcessID == pid and e.th32ProcessID != pid:
                out.append(e.th32ProcessID)
            ok = _k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        _k32.CloseHandle(snap)
    return out


class _Attached:
    """`with _Attached(hwnd, pid):` - this process joined to that window's
    console, and detached again on the way out."""

    def __init__(self, hwnd, pid):
        self.hwnd, self.pid = hwnd, pid

    def __enter__(self):
        _k32.FreeConsole()
        for pid in [self.pid] + _children(self.pid):
            if _k32.AttachConsole(pid):
                if int(_k32.GetConsoleWindow() or 0) == self.hwnd:
                    return self
                _k32.FreeConsole()        # another console: not the one asked for
        raise OSError("cannot reach the console of window %d" % self.hwnd)

    def __exit__(self, *exc):
        _k32.FreeConsole()

    @staticmethod
    def open(name):
        h = _k32.CreateFileW(name, _GENERIC_RW, _SHARE_RW, None, _OPEN_EXISTING, 0, None)
        if not h or h == _INVALID:
            raise OSError("cannot open %s (error %d)" % (name, ctypes.get_last_error()))
        return h


def read_screen(hwnd, pid, max_lines=MAX_LINES):
    """-> (lines, cursor row within them): the console's active screen buffer,
    from `max_lines` above its cursor to its last written row."""
    with _Attached(hwnd, pid):
        h = _Attached.open("CONOUT$")
        try:
            info = _CSBI()
            if not _k32.GetConsoleScreenBufferInfo(h, ctypes.byref(info)):
                raise OSError("cannot read the console (error %d)" % ctypes.get_last_error())
            width = info.dwSize.X
            last = max(info.dwCursorPosition.Y, info.srWindow.Bottom)
            first = max(0, info.dwCursorPosition.Y - max_lines)
            rows = last - first + 1
            buf = ctypes.create_unicode_buffer(width * rows)
            got = wintypes.DWORD()
            _k32.ReadConsoleOutputCharacterW(h, buf, width * rows, _COORD(0, first),
                                             ctypes.byref(got))
            text = buf[:got.value]
        finally:
            _k32.CloseHandle(h)
    lines = [text[i:i + width].rstrip() for i in range(0, len(text), width)]
    cursor = info.dwCursorPosition.Y - first
    while len(lines) > cursor + 1 and not lines[-1]:
        lines.pop()
    return lines, cursor


def _key(char, vk=0, down=True):
    r = _INPUT_RECORD()
    r.EventType = _KEY_EVENT_TYPE
    k = r.Event.KeyEvent
    k.bKeyDown, k.wRepeatCount, k.wVirtualKeyCode = down, 1, vk
    k.UnicodeChar = char
    return r


def type_text(hwnd, pid, text):
    """Keystrokes into the console's input, as though typed there. A newline
    is Enter."""
    records = []
    for ch in text:
        vk, ch = (0x0D, "\r") if ch in "\r\n" else (0, ch)
        records += [_key(ch, vk, True), _key(ch, vk, False)]
    if not records:
        return
    arr = (_INPUT_RECORD * len(records))(*records)
    with _Attached(hwnd, pid):
        h = _Attached.open("CONIN$")
        try:
            wrote = wintypes.DWORD()
            if not _k32.WriteConsoleInputW(h, arr, len(records), ctypes.byref(wrote)):
                raise OSError("cannot type into the console (error %d)"
                              % ctypes.get_last_error())
        finally:
            _k32.CloseHandle(h)


def interrupt(hwnd, pid):
    """Ctrl+C to everything in that console. The reader ignores Ctrl+C itself
    (`serve` sets that), so it is not ended by what it sends."""
    with _Attached(hwnd, pid):
        if not _k32.GenerateConsoleCtrlEvent(_CTRL_C, 0):
            raise OSError("cannot interrupt (error %d)" % ctypes.get_last_error())
        # The console delivers it on threads of its own, a moment later, to
        # whoever is attached then. Detached at once, nobody got it.
        time.sleep(INTERRUPT_SETTLE_S)


def serve(stdin=None, stdout=None):
    """`--reader`: one JSON request per line in, one answer per line out.

    {"op": "read"|"type"|"interrupt", "hwnd", "pid", "text"?} ->
    {"ok": true, "lines", "cursor"} | {"ok": true} | {"ok": false, "error"}"""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    _k32.SetConsoleCtrlHandler(None, True)     # ignore the Ctrl+C `interrupt` sends
    _k32.FreeConsole()                         # the hidden one it was started with
    for line in stdin:
        try:
            req = json.loads(line)
            hwnd, pid = int(req["hwnd"]), int(req["pid"])
            if req["op"] == "read":
                lines, cursor = read_screen(hwnd, pid)
                out = {"ok": True, "lines": lines, "cursor": cursor}
            elif req["op"] == "type":
                type_text(hwnd, pid, req["text"])
                out = {"ok": True}
            elif req["op"] == "interrupt":
                interrupt(hwnd, pid)
                out = {"ok": True}
            else:
                out = {"ok": False, "error": "unknown op %r" % req["op"]}
        except Exception as e:
            out = {"ok": False, "error": str(e)}
        stdout.write(json.dumps(out) + "\n")
        stdout.flush()


class Reader:
    """The app's side of `--reader`: one contained child, requests one at a
    time, started again whenever the last one was ended."""

    def __init__(self):
        self.lock = threading.Lock()
        self.child = None

    def _start(self):
        import studio_procs as procs
        exe = sys.executable
        if os.path.basename(exe).lower() == "pythonw.exe":
            beside = os.path.join(os.path.dirname(exe), "python.exe")
            exe = beside if os.path.isfile(beside) else exe
        self.child = procs.spawn(
            [exe, os.path.abspath(__file__), "--reader"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, creationflags=procs.NO_WINDOW)

    def ask(self, op, hwnd, pid, **kw):
        """One request. Raises OSError with the reader's words when it fails."""
        with self.lock:
            for attempt in (1, 2):
                if self.child is None or self.child.proc.poll() is not None:
                    self._start()
                p = self.child.proc
                try:
                    p.stdin.write(json.dumps(dict(kw, op=op, hwnd=hwnd, pid=pid)) + "\n")
                    p.stdin.flush()
                    line = _readline(p.stdout, READ_TIMEOUT)
                except (OSError, ValueError):
                    line = None
                if line:
                    out = json.loads(line)
                    if not out.get("ok"):
                        raise OSError(out.get("error") or "the reader failed")
                    return out
                # Ended mid-request: most often the console it was attached to
                # closed. One more go with a fresh reader, then say so.
                self.stop()
            raise OSError("the console reader stopped answering")

    def stop(self):
        child, self.child = self.child, None
        if child is not None:
            child.stop(0.5)


def _readline(stream, timeout):
    got = []
    t = threading.Thread(target=lambda: got.append(stream.readline()), daemon=True)
    t.start()
    t.join(timeout)
    return got[0] if got else None


def diff(old, new):
    """How to turn the lines shown into the lines read, cheaply.

    -> (drop, keep, tail): drop `drop` lines from the top, keep the next
    `keep`, and replace everything after them with `tail`. A console that
    scrolled by a few lines is a drop, not a rewrite of three thousand."""
    drop = 0
    if old and new and old[0] != new[0]:
        # the buffer scrolled: find the first new line further down the old
        probe = new[:3]
        for s in range(1, len(old)):
            if old[s:s + len(probe)] == probe:
                drop = s
                break
    shown = old[drop:]
    keep = 0
    for a, b in zip(shown, new):
        if a != b:
            break
        keep += 1
    return drop, keep, new[keep:]


if __name__ == "__main__":
    if sys.argv[1:] == ["--reader"]:
        serve()
    else:
        for w in windows():
            print("%(hwnd)10d  pid %(pid)-6d  %(visible)-5s  %(title)s" % w)
