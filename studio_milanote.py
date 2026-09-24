"""Milanote in a tab: its web app in a browser window of our own, held inside
the Studio Assist window, and files dropped onto the open board.

Milanote has no public API and no MCP server, so this tab has no bridge and no
model: it is a container. What it holds is a Chrome (or Edge) window started
here with `--app=` - no tabs, no address bar - and a profile of its own under
`%LOCALAPPDATA%\\StudioAssistant\\milanote-browser`, so the Milanote sign-in is
made once there and kept. That window is re-parented into the tab's frame
(`SetParent`, `WS_CHILD`) and sized with it.

The profile is ours rather than the user's everyday one for two reasons: a
Chrome already running with that profile would take the launch over, leaving
us no process whose window we can find; and Chrome refuses remote debugging on
the default profile, which uploads need.

Uploading is a drop. Milanote takes a file dragged onto a board and makes a
card of it, so `Browser.upload()` does what a drag from Explorer does, through
the DevTools protocol: `Input.dispatchDragEvent` with the file paths, at the
middle of the page. DevTools listens on a loopback port Chrome picks itself
(`--remote-debugging-port=0`) and writes to `DevToolsActivePort` in the
profile; nothing off this machine can reach it.

Stdlib only: `ctypes` for the window, `urllib` and a small WebSocket client
for DevTools. Every process goes through `studio_procs`, so the browser ends
with the tab or the app however either ends.
"""

import base64
import json
import os
import socket
import struct
import sys
import time
import urllib.parse
import urllib.request

import studio_procs as procs

WINDOWS = sys.platform == "win32"

MILANOTE_URL = os.environ.get("MILANOTE_URL", "https://app.milanote.com/")

BROWSERS = [
    os.path.join(os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                 r"Google\Chrome\Application\chrome.exe"),
    os.path.join(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                 r"Google\Chrome\Application\chrome.exe"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Google\Chrome\Application\chrome.exe"),
    os.path.join(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                 r"Microsoft\Edge\Application\msedge.exe"),
    os.path.join(os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                 r"Microsoft\Edge\Application\msedge.exe"),
]

WINDOW_WAIT = 20.0        # seconds for the browser's first window to appear
PORT_WAIT = 10.0          # ...and for DevToolsActivePort to be written
CLOSE_GRACE = 4.0         # after WM_CLOSE, before the job is terminated


def browser_exe():
    """Chrome, else Edge; STUDIO_MILANOTE_BROWSER names another Chromium."""
    pinned = os.environ.get("STUDIO_MILANOTE_BROWSER")
    if pinned:
        return pinned if os.path.isfile(pinned) else None
    for path in BROWSERS:
        if path and os.path.isfile(path):
            return path
    return None


def profile_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.environ.get("STUDIO_MILANOTE_PROFILE") or os.path.join(
        base, "StudioAssistant", "milanote-browser")


def browser_args(exe, profile, url=MILANOTE_URL):
    return [exe, "--app=" + url, "--user-data-dir=" + profile,
            "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
            "--no-first-run", "--no-default-browser-check",
            "--disable-features=Translate", "--window-position=-32000,-32000"]


def read_port(profile):
    """The DevTools port Chrome chose, from the first line of the file it
    writes into the profile, or None before it has."""
    try:
        with open(os.path.join(profile, "DevToolsActivePort"), encoding="utf-8") as f:
            first = f.readline().strip()
        return int(first) if first.isdigit() else None
    except OSError:
        return None


# ------------------------------------------------------------------ windows

if WINDOWS:
    import ctypes
    from ctypes import wintypes

    _u32 = ctypes.WinDLL("user32", use_last_error=True)
    _ENUM = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    _u32.EnumWindows.argtypes = [_ENUM, wintypes.LPARAM]
    _u32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _u32.IsWindowVisible.argtypes = [wintypes.HWND]
    _u32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _u32.GetWindow.argtypes = [wintypes.HWND, ctypes.c_uint]
    _u32.GetWindow.restype = wintypes.HWND
    _u32.SetParent.argtypes = [wintypes.HWND, wintypes.HWND]
    _u32.SetParent.restype = wintypes.HWND
    _u32.MoveWindow.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, wintypes.BOOL]
    _u32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    _u32.PostMessageW.argtypes = [wintypes.HWND, ctypes.c_uint, wintypes.WPARAM,
                                  wintypes.LPARAM]
    _u32.IsWindow.argtypes = [wintypes.HWND]
    _u32.SetFocus.argtypes = [wintypes.HWND]
    _u32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    _GetLong = getattr(_u32, "GetWindowLongPtrW", _u32.GetWindowLongW)
    _SetLong = getattr(_u32, "SetWindowLongPtrW", _u32.SetWindowLongW)
    _GetLong.argtypes = [wintypes.HWND, ctypes.c_int]
    _GetLong.restype = ctypes.c_ssize_t
    _SetLong.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    _SetLong.restype = ctypes.c_ssize_t

GWL_STYLE, GWL_EXSTYLE = -16, -20
WS_CHILD, WS_POPUP, WS_CAPTION, WS_THICKFRAME = 0x40000000, 0x80000000, 0x00C00000, 0x00040000
WS_SYSMENU, WS_MINIMIZEBOX, WS_MAXIMIZEBOX = 0x00080000, 0x00020000, 0x00010000
WS_EX_APPWINDOW, WS_EX_WINDOWEDGE, WS_EX_DLGMODALFRAME = 0x00040000, 0x100, 0x1
GW_OWNER = 4
WM_CLOSE = 0x0010
SW_SHOW, SW_HIDE = 5, 0
SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER, SWP_NOACTIVATE, SWP_FRAMECHANGED = 1, 2, 4, 0x10, 0x20


def top_windows(pid):
    """Visible, unowned top-level Chromium windows of process `pid`."""
    if not WINDOWS:
        return []
    found = []

    def each(hwnd, _):
        owner = wintypes.DWORD()
        _u32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value != pid or not _u32.IsWindowVisible(hwnd):
            return True
        if _u32.GetWindow(hwnd, GW_OWNER):
            return True                   # a popup or a bubble, not the app window
        name = ctypes.create_unicode_buffer(64)
        _u32.GetClassNameW(hwnd, name, 64)
        if name.value.startswith("Chrome_WidgetWin"):
            found.append(hwnd)
        return True

    _u32.EnumWindows(_ENUM(each), 0)
    return found


def adopt(hwnd, parent):
    """Make `hwnd` a borderless child of `parent`."""
    style = _GetLong(hwnd, GWL_STYLE)
    style &= ~(WS_POPUP | WS_CAPTION | WS_THICKFRAME | WS_SYSMENU
               | WS_MINIMIZEBOX | WS_MAXIMIZEBOX)
    _SetLong(hwnd, GWL_STYLE, style | WS_CHILD)
    ex = _GetLong(hwnd, GWL_EXSTYLE)
    _SetLong(hwnd, GWL_EXSTYLE, ex & ~(WS_EX_APPWINDOW | WS_EX_WINDOWEDGE
                                       | WS_EX_DLGMODALFRAME))
    _u32.SetParent(hwnd, parent)
    _u32.SetWindowPos(hwnd, None, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER
                      | SWP_FRAMECHANGED | SWP_NOACTIVATE)


# ------------------------------------------------------------------ devtools

class DevToolsError(RuntimeError):
    pass


class WebSocket:
    """Just enough of RFC 6455 for DevTools: text frames, client masking,
    fragments and pings. No Origin header, which DevTools would check."""

    def __init__(self, url, timeout=10):
        u = urllib.parse.urlsplit(url)
        self.sock = socket.create_connection((u.hostname, u.port or 80), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        path = u.path + ("?" + u.query if u.query else "")
        self.sock.sendall(("GET %s HTTP/1.1\r\nHost: %s:%s\r\nUpgrade: websocket\r\n"
                           "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
                           "Sec-WebSocket-Version: 13\r\n\r\n"
                           % (path, u.hostname, u.port, key)).encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(1024)
            if not chunk:
                raise DevToolsError("DevTools closed the connection during the handshake")
            head += chunk
        status = head.split(b"\r\n", 1)[0]
        if b" 101 " not in status + b" ":
            raise DevToolsError("DevTools refused the connection: %s"
                                % status.decode("latin-1"))
        self.buf = head.split(b"\r\n\r\n", 1)[1]

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise DevToolsError("DevTools closed the connection")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, text, opcode=1):
        data = text.encode("utf-8") if isinstance(text, str) else text
        n = len(data)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(head + mask + body)

    def recv(self):
        parts = []
        while True:
            b0, b1 = self._read(2)
            fin, opcode, n = b0 & 0x80, b0 & 0x0F, b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else None
            data = self._read(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if opcode == 8:
                raise DevToolsError("DevTools closed the connection")
            if opcode == 9:
                self.send(data, opcode=10)
                continue
            if opcode == 10:
                continue
            parts.append(data)
            if fin:
                return b"".join(parts).decode("utf-8", "replace")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class DevTools:
    """One page's DevTools session: call(method, params) -> result."""

    def __init__(self, ws_url, timeout=15):
        self.ws = WebSocket(ws_url, timeout=timeout)
        self.seq = 0

    def call(self, method, params=None):
        self.seq += 1
        self.ws.send(json.dumps({"id": self.seq, "method": method,
                                 "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") != self.seq:
                continue                  # an event; nothing here subscribes
            if "error" in msg:
                raise DevToolsError("%s: %s" % (method, msg["error"].get("message")))
            return msg.get("result", {})

    def close(self):
        self.ws.close()


def pages(port, timeout=5):
    with urllib.request.urlopen("http://127.0.0.1:%d/json/list" % port,
                                timeout=timeout) as r:
        return [t for t in json.load(r) if t.get("type") == "page"]


def milanote_page(port):
    """The page target showing Milanote, else the first page."""
    found = pages(port)
    for t in found:
        if "milanote.com" in t.get("url", ""):
            return t
    if found:
        return found[0]
    raise DevToolsError("the Milanote window has no page open")


def drop_files(devtools, paths, x, y):
    """What dragging `paths` from Explorer and letting go at (x, y) does."""
    data = {"items": [], "files": list(paths), "dragOperationsMask": 1}
    for kind in ("dragEnter", "dragOver", "drop"):
        devtools.call("Input.dispatchDragEvent",
                      {"type": kind, "x": x, "y": y, "data": data})


# ------------------------------------------------------------------ the browser

class Browser:
    """The Milanote window: started, found, adopted into a frame, sized with
    it, and closed. Every method but start() is safe before start()."""

    def __init__(self, exe=None, profile=None, url=MILANOTE_URL):
        self.exe = exe or browser_exe()
        self.profile = profile or profile_dir()
        self.url = url
        self.child = None
        self.hwnd = None
        self.parent = None                # the frame it is held in, once embedded
        self.inset = (0, 0, 0, 0)         # see measure()
        self.size = (1, 1)                # the parent area last fitted

    def running(self):
        return self.child is not None and self.child.proc.poll() is None

    def start(self):
        """Start the browser and return its window handle. Blocks; call it
        off the UI thread. Raises RuntimeError with a sentence for the user."""
        if not self.exe:
            raise RuntimeError("Milanote opens in Chrome or Edge, and neither is "
                               "installed where this app looks. Install Chrome, or "
                               "set STUDIO_MILANOTE_BROWSER to a Chromium browser.")
        os.makedirs(self.profile, exist_ok=True)
        try:
            os.remove(os.path.join(self.profile, "DevToolsActivePort"))
        except OSError:
            pass
        self.child = procs.spawn(browser_args(self.exe, self.profile, self.url))
        deadline = time.monotonic() + WINDOW_WAIT
        while time.monotonic() < deadline:
            if self.child.proc.poll() is not None:
                raise RuntimeError(
                    "the browser for Milanote closed as it opened. Another copy of it "
                    "may already be using the profile at %s - close that and try again."
                    % self.profile)
            found = top_windows(self.child.pid)
            if found:
                self.hwnd = found[0]
                return self.hwnd
            time.sleep(0.15)
        raise RuntimeError("the browser started, but its Milanote window did not "
                           "appear within %d seconds" % WINDOW_WAIT)

    def embed(self, parent, width, height):
        if self.hwnd and WINDOWS:
            adopt(self.hwnd, parent)
            self.parent = parent
            self.size = (width, height)
            self.fit(width, height)
            _u32.ShowWindow(self.hwnd, SW_SHOW)

    def fit(self, width, height):
        """Fill `width` x `height` of the parent with the page. The window is
        placed so its own title bar and borders fall outside the parent and
        are clipped: an --app window draws a caption of its own (the name,
        minimise, maximise, close) that Windows styles cannot remove."""
        self.size = (width, height)
        # Not before it is embedded: a top-level window moved to (-left, -top)
        # would land on the desktop's corner instead of inside the tab.
        if self.parent and self.hwnd and WINDOWS and _u32.IsWindow(self.hwnd):
            left, top, right, bottom = self.inset
            _u32.MoveWindow(self.hwnd, -left, -top, max(1, width + left + right),
                            max(1, height + top + bottom), True)

    def measure(self):
        """How far the page sits inside the window, in device pixels: (left,
        top, right, bottom), from the page's own outer and inner sizes. The
        sides and the bottom are the frame's invisible resize border; the top
        is that and the drawn title bar. Blocks; off the UI thread."""
        dt = self.page()
        try:
            got = dt.call("Runtime.evaluate", {
                "expression": "[outerWidth - innerWidth, outerHeight - innerHeight, "
                              "devicePixelRatio]", "returnByValue": True})
        finally:
            dt.close()
        dw, dh, ratio = got.get("result", {}).get("value") or (0, 0, 1)
        side = max(0, round(dw * ratio / 2))
        top = max(0, round(dh * ratio) - side)
        self.inset = (side, top, side, side)
        return self.inset

    def focus(self):
        if self.hwnd and WINDOWS and _u32.IsWindow(self.hwnd):
            _u32.SetFocus(self.hwnd)

    def page(self):
        """A DevTools session on the page the window shows. Close it."""
        if not self.running():
            raise RuntimeError("Milanote is not open in this tab")
        port = self.port()
        if not port:
            raise RuntimeError("the Milanote window is not answering on DevTools; "
                               "close the tab and open it again")
        return DevTools(milanote_page(port)["webSocketDebuggerUrl"])

    def reload(self):
        dt = self.page()
        try:
            dt.call("Page.reload", {"ignoreCache": False})
        finally:
            dt.close()

    def port(self, wait=PORT_WAIT):
        deadline = time.monotonic() + wait
        while True:
            port = read_port(self.profile)
            if port or time.monotonic() >= deadline:
                return port
            time.sleep(0.2)

    def upload(self, paths):
        """Drop `paths` onto the board Milanote is showing. Returns a sentence
        saying what was sent. Blocks; call it off the UI thread."""
        paths = [os.path.abspath(p) for p in paths]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise RuntimeError("not a file: %s" % ", ".join(missing))
        dt = self.page()
        try:
            where = dt.call("Runtime.evaluate", {
                "expression": "[innerWidth, innerHeight, location.href]",
                "returnByValue": True}).get("result", {}).get("value") or [800, 600, ""]
            w, h, href = where
            if "milanote.com" not in href:
                raise RuntimeError("the window is not showing Milanote")
            if "/login" in href or "/signup" in href:
                raise RuntimeError("sign in to Milanote in this tab first, then open "
                                   "the board the files should land on")
            drop_files(dt, paths, w // 2, h // 2)
        finally:
            dt.close()
        names = ", ".join(os.path.basename(p) for p in paths)
        return "dropped %d file%s onto the board: %s" % (
            len(paths), "" if len(paths) == 1 else "s", names)

    def release(self):
        """Out of the frame and asked to close, before the frame is destroyed:
        destroying a parent destroys its children, and Chrome, its window gone
        from under it, would be killed rather than closed. On the UI thread."""
        hwnd = self.hwnd
        if hwnd and WINDOWS and _u32.IsWindow(hwnd):
            _u32.ShowWindow(hwnd, SW_HIDE)
            _u32.SetParent(hwnd, None)
            _u32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        self.parent = None

    def close(self, grace=CLOSE_GRACE):
        """WM_CLOSE first, so Chrome saves its session and does not ask to
        restore it next time; then the job, whatever is left of it."""
        hwnd, child = self.hwnd, self.child
        self.hwnd = self.child = None
        if hwnd and WINDOWS:
            _u32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        if child is not None:
            try:
                child.proc.wait(timeout=grace)
            except Exception:
                pass
            child.kill()
