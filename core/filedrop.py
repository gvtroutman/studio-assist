"""Files dragged onto a Tk window: from Explorer, the desktop or a browser.

Tk has no drag and drop of its own, and the stdlib-only rule bars tkinterdnd2.
`accept(window, drop, enter, leave)` registers an OLE drop target on the
window - an `IDropTarget` built here in ctypes - so the drag is seen while it
is still over the window (`enter` / `leave` light it up) and `drop(paths, url)`
gets what was let go:

- **Files** (`CF_HDROP`): Explorer, the desktop, and most browsers for a
  picture they have on disk. A folder is one of the paths; the caller decides
  what a folder means.
- **A link**, when no file came: a picture dragged out of a web page, or a link
  or an address bar. `paths` is empty and `url` holds it.

OLE finds a drop target by walking up from the window under the cursor, so
the registration on a Toplevel covers every widget inside it.

- **On the Tk thread, never into Tk.** `OleInitialize` and `RegisterDragDrop`
  are called on the thread that runs the window, and OLE calls the target
  there, from inside Tk's own message loop, while the drag's source waits.
  Any Tk call made there (a `config`, even an `after`) leaves `_tkinter`
  without its saved thread state, and the next Python callback Tk runs
  aborts the process (`Fatal Python error: PyEval_RestoreThread`). So the
  target only reads the data and notes what happened in `events`; `poll`,
  every `POLL_MS` from `after`, hands them to `enter` / `leave` / `drop`.
- **The objects are never freed.** OLE may call `Release` on a target after it
  was revoked, and a ctypes callback freed while it runs takes the process
  with it. A target is a few hundred bytes, kept in `_HELD` for the process's
  life; closing the window revokes it.
- **Nothing raises into OLE.** A Python exception inside a ctypes callback is
  printed (to nowhere, under pythonw) and the method returns garbage, so every
  method catches everything and answers as if nothing could be dropped.

Stdlib only (`ctypes`). Off 64-bit Windows `accept` returns None: the window
takes no drops and the caller says to click instead.
"""

import re
import sys

WINDOWS = sys.platform == "win32" and sys.maxsize > 2 ** 32
POLL_MS = 40                  # how often `Target.poll` hands OLE's events to Tk
_HELD = []                    # every target made: OLE may call one after revoke
_OLE = []                     # [True] once OleInitialize worked on the Tk thread

if WINDOWS:
    import ctypes
    from ctypes import wintypes

    _ole32 = ctypes.WinDLL("ole32")
    _shell32 = ctypes.WinDLL("shell32")
    _k32 = ctypes.WinDLL("kernel32")
    _u32 = ctypes.WinDLL("user32")

    class GUID(ctypes.Structure):
        _fields_ = [("d1", wintypes.DWORD), ("d2", wintypes.WORD), ("d3", wintypes.WORD),
                    ("d4", ctypes.c_ubyte * 8)]

    class FORMATETC(ctypes.Structure):
        _fields_ = [("cfFormat", ctypes.c_ushort), ("ptd", ctypes.c_void_p),
                    ("dwAspect", wintypes.DWORD), ("lindex", ctypes.c_long),
                    ("tymed", wintypes.DWORD)]

    class STGMEDIUM(ctypes.Structure):
        # The union is a handle or a pointer: one pointer wide, hGlobal here.
        _fields_ = [("tymed", wintypes.DWORD), ("hGlobal", ctypes.c_void_p),
                    ("pUnkForRelease", ctypes.c_void_p)]

    def _iid(d1, d2=0, d3=0, d4=b"\xc0\x00\x00\x00\x00\x00\x00\x46"):
        return bytes(GUID(d1, d2, d3, (ctypes.c_ubyte * 8)(*d4)))

    IID_IUnknown, IID_IDropTarget = _iid(0x00000000), _iid(0x00000122)
    IID_IDataObject = _iid(0x0000010E)

    S_OK, S_FALSE = 0, 1
    E_NOINTERFACE, E_UNEXPECTED = -2147467262, -2147418113     # 0x80004002, 0x8000FFFF
    DROPEFFECT_NONE, DROPEFFECT_COPY, DROPEFFECT_LINK = 0, 1, 4
    CF_UNICODETEXT, CF_HDROP = 13, 15
    DVASPECT_CONTENT, TYMED_HGLOBAL = 1, 1

    _ole32.OleInitialize.argtypes = [ctypes.c_void_p]
    _ole32.OleInitialize.restype = ctypes.c_long
    _ole32.RegisterDragDrop.argtypes = [wintypes.HWND, ctypes.c_void_p]
    _ole32.RegisterDragDrop.restype = ctypes.c_long
    _ole32.RevokeDragDrop.argtypes = [wintypes.HWND]
    _ole32.RevokeDragDrop.restype = ctypes.c_long
    _ole32.ReleaseStgMedium.argtypes = [ctypes.POINTER(STGMEDIUM)]
    _ole32.ReleaseStgMedium.restype = None
    _shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, wintypes.UINT, wintypes.LPWSTR,
                                        wintypes.UINT]
    _shell32.DragQueryFileW.restype = wintypes.UINT
    _k32.GlobalLock.argtypes = [ctypes.c_void_p]
    _k32.GlobalLock.restype = ctypes.c_void_p
    _k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    _k32.GlobalSize.argtypes = [ctypes.c_void_p]
    _k32.GlobalSize.restype = ctypes.c_size_t
    _u32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    _u32.RegisterClipboardFormatW.restype = wintypes.UINT

    CF_URL = _u32.RegisterClipboardFormatW("UniformResourceLocatorW")

    # IDropTarget. POINTL is passed by value: 8 bytes, one register on x64.
    _QI = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID),
                             ctypes.POINTER(ctypes.c_void_p))
    _REF = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)
    _ENTER = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
                                wintypes.DWORD, ctypes.c_int64,
                                ctypes.POINTER(wintypes.DWORD))
    _OVER = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.DWORD,
                               ctypes.c_int64, ctypes.POINTER(wintypes.DWORD))
    _LEAVE = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)

    class _VTable(ctypes.Structure):
        _fields_ = [("QueryInterface", _QI), ("AddRef", _REF), ("Release", _REF),
                    ("DragEnter", _ENTER), ("DragOver", _OVER), ("DragLeave", _LEAVE),
                    ("Drop", _ENTER)]

    class _Object(ctypes.Structure):
        _fields_ = [("vtbl", ctypes.POINTER(_VTable))]

    # IDataObject, called through its own table: GetData 3, QueryGetData 5.
    _GETDATA = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p,
                                  ctypes.POINTER(FORMATETC), ctypes.POINTER(STGMEDIUM))
    _QUERY = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(FORMATETC))


def _method(obj, index, proto):
    table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return proto(table[index])


def _format(cf):
    return FORMATETC(cf, None, DVASPECT_CONTENT, -1, TYMED_HGLOBAL)


def _has(data, cf):
    return _method(data, 5, _QUERY)(data, ctypes.byref(_format(cf))) == S_OK


def _get(data, cf, read):
    """`read(hGlobal)` of the data's `cf`, or None when it has none."""
    medium = STGMEDIUM()
    if _method(data, 3, _GETDATA)(data, ctypes.byref(_format(cf)),
                                  ctypes.byref(medium)) != S_OK:
        return None
    try:
        return read(medium.hGlobal) if medium.hGlobal else None
    finally:
        _ole32.ReleaseStgMedium(ctypes.byref(medium))


def _files(handle):
    count = _shell32.DragQueryFileW(handle, 0xFFFFFFFF, None, 0)
    paths = []
    for i in range(count):
        n = _shell32.DragQueryFileW(handle, i, None, 0)
        buf = ctypes.create_unicode_buffer(n + 1)
        _shell32.DragQueryFileW(handle, i, buf, n + 1)
        paths.append(buf.value)
    return paths


def _text(handle):
    at = _k32.GlobalLock(handle)
    if not at:
        return None
    try:
        return ctypes.wstring_at(at, _k32.GlobalSize(handle) // 2).split("\0", 1)[0]
    finally:
        _k32.GlobalUnlock(handle)


def link(text):
    """`text` when it is one web or data link and nothing else, else None."""
    text = (text or "").strip()
    if "\n" in text or len(text) > 8192:
        return None
    return text if re.match(r"(?i)(https?://|data:image/)\S+$", text) else None


def offered(data):
    """Whether an IDataObject holds anything `read` takes. Files and links
    without reading them (a browser may write the file only when it is asked
    for); plain text is read, as only a link is taken."""
    return (any(_has(data, cf) for cf in (CF_HDROP, CF_URL)) or
            bool(_has(data, CF_UNICODETEXT) and link(_get(data, CF_UNICODETEXT, _text))))


def read(data):
    """(paths, url) from an IDataObject: its files, else the one link it
    carries (a page's picture or link, or text that is nothing but a link)."""
    paths = _get(data, CF_HDROP, _files) or []
    if paths:
        return paths, None
    for cf in (CF_URL, CF_UNICODETEXT):
        url = link(_get(data, cf, _text))
        if url:
            return [], url
    return [], None


class Target:
    """One window's IDropTarget. `pointer` is the COM object OLE holds."""

    def __init__(self, widget, drop, enter=None, leave=None):
        self.widget, self.on_drop, self.on_enter, self.on_leave = widget, drop, enter, leave
        self.hwnd, self.ok, self.refs, self.job = None, False, 1, None
        self.events = []      # ("enter",), ("leave",), ("drop", paths, url) for `poll`
        self.table = _VTable(_QI(self._query), _REF(self._add), _REF(self._release),
                             _ENTER(self._enter), _OVER(self._over), _LEAVE(self._leave),
                             _ENTER(self._drop))
        self.obj = _Object(ctypes.pointer(self.table))
        self.pointer = ctypes.addressof(self.obj)

    # IUnknown
    def _query(self, this, riid, out):
        try:
            if bytes(riid.contents) in (IID_IUnknown, IID_IDropTarget):
                out[0] = self.pointer
                self.refs += 1
                return S_OK
            out[0] = None
            return E_NOINTERFACE
        except Exception:
            return E_NOINTERFACE

    def _add(self, this):
        self.refs += 1
        return self.refs

    def _release(self, this):
        self.refs = max(0, self.refs - 1)
        return self.refs

    # IDropTarget
    def _effect(self, effect):
        allowed = effect[0]
        effect[0] = (DROPEFFECT_NONE if not self.ok else DROPEFFECT_COPY
                     if allowed & DROPEFFECT_COPY else DROPEFFECT_LINK
                     if allowed & DROPEFFECT_LINK else DROPEFFECT_NONE)

    def _call(self, fn, *args):
        if fn is not None:
            try:
                fn(*args)
            except Exception:
                pass

    def _enter(self, this, data, keys, point, effect):
        try:
            self.ok = bool(data) and offered(data)
            self._effect(effect)
            if self.ok:
                self.events.append(("enter",))
            return S_OK
        except Exception:
            self.ok = False
            return E_UNEXPECTED

    def _over(self, this, keys, point, effect):
        try:
            self._effect(effect)
            return S_OK
        except Exception:
            return E_UNEXPECTED

    def _leave(self, this):
        if self.ok:
            self.ok = False
            self.events.append(("leave",))
        return S_OK

    def _drop(self, this, data, keys, point, effect):
        try:
            paths, url = read(data) if data and self.ok else ([], None)
            if not paths and not url:
                self.ok = False
            self._effect(effect)
        except Exception:
            paths, url = [], None
            effect[0] = DROPEFFECT_NONE
        self._leave(this)
        if paths or url:
            self.events.append(("drop", paths, url))
        return S_OK

    def poll(self):
        """On the Tk thread, from `after`: hand on what OLE noted, in order."""
        self.job = None
        while self.events:
            kind, *args = self.events.pop(0)
            self._call({"enter": self.on_enter, "leave": self.on_leave,
                        "drop": self.on_drop}[kind], *args)
        if self.hwnd:
            self.job = self.widget.after(POLL_MS, self.poll)

    def revoke(self, ev=None):
        if ev is not None and ev.widget is not self.widget:
            return
        if self.job is not None:
            try:
                self.widget.after_cancel(self.job)
            except Exception:
                pass
            self.job = None
        if self.hwnd:
            _ole32.RevokeDragDrop(self.hwnd)
            self.hwnd = None


def accept(widget, drop, enter=None, leave=None):
    """Let files and links be dropped anywhere on `widget` (a Toplevel):
    `drop(paths, url)` after the drop, `enter()` / `leave()` as a drag that
    carries something it takes comes over the window and goes (or is let go).
    Call on the Tk thread. -> the Target, or None when this window cannot take
    drops (not 64-bit Windows, or OLE refused)."""
    if not WINDOWS:
        return None
    if not _OLE:
        if _ole32.OleInitialize(None) not in (S_OK, S_FALSE):
            return None       # the thread is in another apartment: no drops here
        _OLE.append(True)
    widget.update_idletasks()
    target = Target(widget, drop, enter, leave)
    _HELD.append(target)
    hwnd = widget.winfo_id()
    if _ole32.RegisterDragDrop(hwnd, target.pointer) != S_OK:
        return None
    target.hwnd = hwnd
    widget.bind("<Destroy>", target.revoke, add="+")
    target.poll()
    return target
