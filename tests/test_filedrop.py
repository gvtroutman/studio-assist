"""core/filedrop: the OLE drop target, called the way OLE calls it.

A real drag needs the mouse, so these build the data a drag carries (a shell
data object holding CF_HDROP or a link) and call the registered target's own
IDropTarget table with it, as DoDragDrop does from the drag's source."""
import gc
import os
import sys
import tempfile
import tkinter as tk
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import core.filedrop as fd

if fd.WINDOWS:
    import ctypes
    from ctypes import wintypes

    _shell32 = ctypes.WinDLL("shell32")
    _k32 = ctypes.WinDLL("kernel32")
    _shell32.SHCreateDataObject.argtypes = [ctypes.c_void_p, wintypes.UINT, ctypes.c_void_p,
                                            ctypes.c_void_p, ctypes.POINTER(fd.GUID),
                                            ctypes.POINTER(ctypes.c_void_p)]
    _shell32.SHCreateDataObject.restype = ctypes.c_long
    _k32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    _k32.GlobalAlloc.restype = ctypes.c_void_p
    _SETDATA = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p,
                                  ctypes.POINTER(fd.FORMATETC),
                                  ctypes.POINTER(fd.STGMEDIUM), wintypes.BOOL)
    _RELEASE = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)


def _global(raw):
    handle = _k32.GlobalAlloc(0x42, len(raw))          # GMEM_MOVEABLE | GMEM_ZEROINIT
    at = fd._k32.GlobalLock(handle)
    ctypes.memmove(at, raw, len(raw))
    fd._k32.GlobalUnlock(handle)
    return handle


def data_object(paths=None, url=None, text=None):
    """A shell IDataObject carrying files, a link and/or plain text."""
    out = ctypes.c_void_p()
    iid = fd.GUID.from_buffer_copy(fd.IID_IDataObject)
    assert _shell32.SHCreateDataObject(None, 0, None, None, ctypes.byref(iid),
                                       ctypes.byref(out)) == 0
    obj = out.value
    put = fd._method(obj, 7, _SETDATA)
    if paths is not None:      # DROPFILES: pFiles, pt.x, pt.y, fNC, fWide
        head = (20).to_bytes(4, "little") + bytes(12) + (1).to_bytes(4, "little")
        names = "".join(p + "\0" for p in paths) + "\0"
        medium = fd.STGMEDIUM(fd.TYMED_HGLOBAL, _global(head + names.encode("utf-16-le")))
        assert put(obj, ctypes.byref(fd._format(fd.CF_HDROP)), ctypes.byref(medium), 1) == 0
    for cf, value in ((fd.CF_URL, url), (fd.CF_UNICODETEXT, text)):
        if value is not None:
            medium = fd.STGMEDIUM(fd.TYMED_HGLOBAL,
                                  _global((value + "\0").encode("utf-16-le")))
            assert put(obj, ctypes.byref(fd._format(cf)), ctypes.byref(medium), 1) == 0
    return obj


def release(obj):
    fd._method(obj, 2, _RELEASE)(obj)


class LinkTests(unittest.TestCase):
    def test_a_link_is_one_web_or_data_address_and_nothing_else(self):
        self.assertEqual(fd.link(" https://example.com/a.jpg \n"), "https://example.com/a.jpg")
        self.assertEqual(fd.link("data:image/png;base64,AAAA"), "data:image/png;base64,AAAA")
        for text in (None, "", "hello", "see https://example.com", "ftp://x/y",
                     "https://a.com\nhttps://b.com"):
            self.assertIsNone(fd.link(text), text)


@unittest.skipUnless(fd.WINDOWS, "OLE drag and drop is Windows-only")
class DropTargetTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(gc.collect)
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.win = tk.Toplevel(self.root)
        self.calls = []
        self.target = fd.accept(self.win, lambda paths, url: self.calls.append(
            ("drop", paths, url)), enter=lambda: self.calls.append("enter"),
            leave=lambda: self.calls.append("leave"))
        self.assertIsNotNone(self.target, "RegisterDragDrop refused the window")

    def drag(self, data, allowed=fd.DROPEFFECT_COPY | fd.DROPEFFECT_LINK, drop=True):
        """Enter, move over and let go (or leave), as OLE calls a target.
        -> the effects it answered."""
        this, effects = self.target.pointer, []
        effect = wintypes.DWORD(allowed)
        fd._method(this, 3, fd._ENTER)(this, data, 0, 0, ctypes.byref(effect))
        effects.append(effect.value)
        effect = wintypes.DWORD(allowed)
        fd._method(this, 4, fd._OVER)(this, 0, 0, ctypes.byref(effect))
        effects.append(effect.value)
        if drop:
            effect = wintypes.DWORD(allowed)
            fd._method(this, 6, fd._ENTER)(this, data, 0, 0, ctypes.byref(effect))
            effects.append(effect.value)
        else:
            fd._method(this, 5, fd._LEAVE)(this)
        self.assertEqual(self.calls, [])         # nothing into Tk from inside OLE
        self.target.poll()                       # as `after` does next
        return effects

    def test_files_light_the_window_and_arrive_after_the_drop(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [os.path.join(folder, "één photo.jpg"), folder]
            data = data_object(paths=paths)
            try:
                effects = self.drag(data)
            finally:
                release(data)
        self.assertEqual(effects, [fd.DROPEFFECT_COPY] * 3)
        self.assertEqual(self.calls, ["enter", "leave", ("drop", paths, None)])

    def test_a_link_alone_arrives_as_the_url(self):
        data = data_object(url="https://example.com/she.jpg")
        try:
            self.drag(data, allowed=fd.DROPEFFECT_LINK)
        finally:
            release(data)
        self.assertEqual(self.calls, ["enter", "leave",
                                      ("drop", [], "https://example.com/she.jpg")])

    def test_files_win_over_the_link_beside_them(self):
        data = data_object(paths=[r"C:\pictures\a.png"], url="https://example.com/a.png")
        try:
            self.drag(data)
        finally:
            release(data)
        self.assertEqual(self.calls[-1], ("drop", [r"C:\pictures\a.png"], None))

    def test_text_that_is_not_a_link_is_refused_and_nothing_lights(self):
        data = data_object(text="a note, not a link")
        try:
            effects = self.drag(data)
        finally:
            release(data)
        self.assertEqual(effects[:2], [fd.DROPEFFECT_NONE] * 2)
        self.assertEqual(self.calls, [])

    def test_a_drag_that_leaves_drops_nothing(self):
        data = data_object(paths=[r"C:\pictures\a.png"])
        try:
            self.drag(data, drop=False)
        finally:
            release(data)
        self.assertEqual(self.calls, ["enter", "leave"])

    def test_it_answers_only_for_its_own_interfaces(self):
        this, out = self.target.pointer, ctypes.c_void_p()
        query = fd._method(this, 0, fd._QI)
        for iid in (fd.IID_IDropTarget, fd.IID_IUnknown):
            out.value = None
            self.assertEqual(query(this, fd.GUID.from_buffer_copy(iid), ctypes.byref(out)), 0)
            self.assertEqual(out.value, this)
        self.assertEqual(query(this, fd.GUID.from_buffer_copy(fd.IID_IDataObject),
                               ctypes.byref(out)), fd.E_NOINTERFACE)
        self.assertIsNone(out.value)

    def test_closing_the_window_revokes_it(self):
        hwnd = self.target.hwnd
        self.assertTrue(hwnd)
        self.assertIsNotNone(self.target.job)
        self.win.destroy()
        self.assertIsNone(self.target.hwnd)
        self.assertIsNone(self.target.job)            # the poll stops with it
        self.assertIn(self.target, fd._HELD)          # kept: OLE may still call it


# OLE calls a target from a message Tk's own loop dispatches. A Win32 timer is
# dispatched the same way, so its TIMERPROC drags onto the target from there.
# A Tk call made at that point aborted the whole process (the app "crashed on
# drop"), so this runs in a process of its own and must exit cleanly.
_IN_MAINLOOP = r"""
import ctypes, os, sys, tkinter as tk
from ctypes import wintypes
sys.path[:0] = [{root!r}, {tests!r}]
import core.filedrop as fd
from test_filedrop import data_object

u32 = ctypes.WinDLL("user32")
TIMERPROC = ctypes.WINFUNCTYPE(None, wintypes.HWND, wintypes.UINT, ctypes.c_size_t,
                               wintypes.DWORD)
u32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, TIMERPROC]
u32.SetTimer.restype = ctypes.c_size_t
u32.KillTimer.argtypes = [wintypes.HWND, ctypes.c_size_t]

root = tk.Tk()
root.withdraw()
win = tk.Toplevel(root)
win.withdraw()
label = tk.Label(win, text="")
label.pack()
seen = []
def dropped(paths, url):
    seen.append(paths)
    label.config(text="dropped")
    root.after(10, root.destroy)
target = fd.accept(win, dropped, enter=lambda: label.config(text="lit"),
                   leave=lambda: label.config(text=""))
data = data_object(paths=[r"C:\pictures\a.png"])

def fired(hwnd, msg, ident, when):
    u32.KillTimer(None, ident)
    this, effect = target.pointer, wintypes.DWORD(fd.DROPEFFECT_COPY)
    fd._method(this, 3, fd._ENTER)(this, data, 0, 0, ctypes.byref(effect))
    fd._method(this, 6, fd._ENTER)(this, data, 0, 0, ctypes.byref(effect))

proc = TIMERPROC(fired)
u32.SetTimer(None, 0, 20, proc)
root.after(5000, root.destroy)
root.mainloop()
print("seen", seen)
"""


@unittest.skipUnless(fd.WINDOWS, "OLE drag and drop is Windows-only")
class DropInsideTheLoopTests(unittest.TestCase):
    def test_a_drop_from_inside_tks_message_loop_does_not_kill_the_app(self):
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        script = _IN_MAINLOOP.format(root=os.path.dirname(here), tests=here)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn(r"seen [['C:\\pictures\\a.png']]", out.stdout)


if __name__ == "__main__":
    unittest.main()
