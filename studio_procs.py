"""Child processes that cannot outlive the process that started them.

Windows does not kill a process's children when it dies, and `Popen.kill()`
ends one process, not the tree under it. A bridge is often a tree: the After
Effects one is `cmd /c npx` -> `node npx-cli` -> `cmd /c after-effects-mcp` ->
`node server.js`, and killing the `cmd` at the top left the `node` at the
bottom running with nobody holding its pipes. So did a window closed from
Task Manager, a crash, and a test run stopped half way.

Every child here is started inside its own **job object** with
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`: everything it starts joins the job, and
the job - the whole tree - ends when `stop()` terminates it or when the last
handle to it closes. That handle belongs to this process, so the kernel closes
it however this process ends, `os._exit` and a hard kill included. The child is
created suspended and resumed only once it is in the job, so not even a
grandchild started in its first instant gets out.

Only what this module started is ever touched: no process is found by name, and
an app a bridge talks to over COM or a socket (Photoshop, After Effects) was
never our child and is never in a job of ours.

Stdlib only (`ctypes`). Off Windows it degrades to plain `Popen` and a
process-group kill.
"""

import atexit
import os
import signal
import subprocess
import sys
import threading
import weakref

WINDOWS = sys.platform == "win32"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CREATE_SUSPENDED = 0x00000004

_live = weakref.WeakSet()          # every Child not yet stopped, for atexit
_live_lock = threading.Lock()

if WINDOWS:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        _NtResumeProcess = ctypes.WinDLL("ntdll").NtResumeProcess
        _NtResumeProcess.argtypes = [wintypes.HANDLE]
    except (OSError, AttributeError):
        _NtResumeProcess = None     # then never start one suspended

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMIT),
                    ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                             ctypes.c_void_p, wintypes.DWORD]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]

    _KILL_ON_JOB_CLOSE = 0x00002000
    _ExtendedLimitInformation = 9


class Job:
    """One kill-on-close job object. Falsy when the OS would not give us one."""

    def __init__(self):
        self.handle = None
        if not WINDOWS:
            return
        h = _k32.CreateJobObjectW(None, None)
        if not h:
            return
        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(h, _ExtendedLimitInformation,
                                            ctypes.byref(info), ctypes.sizeof(info)):
            _k32.CloseHandle(h)
            return
        self.handle = h

    def __bool__(self):
        return self.handle is not None

    def assign(self, proc):
        """Put a Popen in the job. False if it could not be done."""
        if not self.handle:
            return False
        return bool(_k32.AssignProcessToJobObject(self.handle, int(proc._handle)))

    def terminate(self):
        if self.handle:
            _k32.TerminateJobObject(self.handle, 1)

    def close(self):
        """Closing the last handle ends whatever is still in the job."""
        h, self.handle = self.handle, None
        if h:
            _k32.CloseHandle(h)


class Child:
    """A `Popen` and the job holding its tree. `proc` is the plain Popen, so
    anything that reads `.stdin`, `.stdout`, `.poll()` keeps working."""

    def __init__(self, args, creationflags=0, **kw):
        self.job = Job()
        suspend = bool(self.job) and _NtResumeProcess is not None
        if not WINDOWS:
            kw.setdefault("start_new_session", True)   # so stop() can kill the group
        self.proc = subprocess.Popen(
            args, creationflags=creationflags | (CREATE_SUSPENDED if suspend else 0), **kw)
        try:
            if self.job and not self.job.assign(self.proc):
                # Already in a job that forbids nesting (Windows 7). Keep the
                # child; stop() falls back to taskkill /T for its tree.
                self.job.close()
        finally:
            if suspend:
                _NtResumeProcess(int(self.proc._handle))
        with _live_lock:
            _live.add(self)

    @property
    def pid(self):
        return self.proc.pid

    def stop(self, grace=3.0):
        """Ask nicely, then end the whole tree. Safe to call twice.

        Closing stdin is the ask: an MCP bridge and the COM worker both exit
        on end of input. After `grace` seconds whatever is left - the child,
        and anything it started - is terminated, and every handle we hold on
        it is released."""
        p = self.proc
        try:
            if p.stdin:
                p.stdin.close()
        except Exception:
            pass
        if grace and p.poll() is None:
            try:
                p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        self.kill()

    def kill(self):
        """The whole tree, now: the job if there is one, else what we can."""
        p = self.proc
        if self.job:
            self.job.terminate()
        elif p.poll() is None:
            _kill_tree(p)
        try:
            p.wait(timeout=5)
        except Exception:
            pass
        for stream in (p.stdin, p.stdout, p.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass
        self.job.close()
        with _live_lock:
            _live.discard(self)


def _kill_tree(p):
    """Only for a child we could not put in a job."""
    try:
        if WINDOWS:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                           capture_output=True, timeout=10, creationflags=NO_WINDOW)
        else:
            os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        p.kill()
    except Exception:
        pass


def alive(pid):
    """Is `pid` running? Looks, never signals: on Windows `os.kill(pid, 0)`
    is not a probe, it is TerminateProcess."""
    if not WINDOWS:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    h = _k32.OpenProcess(0x00100000, False, pid)        # SYNCHRONIZE
    if not h:
        return False
    try:
        return _k32.WaitForSingleObject(h, 0) == 0x00000102  # WAIT_TIMEOUT
    finally:
        _k32.CloseHandle(h)


if WINDOWS:
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]


def spawn(args, **kw):
    """`Popen(args, **kw)`, contained. Returns a `Child`."""
    return Child(args, **kw)


def live():
    """Children started and not yet stopped - what the doctor and tests count."""
    with _live_lock:
        return [c for c in _live if c.proc.poll() is None]


def stop_all(grace=2.0):
    """Stop every live child at once, each given `grace` to exit on its own.
    In parallel: six tabs closing in turn used to be up to thirty seconds of a
    frozen window."""
    with _live_lock:
        children = list(_live)
    threads = [threading.Thread(target=c.stop, args=(grace,), daemon=True)
               for c in children]
    for t in threads:
        t.start()
    for t in threads:
        t.join(grace + 6)
    for c in children:             # anything whose stop() hung on a pipe
        try:
            c.kill()
        except Exception:
            pass


atexit.register(stop_all, 0.5)


def on_shutdown(callback):
    """Run `callback()` on Ctrl+C, Ctrl+Break and SIGTERM, in the main thread.

    Must be called from the main thread. The callback should start an orderly
    exit; the job objects are what make a disorderly one safe."""
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, lambda signum, frame: callback())
        except (ValueError, OSError):
            pass                   # not the main thread, or not settable here
