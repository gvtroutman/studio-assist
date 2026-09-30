#!/usr/bin/env python3
"""Run the tests where nobody has to watch them: on a Windows desktop of their own.

    python tests/offscreen.py discover -s tests --durations 60
    python tests/offscreen.py tests.test_scene

The arguments are `python -m unittest`'s. The GUI tests open real Tk windows
(a Chat, a Scene Builder, the consoles of the processes they start); run
plainly, each one appears over whatever the user is doing and takes the
keyboard. Started here, the test process and everything it starts live on a
second desktop of the same session that is never switched to: the windows are
real, laid out and drawn as always, and never shown. Output and the exit code
come back as they would from unittest itself.

Everything started is in a kill-on-close job (`core.procs.Job`), so stopping
this process ends the tests too: nothing is left running where it cannot be
seen. Off Windows it is plain `python -m unittest`.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import core.procs as procs  # noqa: E402

DESKTOP = "StudioAssistTests"
GENERIC_ALL = 0x10000000
STARTF_USESTDHANDLES = 0x00000100
INFINITE = 0xFFFFFFFF


def command(argv):
    return [sys.executable, "-m", "unittest"] + list(argv)


def run_unseen(argv):
    """`python -m unittest argv` on the tests' own desktop -> its exit code."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class STARTUPINFO(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
                    ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
                    ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                    ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
                    ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
                    ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                    ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                    ("lpReserved2", ctypes.c_void_p), ("hStdInput", wintypes.HANDLE),
                    ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE)]

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                    ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.CreateDesktopW.restype = wintypes.HANDLE
    user32.CreateDesktopW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p,
                                      wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    user32.CloseDesktop.argtypes = [wintypes.HANDLE]
    k32.CreateProcessW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p,
                                   ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD,
                                   ctypes.c_void_p, wintypes.LPCWSTR,
                                   ctypes.POINTER(STARTUPINFO),
                                   ctypes.POINTER(PROCESS_INFORMATION)]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.ResumeThread.argtypes = [wintypes.HANDLE]
    k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]

    # Opens the desktop if an earlier run (or one alongside) made it already.
    desk = user32.CreateDesktopW(DESKTOP, None, None, 0, GENERIC_ALL, None)
    if not desk:
        raise ctypes.WinError(ctypes.get_last_error())
    job = procs.Job()
    si = STARTUPINFO()
    si.cb = ctypes.sizeof(si)
    si.lpDesktop = DESKTOP
    si.dwFlags = STARTF_USESTDHANDLES
    handles = []
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            h = msvcrt.get_osfhandle(stream.fileno())
            os.set_handle_inheritable(h, True)
        except (OSError, ValueError, AttributeError):
            h = None                        # no such stream here: the child has none
        handles.append(h)
    si.hStdInput, si.hStdOutput, si.hStdError = handles
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    pi = PROCESS_INFORMATION()
    line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command(argv)))
    try:
        if not k32.CreateProcessW(None, line, None, None, True,
                                  procs.CREATE_SUSPENDED if job else 0, None, None,
                                  ctypes.byref(si), ctypes.byref(pi)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if job:
                k32.AssignProcessToJobObject(job.handle, pi.hProcess)
                k32.ResumeThread(pi.hThread)
            k32.WaitForSingleObject(pi.hProcess, INFINITE)
            code = wintypes.DWORD()
            k32.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
            return code.value
        finally:
            k32.CloseHandle(pi.hThread)
            k32.CloseHandle(pi.hProcess)
    finally:
        job.close()                          # whatever the tests left running ends here
        user32.CloseDesktop(desk)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not procs.WINDOWS:
        return subprocess.call(command(argv))
    return run_unseen(argv)


if __name__ == "__main__":
    sys.exit(main())
