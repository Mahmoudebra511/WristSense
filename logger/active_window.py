"""Read the foreground window's process name and title, per operating system.

Only Windows is implemented for now (the project's first target, Section 11).
The interface is deliberately tiny so a macOS or Linux backend can be dropped
in later without touching the logger:

    active_window() -> ActiveWindow | None

Returns ``None`` when the information is unavailable (unsupported OS, no
permission, or a transient failure), which the recorder treats as "no change".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveWindow:
    handle: int
    pid: int
    process: str  # executable base name, e.g. "chrome.exe"
    title: str    # window title, used locally only, never stored


# ---------------------------------------------------------------------------
# Windows backend (ctypes, no third-party dependency)
# ---------------------------------------------------------------------------
if sys.platform == "win32":  # pragma: no cover - exercised only on Windows
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
    ]
    _kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def _process_name(pid: int) -> str:
        handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(1024)
            buf = ctypes.create_unicode_buffer(size.value)
            if _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return buf.value.rsplit("\\", 1)[-1]
            return ""
        finally:
            _kernel32.CloseHandle(handle)

    def _windows_active_window() -> ActiveWindow | None:
        hwnd = _user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = _user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return ActiveWindow(
            handle=int(hwnd),
            pid=int(pid.value),
            process=_process_name(int(pid.value)),
            title=buf.value,
        )

    def active_window() -> ActiveWindow | None:
        try:
            return _windows_active_window()
        except Exception:
            return None

else:  # pragma: no cover - other platforms are not implemented yet

    def active_window() -> ActiveWindow | None:
        return None
