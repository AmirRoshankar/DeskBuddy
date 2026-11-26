"""Double-click entry point; read-only startup, with no torque restoration."""

import ctypes
import os
from pathlib import Path

from .cli import main


def process_running(pid):
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return (
                bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                and code.value == 259
            )
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def launch():
    import sys

    marker = Path("runs/live/process_id.txt")
    if marker.exists():
        try:
            previous = int(marker.read_text(encoding="utf-8-sig").strip())
            if previous != os.getpid() and process_running(previous):
                print(
                    "DeskBuddy is already running. Use the open SO-101 window.",
                    flush=True,
                )
                return
        except (ValueError, OSError):
            pass
    if not Path("configs/local.json").is_file():
        raise RuntimeError("Local configuration missing: configs/local.json")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(os.getpid()), encoding="utf-8")
    sys.argv = [
        "deskbuddy",
        "camera",
        "--config",
        "configs/local.json",
        "--robot-state",
    ]
    print(
        "Opening DeskBuddy. Startup reads camera and encoders; it does not enable torque or move the arm.",
        flush=True,
    )
    try:
        main()
    finally:
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == str(
            os.getpid()
        ):
            marker.unlink()


if __name__ == "__main__":
    launch()
