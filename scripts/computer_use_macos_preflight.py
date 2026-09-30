#!/usr/bin/env python3
"""Read-only macOS GUI-session, display and permission readiness report."""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class Size(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class Rect(ctypes.Structure):
    _fields_ = [("origin", Point), ("size", Size)]


def inspect():
    report = {
        "kind": "macos_computer_use_preflight",
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": sys.platform,
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "invoked_over_ssh": bool(os.environ.get("SSH_CONNECTION")),
        "preflight_checks_passed": False,
        "desktop_actions_tested": False,
        "screenshot_coordinate_mapping_tested": False,
        "scope": "Checks this process identity; another launcher or signed app needs its own permission check and desktop tests.",
    }
    if sys.platform != "darwin":
        report["error"] = "Run this script on the Mac selected for desktop testing."
        return report
    try:
        report["macos_version"] = subprocess.check_output(
            ["/usr/bin/sw_vers", "-productVersion"], text=True, timeout=5).strip()
        report["console_uid"] = Path("/dev/console").stat().st_uid
        report["effective_uid"] = os.geteuid()
        report["same_console_user"] = (report["console_uid"] == report["effective_uid"]
                                       and report["effective_uid"] != 0)
        cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        ax = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        cg.CGPreflightScreenCaptureAccess.argtypes = []
        cg.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
        ax.AXIsProcessTrusted.argtypes = []
        ax.AXIsProcessTrusted.restype = ctypes.c_bool
        report["screen_capture_allowed"] = bool(cg.CGPreflightScreenCaptureAccess())
        report["accessibility_trusted"] = bool(ax.AXIsProcessTrusted())
        cg.CGGetActiveDisplayList.argtypes = [ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
        cg.CGGetActiveDisplayList.restype = ctypes.c_int32
        cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]
        cg.CGDisplayBounds.restype = Rect
        for name in ("CGDisplayPixelsWide", "CGDisplayPixelsHigh"):
            method = getattr(cg, name)
            method.argtypes = [ctypes.c_uint32]
            method.restype = ctypes.c_size_t
        cg.CGDisplayIsMain.argtypes = [ctypes.c_uint32]
        cg.CGDisplayIsMain.restype = ctypes.c_uint32
        cg.CGDisplayCopyDisplayMode.argtypes = [ctypes.c_uint32]
        cg.CGDisplayCopyDisplayMode.restype = ctypes.c_void_p
        cg.CGDisplayModeRelease.argtypes = [ctypes.c_void_p]
        cg.CGDisplayModeRelease.restype = None
        mode_dimensions = {"width": "CGDisplayModeGetWidth", "height": "CGDisplayModeGetHeight",
                           "pixel_width": "CGDisplayModeGetPixelWidth", "pixel_height": "CGDisplayModeGetPixelHeight"}
        for name in mode_dimensions.values():
            method = getattr(cg, name)
            method.argtypes = [ctypes.c_void_p]
            method.restype = ctypes.c_size_t
        identifiers = (ctypes.c_uint32 * 64)()
        count = ctypes.c_uint32()
        error = cg.CGGetActiveDisplayList(64, identifiers, ctypes.byref(count))
        if error:
            raise RuntimeError("CGGetActiveDisplayList returned " + str(error))
        report["displays"] = []
        for identifier in identifiers[:min(count.value, 64)]:
            bounds = cg.CGDisplayBounds(identifier)
            pixels = {"width": cg.CGDisplayPixelsWide(identifier),
                      "height": cg.CGDisplayPixelsHigh(identifier)}
            mode = cg.CGDisplayCopyDisplayMode(identifier)
            try:
                dimensions = ({key: getattr(cg, method)(mode) for key, method in mode_dimensions.items()}
                              if mode else None)
            finally:
                if mode:
                    cg.CGDisplayModeRelease(mode)
            report["displays"].append({
                "id": identifier, "main": bool(cg.CGDisplayIsMain(identifier)),
                "logical_bounds": {"x": bounds.origin.x, "y": bounds.origin.y,
                    "width": bounds.size.width, "height": bounds.size.height},
                "display_reported_pixels": pixels,
                "active_mode": dimensions,
            })
        report["display_list_may_be_truncated"] = count.value >= 64
        report["preflight_checks_passed"] = bool(
            report["same_console_user"] and report["screen_capture_allowed"]
            and report["accessibility_trusted"] and report["displays"]
            and not report["display_list_may_be_truncated"])
    except (OSError, AttributeError, RuntimeError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optionally also save this readiness report")
    args = parser.parse_args()
    report = inspect()
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["preflight_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
