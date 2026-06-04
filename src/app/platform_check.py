import platform
import sys

WINDOWS_10_1903_BUILD = 18362


def is_windows_10_1903_or_newer() -> bool:
    if sys.platform != "win32":
        return False

    version = sys.getwindowsversion()
    if version.major > 10:
        return True
    if version.major == 10 and version.build >= WINDOWS_10_1903_BUILD:
        return True
    return False


def windows_version_text() -> str:
    if sys.platform != "win32":
        return platform.platform()

    version = sys.getwindowsversion()
    return f"Windows {version.major}.{version.minor} build {version.build}"


def windows_version_notice_required() -> bool:
    return sys.platform == "win32" and not is_windows_10_1903_or_newer()