"""Open a local file or folder with the user's configured desktop application."""

from __future__ import annotations

import os
import subprocess
import sys


def open_path(path: str | os.PathLike[str], *, platform: str | None = None) -> None:
    """Open ``path`` using the native shell for Windows, macOS, or Linux.

    ``platform`` is exposed for focused tests; normal callers use the host.
    External commands are passed as argument vectors so paths with spaces or
    shell metacharacters remain a single path argument.
    """
    target = os.fspath(path)
    host = sys.platform if platform is None else platform

    if host == "win32":
        opener = getattr(os, "startfile", None)
        if opener is None:
            raise OSError("Windows file opening is unavailable in this Python build.")
        opener(target)
        return

    if host == "darwin":
        command = "open"
    elif host.startswith("linux"):
        command = "xdg-open"
    else:
        raise OSError(f"Unsupported platform for opening files: {host}")

    try:
        subprocess.run(
            [command, target],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        if not detail:
            detail = f"{command} exited with status {error.returncode}"
        raise OSError(detail) from error
