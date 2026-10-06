"""Platform-specific path opening without shell command interpolation."""

import os
import subprocess
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from vd_test_log.platform_open import open_path


class PlatformOpenTests(unittest.TestCase):
    def test_macos_uses_open_with_one_path_argument(self):
        with patch("vd_test_log.platform_open.subprocess.run") as run:
            open_path(PurePosixPath("/tmp/a file.pdf"), platform="darwin")

        run.assert_called_once_with(
            ["open", "/tmp/a file.pdf"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_linux_uses_xdg_open_with_one_path_argument(self):
        with patch("vd_test_log.platform_open.subprocess.run") as run:
            open_path("/tmp/a file.pdf", platform="linux")

        run.assert_called_once_with(
            ["xdg-open", "/tmp/a file.pdf"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_windows_uses_startfile(self):
        with patch.object(os, "startfile", create=True) as startfile:
            open_path(Path("C:/a file.pdf"), platform="win32")

        startfile.assert_called_once_with(os.fspath(Path("C:/a file.pdf")))

    def test_external_opener_error_is_reported(self):
        failure = subprocess.CalledProcessError(
            4,
            ["open", "/tmp/a.pdf"],
            stderr="Launch Services denied the request",
        )
        with patch("vd_test_log.platform_open.subprocess.run", side_effect=failure):
            with self.assertRaisesRegex(OSError, "Launch Services denied"):
                open_path("/tmp/a.pdf", platform="darwin")

    def test_unknown_platform_is_reported(self):
        with self.assertRaisesRegex(OSError, "Unsupported platform"):
            open_path("/tmp/a.pdf", platform="plan9")


if __name__ == "__main__":
    unittest.main()
