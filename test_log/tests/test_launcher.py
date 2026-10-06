"""Portable shell launcher checks using a mocked Python command."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT / "launch.sh"
FINDER_LAUNCHER = PROJECT / "launch.command"


def find_posix_shell() -> str | None:
    shell = shutil.which("sh") or shutil.which("bash")
    if shell:
        return shell
    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        git_bash = program_files / "Git" / "bin" / "bash.exe"
        if git_bash.is_file():
            return str(git_bash)
    return None


def shell_path(path: Path, shell: str) -> str:
    if os.name != "nt":
        return str(path)
    cygpath = Path(shell).parent.parent / "usr" / "bin" / "cygpath.exe"
    result = subprocess.run(
        [str(cygpath), "-u", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@unittest.skipUnless(find_posix_shell(), "a POSIX shell is required for launcher tests")
class PortableLauncherTests(unittest.TestCase):
    def test_shell_syntax_and_lf_line_endings(self):
        content = LAUNCHER.read_bytes()
        self.assertTrue(content.startswith(b"#!/bin/sh\n"))
        self.assertNotIn(b"\r", content)
        result = subprocess.run(
            [find_posix_shell(), "-n", shell_path(LAUNCHER, find_posix_shell())],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_finder_launcher_has_valid_posix_syntax_and_lf_line_endings(self):
        content = FINDER_LAUNCHER.read_bytes()
        self.assertTrue(content.startswith(b"#!/bin/sh\n"))
        self.assertNotIn(b"\r", content)
        result = subprocess.run(
            [find_posix_shell(), "-n", shell_path(FINDER_LAUNCHER, find_posix_shell())],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_launcher_preserves_caller_folder_and_loads_from_its_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app_folder = root / "vehicle log app"
            app_folder.mkdir()
            launcher = app_folder / "launch.sh"
            shutil.copyfile(LAUNCHER, launcher)
            fake_bin = root / "fake python bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3.12"
            fake_python.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"-c\" ]; then exit 0; fi\n"
                "printf 'cwd=%s\\n' \"$PWD\"\n"
                "printf 'pythonpath=%s\\n' \"$PYTHONPATH\"\n"
                "for argument do printf 'arg=%s\\n' \"$argument\"; done\n",
                encoding="utf-8",
                newline="\n",
            )
            fake_python.chmod(0o755)
            foreign_folder = root / "different working folder"
            foreign_folder.mkdir()
            environment = os.environ.copy()
            environment["PATH"] = str(fake_bin) + os.pathsep + environment.get("PATH", "")
            data_dir = "Driver A's test data"

            result = subprocess.run(
                [
                    find_posix_shell(),
                    shell_path(launcher, find_posix_shell()),
                    "--help",
                    "--data-dir",
                    data_dir,
                ],
                cwd=foreign_folder,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertTrue(
                lines[0].replace("\\", "/").endswith("/different working folder"),
                msg=f"launcher output was {result.stdout!r}; stderr was {result.stderr!r}",
            )
            self.assertTrue(
                lines[1].replace("\\", "/").split(":", 1)[0].endswith("/vehicle log app"),
                msg=f"launcher output was {result.stdout!r}; stderr was {result.stderr!r}",
            )
            self.assertEqual(
                lines[2:],
                [
                    "arg=-m",
                    "arg=vd_test_log",
                    "arg=--help",
                    "arg=--data-dir",
                    f"arg={data_dir}",
                ],
            )

    def test_finder_launcher_forwards_arguments_and_preserves_caller_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app_folder = root / "driver's vehicle log app"
            app_folder.mkdir()
            shutil.copyfile(FINDER_LAUNCHER, app_folder / "launch.command")
            shutil.copyfile(LAUNCHER, app_folder / "launch.sh")
            fake_bin = root / "fake python bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3.12"
            fake_python.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"-c\" ]; then exit 0; fi\n"
                "printf 'cwd=%s\\n' \"$PWD\"\n"
                "printf 'pythonpath=%s\\n' \"$PYTHONPATH\"\n"
                "for argument do printf 'arg=%s\\n' \"$argument\"; done\n",
                encoding="utf-8",
                newline="\n",
            )
            fake_python.chmod(0o755)
            foreign_folder = root / "different working folder"
            foreign_folder.mkdir()
            environment = os.environ.copy()
            environment["PATH"] = str(fake_bin) + os.pathsep + environment.get("PATH", "")
            data_dir = "Driver A's test data"

            result = subprocess.run(
                [
                    find_posix_shell(),
                    shell_path(app_folder / "launch.command", find_posix_shell()),
                    "--data-dir",
                    data_dir,
                ],
                cwd=foreign_folder,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertTrue(
                lines[0].replace("\\", "/").endswith("/different working folder"),
                msg=f"launcher output was {result.stdout!r}; stderr was {result.stderr!r}",
            )
            self.assertTrue(
                lines[1].replace("\\", "/").split(":", 1)[0].endswith("/driver's vehicle log app"),
                msg=f"launcher output was {result.stdout!r}; stderr was {result.stderr!r}",
            )
            self.assertEqual(
                lines[2:],
                [
                    "arg=-m",
                    "arg=vd_test_log",
                    "arg=--data-dir",
                    f"arg={data_dir}",
                ],
            )
            self.assertNotIn("Press Return", result.stdout + result.stderr)

    def test_finder_launcher_reports_failure_without_waiting_on_piped_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app_folder = root / "vehicle log app"
            app_folder.mkdir()
            shutil.copyfile(FINDER_LAUNCHER, app_folder / "launch.command")
            shutil.copyfile(LAUNCHER, app_folder / "launch.sh")
            fake_bin = root / "fake python bin"
            fake_bin.mkdir()
            for candidate in ("python3.12", "python3"):
                fake_python = fake_bin / candidate
                fake_python.write_text(
                    "#!/bin/sh\nexit 1\n",
                    encoding="utf-8",
                    newline="\n",
                )
                fake_python.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = str(fake_bin) + os.pathsep + environment.get("PATH", "")

            result = subprocess.run(
                [find_posix_shell(), shell_path(app_folder / "launch.command", find_posix_shell())],
                cwd=root,
                env=environment,
                input="",
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Python 3.12 or newer with Tkinter and sqlite3 is required.", result.stderr)
            self.assertIn("could not be started", result.stderr)
            self.assertNotIn("Press Return", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
