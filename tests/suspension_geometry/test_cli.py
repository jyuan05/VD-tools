"""Subprocess checks for the installed and module-based suspension CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "suspension_geometry"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)


def test_installed_entrypoint_and_module_help_show_validate_command() -> None:
    for command in (
        ("vd-suspension", "--help"),
        (sys.executable, "-m", "suspension_geometry.cli", "--help"),
    ):
        result = _run(*command)
        assert result.returncode == 0, result.stderr
        assert "validate" in result.stdout
        validate_help = _run(*command[:-1], "validate", "--help")
        assert validate_help.returncode == 0, validate_help.stderr
        assert "--kind" in validate_help.stdout


def test_geometry_and_resolved_study_configs_validate() -> None:
    geometry = CONFIGS / "geometries" / "synthetic_direct.yaml"
    study = CONFIGS / "studies" / "direct_heave.yaml"

    geometry_result = _run("vd-suspension", "validate", str(geometry))
    assert geometry_result.returncode == 0, geometry_result.stderr
    assert "geometry" in geometry_result.stdout.lower()

    study_result = _run("vd-suspension", "validate", str(study), "--kind", "study")
    assert study_result.returncode == 0, study_result.stderr
    assert "study" in study_result.stdout.lower()


def test_setup_validation_accepts_matching_geometry() -> None:
    setup = CONFIGS / "setups" / "synthetic_direct_loaded.yaml"
    geometry = CONFIGS / "geometries" / "synthetic_direct.yaml"

    result = _run(
        "vd-suspension",
        "validate",
        str(setup),
        "--kind",
        "setup",
        "--geometry",
        str(geometry),
    )

    assert result.returncode == 0, result.stderr
    assert "setup" in result.stdout.lower()


def test_invalid_yaml_reports_path_without_traceback(tmp_path: Path) -> None:
    malformed = tmp_path / "broken.yaml"
    malformed.write_text("schema_version: [\n", encoding="utf-8")

    result = _run("vd-suspension", "validate", str(malformed))

    assert result.returncode != 0
    assert str(malformed) in result.stderr
    assert "invalid YAML" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_config_reports_path_without_traceback(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    result = _run("vd-suspension", "validate", str(missing))

    assert result.returncode != 0
    assert str(missing) in result.stderr
    assert "Traceback" not in result.stderr


def test_unavailable_run_and_plot_fail_without_creating_artifacts(tmp_path: Path) -> None:
    study = CONFIGS / "studies" / "direct_heave.yaml"
    output = tmp_path / "not-created"

    for command in ("run", "plot"):
        result = _run("vd-suspension", command, str(study), "--output", str(output))
        assert result.returncode != 0
        assert "not implemented" in result.stderr.lower()
        assert "Traceback" not in result.stderr
        assert not output.exists()
