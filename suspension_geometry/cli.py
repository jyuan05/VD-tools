"""Command-line configuration validation for suspension geometry studies."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .config import ConfigError, load_geometry, load_setup, resolve_study


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vd-suspension",
        description="Validate suspension geometry, setup, and study configuration files.",
        epilog=(
            "Study validation resolves its geometry and setup files and checks their schemas, units, "
            "references, and sample limits. It does not compute numeric travel or rate maps, check "
            "metric or plot names for supported values, or create plots and exports."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a configuration file")
    validate.add_argument("path", help="YAML configuration file")
    validate.add_argument(
        "--kind",
        choices=("geometry", "setup", "study"),
        default="geometry",
        help="configuration type (default: geometry)",
    )
    validate.add_argument(
        "--geometry",
        metavar="GEOMETRY_PATH",
        help="geometry file used to validate active setup anti-roll bars",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the available CLI command and return its process status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"run", "plot"}:
        print(
            f"vd-suspension: {arguments[0]} is not implemented yet; configuration validation is available.",
            file=sys.stderr,
        )
        return 2

    parser = _parser()
    options = parser.parse_args(arguments)
    if options.command != "validate":
        parser.error("unsupported command")
    if options.geometry is not None and options.kind != "setup":
        parser.error("--geometry can only be used with --kind setup")

    try:
        if options.kind == "geometry":
            load_geometry(options.path)
        elif options.kind == "setup":
            geometry = load_geometry(options.geometry) if options.geometry is not None else None
            load_setup(options.path, geometry=geometry)
        else:
            resolve_study(options.path)
    except ConfigError as error:
        print(f"vd-suspension: {error}", file=sys.stderr)
        return 2

    print(f"Valid {options.kind} configuration structure: {options.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
