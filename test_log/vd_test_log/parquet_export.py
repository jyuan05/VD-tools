"""Build typed whole-log Parquet tables from an immutable repository snapshot."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import stat
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .models import ParquetSnapshot


_STRING_OUTING_FIELDS = (
    "front_wing_height",
    "rw_setting",
    "front_spring_rate",
    "rear_spring_rate",
    "rear_arb_blade_setting",
    "rear_arb_motion_ratio_setting",
    "sprocket_size",
    "engine_tune",
)
_FLOAT_OUTING_FIELDS = (
    "front_damping_ratio",
    "rear_damping_ratio",
    "diff_ramp_angle",
    "diff_preload",
)
_CORNER_FIELDS = ("camber", "toe", "pressure", "corner_weight")
_CORNERS = ("FL", "FR", "RL", "RR")
_TABLE_FILENAMES = {
    "outings": "outings.parquet",
    "laps": "laps.parquet",
    "attachments": "attachments.parquet",
}


def _nullable_text(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _arrow_schemas(pa):
    def field(name, arrow_type, *, nullable=False):
        return pa.field(name, arrow_type, nullable=nullable)

    text = pa.string()
    integer = pa.int64()
    floating = pa.float64()
    outing_fields = [
        field("outing_uuid", text),
        field("test_day_uuid", text),
        field("test_date", pa.date32()),
        field("test_location", text),
        field("weather", text, nullable=True),
        field("test_day_notes", text, nullable=True),
        field("user_label", text),
        field("outing_id", text, nullable=True),
        field("settings_text", text),
        field("outing_notes", text, nullable=True),
        field("sort_order", integer),
        field("default_event_layout_uuid", text, nullable=True),
        field("default_track_name", text, nullable=True),
        field("default_layout_name", text, nullable=True),
        field("default_event_name", text, nullable=True),
        field("default_event_type", text, nullable=True),
        field("default_event_length_m", floating, nullable=True),
        field("default_driver", text, nullable=True),
        field("created_at", text),
    ]
    outing_fields.extend(
        field(name, text, nullable=True) for name in _STRING_OUTING_FIELDS[:6]
    )
    outing_fields.extend(field(name, floating, nullable=True) for name in _FLOAT_OUTING_FIELDS)
    outing_fields.extend(
        field(name, text, nullable=True) for name in _STRING_OUTING_FIELDS[6:]
    )
    outing_fields.extend(
        field(f"{corner}_{name}", floating, nullable=True)
        for corner in _CORNERS
        for name in _CORNER_FIELDS
    )

    lap_fields = [
        field("lap_uuid", text),
        field("outing_uuid", text),
        field("event_layout_uuid", text),
        field("sequence", integer),
        field("time_ms", integer),
        field("status", text),
        field("driver", text, nullable=True),
        field("time_of_day", text, nullable=True),
        field("notes", text, nullable=True),
        field("track_name", text),
        field("layout_name", text),
        field("event_name", text, nullable=True),
        field("event_type", text, nullable=True),
        field("length_m", floating, nullable=True),
        field("created_at", text),
    ]
    attachment_fields = [
        field("attachment_uuid", text),
        field("owner_type", text),
        field("owner_uuid", text),
        field("role", text),
        field("original_name", text),
        field("relative_path", text),
        field("created_at", text),
    ]
    return {
        "outings": pa.schema(outing_fields),
        "laps": pa.schema(lap_fields),
        "attachments": pa.schema(attachment_fields),
    }


def _outing_rows(snapshot: ParquetSnapshot) -> list[dict[str, Any]]:
    days_by_id = {day.id: day for day in snapshot.days}
    events_by_id = {event.id: event for event in snapshot.event_layouts}
    rows: list[dict[str, Any]] = []
    for setup in snapshot.setups:
        day = days_by_id.get(setup.test_day_id)
        if day is None:
            raise ValueError(f"Outing {setup.id} references a missing test day.")
        default_event = None
        if setup.event_layout_id is not None:
            default_event = events_by_id.get(setup.event_layout_id)
            if default_event is None:
                raise ValueError(f"Outing {setup.id} references a missing event/layout.")

        settings = json.loads(setup.structured_settings_json or "{}")
        row: dict[str, Any] = {
            "outing_uuid": setup.id,
            "test_day_uuid": day.id,
            "test_date": date.fromisoformat(day.date),
            "test_location": day.location,
            "weather": day.weather,
            "test_day_notes": day.notes,
            "user_label": setup.name,
            "outing_id": setup.setup_code,
            "settings_text": setup.settings_text,
            "outing_notes": setup.notes,
            "sort_order": setup.order,
            "default_event_layout_uuid": setup.event_layout_id,
            "default_track_name": default_event.track_name if default_event else None,
            "default_layout_name": default_event.layout_name if default_event else None,
            "default_event_name": default_event.event_name if default_event else None,
            "default_event_type": default_event.event_type if default_event else None,
            "default_event_length_m": default_event.length_m if default_event else None,
            "default_driver": setup.driver,
            "created_at": setup.created_at,
        }
        for name in _STRING_OUTING_FIELDS:
            value = settings.get(name)
            row[name] = _nullable_text(value) if name == "engine_tune" else value
        for name in _FLOAT_OUTING_FIELDS:
            row[name] = settings.get(name)
        corners = settings.get("corners", {})
        for corner in _CORNERS:
            corner_settings = corners.get(corner, {}) if isinstance(corners, dict) else {}
            for name in _CORNER_FIELDS:
                row[f"{corner}_{name}"] = corner_settings.get(name)
        rows.append(row)
    return rows


def _lap_rows(snapshot: ParquetSnapshot) -> list[dict[str, Any]]:
    setups_by_id = {setup.id: setup for setup in snapshot.setups}
    events_by_id = {event.id: event for event in snapshot.event_layouts}
    rows: list[dict[str, Any]] = []
    for lap in snapshot.laps:
        setup = setups_by_id.get(lap.setup_id)
        event = events_by_id.get(lap.event_layout_id)
        if setup is None or event is None:
            raise ValueError(f"Lap {lap.id} references a missing outing or event/layout.")
        rows.append(
            {
                "lap_uuid": lap.id,
                "outing_uuid": setup.id,
                "event_layout_uuid": event.id,
                "sequence": lap.sequence,
                "time_ms": lap.time_ms,
                "status": lap.status,
                "driver": lap.driver,
                "time_of_day": lap.time_of_day,
                "notes": lap.notes,
                "track_name": event.track_name,
                "layout_name": event.layout_name,
                "event_name": event.event_name,
                "event_type": event.event_type,
                "length_m": event.length_m,
                "created_at": lap.created_at,
            }
        )
    return rows


def _attachment_rows(snapshot: ParquetSnapshot) -> list[dict[str, Any]]:
    rows = []
    for attachment in snapshot.attachments:
        owner_type = "outing" if attachment.owner_type == "setup" else attachment.owner_type
        rows.append(
            {
                "attachment_uuid": attachment.id,
                "owner_type": owner_type,
                "owner_uuid": attachment.owner_id,
                "role": attachment.role,
                "original_name": attachment.original_name,
                "relative_path": attachment.relative_path,
                "created_at": attachment.created_at,
            }
        )
    return rows


def build_parquet_tables(snapshot: ParquetSnapshot) -> dict[str, object]:
    """Build explicit-schema Arrow tables, importing PyArrow only on demand."""
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError(
            "Parquet export requires optional PyArrow. From the test_log folder, run "
            "`python -m pip install -r requirements-parquet.txt` using the same Python "
            "interpreter that launches this app, then retry."
        ) from error

    rows_by_name = {
        "outings": _outing_rows(snapshot),
        "laps": _lap_rows(snapshot),
        "attachments": _attachment_rows(snapshot),
    }
    schemas = _arrow_schemas(pa)
    return {
        name: pa.Table.from_pylist(rows, schema=schemas[name])
        for name, rows in rows_by_name.items()
    }


def _write_table(table: object, destination: Path) -> None:
    """Write one explicit-schema Arrow table using the optional Parquet engine."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Parquet export requires optional PyArrow. From the test_log folder, run "
            "`python -m pip install -r requirements-parquet.txt` using the same Python "
            "interpreter that launches this app, then retry."
        ) from error
    pq.write_table(table, destination, version="1.0", compression="snappy")


def _lexists(path: Path) -> bool:
    """Return whether a directory entry exists, including a dangling symlink."""
    return os.path.lexists(os.fspath(path)) or path.is_symlink()


def _manifest(tables: dict[str, object]) -> dict[str, Any]:
    return {
        "dataset_format": "vd-test-log-parquet",
        "dataset_format_version": 1,
        "tables": {
            name: {
                "file": _TABLE_FILENAMES[name],
                "row_count": table.num_rows,
            }
            for name, table in tables.items()
        },
        "units": {
            "test_date": "ISO 8601 calendar date, stored as Arrow date32",
            "default_event_length_m": "m",
            "length_m": "m",
            "front_wing_height": "choice (1 = lowest)",
            "rw_setting": "choice (1 = highest downforce)",
            "front_spring_rate": "lb/in",
            "rear_spring_rate": "lb/in",
            "front_damping_ratio": "dimensionless",
            "rear_damping_ratio": "dimensionless",
            "rear_arb_blade_setting": "choice (1 = stiffest, 6 = softest)",
            "rear_arb_motion_ratio_setting": "choice (MR1 or MR2)",
            "diff_ramp_angle": "degrees",
            "diff_preload": "ft-lb",
            "sprocket_size": "teeth text",
            "engine_tune": "tune name/version text",
            **{
                f"{corner}_{field}": unit
                for corner in _CORNERS
                for field, unit in (
                    ("camber", "degrees"),
                    ("toe", "degrees"),
                    ("pressure", "PSI"),
                    ("corner_weight", "lb"),
                )
            },
            "time_ms": "ms (integer)",
            "sequence": "lap order (integer)",
        },
        "attachments": {
            "content_included": False,
            "relative_path_base": "source application data folder",
        },
    }


def _remove_owned_staging(staging: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        info = staging.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        return
    shutil.rmtree(staging)


def _atomic_publish_oserror(error_number: int, destination: Path, platform: str) -> OSError:
    destination_name = os.fspath(destination)
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        return FileExistsError(
            error_number,
            "The Parquet dataset destination appeared during export and was left untouched.",
            destination_name,
        )

    unsupported_errors = {
        getattr(errno, name)
        for name in ("EINVAL", "ENOSYS", "ENOTSUP", "EOPNOTSUPP")
        if hasattr(errno, name)
    }
    if error_number in unsupported_errors:
        message = (
            f"Atomic no-overwrite directory publication is not supported by {platform} "
            "or this filesystem. Choose a supported local filesystem and a different output folder."
        )
        return OSError(error_number, message, destination_name)
    return OSError(error_number, os.strerror(error_number), destination_name)


def _rename_directory_noreplace(
    source: Path,
    destination: Path,
    *,
    platform: str | None = None,
    libc: Any | None = None,
) -> None:
    """Atomically publish a directory only when its destination is absent."""
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        # Python's Windows os.rename fails when dst exists; os.replace would not.
        os.rename(source, destination)
        return

    if platform == "darwin":
        symbol = "renamex_np"
        argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (os.fsencode(source), os.fsencode(destination), 0x00000004)  # RENAME_EXCL
    elif platform.startswith("linux"):
        symbol = "renameat2"
        argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        arguments = (
            -100,  # AT_FDCWD
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            0x00000001,  # RENAME_NOREPLACE
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            f"Atomic no-overwrite directory publication is unavailable on {platform}; "
            "choose a supported local filesystem and a different output folder.",
            os.fspath(destination),
        )

    if libc is None:
        libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = getattr(libc, symbol)
    except AttributeError as error:
        raise OSError(
            errno.ENOTSUP,
            f"Atomic no-overwrite directory publication is unavailable on {platform}; "
            "choose a supported local filesystem and a different output folder.",
            os.fspath(destination),
        ) from error

    rename.argtypes = argtypes
    rename.restype = ctypes.c_int
    ctypes.set_errno(0)
    if rename(*arguments) != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise _atomic_publish_oserror(error_number, destination, platform)


def write_parquet_dataset(
    snapshot: ParquetSnapshot,
    destination: Path,
    *,
    guard_destination: Callable[[Path], None],
) -> Path:
    """Write a complete new dataset beside its target, then publish it atomically."""
    destination = Path(destination).expanduser()
    guard_destination(destination)
    if _lexists(destination):
        raise FileExistsError(f"The Parquet dataset destination already exists: {destination}")
    if not destination.name or destination.name in {".", ".."}:
        raise ValueError("Choose a named directory for the Parquet dataset.")

    # Load PyArrow and materialize the immutable snapshot before creating output paths.
    tables = build_parquet_tables(snapshot)
    parent = destination.parent
    staging = parent / f".{destination.name}.staging-{uuid4().hex}"
    guard_destination(staging)
    parent.mkdir(parents=True, exist_ok=True)
    guard_destination(staging)

    staging_identity: tuple[int, int] | None = None
    try:
        staging.mkdir()
        info = staging.lstat()
        staging_identity = (info.st_dev, info.st_ino)
        guard_destination(staging)

        for name, table in tables.items():
            _write_table(table, staging / _TABLE_FILENAMES[name])

        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(_manifest(tables), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

        guard_destination(destination)
        if _lexists(destination):
            raise FileExistsError(f"The Parquet dataset destination already exists: {destination}")
        _rename_directory_noreplace(staging, destination)
        staging_identity = None
        return destination
    except BaseException as error:
        try:
            _remove_owned_staging(staging, staging_identity)
        except OSError as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(f"Could not remove exporter-owned staging directory: {cleanup_error}")
        raise
