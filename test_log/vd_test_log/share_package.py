"""Portable JSON-and-blob packages for sharing offline test logs.

The public records and callable signatures in this module are shared with the
log-merge backend. Package blobs use generated ZIP member names rather than
paths from the source machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Callable, TYPE_CHECKING
import zlib
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from .setup_settings import normalise_setup_settings_json
from .validation import (
    ValidationError,
    validate_day,
    validate_event_layout,
    validate_lap,
    validate_setup,
    validate_staged_attachment,
)

from .models import Attachment, EventLayout, Lap, Setup, StagedAttachment, TestDay

if TYPE_CHECKING:
    from .attachments import AttachmentStore
    from .repository import SQLiteRepository


MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ATTACHMENT_BYTES = 512 * 1024 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 2 * 1024 * 1024 * 1024
MAX_RECORD_COUNT = 250_000
MAX_MEMBER_COUNT = 50_000
STREAM_CHUNK_BYTES = 1024 * 1024
PACKAGE_FORMAT = "vd-test-log-share-package"
PACKAGE_VERSION = 1


class PackageError(ValueError):
    """A package is invalid, incomplete, unsafe, or could not be written."""


class PackageConflictError(PackageError):
    """A package import conflicts with records already present in the log."""


class PackageChangedError(PackageError):
    """A package changed after the user previewed it."""


@dataclass(frozen=True, slots=True)
class PackageCounts:
    days: int
    setups: int
    laps: int
    event_layouts: int
    attachments: int


@dataclass(frozen=True, slots=True)
class PackageConflict:
    kind: str
    record_id: str
    description: str


@dataclass(frozen=True, slots=True)
class ImportPreview:
    source_label: str
    fingerprint: str
    incoming: PackageCounts
    added: PackageCounts
    skipped: PackageCounts
    conflicts: tuple[PackageConflict, ...]
    separate_copy: bool


@dataclass(frozen=True, slots=True)
class ImportResult:
    source_label: str
    fingerprint: str
    added: PackageCounts
    skipped: PackageCounts
    backup_path: Path | None


@dataclass(frozen=True, slots=True)
class PackageAttachment:
    record: Attachment
    member: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LogSnapshot:
    days: tuple[TestDay, ...]
    setups: tuple[Setup, ...]
    laps: tuple[Lap, ...]
    event_layouts: tuple[EventLayout, ...]
    attachments: tuple[Attachment, ...]


@dataclass(frozen=True, slots=True)
class LoadedPackage:
    source: Path
    source_label: str
    fingerprint: str
    days: tuple[TestDay, ...]
    setups: tuple[Setup, ...]
    laps: tuple[Lap, ...]
    event_layouts: tuple[EventLayout, ...]
    attachments: tuple[PackageAttachment, ...]


def capture_log(repository: SQLiteRepository) -> LogSnapshot:
    """Return all saved records, including archived events and attachments."""
    days = tuple(repository.list_days())
    setups = tuple(
        setup
        for day in days
        for setup in repository.list_setups(day.id)
    )
    laps = tuple(
        lap
        for setup in setups
        for lap in repository.list_laps(setup.id)
    )
    event_layouts = tuple(repository.list_event_layouts(include_archived=True))
    attachments = tuple(
        attachment
        for owner_type, identifiers in (
            ("day", (record.id for record in days)),
            ("setup", (record.id for record in setups)),
            ("lap", (record.id for record in laps)),
            ("event_layout", (record.id for record in event_layouts)),
        )
        for identifier in identifiers
        for attachment in repository.list_attachments(owner_type, identifier)
    )
    snapshot = LogSnapshot(days, setups, laps, event_layouts, attachments)
    try:
        _validate_records(snapshot.days, snapshot.setups, snapshot.laps,
                          snapshot.event_layouts, snapshot.attachments)
    except PackageError:
        raise
    return snapshot


def read_package(source: Path) -> LoadedPackage:
    """Read and fully validate a portable log package without extracting it."""
    package_path = Path(source).expanduser()
    try:
        if not package_path.is_file():
            raise PackageError("Choose an existing log package file.")
        with ZipFile(package_path, "r") as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBER_COUNT:
                raise PackageError("The package contains too many archive members.")
            info_by_name: dict[str, ZipInfo] = {}
            for info in infos:
                _validate_zip_info(info)
                if info.filename in info_by_name:
                    raise PackageError(f"The package contains a duplicate member: {info.filename}")
                info_by_name[info.filename] = info
            manifest_info = info_by_name.get("manifest.json")
            if manifest_info is None:
                raise PackageError("The package is missing its manifest.json file.")
            if manifest_info.file_size > MAX_MANIFEST_BYTES:
                raise PackageError("The package manifest is larger than the supported limit.")
            manifest_bytes, manifest_size, _ = _read_member(
                archive, manifest_info, max_bytes=MAX_MANIFEST_BYTES
            )
            if manifest_size != manifest_info.file_size:
                raise PackageError("The package manifest size is inconsistent.")
            manifest = _decode_manifest(manifest_bytes)

            days = _parse_records(manifest["days"], TestDay, "days", validate_day)
            setups = _parse_records(manifest["setups"], Setup, "setups", validate_setup)
            laps = _parse_records(manifest["laps"], Lap, "laps", validate_lap)
            event_layouts = _parse_records(
                manifest["event_layouts"],
                EventLayout,
                "event_layouts",
                validate_event_layout,
            )
            attachments = _parse_package_attachments(manifest["attachments"])
            expected_members = {"manifest.json", *(item.member for item in attachments)}
            if set(info_by_name) != expected_members:
                unexpected = sorted(set(info_by_name) - expected_members)
                missing = sorted(expected_members - set(info_by_name))
                if unexpected:
                    raise PackageError(f"The package contains an unknown archive member: {unexpected[0]}")
                raise PackageError(f"The package is missing archive member: {missing[0]}")
            if len(info_by_name) != len(attachments) + 1:
                raise PackageError("The package has an invalid number of archive members.")
            _validate_records(days, setups, laps, event_layouts, (item.record for item in attachments))

            declared_total = 0
            actual_total = 0
            for index, item in enumerate(attachments):
                if item.member != _blob_member(index):
                    raise PackageError("The package uses an invalid or non-generated blob member name.")
                if item.size > MAX_ATTACHMENT_BYTES:
                    raise PackageError(f"An attachment exceeds the {MAX_ATTACHMENT_BYTES}-byte limit.")
                declared_total += item.size
                if declared_total > MAX_TOTAL_ATTACHMENT_BYTES:
                    raise PackageError("The package attachments exceed the total size limit.")
                info = info_by_name[item.member]
                if info.file_size > MAX_ATTACHMENT_BYTES:
                    raise PackageError(f"An attachment exceeds the {MAX_ATTACHMENT_BYTES}-byte limit.")
                if info.file_size != item.size:
                    raise PackageError(f"Attachment {item.record.id} has an inconsistent declared size.")
                _, actual_size, actual_hash = _read_member(
                    archive,
                    info,
                    max_bytes=min(MAX_ATTACHMENT_BYTES, MAX_TOTAL_ATTACHMENT_BYTES - actual_total),
                    collect=False,
                )
                actual_total += actual_size
                if actual_size != item.size:
                    raise PackageError(f"Attachment {item.record.id} has an invalid size.")
                if actual_hash != item.sha256:
                    raise PackageError(f"Attachment {item.record.id} failed its SHA-256 check.")

        fingerprint = _fingerprint(days, setups, laps, event_layouts, attachments)
        return LoadedPackage(
            source=package_path,
            source_label=manifest["source_label"],
            fingerprint=fingerprint,
            days=days,
            setups=setups,
            laps=laps,
            event_layouts=event_layouts,
            attachments=attachments,
        )
    except PackageError:
        raise
    except (BadZipFile, OSError, RuntimeError, EOFError, ValueError, TypeError, zlib.error) as error:
        raise PackageError(f"Could not read log package: {error}") from error


def write_package(
    destination: Path,
    snapshot: LogSnapshot,
    store: AttachmentStore,
    source_label: str = "",
    *,
    guard_destination: Callable[[Path], None] | None = None,
) -> Path:
    """Write an atomic JSON-and-blob ZIP package beside its destination."""
    target = Path(destination).expanduser()
    if not target.is_absolute():
        target = Path(os.path.abspath(target))
    if not isinstance(source_label, str):
        raise PackageError("The optional source label must be text.")
    if guard_destination is not None:
        guard_destination(target)
    if not target.parent.is_dir():
        raise PackageError("Choose a destination inside an existing folder.")
    if target.exists() and target.is_dir():
        raise PackageError("Choose a package filename, not a folder.")

    try:
        record_arrays = _snapshot_record_arrays(snapshot)
        _validate_records(
            snapshot.days,
            snapshot.setups,
            snapshot.laps,
            snapshot.event_layouts,
            snapshot.attachments,
        )
        if len(snapshot.attachments) + 1 > MAX_MEMBER_COUNT:
            raise PackageError("The log contains too many records or attachments for one package.")
        package_attachments: list[PackageAttachment] = []
        source_paths: list[Path] = []
        total_size = 0
        for index, attachment in enumerate(sorted(snapshot.attachments, key=_attachment_sort_key)):
            try:
                source_path = store.resolve(attachment)
                size, digest = _hash_file(source_path, MAX_ATTACHMENT_BYTES)
            except (OSError, ValueError) as error:
                raise PackageError(
                    f"Could not read attachment {attachment.original_name!r}: {error}"
                ) from error
            total_size += size
            if total_size > MAX_TOTAL_ATTACHMENT_BYTES:
                raise PackageError("The log attachments exceed the total package size limit.")
            package_attachments.append(
                PackageAttachment(attachment, _blob_member(index), size, digest)
            )
            source_paths.append(source_path)

        manifest = {
            "format": PACKAGE_FORMAT,
            "version": PACKAGE_VERSION,
            "source_label": source_label,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            **record_arrays,
            "attachments": [
                {
                    "record": asdict(item.record),
                    "member": item.member,
                    "size": item.size,
                    "sha256": item.sha256,
                }
                for item in package_attachments
            ],
        }
        manifest_bytes = _encode_json(manifest)
        if len(manifest_bytes) > MAX_MANIFEST_BYTES:
            raise PackageError("The log manifest exceeds the supported size limit.")
        if len(package_attachments) + 1 > MAX_MEMBER_COUNT:
            raise PackageError("The log contains too many records or attachments for one package.")

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w+b") as output_stream:
                with ZipFile(
                    output_stream,
                    "w",
                    compression=ZIP_DEFLATED,
                    allowZip64=True,
                ) as archive:
                    archive.writestr("manifest.json", manifest_bytes)
                    written_total = 0
                    for item, source_path in zip(package_attachments, source_paths):
                        with source_path.open("rb") as input_stream:
                            with archive.open(item.member, "w", force_zip64=True) as blob_stream:
                                size, digest = _copy_and_hash(
                                    input_stream,
                                    blob_stream,
                                    max_bytes=MAX_ATTACHMENT_BYTES,
                                    total_remaining=MAX_TOTAL_ATTACHMENT_BYTES - written_total,
                                )
                        written_total += size
                        if size != item.size or digest != item.sha256:
                            raise PackageError(
                                f"Attachment {item.record.original_name!r} changed while the package was written."
                            )
                output_stream.flush()
                os.fsync(output_stream.fileno())
            if guard_destination is not None:
                guard_destination(target)
            os.replace(temporary, target)
            return target
        finally:
            if temporary.exists():
                temporary.unlink()
    except PackageError:
        raise
    except (OSError, BadZipFile, ValueError, TypeError) as error:
        raise PackageError(f"Could not write log package: {error}") from error


def copy_package_attachment(
    package: LoadedPackage, item: PackageAttachment, destination: Path
) -> None:
    """Stream one validated blob into a caller-chosen generated path."""
    if item not in package.attachments:
        raise PackageError("The requested attachment is not part of this package.")
    expected_index = package.attachments.index(item)
    if item.member != _blob_member(expected_index):
        raise PackageError("The package attachment does not have a generated member name.")
    if item.size > MAX_ATTACHMENT_BYTES:
        raise PackageError(f"An attachment exceeds the {MAX_ATTACHMENT_BYTES}-byte limit.")

    target = Path(destination)
    created_destination = False
    try:
        with ZipFile(package.source, "r") as archive:
            info = archive.getinfo(item.member)
            _validate_zip_info(info)
            if info.file_size != item.size:
                raise PackageError(f"Attachment {item.record.id} has an inconsistent declared size.")
            with archive.open(info, "r") as source_stream:
                with target.open("xb") as destination_stream:
                    created_destination = True
                    size, digest = _copy_and_hash(
                        source_stream,
                        destination_stream,
                        max_bytes=min(MAX_ATTACHMENT_BYTES, MAX_TOTAL_ATTACHMENT_BYTES),
                    )
            if size != item.size:
                raise PackageError(f"Attachment {item.record.id} has an invalid size.")
            if digest != item.sha256:
                raise PackageError(f"Attachment {item.record.id} failed its SHA-256 check.")
    except PackageError:
        if created_destination:
            target.unlink(missing_ok=True)
        raise
    except (BadZipFile, OSError, RuntimeError, EOFError, ValueError, zlib.error) as error:
        if created_destination:
            target.unlink(missing_ok=True)
        raise PackageError(f"Could not copy package attachment: {error}") from error


def semantic_record(
    record: TestDay | Setup | Lap | EventLayout | Attachment,
    *,
    attachment_sha256: str | None = None,
) -> dict[str, object]:
    """Return canonical substantive fields for comparison and fingerprinting."""
    values = asdict(record)
    if isinstance(record, Setup):
        values.pop("order", None)
        _validate_setup_json(record.structured_settings_json)
        values["structured_settings_json"] = normalise_setup_settings_json(
            record.structured_settings_json
        )
    elif isinstance(record, Lap):
        values.pop("sequence", None)
    elif isinstance(record, Attachment):
        values.pop("relative_path", None)
        if attachment_sha256 is not None:
            values["sha256"] = attachment_sha256
    return values


def _snapshot_record_arrays(snapshot: LogSnapshot) -> dict[str, list[dict[str, object]]]:
    if not isinstance(snapshot, LogSnapshot):
        raise PackageError("A log snapshot is required to write a package.")
    return {
        "days": [asdict(item) for item in sorted(snapshot.days, key=lambda item: item.id)],
        "setups": [
            asdict(item)
            for item in sorted(snapshot.setups, key=lambda item: (item.test_day_id, item.order, item.id))
        ],
        "laps": [
            asdict(item)
            for item in sorted(snapshot.laps, key=lambda item: (item.setup_id, item.sequence, item.id))
        ],
        "event_layouts": [
            asdict(item) for item in sorted(snapshot.event_layouts, key=lambda item: item.id)
        ],
    }


def _validate_records(days, setups, laps, event_layouts, attachments) -> None:
    record_groups = (
        ("days", tuple(days), TestDay, validate_day),
        ("setups", tuple(setups), Setup, validate_setup),
        ("laps", tuple(laps), Lap, validate_lap),
        ("event layouts", tuple(event_layouts), EventLayout, validate_event_layout),
    )
    total_records = sum(len(group[1]) for group in record_groups)
    attachments = tuple(attachments)
    total_records += len(attachments)
    if total_records > MAX_RECORD_COUNT:
        raise PackageError("The package contains too many records.")

    ids_by_kind: dict[type, set[str]] = {}
    for label, records, record_type, validator in record_groups:
        identifiers: set[str] = set()
        for record in records:
            if not isinstance(record, record_type):
                raise PackageError(f"The package contains an invalid {label} record.")
            try:
                if isinstance(record, Setup):
                    _validate_setup_json(record.structured_settings_json)
                validator(record)
            except (ValidationError, TypeError, ValueError) as error:
                raise PackageError(f"Invalid {label} record {getattr(record, 'id', '')!r}: {error}") from error
            if record.id in identifiers:
                raise PackageError(f"The package contains duplicate {label} ID {record.id!r}.")
            identifiers.add(record.id)
        ids_by_kind[record_type] = identifiers

    setup_order_keys: set[tuple[str, int]] = set()
    for setup in setups:
        if setup.test_day_id not in ids_by_kind[TestDay]:
            raise PackageError(f"Setup {setup.id!r} refers to a missing test day.")
        if setup.event_layout_id is not None and setup.event_layout_id not in ids_by_kind[EventLayout]:
            raise PackageError(f"Setup {setup.id!r} refers to a missing event/layout.")
        key = (setup.test_day_id, setup.order)
        if key in setup_order_keys:
            raise PackageError(f"The package contains duplicate setup order {setup.order} for day {setup.test_day_id!r}.")
        setup_order_keys.add(key)

    lap_sequence_keys: set[tuple[str, int]] = set()
    for lap in laps:
        if lap.setup_id not in ids_by_kind[Setup]:
            raise PackageError(f"Lap {lap.id!r} refers to a missing setup.")
        if lap.event_layout_id not in ids_by_kind[EventLayout]:
            raise PackageError(f"Lap {lap.id!r} refers to a missing event/layout.")
        key = (lap.setup_id, lap.sequence)
        if key in lap_sequence_keys:
            raise PackageError(f"The package contains duplicate lap sequence {lap.sequence} for setup {lap.setup_id!r}.")
        lap_sequence_keys.add(key)

    attachment_ids: set[str] = set()
    attachment_paths: set[str] = set()
    owner_ids = {
        "day": ids_by_kind[TestDay],
        "setup": ids_by_kind[Setup],
        "lap": ids_by_kind[Lap],
        "event_layout": ids_by_kind[EventLayout],
    }
    for attachment in attachments:
        if not isinstance(attachment, Attachment):
            raise PackageError("The package contains invalid attachment metadata.")
        if not isinstance(attachment.owner_type, str) or attachment.owner_type not in owner_ids:
            raise PackageError(f"Attachment {attachment.id!r} has an unknown owner type.")
        try:
            validate_staged_attachment(
                StagedAttachment(
                    attachment.role, attachment.original_name, attachment.relative_path
                ),
                attachment.owner_type,
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise PackageError(f"Invalid attachment metadata {attachment.id!r}: {error}") from error
        if not isinstance(attachment.id, str) or not attachment.id.strip():
            raise PackageError("Every attachment needs a record ID.")
        if not isinstance(attachment.owner_id, str) or attachment.owner_id not in owner_ids.get(attachment.owner_type, set()):
            raise PackageError(f"Attachment {attachment.id!r} refers to a missing owner.")
        _validate_relative_metadata_path(attachment.relative_path)
        if not isinstance(attachment.created_at, str) or not _is_aware_timestamp(attachment.created_at):
            raise PackageError(f"Attachment {attachment.id!r} has an invalid creation timestamp.")
        if attachment.id in attachment_ids:
            raise PackageError(f"The package contains duplicate attachment ID {attachment.id!r}.")
        if attachment.relative_path in attachment_paths:
            raise PackageError(f"The package contains duplicate attachment path {attachment.relative_path!r}.")
        attachment_ids.add(attachment.id)
        attachment_paths.add(attachment.relative_path)


def _parse_records(raw_records, record_type, label, validator):
    if not isinstance(raw_records, list):
        raise PackageError(f"The package {label} field must be an array.")
    expected_fields = {item.name for item in fields(record_type)}
    parsed = []
    for raw in raw_records:
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise PackageError(f"A {label} record has missing or unknown fields.")
        try:
            record = record_type(**raw)
            validator(record)
        except (TypeError, ValidationError, ValueError) as error:
            raise PackageError(f"The package contains an invalid {label} record: {error}") from error
        parsed.append(record)
    return tuple(parsed)


def _parse_package_attachments(raw_attachments) -> tuple[PackageAttachment, ...]:
    if not isinstance(raw_attachments, list):
        raise PackageError("The package attachments field must be an array.")
    result: list[PackageAttachment] = []
    expected_fields = {item.name for item in fields(Attachment)}
    for raw in raw_attachments:
        if not isinstance(raw, dict) or set(raw) != {"record", "member", "size", "sha256"}:
            raise PackageError("An attachment entry has missing or unknown fields.")
        record_data = raw["record"]
        if not isinstance(record_data, dict) or set(record_data) != expected_fields:
            raise PackageError("An attachment record has missing or unknown fields.")
        if not isinstance(raw["member"], str) or not _is_generated_blob_member(raw["member"]):
            raise PackageError("An attachment has an invalid archive member name.")
        if isinstance(raw["size"], bool) or not isinstance(raw["size"], int) or raw["size"] < 0:
            raise PackageError("An attachment has an invalid declared size.")
        if not isinstance(raw["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", raw["sha256"]) is None:
            raise PackageError("An attachment has an invalid SHA-256 value.")
        try:
            record = Attachment(**record_data)
        except TypeError as error:
            raise PackageError(f"An attachment record is invalid: {error}") from error
        result.append(PackageAttachment(record, raw["member"], raw["size"], raw["sha256"]))
    return tuple(result)


def _decode_manifest(payload: bytes) -> dict[str, object]:
    try:
        decoded = payload.decode("utf-8")
        manifest = json.loads(
            decoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        raise PackageError(f"The package manifest is not strict UTF-8 JSON: {error}") from error
    expected = {
        "format", "version", "source_label", "exported_at", "days", "setups",
        "laps", "event_layouts", "attachments",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise PackageError("The package manifest has missing or unknown fields.")
    if manifest["format"] != PACKAGE_FORMAT:
        raise PackageError("The file is not a supported test log package.")
    if type(manifest["version"]) is not int or manifest["version"] != PACKAGE_VERSION:
        raise PackageError(f"The package version {manifest['version']!r} is not supported.")
    if not isinstance(manifest["source_label"], str):
        raise PackageError("The package source label must be text.")
    if not isinstance(manifest["exported_at"], str) or not _is_aware_timestamp(manifest["exported_at"]):
        raise PackageError("The package export timestamp is invalid.")
    arrays = ("days", "setups", "laps", "event_layouts", "attachments")
    if any(not isinstance(manifest[name], list) for name in arrays):
        raise PackageError("Package record fields must be arrays.")
    if sum(len(manifest[name]) for name in arrays) > MAX_RECORD_COUNT:
        raise PackageError("The package contains too many records.")
    if len(manifest["attachments"]) + 1 > MAX_MEMBER_COUNT:
        raise PackageError("The package contains too many archive members.")
    return manifest


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_setup_json(value: str) -> None:
    try:
        json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        raise PackageError(f"Structured setup settings are not strict JSON: {error}") from error


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON value {value} is not permitted")


def _encode_json(value) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PackageError(f"The log contains a value that cannot be packaged: {error}") from error


def _blob_member(index: int) -> str:
    return f"blobs/{index:08d}"


def _is_generated_blob_member(value: str) -> bool:
    return re.fullmatch(r"blobs/[0-9]{8}", value) is not None


def _attachment_sort_key(attachment: Attachment):
    return (attachment.owner_type, attachment.owner_id, attachment.created_at, attachment.id)


def _validate_relative_metadata_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise PackageError("Attachment paths must be relative POSIX paths.")
    parts = value.split("/")
    if value.startswith("/") or any(part in ("", ".", "..") or ":" in part for part in parts):
        raise PackageError("Attachment paths cannot be absolute or contain traversal segments.")


def _is_aware_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.tzinfo is not None and parsed.utcoffset() is not None
    except (TypeError, ValueError):
        return False


def _validate_zip_info(info: ZipInfo) -> None:
    name = info.filename
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise PackageError("The package contains an invalid archive path.")
    parts = name.split("/")
    if name.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise PackageError("The package contains an absolute or traversal archive path.")
    if info.is_dir():
        raise PackageError("The package cannot contain directory entries.")
    if info.flag_bits & 0x1:
        raise PackageError("Encrypted package members are not supported.")
    if info.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
        raise PackageError("The package uses an unsupported compression method.")
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode) or (file_type and file_type != stat.S_IFREG):
        raise PackageError("The package cannot contain symlinks or special files.")


def _read_member(archive: ZipFile, info: ZipInfo, *, max_bytes: int, collect: bool = True):
    count = 0
    digest = hashlib.sha256()
    content = bytearray() if collect else None
    with archive.open(info, "r") as stream:
        while True:
            chunk = stream.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            count += len(chunk)
            if count > max_bytes:
                raise PackageError("A package member exceeds its actual streamed size limit.")
            digest.update(chunk)
            if content is not None:
                content.extend(chunk)
    return (bytes(content) if content is not None else None), count, digest.hexdigest()


def _copy_and_hash(
    source_stream,
    destination_stream,
    *,
    max_bytes: int,
    total_remaining: int | None = None,
):
    count = 0
    digest = hashlib.sha256()
    while True:
        chunk = source_stream.read(STREAM_CHUNK_BYTES)
        if not chunk:
            break
        count += len(chunk)
        if count > max_bytes:
            raise PackageError("An attachment exceeds its actual streamed size limit.")
        if total_remaining is not None and count > total_remaining:
            raise PackageError("The attachments exceed their actual total streamed size limit.")
        digest.update(chunk)
        destination_stream.write(chunk)
    return count, digest.hexdigest()


def _hash_file(path: Path, max_bytes: int) -> tuple[int, str]:
    if not path.is_file():
        raise PackageError("The stored attachment is not a regular file.")
    count = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            count += len(chunk)
            if count > max_bytes:
                raise PackageError("An attachment exceeds its actual streamed size limit.")
            digest.update(chunk)
    return count, digest.hexdigest()


def _fingerprint(days, setups, laps, event_layouts, attachments) -> str:
    payload = {
        "days": sorted((semantic_record(item) for item in days), key=lambda item: item["id"]),
        "setups": sorted((semantic_record(item) for item in setups), key=lambda item: item["id"]),
        "laps": sorted((semantic_record(item) for item in laps), key=lambda item: item["id"]),
        "event_layouts": sorted((semantic_record(item) for item in event_layouts), key=lambda item: item["id"]),
        "attachments": sorted(
            (
                semantic_record(item.record, attachment_sha256=item.sha256)
                for item in attachments
            ),
            key=lambda item: item["id"],
        ),
    }
    return hashlib.sha256(_encode_json(payload)).hexdigest()
