"""Stable-ID planning and transactional application for offline log packages."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid5

from .attachments import AttachmentStore
from .models import Attachment, EventLayout, Lap, Setup, StagedAttachment, TestDay
from .repository import SQLiteRepository
from .share_package import (
    ImportPreview,
    LoadedPackage,
    PackageAttachment,
    PackageConflict,
    PackageConflictError,
    PackageCounts,
    copy_package_attachment,
    capture_log,
    semantic_record,
)

if TYPE_CHECKING:
    from .services import TestLogServices


_COPY_NAMESPACE = UUID("31f64f82-06f1-5a6c-9f87-6d928d7beafa")


@dataclass(frozen=True, slots=True)
class ImportPlan:
    days: tuple[TestDay, ...]
    setups: tuple[Setup, ...]
    laps: tuple[Lap, ...]
    event_layouts: tuple[EventLayout, ...]
    attachments: tuple[PackageAttachment, ...]
    added: PackageCounts
    skipped: PackageCounts
    conflicts: tuple[PackageConflict, ...]


def _counts(package: LoadedPackage) -> PackageCounts:
    return PackageCounts(
        days=len(package.days),
        setups=len(package.setups),
        laps=len(package.laps),
        event_layouts=len(package.event_layouts),
        attachments=len(package.attachments),
    )


def _mapped_id(fingerprint: str, kind: str, source_id: str) -> str:
    return uuid5(_COPY_NAMESPACE, f"{fingerprint}:{kind}:{source_id}").hex


def _remap_package(
    package: LoadedPackage, separate_copy: bool
) -> tuple[
    tuple[TestDay, ...],
    tuple[Setup, ...],
    tuple[Lap, ...],
    tuple[EventLayout, ...],
    tuple[PackageAttachment, ...],
]:
    if separate_copy:
        ids = {
            kind: {
                record.id: _mapped_id(package.fingerprint, kind, record.id)
                for record in records
            }
            for kind, records in (
                ("day", package.days),
                ("setup", package.setups),
                ("lap", package.laps),
                ("event_layout", package.event_layouts),
                ("attachment", tuple(item.record for item in package.attachments)),
            )
        }
    else:
        ids = {
            kind: {record.id: record.id for record in records}
            for kind, records in (
                ("day", package.days),
                ("setup", package.setups),
                ("lap", package.laps),
                ("event_layout", package.event_layouts),
                ("attachment", tuple(item.record for item in package.attachments)),
            )
        }

    days = tuple(replace(record, id=ids["day"][record.id]) for record in package.days)
    event_layouts = tuple(
        replace(record, id=ids["event_layout"][record.id])
        for record in package.event_layouts
    )
    setups = tuple(
        replace(
            record,
            id=ids["setup"][record.id],
            test_day_id=ids["day"][record.test_day_id],
            event_layout_id=(
                ids["event_layout"][record.event_layout_id]
                if record.event_layout_id is not None
                else None
            ),
        )
        for record in package.setups
    )
    laps = tuple(
        replace(
            record,
            id=ids["lap"][record.id],
            setup_id=ids["setup"][record.setup_id],
            event_layout_id=ids["event_layout"][record.event_layout_id],
        )
        for record in package.laps
    )
    attachments = tuple(
        replace(
            item,
            record=replace(
                item.record,
                id=ids["attachment"][item.record.id],
                owner_id=ids[item.record.owner_type][item.record.owner_id],
            ),
        )
        for item in package.attachments
    )
    return days, setups, laps, event_layouts, attachments


def _attachment_digest(store: AttachmentStore, record: Attachment) -> str:
    digest = hashlib.sha256()
    with store.resolve(record).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _make_conflict(kind: str, record_id: str, description: str) -> PackageConflict:
    return PackageConflict(kind=kind, record_id=record_id, description=description)


def _raise_conflicts(conflicts: tuple[PackageConflict, ...]) -> None:
    details = "; ".join(conflict.description for conflict in conflicts[:3])
    remaining = len(conflicts) - min(3, len(conflicts))
    if remaining:
        details += f"; and {remaining} more conflict(s)"
    error = PackageConflictError(
        f"The package has {len(conflicts)} conflict(s); no records were imported. {details}"
    )
    error.conflicts = conflicts
    raise error


def plan_import(
    repository: SQLiteRepository,
    store: AttachmentStore,
    package: LoadedPackage,
    *,
    separate_copy: bool = False,
) -> ImportPlan:
    """Compare incoming stable IDs with current rows and plan additions only."""
    current = capture_log(repository)
    days, setups, laps, events, attachments = _remap_package(package, separate_copy)

    current_records: dict[str, dict[str, object]] = {
        "day": {record.id: record for record in current.days},
        "setup": {record.id: record for record in current.setups},
        "lap": {record.id: record for record in current.laps},
        "event_layout": {record.id: record for record in current.event_layouts},
    }
    incoming_records: tuple[tuple[str, tuple[object, ...]], ...] = (
        ("day", days),
        ("setup", setups),
        ("lap", laps),
        ("event_layout", events),
    )
    additions: dict[str, list[object]] = {kind: [] for kind, _ in incoming_records}
    skipped: dict[str, int] = {kind: 0 for kind, _ in incoming_records}
    conflicts: list[PackageConflict] = []

    for kind, records in incoming_records:
        existing_by_id = current_records[kind]
        for record in records:
            identifier = record.id
            existing = existing_by_id.get(identifier)
            if existing is None:
                additions[kind].append(record)
            elif semantic_record(record) == semantic_record(existing):
                skipped[kind] += 1
            else:
                conflicts.append(
                    _make_conflict(
                        kind,
                        identifier,
                        f"{kind.replace('_', ' ').title()} {identifier} already exists with different saved fields.",
                    )
                )

    current_attachments = {record.id: record for record in current.attachments}
    new_attachments: list[PackageAttachment] = []
    skipped_attachment_count = 0
    for item in attachments:
        record = item.record
        existing = current_attachments.get(record.id)
        if existing is None:
            if (
                record.owner_type == "event_layout"
                and repository.get_event_layout(record.owner_id) is not None
                and repository.event_is_referenced(record.owner_id)
            ):
                conflicts.append(
                    _make_conflict(
                        "event_layout_attachment",
                        record.owner_id,
                        f"A new map or file for referenced event/layout {record.owner_id} conflicts with its saved history.",
                    )
                )
            else:
                new_attachments.append(item)
            continue

        if semantic_record(record) != semantic_record(existing):
            conflicts.append(
                _make_conflict(
                    "attachment",
                    record.id,
                    f"Attachment {record.id} already exists with different saved metadata.",
                )
            )
            continue
        try:
            existing_digest = _attachment_digest(store, existing)
        except (OSError, ValueError) as error:
            conflicts.append(
                _make_conflict(
                    "attachment",
                    record.id,
                    f"The existing file for attachment {record.id} cannot be read: {error}.",
                )
            )
            continue
        if existing_digest != item.sha256:
            conflicts.append(
                _make_conflict(
                    "attachment",
                    record.id,
                    f"Attachment {record.id} has different file contents.",
                )
            )
        else:
            skipped_attachment_count += 1

    added = PackageCounts(
        days=len(additions["day"]),
        setups=len(additions["setup"]),
        laps=len(additions["lap"]),
        event_layouts=len(additions["event_layout"]),
        attachments=len(new_attachments),
    )
    skipped_counts = PackageCounts(
        days=skipped["day"],
        setups=skipped["setup"],
        laps=skipped["lap"],
        event_layouts=skipped["event_layout"],
        attachments=skipped_attachment_count,
    )
    new_setups = tuple(
        record
        for _, record in sorted(
            enumerate(additions["setup"]),
            key=lambda pair: (pair[1].test_day_id, pair[1].order, pair[0]),
        )
    )
    new_laps = tuple(
        record
        for _, record in sorted(
            enumerate(additions["lap"]),
            key=lambda pair: (pair[1].setup_id, pair[1].sequence, pair[0]),
        )
    )
    return ImportPlan(
        days=tuple(additions["day"]),
        setups=new_setups,
        laps=new_laps,
        event_layouts=tuple(additions["event_layout"]),
        attachments=tuple(new_attachments),
        added=added,
        skipped=skipped_counts,
        conflicts=tuple(conflicts),
    )


def preview_from_plan(
    package: LoadedPackage, plan: ImportPlan, *, separate_copy: bool
) -> ImportPreview:
    return ImportPreview(
        source_label=package.source_label,
        fingerprint=package.fingerprint,
        incoming=_counts(package),
        added=plan.added,
        skipped=plan.skipped,
        conflicts=plan.conflicts,
        separate_copy=separate_copy,
    )


def apply_import_plan(
    repository: SQLiteRepository,
    store: AttachmentStore,
    package: LoadedPackage,
    plan: ImportPlan,
) -> None:
    """Copy only planned new blobs, then insert their rows in one DB batch."""
    staged: list[StagedAttachment] = []
    imported_attachments: list[Attachment] = []
    package_items_by_member = {item.member: item for item in package.attachments}
    try:
        with tempfile.TemporaryDirectory(prefix="vd-test-log-import-") as temporary:
            temporary_root = Path(temporary)
            for index, item in enumerate(plan.attachments):
                spool_path = temporary_root / f"{index:08d}-{item.record.id}.blob"
                source_item = package_items_by_member.get(item.member)
                if source_item is None:
                    raise ValueError("An import attachment is not present in the loaded package.")
                copy_package_attachment(package, source_item, spool_path)
                copied = store.stage(spool_path, item.record.role)
                copied = replace(copied, original_name=item.record.original_name)
                staged.append(copied)
                imported_attachments.append(
                    replace(item.record, relative_path=copied.relative_path)
                )
        repository.apply_import_batch(
            days=plan.days,
            setups=plan.setups,
            laps=plan.laps,
            event_layouts=plan.event_layouts,
            attachments=tuple(imported_attachments),
        )
    except BaseException as error:
        failures = store.remove_staged(staged)
        if failures and hasattr(error, "add_note"):
            error.add_note(
                "Could not remove staged import files: "
                + ", ".join(str(path) for path in failures)
            )
        raise


def raise_plan_conflicts(plan: ImportPlan) -> None:
    if plan.conflicts:
        _raise_conflicts(plan.conflicts)

