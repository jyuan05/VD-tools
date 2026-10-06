import dataclasses
import hashlib
import json
import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

from vd_test_log.attachments import AttachmentStore
from vd_test_log.models import (
    Attachment,
    DataPaths,
    EventLayout,
    Lap,
    Setup,
    StagedAttachment,
    TestDay,
)
from vd_test_log.repository import SQLiteRepository
from vd_test_log.share_package import (
    PackageError,
    LogSnapshot,
    capture_log,
    copy_package_attachment,
    read_package,
    semantic_record,
    write_package,
)


CREATED_AT = "2026-10-05T10:15:30+00:00"


class SharePackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_root = self.root / "data"
        self.data_root.mkdir()
        self.repository = None
        self.paths = DataPaths(
            root=self.data_root,
            database=self.data_root / "test_log.sqlite3",
            attachments=self.data_root / "attachments",
            lock_file=self.data_root / ".test_log.lock",
            backup_root=self.data_root / "backups",
        )
        self.store = AttachmentStore(self.paths)

    def tearDown(self):
        if self.repository is not None:
            self.repository.close()
        self.temporary_directory.cleanup()

    def make_snapshot(self):
        day = TestDay(
            id="day-1",
            date="2026-10-05",
            location="Bruntingthorpe",
            weather=None,
            notes="offline notes",
            created_at=CREATED_AT,
        )
        setup = Setup(
            id="setup-1",
            test_day_id=day.id,
            name="Baseline",
            setup_code="A01",
            settings_text="Cold pressures: 20 psi",
            notes=None,
            order=1,
            created_at=CREATED_AT,
            event_layout_id="event-1",
            driver="Driver 1",
            structured_settings_json=(
                '{"corners":{"FL":{"toe":0.1}},"front_spring_rate":"250"}'
            ),
        )
        event = EventLayout(
            id="event-1",
            track_name="Bruntingthorpe",
            layout_name="Short Loop",
            event_name="Handling Day",
            event_type="test",
            length_m=1234.5,
            notes="Archived event",
            archived=True,
            created_at=CREATED_AT,
        )
        lap = Lap(
            id="lap-1",
            setup_id=setup.id,
            event_layout_id=event.id,
            sequence=1,
            time_ms=42_318,
            status="valid",
            driver="Historical Driver",
            time_of_day="10:15",
            notes="event snapshot",
            created_at=CREATED_AT,
        )
        attachment_rows = (
            ("day", day.id, "file", "day.bin", b"day bytes"),
            ("setup", setup.id, "file", "setup.txt", b"setup bytes"),
            ("lap", lap.id, "file", "lap.csv", b"lap bytes"),
            ("event_layout", event.id, "map", "event map.png", b"map bytes"),
        )
        attachments = []
        for index, (owner_type, owner_id, role, name, content) in enumerate(attachment_rows):
            relative_path = f"attachments/{index}-{name.replace(' ', '_')}"
            file_path = self.data_root / Path(relative_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)
            attachments.append(
                Attachment(
                    id=f"attachment-{index + 1}",
                    owner_type=owner_type,
                    owner_id=owner_id,
                    role=role,
                    original_name=name,
                    relative_path=relative_path,
                    created_at=CREATED_AT,
                )
            )
        return LogSnapshot(
            days=(day,),
            setups=(setup,),
            laps=(lap,),
            event_layouts=(event,),
            attachments=tuple(attachments),
        )

    def make_repository(self):
        repository = SQLiteRepository.open(self.paths.database)
        self.repository = repository
        day = TestDay("day-1", "2026-10-05", "Bruntingthorpe", None, None, CREATED_AT)
        event = EventLayout(
            "event-1", "Bruntingthorpe", "Short Loop", "Handling Day", "test",
            1234.5, None, False, CREATED_AT,
        )
        setup = Setup(
            "setup-1", day.id, "Baseline", "A01", "Cold pressures: 20 psi", None,
            1, CREATED_AT, event.id, "Driver 1", "{}",
        )
        lap = Lap(
            "lap-1", setup.id, event.id, 1, 42_318, "valid", "Historical Driver",
            "10:15", "snapshot", CREATED_AT,
        )
        repository.save_day(day, attachments=(StagedAttachment("file", "day.bin", "attachments/day.bin"),))
        repository.save_event_layout(
            event, attachments=(StagedAttachment("map", "map.png", "attachments/map.png"),)
        )
        repository.save_setup(
            setup, attachments=(StagedAttachment("file", "setup.txt", "attachments/setup.txt"),)
        )
        repository.save_lap(
            lap, attachments=(StagedAttachment("file", "lap.csv", "attachments/lap.csv"),)
        )
        with repository._connection:
            repository._connection.execute(
                "UPDATE event_layouts SET archived = 1 WHERE id = ?", (event.id,)
            )
        self.paths.attachments.mkdir(parents=True, exist_ok=True)
        for name, data in (
            ("day.bin", b"day bytes"),
            ("map.png", b"map bytes"),
            ("setup.txt", b"setup bytes"),
            ("lap.csv", b"lap bytes"),
        ):
            (self.paths.attachments / name).write_bytes(data)
        return repository

    def test_capture_includes_archived_layouts_and_attachments_for_each_owner(self):
        repository = self.make_repository()
        try:
            snapshot = capture_log(repository)
        finally:
            repository.close()
            self.repository = None

        self.assertEqual([item.id for item in snapshot.days], ["day-1"])
        self.assertEqual([item.id for item in snapshot.setups], ["setup-1"])
        self.assertEqual([item.id for item in snapshot.laps], ["lap-1"])
        self.assertEqual([item.id for item in snapshot.event_layouts], ["event-1"])
        self.assertTrue(snapshot.event_layouts[0].archived)
        self.assertEqual(
            {(item.owner_type, item.owner_id, item.role) for item in snapshot.attachments},
            {
                ("day", "day-1", "file"),
                ("setup", "setup-1", "file"),
                ("lap", "lap-1", "file"),
                ("event_layout", "event-1", "map"),
            },
        )

    def test_package_round_trip_keeps_records_and_streamed_blob_members(self):
        snapshot = self.make_snapshot()
        target = self.root / "portable package.zip"

        write_package(target, snapshot, self.store, "BFR 2026")
        package = read_package(target)

        self.assertEqual(package.source_label, "BFR 2026")
        self.assertEqual(package.days, snapshot.days)
        self.assertEqual(package.setups, snapshot.setups)
        self.assertEqual(package.laps, snapshot.laps)
        self.assertEqual(package.event_layouts, snapshot.event_layouts)
        self.assertEqual(
            {item.record.id: item.record for item in package.attachments},
            {item.id: item for item in snapshot.attachments},
        )
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(names[0], "manifest.json")
            self.assertEqual(set(names[1:]), {item.member for item in package.attachments})
            self.assertTrue(all("\\" not in name for name in names))
            self.assertTrue(all(name.startswith("blobs/") for name in names[1:]))

        for item in package.attachments:
            content = (self.data_root / item.record.relative_path).read_bytes()
            self.assertEqual(item.size, len(content))
            self.assertEqual(item.sha256, hashlib.sha256(content).hexdigest())
        copied = self.root / "generated-copy.bin"
        copy_package_attachment(package, package.attachments[0], copied)
        self.assertEqual(copied.read_bytes(), b"day bytes")

    def test_fingerprint_ignores_source_label_export_time_and_local_presentation_order(self):
        first_target = self.root / "first.zip"
        second_target = self.root / "second.zip"
        snapshot = self.make_snapshot()
        changed_order = dataclasses.replace(
            snapshot,
            setups=(
                dataclasses.replace(
                    snapshot.setups[0],
                    order=12,
                    structured_settings_json=(
                        '{ "front_spring_rate" : "250", "corners" : {"FL":{"toe":0.1} } }'
                    ),
                ),
            ),
            laps=(dataclasses.replace(snapshot.laps[0], sequence=9),),
        )

        write_package(first_target, snapshot, self.store, "source A")
        write_package(second_target, changed_order, self.store, "source B")

        self.assertEqual(read_package(first_target).fingerprint, read_package(second_target).fingerprint)

        stored_target = self.root / "stored-compression.zip"
        with zipfile.ZipFile(first_target) as original, zipfile.ZipFile(
            stored_target, "w", compression=zipfile.ZIP_STORED
        ) as stored:
            for item in original.infolist():
                stored.writestr(item.filename, original.read(item.filename))
        self.assertEqual(read_package(first_target).fingerprint, read_package(stored_target).fingerprint)

    def test_empty_snapshot_is_a_valid_package_without_blob_members(self):
        target = self.root / "empty.zip"
        empty = LogSnapshot((), (), (), (), ())

        write_package(target, empty, self.store)
        package = read_package(target)

        self.assertEqual(package.days, ())
        self.assertEqual(package.attachments, ())
        with zipfile.ZipFile(target) as archive:
            self.assertEqual(archive.namelist(), ["manifest.json"])

    def test_fingerprint_semantics_ignore_local_order_and_paths_but_include_blob_hash(self):
        snapshot = self.make_snapshot()
        setup = snapshot.setups[0]
        lap = snapshot.laps[0]
        attachment = snapshot.attachments[0]
        changed_setup = dataclasses.replace(
            setup,
            order=29,
            structured_settings_json=(
                '{ "front_spring_rate" : "250", "corners" : {"FL":{"toe":0.1} } }'
            ),
        )
        changed_lap = dataclasses.replace(lap, sequence=17)
        moved_attachment = dataclasses.replace(
            attachment, relative_path="attachments/on-another-machine.bin"
        )

        self.assertEqual(semantic_record(setup), semantic_record(changed_setup))
        self.assertEqual(semantic_record(lap), semantic_record(changed_lap))
        self.assertEqual(
            semantic_record(attachment, attachment_sha256="abc"),
            semantic_record(moved_attachment, attachment_sha256="abc"),
        )
        self.assertNotEqual(
            semantic_record(attachment, attachment_sha256="abc"),
            semantic_record(attachment, attachment_sha256="def"),
        )

    def test_failed_export_keeps_existing_destination_and_removes_sibling_temp(self):
        snapshot = self.make_snapshot()
        target = self.root / "already-there.zip"
        target.write_bytes(b"preserve this package")
        missing = dataclasses.replace(
            snapshot.attachments[0], relative_path="attachments/missing.bin"
        )
        snapshot = dataclasses.replace(snapshot, attachments=(missing,) + snapshot.attachments[1:])

        with self.assertRaises((PackageError, OSError)):
            write_package(target, snapshot, self.store, "source")

        self.assertEqual(target.read_bytes(), b"preserve this package")
        self.assertEqual(sorted(path.name for path in self.root.glob(".already-there.zip.*.tmp")), [])

    def test_failed_atomic_replace_keeps_existing_destination_and_cleans_temp(self):
        target = self.root / "existing.zip"
        target.write_bytes(b"keep prior export")

        with patch("vd_test_log.share_package.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaises(PackageError):
                write_package(target, self.make_snapshot(), self.store)

        self.assertEqual(target.read_bytes(), b"keep prior export")
        self.assertEqual(list(self.root.glob(".existing.zip.*.tmp")), [])

    def test_destination_guard_runs_before_writing_or_replacing(self):
        target = self.root / "protected.zip"
        target.write_bytes(b"preserve")
        observed = []

        def reject(destination):
            observed.append(Path(destination))
            raise PackageError("protected destination")

        with self.assertRaisesRegex(PackageError, "protected destination"):
            write_package(target, self.make_snapshot(), self.store, guard_destination=reject)

        self.assertEqual(observed, [target])
        self.assertEqual(target.read_bytes(), b"preserve")
        self.assertEqual(list(self.root.glob(".protected.zip.*.tmp")), [])

    def test_reader_rejects_duplicate_json_keys_and_unknown_manifest_fields(self):
        duplicate = self.root / "duplicate-key.zip"
        with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "manifest.json",
                b'{"format":"vd-test-log-share-package","format":"vd-test-log-share-package","version":1}',
            )
        with self.assertRaises(PackageError):
            read_package(duplicate)

        valid = self.root / "valid.zip"
        write_package(valid, self.make_snapshot(), self.store)
        malformed = self.root / "unknown-field.zip"
        self._rewrite_manifest(valid, malformed, lambda manifest: manifest.update({"surprise": 1}))
        with self.assertRaises(PackageError):
            read_package(malformed)

        unsupported = self.root / "unsupported-version.zip"
        self._rewrite_manifest(valid, unsupported, lambda manifest: manifest.update({"version": 2}))
        with self.assertRaises(PackageError):
            read_package(unsupported)

    def test_reader_rejects_broken_references_duplicate_ids_and_hostile_members(self):
        valid = self.root / "valid.zip"
        write_package(valid, self.make_snapshot(), self.store)

        broken_reference = self.root / "broken-reference.zip"
        self._rewrite_manifest(
            valid,
            broken_reference,
            lambda manifest: manifest["setups"][0].update({"test_day_id": "missing-day"}),
        )
        with self.assertRaises(PackageError):
            read_package(broken_reference)

        duplicate_id = self.root / "duplicate-id.zip"
        self._rewrite_manifest(
            valid,
            duplicate_id,
            lambda manifest: manifest["days"].append(dict(manifest["days"][0])),
        )
        with self.assertRaises(PackageError):
            read_package(duplicate_id)

        hostile_member = self.root / "hostile-member.zip"
        with zipfile.ZipFile(valid) as original, zipfile.ZipFile(hostile_member, "w") as altered:
            for item in original.infolist():
                altered.writestr(item, original.read(item.filename))
            altered.writestr("../escape.bin", b"must not be extracted")
        with self.assertRaises(PackageError):
            read_package(hostile_member)

        duplicate_setup_json = self.root / "duplicate-setup-json.zip"
        self._rewrite_manifest(
            valid,
            duplicate_setup_json,
            lambda manifest: manifest["setups"][0].update(
                {"structured_settings_json": '{"front_spring_rate":"250","front_spring_rate":"350"}'}
            ),
        )
        with self.assertRaises(PackageError):
            read_package(duplicate_setup_json)

    def test_reader_rejects_duplicate_zip_members_and_symlinks(self):
        valid = self.root / "valid.zip"
        write_package(valid, self.make_snapshot(), self.store)
        duplicate = self.root / "duplicate-member.zip"
        with zipfile.ZipFile(valid) as original, zipfile.ZipFile(duplicate, "w") as altered:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for item in original.infolist():
                    altered.writestr(item, original.read(item.filename))
                altered.writestr("manifest.json", original.read("manifest.json"))
        with self.assertRaises(PackageError):
            read_package(duplicate)

        symlink = self.root / "symlink-member.zip"
        link_info = zipfile.ZipInfo("manifest.json")
        link_info.create_system = 3
        link_info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(symlink, "w") as archive:
            archive.writestr(link_info, b"manifest.json")
        with self.assertRaises(PackageError):
            read_package(symlink)

    def test_reader_rejects_missing_and_hash_mismatched_blob_members(self):
        valid = self.root / "valid.zip"
        write_package(valid, self.make_snapshot(), self.store)
        package = read_package(valid)

        missing = self.root / "missing-blob.zip"
        with zipfile.ZipFile(valid) as original, zipfile.ZipFile(missing, "w") as altered:
            for item in original.infolist():
                if item.filename not in {blob.member for blob in package.attachments}:
                    altered.writestr(item, original.read(item.filename))
        with self.assertRaises(PackageError):
            read_package(missing)

        mismatched = self.root / "mismatched-hash.zip"
        self._rewrite_blob(valid, mismatched, package.attachments[0].member, b"tampered")
        with self.assertRaises(PackageError):
            read_package(mismatched)

    def test_copy_revalidates_source_and_removes_only_its_failed_destination(self):
        valid = self.root / "valid.zip"
        write_package(valid, self.make_snapshot(), self.store)
        package = read_package(valid)
        changed = self.root / "changed-after-preview.zip"
        item = package.attachments[0]
        self._rewrite_blob(valid, changed, item.member, b"changed bytes")
        changed_package = dataclasses.replace(package, source=changed)
        destination = self.root / "partial-copy.bin"

        existing = self.root / "preserved-copy.bin"
        existing.write_bytes(b"previous file")
        with self.assertRaises(PackageError):
            copy_package_attachment(package, item, existing)
        self.assertEqual(existing.read_bytes(), b"previous file")

        with self.assertRaises(PackageError):
            copy_package_attachment(changed_package, item, destination)

        self.assertFalse(destination.exists())

    def test_actual_streamed_attachment_limit_is_enforced(self):
        valid = self.root / "valid.zip"
        write_package(valid, self.make_snapshot(), self.store)

        with patch("vd_test_log.share_package.MAX_ATTACHMENT_BYTES", 4):
            with self.assertRaises(PackageError):
                read_package(valid)

    def _rewrite_manifest(self, source, target, mutate):
        with zipfile.ZipFile(source) as original:
            manifest = json.loads(original.read("manifest.json"))
            mutate(manifest)
            replacement = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as altered:
                altered.writestr("manifest.json", replacement)
                for item in original.infolist():
                    if item.filename != "manifest.json":
                        altered.writestr(item, original.read(item.filename))

    def _rewrite_blob(self, source, target, member, content):
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(
            target, "w", compression=zipfile.ZIP_DEFLATED
        ) as altered:
            for item in original.infolist():
                altered.writestr(
                    item,
                    content if item.filename == member else original.read(item.filename),
                )


if __name__ == "__main__":
    unittest.main()
