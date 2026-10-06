import dataclasses
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from vd_test_log.models import (
    DataPaths,
    EventLayout,
    Lap,
    Setup,
    TestDay,
    new_id,
    utc_now_iso,
)
from vd_test_log.services import (
    BackupInProgressError,
    ServicesClosedError,
    TestLogServices,
    UnsafeSharePackageDestinationError,
)
from vd_test_log.share_package import PackageChangedError, PackageConflictError, PackageError


class SharePackageImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self._services = []
        self.central = self.open_services("central")
        self._input_number = 0

    def tearDown(self):
        for service in reversed(self._services):
            service.close()
        self.temporary_directory.cleanup()

    def open_services(self, name):
        root = self.root / name
        paths = DataPaths(
            root=root,
            database=root / "test_log.sqlite3",
            attachments=root / "attachments",
            lock_file=root / ".test_log.lock",
            backup_root=root / "backups",
        )
        service = TestLogServices.open(paths)
        self._services.append(service)
        return service

    def make_day(self, *, identifier=None, date="2026-10-05", location="Bruntingthorpe"):
        return TestDay(
            id=identifier or new_id(),
            date=date,
            location=location,
            weather="dry",
            notes="morning session",
            created_at=utc_now_iso(),
        )

    def make_event(self, *, identifier=None):
        return EventLayout(
            id=identifier or new_id(),
            track_name="Bruntingthorpe",
            layout_name="Short Loop",
            event_name="Handling Day",
            event_type="test",
            length_m=1234.5,
            notes="dry surface",
            archived=False,
            created_at=utc_now_iso(),
        )

    def make_setup(self, day_id, event_id, *, identifier=None, order=1, name="Baseline"):
        return Setup(
            id=identifier or new_id(),
            test_day_id=day_id,
            name=name,
            setup_code="A01",
            settings_text="Cold pressures: 20 psi",
            notes="Keep the front settings",
            order=order,
            created_at=utc_now_iso(),
            event_layout_id=event_id,
            driver="Setup driver",
            structured_settings_json='{"rear_spring_rate":"300","diff_preload":0}',
        )

    def make_lap(self, setup_id, event_id, *, identifier=None, sequence=1, time_ms=42318):
        return Lap(
            id=identifier or new_id(),
            setup_id=setup_id,
            event_layout_id=event_id,
            sequence=sequence,
            time_ms=time_ms,
            status="valid",
            driver="Historical lap driver",
            time_of_day="10:24:11",
            notes="clean lap",
            created_at=utc_now_iso(),
        )

    def attach(self, service, record, owner_type, *, name, content, role="file"):
        self._input_number += 1
        source = self.root / "inputs" / str(self._input_number) / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
        staged = service.stage_attachment(source, role)
        if owner_type == "day":
            service.save_day(record, (staged,))
        elif owner_type == "setup":
            service.save_setup(record, (staged,))
        elif owner_type == "lap":
            service.save_lap(record, (staged,))
        elif owner_type == "event_layout":
            service.save_event_layout(record, (staged,))
        else:
            raise AssertionError(owner_type)

    def create_complete_source(self):
        source = self.open_services("source")
        day = self.make_day()
        self.attach(source, day, "day", name="day-notes.txt", content=b"day notes\n")

        event = self.make_event()
        self.attach(
            source,
            event,
            "event_layout",
            name="course-map.png",
            content=b"map image bytes",
            role="map",
        )
        setup = self.make_setup(day.id, event.id)
        self.attach(source, setup, "setup", name="setup-data.csv", content=b"front,2\n")
        lap = self.make_lap(setup.id, event.id)
        self.attach(source, lap, "lap", name="lap-video.txt", content=b"lap evidence\n")
        source.archive_event_layout(event.id, True)
        return source, {
            "day": source.get_day(day.id),
            "event": source.get_event_layout(event.id),
            "setup": source.get_setup(setup.id),
            "lap": source.get_lap(lap.id),
            "attachments": tuple(
                attachment
                for owner_type, owner_id in (
                    ("day", day.id),
                    ("event_layout", event.id),
                    ("setup", setup.id),
                    ("lap", lap.id),
                )
                for attachment in source.list_attachments(owner_type, owner_id)
            ),
        }

    def test_import_preserves_record_and_attachment_ids_and_archived_history(self):
        source, expected = self.create_complete_source()
        package_path = self.root / "source-package.zip"
        source.export_share_package(package_path, source_label="driver 17")

        duplicate_visible_day = self.make_day()
        self.central.save_day(duplicate_visible_day)
        preview = self.central.preview_share_package(package_path)
        self.assertEqual(preview.source_label, "driver 17")
        self.assertEqual((preview.incoming.days, preview.incoming.setups, preview.incoming.laps), (1, 1, 1))
        self.assertEqual((preview.incoming.event_layouts, preview.incoming.attachments), (1, 4))
        result = self.central.import_share_package(
            package_path,
            expected_fingerprint=preview.fingerprint,
        )

        self.assertEqual(result.added, preview.added)
        self.assertIsNotNone(result.backup_path)
        self.assertTrue(result.backup_path.is_dir())
        self.assertIsNotNone(self.central.get_day(expected["day"].id))
        self.assertIsNotNone(self.central.get_day(duplicate_visible_day.id))
        self.assertEqual(self.central.get_day(expected["day"].id), expected["day"])
        self.assertEqual(self.central.get_setup(expected["setup"].id), expected["setup"])
        self.assertEqual(self.central.get_lap(expected["lap"].id), expected["lap"])
        self.assertEqual(self.central.get_event_layout(expected["event"].id), expected["event"])

        actual_attachments = tuple(
            attachment
            for owner_type, owner_id in (
                ("day", expected["day"].id),
                ("event_layout", expected["event"].id),
                ("setup", expected["setup"].id),
                ("lap", expected["lap"].id),
            )
            for attachment in self.central.list_attachments(owner_type, owner_id)
        )
        self.assertEqual(
            {item.id for item in actual_attachments},
            {item.id for item in expected["attachments"]},
        )
        self.assertEqual(
            {(item.id, item.owner_type, item.owner_id, item.role, item.original_name, item.created_at)
             for item in actual_attachments},
            {(item.id, item.owner_type, item.owner_id, item.role, item.original_name, item.created_at)
             for item in expected["attachments"]},
        )
        self.assertEqual(
            sorted(self.central.resolve_attachment(item).read_bytes() for item in actual_attachments),
            sorted(source.resolve_attachment(item).read_bytes() for item in expected["attachments"]),
        )

    def test_reexport_is_a_noop_after_local_order_changes_and_new_children_append(self):
        source, expected = self.create_complete_source()
        first_package = self.root / "first.zip"
        source.export_share_package(first_package, source_label="first label")
        first_result = self.central.import_share_package(first_package)
        self.assertIsNotNone(first_result.backup_path)

        day = self.central.get_day(expected["day"].id)
        local_setup = self.make_setup(day.id, expected["event"].id, order=1, name="Central only")
        self.central.save_setup(local_setup)
        self.central.archive_event_layout(expected["event"].id, False)
        local_lap = self.make_lap(expected["setup"].id, expected["event"].id, sequence=1)
        self.central.save_lap(local_lap)
        self.central.archive_event_layout(expected["event"].id, True)
        self.assertEqual(self.central.get_setup(expected["setup"].id).order, 2)
        self.assertEqual(self.central.get_lap(expected["lap"].id).sequence, 2)

        reexported_package = self.root / "reexported.zip"
        source.export_share_package(reexported_package, source_label="changed label")
        no_op_preview = self.central.preview_share_package(reexported_package)
        self.assertEqual(no_op_preview.fingerprint, first_result.fingerprint)
        self.assertFalse(no_op_preview.conflicts)
        self.assertEqual(no_op_preview.added.days, 0)
        self.assertEqual(no_op_preview.added.setups, 0)
        self.assertEqual(no_op_preview.added.laps, 0)
        no_op_result = self.central.import_share_package(
            reexported_package,
            expected_fingerprint=no_op_preview.fingerprint,
        )
        self.assertIsNone(no_op_result.backup_path)

        source.archive_event_layout(expected["event"].id, False)
        new_source_lap = self.make_lap(expected["setup"].id, expected["event"].id, sequence=2, time_ms=41900)
        source.save_lap(new_source_lap)
        source.archive_event_layout(expected["event"].id, True)
        setup_file = self.root / "incremental-setup.txt"
        setup_file.write_bytes(b"new setup attachment")
        staged = source.stage_attachment(setup_file, "file")
        source.save_setup(source.get_setup(expected["setup"].id), (staged,))
        incremental_package = self.root / "incremental.zip"
        source.export_share_package(incremental_package)

        incremental_preview = self.central.preview_share_package(incremental_package)
        self.assertEqual(incremental_preview.added.laps, 1)
        self.assertEqual(incremental_preview.added.attachments, 1)
        self.assertFalse(incremental_preview.conflicts)
        incremental_result = self.central.import_share_package(
            incremental_package,
            expected_fingerprint=incremental_preview.fingerprint,
        )
        self.assertIsNotNone(incremental_result.backup_path)
        imported_lap = self.central.get_lap(new_source_lap.id)
        self.assertEqual(imported_lap.sequence, 3)
        self.assertEqual(self.central.get_lap(expected["lap"].id).sequence, 2)
        self.assertEqual(self.central.get_lap(local_lap.id).sequence, 1)
        self.assertEqual(len(self.central.list_attachments("setup", expected["setup"].id)), 2)

    def test_changed_same_id_blocks_all_rows_and_does_not_create_a_backup(self):
        source = self.open_services("conflicting-source")
        shared_day = self.make_day()
        source.save_day(shared_day)
        new_day = self.make_day(date="2026-10-06", location="Mallory Park")
        source.save_day(new_day)
        package_path = self.root / "conflict.zip"
        source.export_share_package(package_path)

        self.central.save_day(dataclasses.replace(shared_day, location="Central edit"))
        before_ids = {record.id for record in self.central.list_days()}
        preview = self.central.preview_share_package(package_path)
        self.assertTrue(any(conflict.record_id == shared_day.id for conflict in preview.conflicts))
        self.assertEqual(preview.added.days, 1)
        with self.assertRaises(PackageConflictError):
            self.central.import_share_package(package_path, expected_fingerprint=preview.fingerprint)

        self.assertEqual({record.id for record in self.central.list_days()}, before_ids)
        self.assertFalse(self.central.paths.backup_root.exists())

    def test_changed_attachment_bytes_with_the_same_id_are_a_conflict(self):
        source, expected = self.create_complete_source()
        original_package = self.root / "original-attachment.zip"
        source.export_share_package(original_package)
        self.central.import_share_package(original_package)
        changed_attachment = expected["attachments"][0]
        source.resolve_attachment(changed_attachment).write_bytes(b"changed attachment bytes")
        changed_package = self.root / "changed-attachment.zip"
        source.export_share_package(changed_package)

        preview = self.central.preview_share_package(changed_package)
        self.assertTrue(any(item.record_id == changed_attachment.id for item in preview.conflicts))
        with self.assertRaises(PackageConflictError):
            self.central.import_share_package(
                changed_package,
                expected_fingerprint=preview.fingerprint,
            )

        central_attachment = next(
            item
            for owner_type, owner_id in (
                ("day", expected["day"].id),
                ("event_layout", expected["event"].id),
                ("setup", expected["setup"].id),
                ("lap", expected["lap"].id),
            )
            for item in self.central.list_attachments(owner_type, owner_id)
            if item.id == changed_attachment.id
        )
        self.assertNotEqual(
            self.central.resolve_attachment(central_attachment).read_bytes(),
            b"changed attachment bytes",
        )

    def test_separate_copy_is_deterministic_and_detects_edits_to_its_copy(self):
        source, expected = self.create_complete_source()
        package_path = self.root / "copy-source.zip"
        source.export_share_package(package_path)
        self.central.import_share_package(package_path)

        preview = self.central.preview_share_package(package_path, separate_copy=True)
        self.assertFalse(preview.conflicts)
        first_copy = self.central.import_share_package(
            package_path,
            separate_copy=True,
            expected_fingerprint=preview.fingerprint,
        )
        self.assertEqual(first_copy.added.days, 1)
        copied_days = [day for day in self.central.list_days() if day.id != expected["day"].id]
        self.assertEqual(len(copied_days), 1)
        copied_day = copied_days[0]
        copied_setups = self.central.list_setups(copied_day.id)
        self.assertEqual(len(copied_setups), 1)
        copied_setup = copied_setups[0]
        self.assertNotEqual(copied_setup.id, expected["setup"].id)
        self.assertNotEqual(copied_setup.event_layout_id, expected["event"].id)
        copied_laps = self.central.list_laps(copied_setup.id)
        self.assertEqual(len(copied_laps), 1)
        self.assertEqual(copied_laps[0].event_layout_id, copied_setup.event_layout_id)
        copied_attachment_ids = {
            item.id
            for owner_type, owner_id in (
                ("day", copied_day.id),
                ("setup", copied_setup.id),
                ("lap", copied_laps[0].id),
                ("event_layout", copied_setup.event_layout_id),
            )
            for item in self.central.list_attachments(owner_type, owner_id)
        }
        original_attachment_ids = {item.id for item in expected["attachments"]}
        self.assertEqual(len(copied_attachment_ids), 4)
        self.assertTrue(copied_attachment_ids.isdisjoint(original_attachment_ids))

        second_preview = self.central.preview_share_package(package_path, separate_copy=True)
        self.assertFalse(second_preview.conflicts)
        self.assertEqual(second_preview.added.days, 0)
        second_copy = self.central.import_share_package(
            package_path,
            separate_copy=True,
            expected_fingerprint=second_preview.fingerprint,
        )
        self.assertIsNone(second_copy.backup_path)

        self.central.save_day(dataclasses.replace(copied_day, location="Edited central copy"))
        edited_preview = self.central.preview_share_package(package_path, separate_copy=True)
        self.assertTrue(any(conflict.record_id == copied_day.id for conflict in edited_preview.conflicts))
        with self.assertRaises(PackageConflictError):
            self.central.import_share_package(
                package_path,
                separate_copy=True,
                expected_fingerprint=edited_preview.fingerprint,
            )

    def test_spooled_imports_preserve_safe_suffixes_for_all_owners_and_ids(self):
        source, expected = self.create_complete_source()
        malicious_id = "../../../../outside"
        original_day_attachment = source.list_attachments("day", expected["day"].id)[0]
        source._repository._connection.execute(
            "UPDATE attachments SET id = ? WHERE id = ?",
            (malicious_id, original_day_attachment.id),
        )
        source._repository._connection.commit()
        package_path = self.root / "odd-attachment-id.zip"
        source.export_share_package(package_path)

        spool_parent = self.root / "isolated-spool-parent"
        spool_parent.mkdir()
        escaped_spool_file = self.root / "outside.blob"
        with patch.object(tempfile, "tempdir", str(spool_parent)):
            self.central.import_share_package(package_path)
            self.assertFalse(escaped_spool_file.exists())

            self.central.import_share_package(package_path, separate_copy=True)

        self.assertEqual(
            self.central.list_attachments("day", expected["day"].id)[0].id,
            malicious_id,
        )
        all_attachments = tuple(
            attachment
            for service in (self.central,)
            for owner_type, owners in (
                ("day", service.list_days()),
                ("setup", tuple(
                    setup
                    for day in service.list_days()
                    for setup in service.list_setups(day.id)
                )),
                ("lap", tuple(
                    lap
                    for day in service.list_days()
                    for setup in service.list_setups(day.id)
                    for lap in service.list_laps(setup.id)
                )),
                ("event_layout", service.list_event_layouts(include_archived=True)),
            )
            for owner in owners
            for attachment in service.list_attachments(owner_type, owner.id)
        )
        self.assertEqual(len(all_attachments), 8)
        self.assertEqual(
            {attachment.owner_type for attachment in all_attachments},
            {"day", "setup", "lap", "event_layout"},
        )
        for attachment in all_attachments:
            self.assertEqual(
                Path(attachment.relative_path).suffix,
                Path(attachment.original_name).suffix,
                attachment.original_name,
            )

    def test_import_reuses_one_zip_reader_for_all_attachment_blobs(self):
        source, _ = self.create_complete_source()
        package_path = self.root / "single-reader.zip"
        source.export_share_package(package_path)

        with patch("vd_test_log.share_package.ZipFile", wraps=ZipFile) as zip_factory:
            result = self.central.import_share_package(package_path)

        self.assertEqual(result.added.attachments, 4)
        self.assertEqual(zip_factory.call_count, 2)

    def test_new_map_on_already_referenced_event_is_a_preview_conflict(self):
        source = self.open_services("event-source")
        day = self.make_day()
        source.save_day(day)
        event = self.make_event()
        source.save_event_layout(event)
        first_package = self.root / "event-before-map.zip"
        source.export_share_package(first_package)
        self.central.import_share_package(first_package)
        self.central.save_setup(self.make_setup(day.id, event.id))

        map_file = self.root / "late-map.png"
        map_file.write_bytes(b"late map bytes")
        staged = source.stage_attachment(map_file, "map")
        source.save_event_layout(source.get_event_layout(event.id), (staged,))
        second_package = self.root / "event-after-map.zip"
        source.export_share_package(second_package)

        preview = self.central.preview_share_package(second_package)
        self.assertTrue(preview.conflicts)
        with self.assertRaises(PackageConflictError):
            self.central.import_share_package(second_package, expected_fingerprint=preview.fingerprint)
        self.assertEqual(self.central.list_attachments("event_layout", event.id), [])

    def test_expected_fingerprint_and_service_lifecycle_are_rechecked(self):
        source = self.open_services("lifecycle-source")
        source.save_day(self.make_day())
        old_package = self.root / "old.zip"
        source.export_share_package(old_package)
        old_fingerprint = source.preview_share_package(old_package).fingerprint
        source.save_day(self.make_day(date="2026-10-06"))
        changed_package = self.root / "new.zip"
        source.export_share_package(changed_package)

        with self.assertRaises(PackageChangedError):
            self.central.import_share_package(changed_package, expected_fingerprint=old_fingerprint)
        self.assertEqual(self.central.list_days(), [])

        self.central.close()
        with self.assertRaises(ServicesClosedError):
            self.central.preview_share_package(old_package)

    def test_conflicts_are_recomputed_between_preview_and_apply(self):
        source = self.open_services("changed-central-source")
        incoming = self.make_day()
        source.save_day(incoming)
        package_path = self.root / "apply-recheck.zip"
        source.export_share_package(package_path)
        preview = self.central.preview_share_package(package_path)
        self.assertFalse(preview.conflicts)
        self.assertEqual(preview.added.days, 1)

        local_edit = dataclasses.replace(incoming, location="Central edit after preview")
        self.central.save_day(local_edit)
        with self.assertRaises(PackageConflictError):
            self.central.import_share_package(
                package_path,
                expected_fingerprint=preview.fingerprint,
            )

        self.assertEqual(self.central.get_day(incoming.id), local_edit)
        self.assertFalse(self.central.paths.backup_root.exists())

    def test_export_rejects_database_and_attachment_aliases(self):
        database_contents = self.central.paths.database.read_bytes()
        with self.assertRaises(UnsafeSharePackageDestinationError):
            self.central.export_share_package(self.central.paths.database)
        with self.assertRaises(UnsafeSharePackageDestinationError):
            self.central.export_share_package(self.central.paths.attachments / "package.zip")
        with self.assertRaises(UnsafeSharePackageDestinationError):
            self.central.export_share_package(self.central.paths.backup_root / "package.zip")
        self.assertEqual(self.central.paths.database.read_bytes(), database_contents)

        day = self.make_day()
        self.central.save_day(day)
        input_file = self.root / "alias-source.txt"
        input_file.write_bytes(b"managed attachment")
        staged = self.central.stage_attachment(input_file, "file")
        self.central.save_day(day, (staged,))
        managed_file = self.central.resolve_attachment(
            self.central.list_attachments("day", day.id)[0]
        )
        alias = self.root / "attachment-hardlink.zip"
        try:
            os.link(managed_file, alias)
        except OSError as error:
            self.skipTest(f"The host cannot create a hard link for alias coverage: {error}")
        try:
            with self.assertRaises(UnsafeSharePackageDestinationError):
                self.central.export_share_package(alias)
        finally:
            alias.unlink(missing_ok=True)

    def test_busy_backup_barrier_rejects_import_before_backup_or_staging(self):
        source, _ = self.create_complete_source()
        package_path = self.root / "busy.zip"
        source.export_share_package(package_path)

        self.central._backup_active = True
        try:
            with self.assertRaises(BackupInProgressError):
                self.central.import_share_package(package_path)
        finally:
            self.central._backup_active = False
        self.assertEqual(self.central.list_days(), [])
        self.assertFalse(self.central.paths.backup_root.exists())

    def test_transaction_failure_rolls_back_rows_and_removes_staged_blobs(self):
        source, _ = self.create_complete_source()
        package_path = self.root / "forced-rollback.zip"
        source.export_share_package(package_path)
        self.central._repository._connection.execute(
            """
            CREATE TRIGGER fail_import_attachment
            BEFORE INSERT ON attachments
            WHEN NEW.original_name = 'lap-video.txt'
            BEGIN
                SELECT RAISE(ABORT, 'forced import test failure');
            END
            """
        )

        with self.assertRaises((sqlite3.IntegrityError, PackageError)):
            self.central.import_share_package(package_path)

        self.assertEqual(self.central.list_days(), [])
        self.assertEqual(self.central.list_event_layouts(include_archived=True), [])
        self.assertEqual(list(self.central.paths.attachments.glob("**/*")), [])


if __name__ == "__main__":
    unittest.main()
