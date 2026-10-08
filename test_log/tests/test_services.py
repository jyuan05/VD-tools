import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from vd_test_log.attachments import AttachmentMissingError
from vd_test_log.models import DataPaths, EventLayout, Lap, Setup, TestDay, new_id, utc_now_iso
from vd_test_log.services import TestLogServices
from vd_test_log.validation import ValidationError


class TestLogServicesTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.paths = DataPaths(
            root=self.root / "data",
            database=self.root / "data" / "test_log.sqlite3",
            attachments=self.root / "data" / "attachments",
            lock_file=self.root / "data" / ".test_log.lock",
            backup_root=self.root / "data" / "backups",
        )
        self.services = TestLogServices.open(self.paths)

    def tearDown(self):
        self.services.close()
        self.temporary_directory.cleanup()

    def make_day(self):
        return TestDay(
            id=new_id(),
            date="2026-10-05",
            location="Bruntingthorpe",
            weather=None,
            notes=None,
            created_at=utc_now_iso(),
        )

    def make_setup(self, day_id, order=1):
        return Setup(
            id=new_id(),
            test_day_id=day_id,
            name="Baseline",
            setup_code="A01",
            settings_text="Cold pressures: 20 psi",
            notes="Keep the front settings",
            order=order,
            created_at=utc_now_iso(),
        )

    def make_event(self):
        return EventLayout(
            id=new_id(),
            track_name="Bruntingthorpe",
            layout_name="Short Loop",
            event_name="Handling Day",
            event_type="test",
            length_m=1234.5,
            notes="Dry surface",
            archived=False,
            created_at=utc_now_iso(),
        )

    def test_setup_duplicate_has_new_order_and_independent_file_owner(self):
        day = self.make_day()
        self.services.save_day(day)
        source = self.root / "setup.txt"
        source.write_text("front = 2", encoding="utf-8")
        staged = self.services.stage_attachment(source, "file")
        original = self.make_setup(day.id)
        self.services.save_setup(original, (staged,))

        duplicate = self.services.duplicate_setup(original.id)

        self.assertNotEqual(duplicate.id, original.id)
        self.assertEqual(duplicate.name, "Baseline (copy)")
        self.assertIsNone(duplicate.setup_code)
        self.assertEqual(duplicate.settings_text, original.settings_text)
        self.assertEqual(duplicate.notes, original.notes)
        self.assertEqual(duplicate.order, 2)
        original_attachment = self.services.list_attachments("setup", original.id)[0]
        copied_attachment = self.services.list_attachments("setup", duplicate.id)[0]
        self.assertNotEqual(original_attachment.relative_path, copied_attachment.relative_path)
        self.assertEqual(
            self.services.resolve_attachment(original_attachment).read_bytes(),
            self.services.resolve_attachment(copied_attachment).read_bytes(),
        )
        self.assertEqual(self.services.discard_staged((staged,)), [])
        self.assertTrue(self.services.resolve_attachment(original_attachment).is_file())

    def test_copy_setup_to_another_day_preserves_metadata_order_and_independent_files_without_laps(self):
        source_day = self.make_day()
        destination_day = dataclasses.replace(self.make_day(), date="2026-10-06")
        self.services.save_day(source_day)
        self.services.save_day(destination_day)
        event = self.services.save_event_layout(self.make_event())
        destination_existing = self.services.save_setup(
            self.make_setup(destination_day.id, order=4)
        )

        source_file = self.root / "setup-evidence.txt"
        source_file.write_bytes(b"original setup evidence")
        staged = self.services.stage_attachment(source_file, "file")
        original = dataclasses.replace(
            self.make_setup(source_day.id),
            name="Baseline",
            setup_code="A01",
            settings_text="Cold pressures: 20 psi",
            notes="Keep the front settings",
            event_layout_id=event.id,
            driver="Driver Two",
            structured_settings_json=(
                '{"rear_spring_rate":"300","diff_preload":0,"engine_tune":"  Tune v2  "}'
            ),
        )
        original = self.services.save_setup(original, (staged,))
        original_lap = Lap(
            id=new_id(),
            setup_id=original.id,
            event_layout_id=event.id,
            sequence=1,
            time_ms=42318,
            status="valid",
            driver="Driver Two",
            time_of_day=None,
            notes=None,
            created_at=utc_now_iso(),
        )
        self.services.save_lap(original_lap)
        source_attachment_before = self.services.list_attachments("setup", original.id)[0]
        source_stored = self.services.resolve_attachment(source_attachment_before)

        copied = self.services.copy_setup_to_day(original.id, destination_day.id)

        self.assertEqual(
            (
                copied.test_day_id,
                copied.name,
                copied.setup_code,
                copied.settings_text,
                copied.notes,
                copied.event_layout_id,
                copied.driver,
                copied.structured_settings_json,
            ),
            (
                destination_day.id,
                original.name,
                original.setup_code,
                original.settings_text,
                original.notes,
                event.id,
                "Driver Two",
                original.structured_settings_json,
            ),
        )
        self.assertNotEqual(copied.id, original.id)
        self.assertNotEqual(copied.created_at, original.created_at)
        self.assertEqual(copied.order, 2)
        self.assertEqual(self.services.get_setup(original.id), original)
        self.assertEqual(self.services.list_laps(original.id), [original_lap])
        self.assertEqual(self.services.list_laps(copied.id), [])
        self.assertEqual(
            self.services.list_setups(destination_day.id),
            [destination_existing, copied],
        )

        source_attachment_after = self.services.list_attachments("setup", original.id)[0]
        copied_attachment = self.services.list_attachments("setup", copied.id)[0]
        copied_stored = self.services.resolve_attachment(copied_attachment)
        self.assertEqual(source_attachment_after, source_attachment_before)
        self.assertNotEqual(copied_attachment.id, source_attachment_before.id)
        self.assertEqual(copied_attachment.role, source_attachment_before.role)
        self.assertEqual(copied_attachment.original_name, source_attachment_before.original_name)
        self.assertNotEqual(copied_attachment.relative_path, source_attachment_before.relative_path)
        self.assertEqual(source_stored.read_bytes(), b"original setup evidence")
        self.assertEqual(copied_stored.read_bytes(), b"original setup evidence")
        source_stored.write_bytes(b"changed source copy")
        self.assertEqual(copied_stored.read_bytes(), b"original setup evidence")

    def test_copy_setup_to_another_day_rejects_missing_source_destination_and_same_day_before_copying(self):
        source_day = self.make_day()
        destination_day = dataclasses.replace(self.make_day(), date="2026-10-06")
        self.services.save_day(source_day)
        self.services.save_day(destination_day)
        original = self.make_setup(source_day.id)
        self.services.save_setup(original)

        with self.subTest(case="missing source"):
            with self.assertRaises(ValidationError):
                self.services.copy_setup_to_day("missing-setup-id", destination_day.id)
        with self.subTest(case="missing destination"):
            with self.assertRaises(ValidationError):
                self.services.copy_setup_to_day(original.id, "missing-day-id")
        with self.subTest(case="same day"):
            with self.assertRaises(ValidationError):
                self.services.copy_setup_to_day(original.id, source_day.id)

        self.assertEqual(self.services.list_setups(source_day.id), [original])
        self.assertEqual(self.services.list_setups(destination_day.id), [])

    def test_copy_setup_to_another_day_cleans_partial_attachment_copies_on_missing_source_file(self):
        source_day = self.make_day()
        destination_day = dataclasses.replace(self.make_day(), date="2026-10-06")
        self.services.save_day(source_day)
        self.services.save_day(destination_day)
        first_source = self.root / "first-setup.txt"
        second_source = self.root / "second-setup.txt"
        first_source.write_text("first", encoding="utf-8")
        second_source.write_text("second", encoding="utf-8")
        staged = (
            self.services.stage_attachment(first_source, "file"),
            self.services.stage_attachment(second_source, "file"),
        )
        original = self.make_setup(source_day.id)
        self.services.save_setup(original, staged)
        attachments = self.services.list_attachments("setup", original.id)
        self.services.resolve_attachment(attachments[-1]).unlink()
        files_before = set(self.paths.attachments.iterdir())

        with self.assertRaises(AttachmentMissingError):
            self.services.copy_setup_to_day(original.id, destination_day.id)

        self.assertEqual(self.services.list_setups(source_day.id), [original])
        self.assertEqual(self.services.list_setups(destination_day.id), [])
        self.assertEqual(set(self.paths.attachments.iterdir()), files_before)
        self.assertTrue(self.services.resolve_attachment(attachments[0]).is_file())


    def test_event_duplicate_is_active_and_owns_a_separate_map_copy(self):
        source = self.root / "short-loop.map"
        source.write_bytes(b"map data")
        staged = self.services.stage_attachment(source, "map")
        original = self.services.save_event_layout(self.make_event(), (staged,))
        self.services.archive_event_layout(original.id)

        duplicate = self.services.duplicate_event_layout(original.id)

        self.assertFalse(duplicate.archived)
        original_map = self.services.list_attachments("event_layout", original.id)[0]
        copied_map = self.services.list_attachments("event_layout", duplicate.id)[0]
        self.assertEqual(original_map.role, "map")
        self.assertEqual(copied_map.role, "map")
        self.assertNotEqual(original_map.relative_path, copied_map.relative_path)
        self.assertEqual(
            self.services.resolve_attachment(original_map).read_bytes(),
            self.services.resolve_attachment(copied_map).read_bytes(),
        )

    def test_duplicate_failure_removes_partial_copies_and_creates_no_record(self):
        day = self.make_day()
        self.services.save_day(day)
        first_source = self.root / "first.txt"
        second_source = self.root / "second.txt"
        first_source.write_text("first", encoding="utf-8")
        second_source.write_text("second", encoding="utf-8")
        staged = (
            self.services.stage_attachment(first_source, "file"),
            self.services.stage_attachment(second_source, "file"),
        )
        original = self.make_setup(day.id)
        self.services.save_setup(original, staged)
        attachments = self.services.list_attachments("setup", original.id)
        self.services.resolve_attachment(attachments[-1]).unlink()
        files_before = set(self.paths.attachments.iterdir())

        with self.assertRaises(AttachmentMissingError):
            self.services.duplicate_setup(original.id)

        self.assertEqual(self.services.list_setups(day.id), [original])
        self.assertEqual(set(self.paths.attachments.iterdir()), files_before)

    def test_failed_save_cleans_only_staged_copy_and_leaves_no_record(self):
        source = self.root / "notes.txt"
        source.write_text("source stays", encoding="utf-8")
        staged = self.services.stage_attachment(source, "file")
        invalid_day = dataclasses.replace(self.make_day(), date="2026-02-30")
        copied_path = self.paths.root / staged.relative_path

        with self.assertRaises(ValidationError):
            self.services.save_day(invalid_day, (staged,))

        self.assertFalse(copied_path.exists())
        self.assertEqual(self.services.list_days(), [])
        self.assertEqual(source.read_text(encoding="utf-8"), "source stays")
        self.assertEqual(self.services.discard_staged((staged,)), [])

    def test_committed_delete_removes_owned_attachment_files(self):
        day = self.make_day()
        self.services.save_day(day)
        source = self.root / "setup.txt"
        source.write_text("setup", encoding="utf-8")
        staged = self.services.stage_attachment(source, "file")
        setup = self.make_setup(day.id)
        self.services.save_setup(setup, (staged,))
        attachment = self.services.list_attachments("setup", setup.id)[0]
        stored = self.services.resolve_attachment(attachment)

        failures = self.services.delete_setup(setup.id)

        self.assertEqual(failures, [])
        self.assertFalse(stored.exists())
        self.assertIsNone(self.services.get_setup(setup.id))


    def test_duplicate_setup_copies_event_driver_and_structured_settings(self):
        required_fields = {"event_layout_id", "driver", "structured_settings_json"}
        self.assertTrue(required_fields <= Setup.__dataclass_fields__.keys())
        day = self.make_day()
        event = self.make_event()
        self.services.save_day(day)
        self.services.save_event_layout(event)
        original = dataclasses.replace(
            self.make_setup(day.id),
            event_layout_id=event.id,
            driver="Driver Two",
            structured_settings_json='{"rear_spring_rate":"300","diff_preload":0}',
        )
        self.services.save_setup(original)

        duplicate = self.services.duplicate_setup(original.id)

        self.assertEqual(duplicate.event_layout_id, event.id)
        self.assertEqual(duplicate.driver, "Driver Two")
        self.assertEqual(
            json.loads(duplicate.structured_settings_json),
            {"rear_spring_rate": "300", "diff_preload": 0},
        )
        self.assertNotEqual(duplicate.id, original.id)
        self.assertEqual(duplicate.order, 2)
        self.assertEqual(self.services.list_attachments("setup", duplicate.id), [])


if __name__ == "__main__":
    unittest.main()
