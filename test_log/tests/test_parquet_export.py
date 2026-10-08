import dataclasses
import ctypes
import errno
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

PYARROW_AVAILABLE = pa is not None

from vd_test_log.models import (
    DataPaths,
    EventLayout,
    Lap,
    Setup,
    StagedAttachment,
    TestDay,
)
from vd_test_log import parquet_export
from vd_test_log.parquet_export import build_parquet_tables
from vd_test_log.repository import SQLiteRepository
from vd_test_log.services import TestLogServices


CREATED_AT = "2026-10-08T12:00:00+00:00"


class FakeCFunction:
    def __init__(self, implementation):
        self.argtypes = None
        self.restype = None
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


class FakeLibC:
    def __init__(self, **functions):
        self.__dict__.update(functions)


class AtomicDirectoryPublishTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "staging"
        self.source.mkdir()
        self.destination = self.root / "dataset"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_linux_uses_renameat2_with_noreplace_flag(self):
        calls = []
        libc = FakeLibC(renameat2=FakeCFunction(lambda *args: calls.append(args) or 0))

        parquet_export._rename_directory_noreplace(
            self.source, self.destination, platform="linux", libc=libc
        )

        self.assertEqual(
            calls,
            [
                (
                    -100,
                    os.fsencode(self.source),
                    -100,
                    os.fsencode(self.destination),
                    1,
                )
            ],
        )

    def test_macos_uses_renamex_np_exclusive_flag(self):
        calls = []
        libc = FakeLibC(renamex_np=FakeCFunction(lambda *args: calls.append(args) or 0))

        parquet_export._rename_directory_noreplace(
            self.source, self.destination, platform="darwin", libc=libc
        )

        self.assertEqual(
            calls,
            [(os.fsencode(self.source), os.fsencode(self.destination), 4)],
        )

    def test_linux_existing_destination_errno_becomes_file_exists_error(self):
        def reject_existing_destination(*args):
            ctypes.set_errno(errno.EEXIST)
            return -1

        libc = FakeLibC(renameat2=FakeCFunction(reject_existing_destination))
        with patch.object(parquet_export.os, "rename", side_effect=AssertionError("unsafe fallback")):
            with self.assertRaises(FileExistsError) as raised:
                parquet_export._rename_directory_noreplace(
                    self.source, self.destination, platform="linux", libc=libc
                )

        self.assertEqual(raised.exception.errno, errno.EEXIST)
        self.assertTrue(self.source.is_dir())

    def test_unsupported_filesystem_fails_actionably_without_fallback(self):
        def reject_unsupported_filesystem(*args):
            ctypes.set_errno(errno.EINVAL)
            return -1

        libc = FakeLibC(renameat2=FakeCFunction(reject_unsupported_filesystem))
        with patch.object(parquet_export.os, "rename", side_effect=AssertionError("unsafe fallback")):
            with self.assertRaisesRegex(OSError, "Atomic no-overwrite") as raised:
                parquet_export._rename_directory_noreplace(
                    self.source, self.destination, platform="linux", libc=libc
                )

        self.assertEqual(raised.exception.errno, errno.EINVAL)
        self.assertTrue(self.source.is_dir())

    def test_missing_platform_api_fails_closed(self):
        with patch.object(parquet_export.os, "rename", side_effect=AssertionError("unsafe fallback")):
            with self.assertRaisesRegex(OSError, "Atomic no-overwrite") as raised:
                parquet_export._rename_directory_noreplace(
                    self.source, self.destination, platform="darwin", libc=object()
                )

        self.assertEqual(raised.exception.errno, errno.ENOTSUP)
        self.assertTrue(self.source.is_dir())

    @unittest.skipUnless(os.name == "nt", "Windows os.rename contract check")
    def test_windows_rename_rejects_existing_empty_destination(self):
        self.destination.mkdir()

        with self.assertRaises(FileExistsError):
            parquet_export._rename_directory_noreplace(
                self.source, self.destination, platform="win32"
            )

        self.assertTrue(self.source.is_dir())
        self.assertTrue(self.destination.is_dir())
        self.assertEqual(list(self.destination.iterdir()), [])


def make_day(identifier="day-1", **changes):
    return dataclasses.replace(
        TestDay(
            id=identifier,
            date="2026-10-08",
            location="Bruntingthorpe",
            weather=None,
            notes=None,
            created_at=CREATED_AT,
        ),
        **changes,
    )


def make_setup(identifier="setup-1", day_id="day-1", order=1, **changes):
    return dataclasses.replace(
        Setup(
            id=identifier,
            test_day_id=day_id,
            name="Baseline",
            setup_code="A01",
            settings_text="Cold tyre pressures: 20 psi",
            notes=None,
            order=order,
            created_at=CREATED_AT,
        ),
        **changes,
    )


def make_event(identifier="event-1", **changes):
    return dataclasses.replace(
        EventLayout(
            id=identifier,
            track_name="Bruntingthorpe",
            layout_name="Short Loop",
            event_name="Handling Day",
            event_type="test",
            length_m=1234.5,
            notes=None,
            archived=False,
            created_at=CREATED_AT,
        ),
        **changes,
    )


def make_lap(identifier, setup_id, event_id, sequence, status, **changes):
    return dataclasses.replace(
        Lap(
            id=identifier,
            setup_id=setup_id,
            event_layout_id=event_id,
            sequence=sequence,
            time_ms=42_318 + sequence,
            status=status,
            driver="Saved Driver",
            time_of_day=None,
            notes=None,
            created_at=CREATED_AT,
        ),
        **changes,
    )


@unittest.skipUnless(PYARROW_AVAILABLE, "PyArrow table readback requires the optional dependency")
class ParquetExportTableTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "test_log.sqlite3"
        self.repository = SQLiteRepository.open(self.database_path)

    def tearDown(self):
        self.repository.close()
        self.temporary_directory.cleanup()

    def assert_schema_fields(self, table, expected):
        self.assertEqual(
            [(field.name, field.type) for field in table.schema],
            expected,
        )

    def test_empty_log_builds_all_tables_with_explicit_typed_schemas(self):
        tables = build_parquet_tables(self.repository.capture_parquet_snapshot())

        self.assertEqual(set(tables), {"outings", "laps", "attachments"})
        self.assertEqual(tuple(table.num_rows for table in tables.values()), (0, 0, 0))
        self.assert_schema_fields(
            tables["outings"],
            [
                ("outing_uuid", pa.string()),
                ("test_day_uuid", pa.string()),
                ("test_date", pa.date32()),
                ("test_location", pa.string()),
                ("weather", pa.string()),
                ("test_day_notes", pa.string()),
                ("user_label", pa.string()),
                ("outing_id", pa.string()),
                ("settings_text", pa.string()),
                ("outing_notes", pa.string()),
                ("sort_order", pa.int64()),
                ("default_event_layout_uuid", pa.string()),
                ("default_track_name", pa.string()),
                ("default_layout_name", pa.string()),
                ("default_event_name", pa.string()),
                ("default_event_type", pa.string()),
                ("default_event_length_m", pa.float64()),
                ("default_driver", pa.string()),
                ("created_at", pa.string()),
                ("front_wing_height", pa.string()),
                ("rw_setting", pa.string()),
                ("front_spring_rate", pa.string()),
                ("rear_spring_rate", pa.string()),
                ("rear_arb_blade_setting", pa.string()),
                ("rear_arb_motion_ratio_setting", pa.string()),
                ("front_damping_ratio", pa.float64()),
                ("rear_damping_ratio", pa.float64()),
                ("diff_ramp_angle", pa.float64()),
                ("diff_preload", pa.float64()),
                ("sprocket_size", pa.string()),
                ("engine_tune", pa.string()),
                *[
                    (f"{corner}_{field}", pa.float64())
                    for corner in ("FL", "FR", "RL", "RR")
                    for field in ("camber", "toe", "pressure", "corner_weight")
                ],
            ],
        )
        self.assert_schema_fields(
            tables["laps"],
            [
                ("lap_uuid", pa.string()),
                ("outing_uuid", pa.string()),
                ("event_layout_uuid", pa.string()),
                ("sequence", pa.int64()),
                ("time_ms", pa.int64()),
                ("status", pa.string()),
                ("driver", pa.string()),
                ("time_of_day", pa.string()),
                ("notes", pa.string()),
                ("track_name", pa.string()),
                ("layout_name", pa.string()),
                ("event_name", pa.string()),
                ("event_type", pa.string()),
                ("length_m", pa.float64()),
                ("created_at", pa.string()),
            ],
        )
        self.assert_schema_fields(
            tables["attachments"],
            [
                ("attachment_uuid", pa.string()),
                ("owner_type", pa.string()),
                ("owner_uuid", pa.string()),
                ("role", pa.string()),
                ("original_name", pa.string()),
                ("relative_path", pa.string()),
                ("created_at", pa.string()),
            ],
        )
        self.assertEqual(tables["outings"].schema.field("test_date").type, pa.date32())
        self.assertEqual(tables["laps"].schema.field("time_ms").type, pa.int64())
        self.assertTrue(tables["outings"].schema.field("engine_tune").nullable)
        self.assertFalse(tables["outings"].schema.field("outing_uuid").nullable)

    def test_snapshot_tables_preserve_saved_values_history_and_attachment_references(self):
        day = make_day(weather=None, notes="", date="2026-10-08")
        event = make_event(event_name=None, event_type="test", length_m=None)
        historical_event = make_event(
            "event-old",
            layout_name="Old Loop",
            event_name="Prior Test",
            event_type=None,
            length_m=None,
            archived=False,
        )
        settings = (
            '{"front_wing_height":"5","rw_setting":"LD",'
            '"front_spring_rate":"350","front_damping_ratio":0.82,'
            '"rear_spring_rate":"250","rear_damping_ratio":0.73,'
            '"rear_arb_blade_setting":"OFF","rear_arb_motion_ratio_setting":"MR2",'
            '"diff_ramp_angle":45,"diff_preload":3.5,"sprocket_size":"13T",'
            '"engine_tune":"  Tune v2  ","corners":{'
            '"FL":{"camber":-1.2,"toe":0.1,"pressure":12.5,"corner_weight":210},'
            '"FR":{"camber":-1.1,"toe":0.2,"pressure":12.6,"corner_weight":211},'
            '"RL":{"camber":-0.8,"toe":-0.1,"pressure":13.0,"corner_weight":220},'
            '"RR":{"camber":-0.9,"toe":-0.2,"pressure":13.1,"corner_weight":221}}}'
        )
        outing = make_setup(
            day_id=day.id,
            name="Wet baseline",
            setup_code="B-7",
            notes="Friday run",
            event_layout_id=event.id,
            driver="Default Driver",
            structured_settings_json=settings,
        )
        zero_lap_outing = make_setup(
            "setup-no-laps",
            day.id,
            order=2,
            name="No laps yet",
            structured_settings_json="{}",
        )
        valid_lap = make_lap(
            "lap-valid", outing.id, event.id, 1, "valid", time_ms=38_765,
            driver="Saved Driver", time_of_day="10:15", notes="Best line",
        )
        invalid_lap = make_lap(
            "lap-invalid", outing.id, historical_event.id, 2, "invalid", time_ms=39_001,
            driver=None, time_of_day="bad clock text", notes=None,
        )
        day_file = StagedAttachment("file", "weather.bin", "days/day-1/weather.bin")
        outing_file = StagedAttachment("file", "setup.parquet", "setups/setup-1/setup.parquet")
        lap_file = StagedAttachment("file", "trace.bin", "laps/lap-valid/trace.bin")
        map_file = StagedAttachment("map", "old-loop.map", "events/event-old/old-loop.map")

        self.repository.save_day(day, (day_file,))
        self.repository.save_event_layout(event)
        self.repository.save_event_layout(historical_event, (map_file,))
        self.repository.save_setup(outing, (outing_file,))
        self.repository.save_setup(zero_lap_outing)
        self.repository.save_lap(valid_lap, (lap_file,))
        self.repository.save_lap(invalid_lap)
        self.repository.archive_event_layout(historical_event.id)
        tables = build_parquet_tables(self.repository.capture_parquet_snapshot())
        outings = {row["outing_uuid"]: row for row in tables["outings"].to_pylist()}
        laps = tables["laps"].to_pylist()
        attachments = tables["attachments"].to_pylist()

        self.assertEqual(set(outings), {outing.id, zero_lap_outing.id})
        self.assertEqual(outings[outing.id]["test_date"], date(2026, 10, 8))
        self.assertEqual(outings[outing.id]["user_label"], "Wet baseline")
        self.assertEqual(outings[outing.id]["outing_id"], "B-7")
        self.assertEqual(outings[outing.id]["default_event_layout_uuid"], event.id)
        self.assertEqual(outings[outing.id]["default_track_name"], "Bruntingthorpe")
        self.assertEqual(outings[outing.id]["default_layout_name"], "Short Loop")
        self.assertEqual(outings[outing.id]["default_driver"], "Default Driver")
        self.assertEqual(outings[outing.id]["rear_arb_blade_setting"], "OFF")
        self.assertEqual(outings[outing.id]["rear_arb_motion_ratio_setting"], "MR2")
        self.assertEqual(outings[outing.id]["diff_preload"], 3.5)
        self.assertEqual(outings[outing.id]["sprocket_size"], "13T")
        self.assertEqual(outings[outing.id]["engine_tune"], "Tune v2")
        self.assertEqual(outings[outing.id]["FL_camber"], -1.2)
        self.assertEqual(outings[outing.id]["RR_corner_weight"], 221.0)
        self.assertIsNone(outings[zero_lap_outing.id]["engine_tune"])
        self.assertEqual(
            [(row["lap_uuid"], row["time_ms"], row["status"]) for row in laps],
            [("lap-valid", 38_765, "valid"), ("lap-invalid", 39_001, "invalid")],
        )
        self.assertEqual(laps[0]["event_layout_uuid"], event.id)
        self.assertEqual(laps[0]["driver"], "Saved Driver")
        self.assertEqual(laps[0]["track_name"], "Bruntingthorpe")
        self.assertEqual(laps[1]["event_layout_uuid"], historical_event.id)
        self.assertEqual(laps[1]["layout_name"], "Old Loop")
        self.assertEqual(laps[1]["event_name"], "Prior Test")
        self.assertEqual(laps[1]["time_of_day"], "bad clock text")
        self.assertIsNone(laps[1]["driver"])
        self.assertEqual(
            {(row["owner_type"], row["owner_uuid"]) for row in attachments},
            {
                ("day", day.id),
                ("outing", outing.id),
                ("lap", valid_lap.id),
                ("event_layout", historical_event.id),
            },
        )
        outing_attachment = next(row for row in attachments if row["owner_type"] == "outing")
        self.assertEqual(outing_attachment["original_name"], "setup.parquet")
        self.assertEqual(outing_attachment["relative_path"], "setups/setup-1/setup.parquet")

        with tempfile.TemporaryDirectory() as directory:
            for name, table in tables.items():
                destination = Path(directory) / f"{name}.parquet"
                pq.write_table(table, destination)
                restored = pq.read_table(destination)
                self.assertEqual(restored.schema, table.schema)
                self.assertEqual(restored.to_pylist(), table.to_pylist())


class ParquetDatasetServiceTests(unittest.TestCase):
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

    @staticmethod
    def create_junction(link, target):
        quote = lambda value: str(value).replace("'", "''")
        command = (
            f"New-Item -ItemType Junction -Path '{quote(link)}' "
            f"-Target '{quote(target)}' | Out-Null"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode:
            raise OSError(result.stderr or result.stdout or "Could not create junction.")

    @unittest.skipUnless(PYARROW_AVAILABLE, "PyArrow dataset writing requires the optional dependency")
    def test_export_writes_three_tables_and_manifest_into_new_directory(self):
        day = make_day()
        outing = make_setup(day_id=day.id)
        self.services.save_day(day)
        self.services.save_setup(outing)
        destination = self.root / "exports" / "october-test"

        result = self.services.export_parquet(destination)

        self.assertEqual(result, destination)
        self.assertEqual(
            {path.name for path in destination.iterdir()},
            {"outings.parquet", "laps.parquet", "attachments.parquet", "manifest.json"},
        )
        self.assertEqual(pq.read_table(destination / "outings.parquet").num_rows, 1)
        self.assertEqual(pq.read_table(destination / "laps.parquet").num_rows, 0)
        self.assertEqual(pq.read_table(destination / "attachments.parquet").num_rows, 0)
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset_format"], "vd-test-log-parquet")
        self.assertEqual(manifest["dataset_format_version"], 1)
        self.assertEqual(
            manifest["tables"],
            {
                "outings": {"file": "outings.parquet", "row_count": 1},
                "laps": {"file": "laps.parquet", "row_count": 0},
                "attachments": {"file": "attachments.parquet", "row_count": 0},
            },
        )
        self.assertEqual(manifest["units"]["front_spring_rate"], "lb/in")
        self.assertEqual(manifest["units"]["diff_preload"], "ft-lb")
        self.assertEqual(manifest["units"]["FL_pressure"], "PSI")
        self.assertEqual(manifest["units"]["time_ms"], "ms (integer)")
        self.assertFalse(manifest["attachments"]["content_included"])
        self.assertEqual(
            manifest["attachments"]["relative_path_base"],
            "source application data folder",
        )

    def test_missing_pyarrow_is_actionable_and_creates_no_output(self):
        script = textwrap.dedent(
            r"""
            import sys
            from pathlib import Path

            class BlockPyArrow:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "pyarrow" or fullname.startswith("pyarrow."):
                        raise ModuleNotFoundError("PyArrow intentionally blocked by test")

            sys.meta_path.insert(0, BlockPyArrow())
            from vd_test_log.models import DataPaths
            from vd_test_log.services import TestLogServices

            root = Path(sys.argv[1])
            paths = DataPaths(
                root=root / "data",
                database=root / "data" / "test_log.sqlite3",
                attachments=root / "data" / "attachments",
                lock_file=root / "data" / ".test_log.lock",
                backup_root=root / "data" / "backups",
            )
            services = TestLogServices.open(paths)
            destination = root / "not-created" / "dataset"
            try:
                try:
                    services.export_parquet(destination)
                except Exception as error:
                    print(f"error={type(error).__name__}: {error}")
                print(f"parent_exists={destination.parent.exists()}")
            finally:
                services.close()
            """
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script, str(self.root / "blocked")],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("optional PyArrow", completed.stdout)
        self.assertIn("pip install -r requirements-parquet.txt", completed.stdout)
        self.assertIn("parent_exists=False", completed.stdout)

    def test_existing_destination_is_rejected_without_modification(self):
        destination = self.root / "already-here"
        destination.mkdir()
        sentinel = destination / "preserve.txt"
        sentinel.write_text("keep this destination", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.services.export_parquet(destination)

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep this destination")
        self.assertEqual(list(self.root.glob(".already-here.staging-*")), [])

    def test_dangling_symlink_destination_is_rejected_and_preserved(self):
        destination = self.root / "dangling-dataset"
        try:
            destination.symlink_to(self.root / "missing-target", target_is_directory=True)
        except OSError as error:
            self.skipTest(f"symlinks are unavailable in this environment: {error}")

        with self.assertRaises(FileExistsError):
            self.services.export_parquet(destination)

        self.assertTrue(destination.is_symlink())
        self.assertEqual(list(self.root.glob(".dangling-dataset.staging-*")), [])

    def test_lexists_rejects_a_dangling_final_entry_before_creating_staging(self):
        destination = self.root / "dangling-dataset"
        with patch("vd_test_log.parquet_export.os.path.lexists", return_value=True):
            with self.assertRaises(FileExistsError):
                self.services.export_parquet(destination)

        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".dangling-dataset.staging-*")), [])

    def test_protected_database_attachments_backup_and_lock_paths_are_rejected(self):
        target_factories = (
            lambda paths: paths.database,
            lambda paths: Path(f"{paths.database}-wal"),
            lambda paths: paths.lock_file,
            lambda paths: paths.attachments / "dataset",
            lambda paths: paths.backup_root / "existing-backup" / "dataset",
        )
        for index, target_factory in enumerate(target_factories):
            with self.subTest(target_index=index):
                destination = target_factory(self.paths)

                with self.assertRaises(ValueError):
                    self.services.export_parquet(destination)

                self.assertFalse(destination.is_dir())
                if destination in (self.paths.database, self.paths.lock_file):
                    self.assertTrue(destination.is_file())
                elif destination == Path(f"{self.paths.database}-wal"):
                    self.assertFalse(destination.exists())
                else:
                    self.assertFalse(destination.exists())

    def test_rejects_hard_link_to_a_managed_attachment(self):
        managed = self.paths.attachments / "owned.bin"
        managed.write_bytes(b"preserve managed content")
        alias = self.root / "attachment-hard-link"
        try:
            os.link(managed, alias)
        except (AttributeError, NotImplementedError, OSError) as error:
            self.skipTest(f"hard links are unavailable in this environment: {error}")

        with self.assertRaises(ValueError):
            self.services.export_parquet(alias)

        self.assertEqual(managed.read_bytes(), b"preserve managed content")
        self.assertEqual(alias.read_bytes(), b"preserve managed content")

    def test_rejects_destination_through_attachment_junction(self):
        junction = self.root / "attachment-alias"
        try:
            self.create_junction(junction, self.paths.attachments)
        except OSError as error:
            self.skipTest(f"junctions are unavailable in this environment: {error}")
        destination = junction / "dataset"

        with self.assertRaises(ValueError):
            self.services.export_parquet(destination)

        self.assertFalse(destination.exists())

    @unittest.skipUnless(PYARROW_AVAILABLE, "PyArrow dataset writing requires the optional dependency")
    def test_write_failure_cleans_only_staging_and_never_publishes_partial_dataset(self):
        destination = self.root / "exports" / "failed-dataset"
        with patch(
            "vd_test_log.parquet_export._write_table",
            side_effect=[None, OSError("disk full")],
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.services.export_parquet(destination)

        self.assertFalse(destination.exists())
        self.assertEqual(
            list(destination.parent.glob(f".{destination.name}.staging-*")),
            [],
        )

    @unittest.skipUnless(PYARROW_AVAILABLE, "PyArrow dataset writing requires the optional dependency")
    def test_concurrent_empty_destination_is_rejected_and_preserved(self):
        destination = self.root / "exports" / "raced-dataset"

        def race_to_create_empty_directory(*args):
            # This runs inside the atomic operation, after the exporter's
            # final absence check, like a concurrent creator on Unix.
            Path(os.fsdecode(args[3])).mkdir()
            ctypes.set_errno(errno.EEXIST)
            return -1

        fake_libc = FakeLibC(renameat2=FakeCFunction(race_to_create_empty_directory))
        real_publish = parquet_export._rename_directory_noreplace

        def linux_no_replace_publish(source, target):
            return real_publish(source, target, platform="linux", libc=fake_libc)

        with patch.object(
            parquet_export,
            "_rename_directory_noreplace",
            side_effect=linux_no_replace_publish,
        ):
            with self.assertRaises(FileExistsError):
                self.services.export_parquet(destination)

        self.assertTrue(destination.is_dir())
        self.assertEqual(list(destination.iterdir()), [])
        self.assertEqual(
            list(destination.parent.glob(f".{destination.name}.staging-*")),
            [],
        )


if __name__ == "__main__":
    unittest.main()
