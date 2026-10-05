from __future__ import annotations

import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path

from vd_test_log.repository import SQLiteRepository
from vd_test_log.validation import ValidationError


def create_schema_one_database(path: Path):
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE test_days (
                id TEXT PRIMARY KEY, date TEXT NOT NULL, location TEXT NOT NULL,
                weather TEXT, notes TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE setups (
                id TEXT PRIMARY KEY,
                test_day_id TEXT NOT NULL REFERENCES test_days(id) ON DELETE CASCADE,
                name TEXT NOT NULL, setup_code TEXT, settings_text TEXT NOT NULL,
                notes TEXT, sort_order INTEGER NOT NULL CHECK (sort_order > 0),
                created_at TEXT NOT NULL, UNIQUE (test_day_id, sort_order)
            );
            CREATE TABLE event_layouts (
                id TEXT PRIMARY KEY, track_name TEXT NOT NULL, layout_name TEXT NOT NULL,
                event_name TEXT, event_type TEXT, length_m REAL, notes TEXT,
                archived INTEGER NOT NULL CHECK (archived IN (0, 1)),
                created_at TEXT NOT NULL
            );
            CREATE TABLE laps (
                id TEXT PRIMARY KEY,
                setup_id TEXT NOT NULL REFERENCES setups(id) ON DELETE CASCADE,
                event_layout_id TEXT NOT NULL REFERENCES event_layouts(id) ON DELETE RESTRICT,
                sequence INTEGER NOT NULL CHECK (sequence > 0),
                time_ms INTEGER NOT NULL CHECK (time_ms > 0),
                status TEXT NOT NULL CHECK (status IN ('valid', 'invalid')),
                driver TEXT, time_of_day TEXT, notes TEXT, created_at TEXT NOT NULL,
                UNIQUE (setup_id, sequence)
            );
            CREATE TABLE attachments (
                id TEXT PRIMARY KEY,
                owner_type TEXT NOT NULL,
                day_owner_id TEXT REFERENCES test_days(id) ON DELETE CASCADE,
                setup_owner_id TEXT REFERENCES setups(id) ON DELETE CASCADE,
                lap_owner_id TEXT REFERENCES laps(id) ON DELETE CASCADE,
                event_layout_owner_id TEXT REFERENCES event_layouts(id) ON DELETE CASCADE,
                role TEXT NOT NULL, original_name TEXT NOT NULL,
                relative_path TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO test_days VALUES (?, ?, ?, ?, ?, ?)",
            ("day-1", "2026-10-05", "Bruntingthorpe", None, "day notes", "day-created"),
        )
        events = (
            ("event-a", "Short Loop", 0),
            ("event-b", "Long Loop", 0),
            ("event-archived", "Archived Loop", 1),
        )
        for event_id, layout, archived in events:
            connection.execute(
                """
                INSERT INTO event_layouts
                    (id, track_name, layout_name, archived, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_id, "Bruntingthorpe", layout, archived, f"{event_id}-created"),
            )

        setup_ids = (
            "setup-empty",
            "setup-uniform",
            "setup-mixed-drivers",
            "setup-mixed-events",
            "setup-mixed-null-driver",
            "setup-archived",
        )
        for order, setup_id in enumerate(setup_ids, start=1):
            connection.execute(
                """
                INSERT INTO setups
                    (id, test_day_id, name, setup_code, settings_text, notes, sort_order, created_at)
                VALUES (?, 'day-1', ?, ?, ?, ?, ?, ?)
                """,
                (
                    setup_id,
                    setup_id,
                    f"code-{order}",
                    f"settings-{order}",
                    f"setup notes {order}",
                    order,
                    f"setup-{order}-created",
                ),
            )

        lap_records = (
            ("lap-uniform-a", "setup-uniform", "event-a", 1, 42_001, "valid", "Ada"),
            ("lap-uniform-b", "setup-uniform", "event-a", 2, 42_111, "invalid", "Ada"),
            ("lap-driver-a", "setup-mixed-drivers", "event-a", 1, 43_001, "valid", "Ada"),
            ("lap-driver-b", "setup-mixed-drivers", "event-a", 2, 43_111, "valid", "Bea"),
            ("lap-event-a", "setup-mixed-events", "event-a", 1, 44_001, "valid", "Cal"),
            ("lap-event-b", "setup-mixed-events", "event-b", 2, 44_111, "invalid", "Cal"),
            ("lap-null-driver", "setup-mixed-null-driver", "event-a", 1, 45_001, "valid", None),
            ("lap-text-driver", "setup-mixed-null-driver", "event-a", 2, 45_111, "valid", "Dee"),
            ("lap-archived-a", "setup-archived", "event-archived", 1, 46_001, "valid", None),
            ("lap-archived-b", "setup-archived", "event-archived", 2, 46_111, "valid", None),
        )
        for index, record in enumerate(lap_records, start=1):
            lap_id, setup_id, event_id, sequence, time_ms, status, driver = record
            connection.execute(
                """
                INSERT INTO laps
                    (id, setup_id, event_layout_id, sequence, time_ms, status, driver,
                     time_of_day, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lap_id,
                    setup_id,
                    event_id,
                    sequence,
                    time_ms,
                    status,
                    driver,
                    f"10:{index:02d}",
                    f"lap notes {index}",
                    f"lap-{index}-created",
                ),
            )

        relative_path = "attachments/lap-uniform-a/video.bin"
        connection.execute(
            """
            INSERT INTO attachments
                (id, owner_type, lap_owner_id, role, original_name, relative_path, created_at)
            VALUES (?, 'lap', ?, 'file', ?, ?, ?)
            """,
            ("attachment-lap", "lap-uniform-a", "video.bin", relative_path, "attachment-created"),
        )
        connection.execute(
            """
            INSERT INTO attachments
                (id, owner_type, setup_owner_id, role, original_name, relative_path, created_at)
            VALUES (?, 'setup', ?, 'file', ?, ?, ?)
            """,
            (
                "attachment-setup",
                "setup-uniform",
                "setup-notes.txt",
                "attachments/setup-uniform/notes.txt",
                "attachment-setup-created",
            ),
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    attachment_file = path.parent / relative_path
    attachment_file.parent.mkdir(parents=True, exist_ok=True)
    attachment_file.write_bytes(b"preserve attachment bytes")
    with closing(sqlite3.connect(path)) as connection:
        old_lap_columns = (
            "id", "setup_id", "event_layout_id", "sequence", "time_ms", "status",
            "driver", "time_of_day", "notes", "created_at",
        )
        old_setup_columns = (
            "id", "test_day_id", "name", "setup_code", "settings_text",
            "notes", "sort_order", "created_at",
        )
        old_attachment_columns = (
            "id", "owner_type", "day_owner_id", "setup_owner_id", "lap_owner_id",
            "event_layout_owner_id", "role", "original_name", "relative_path", "created_at",
        )
        before_laps = connection.execute(
            f"SELECT {', '.join(old_lap_columns)} FROM laps ORDER BY id"
        ).fetchall()
        before_setups = connection.execute(
            f"SELECT {', '.join(old_setup_columns)} FROM setups ORDER BY id"
        ).fetchall()
        before_attachments = connection.execute(
            f"SELECT {', '.join(old_attachment_columns)} FROM attachments ORDER BY id"
        ).fetchall()
    return before_laps, before_setups, before_attachments, attachment_file


class RepositoryMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "schema-one.sqlite3"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_schema_one_backfills_only_independently_uniform_setup_history(self):
        before_laps, before_setups, before_attachments, attachment_file = (
            create_schema_one_database(self.database_path)
        )
        repository = SQLiteRepository.open(self.database_path)
        try:
            with closing(sqlite3.connect(self.database_path)) as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, 2, "schema 1 should migrate to schema 2")
            if version != 2:
                return

            expected_defaults = {
                "setup-empty": (None, None),
                "setup-uniform": ("event-a", "Ada"),
                "setup-mixed-drivers": ("event-a", None),
                "setup-mixed-events": (None, "Cal"),
                "setup-mixed-null-driver": ("event-a", None),
                "setup-archived": ("event-archived", None),
            }
            for setup_id, expected in expected_defaults.items():
                with self.subTest(setup_id=setup_id):
                    setup = repository.get_setup(setup_id)
                    self.assertEqual((setup.event_layout_id, setup.driver), expected)
                    self.assertEqual(setup.structured_settings_json, "{}")
        finally:
            repository.close()

        with closing(sqlite3.connect(self.database_path)) as connection:
            old_lap_columns = (
                "id", "setup_id", "event_layout_id", "sequence", "time_ms", "status",
                "driver", "time_of_day", "notes", "created_at",
            )
            old_setup_columns = (
                "id", "test_day_id", "name", "setup_code", "settings_text",
                "notes", "sort_order", "created_at",
            )
            old_attachment_columns = (
                "id", "owner_type", "day_owner_id", "setup_owner_id", "lap_owner_id",
                "event_layout_owner_id", "role", "original_name", "relative_path", "created_at",
            )
            self.assertEqual(
                connection.execute(
                    f"SELECT {', '.join(old_lap_columns)} FROM laps ORDER BY id"
                ).fetchall(),
                before_laps,
            )
            self.assertEqual(
                connection.execute(
                    f"SELECT {', '.join(old_setup_columns)} FROM setups ORDER BY id"
                ).fetchall(),
                before_setups,
            )
            self.assertEqual(
                connection.execute(
                    f"SELECT {', '.join(old_attachment_columns)} FROM attachments ORDER BY id"
                ).fetchall(),
                before_attachments,
            )
        self.assertEqual(attachment_file.read_bytes(), b"preserve attachment bytes")

    def test_migration_failure_rolls_back_added_columns_and_version(self):
        create_schema_one_database(self.database_path)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                CREATE TRIGGER fail_setup_default_backfill
                BEFORE UPDATE OF event_layout_id, driver ON setups
                BEGIN
                    SELECT RAISE(ABORT, 'intentional migration failure');
                END
                """
            )
            connection.commit()

        opened_repository = None
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                opened_repository = SQLiteRepository.open(self.database_path)
        finally:
            if opened_repository is not None:
                opened_repository.close()

        with closing(sqlite3.connect(self.database_path)) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(setups)")
            }
        self.assertEqual(version, 1)
        self.assertNotIn("event_layout_id", columns)
        self.assertNotIn("driver", columns)
        self.assertNotIn("structured_settings_json", columns)

    def test_incomplete_version_one_database_is_refused_without_changes(self):
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("CREATE TABLE sentinel(value TEXT)")
            connection.execute("INSERT INTO sentinel(value) VALUES ('keep me')")
            connection.execute("PRAGMA user_version = 1")
            connection.commit()
        before = self.database_path.read_bytes()

        opened_repository = None
        try:
            with self.assertRaises(ValidationError):
                opened_repository = SQLiteRepository.open(self.database_path)
        finally:
            if opened_repository is not None:
                opened_repository.close()

        self.assertEqual(self.database_path.read_bytes(), before)
        with closing(sqlite3.connect(self.database_path)) as connection:
            self.assertEqual(
                connection.execute("SELECT value FROM sentinel").fetchone()[0],
                "keep me",
            )


    def test_migrated_database_protects_setup_only_event_reference(self):
        create_schema_one_database(self.database_path)
        repository = SQLiteRepository.open(self.database_path)
        try:
            with repository._connection:
                repository._connection.execute(
                    """
                    INSERT INTO attachments(
                        id, owner_type, event_layout_owner_id, role,
                        original_name, relative_path, created_at
                    )
                    VALUES (?, 'event_layout', ?, 'map', ?, ?, ?)
                    """,
                    (
                        "setup-only-map",
                        "event-b",
                        "map.png",
                        "attachments/event-b/map.png",
                        "map-created",
                    ),
                )
                repository._connection.execute(
                    "UPDATE setups SET event_layout_id = ? WHERE id = ?",
                    ("event-b", "setup-empty"),
                )

            archived = repository.archive_event_layout("event-b")
            self.assertTrue(archived.archived)
            with self.assertRaises(sqlite3.IntegrityError):
                with repository._connection:
                    repository._connection.execute(
                        "UPDATE event_layouts SET layout_name = ? WHERE id = ?",
                        ("Changed layout", "event-b"),
                    )
            with self.assertRaises(sqlite3.IntegrityError):
                with repository._connection:
                    repository._connection.execute(
                        "UPDATE attachments SET original_name = ? WHERE id = ?",
                        ("changed.png", "setup-only-map"),
                    )
            with self.assertRaises(sqlite3.IntegrityError):
                with repository._connection:
                    repository._connection.execute(
                        "DELETE FROM attachments WHERE id = ?",
                        ("setup-only-map",),
                    )
            with self.assertRaises(sqlite3.IntegrityError):
                with repository._connection:
                    repository._connection.execute(
                        """
                        INSERT INTO attachments(
                            id, owner_type, event_layout_owner_id, role,
                            original_name, relative_path, created_at
                        )
                        VALUES (?, 'event_layout', ?, 'map', ?, ?, ?)
                        """,
                        (
                            "setup-only-map-2",
                            "event-b",
                            "second map.png",
                            "attachments/event-b/second-map.png",
                            "map-created-2",
                        ),
                    )
        finally:
            repository.close()


if __name__ == "__main__":
    unittest.main()
