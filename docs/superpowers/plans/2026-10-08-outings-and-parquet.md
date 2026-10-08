# Outings and Parquet Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement task by task. Use one Luna Max implementer with the parent as harness and reviewer; do not spawn child agents.

**Goal:** Rename the user-facing setup concept to Outing, add the optional engine-tune field and multi-file attachments, and export a typed whole-log Parquet dataset without changing offline sharing or CSV compatibility.

**Architecture:** Keep `Setup` and its SQLite relationships as the internal record model. Add `engine_tune` to the normalized structured-settings JSON behind a SQLite version fence. Capture an immutable record snapshot, then lazily load PyArrow to build and atomically publish a three-table Parquet dataset.

**Tech Stack:** Python 3.12+, standard library, Tkinter, SQLite, optional PyArrow.

**Spec:** `docs/superpowers/specs/2026-10-08-outings-and-parquet-design.md`

## Global Constraints

- Keep SQLite table names and internal `Setup`, `setup_id`, and `setup` attachment-owner identifiers unchanged.
- User-facing setup terminology becomes Outing; retain the existing CSV column names and their order.
- Add only the optional normalized JSON key `engine_tune`; blank or missing values display blank and export as Parquet null.
- Preserve all current optional vehicle choices and units, default driver/event at outing, and lap time/status/history behavior.
- Stage multi-file attachments as an all-or-nothing batch and keep payloads opaque.
- Export a whole-log directory with `outings.parquet`, `laps.parquet`, `attachments.parquet`, and `manifest.json`; keep typed schemas for empty tables.
- PyArrow stays optional and lazily imported; add only `test_log/requirements-parquet.txt`, with no runtime installation or network calls.
- Preserve ZIP sharing, IDs, conflict rules, backups, and the existing 512 MiB per-file / 2 GiB aggregate attachment limits.
- Keep output writes out of managed paths and publish the complete dataset from a sibling staging directory.

## Review Focus

1. After upgrade to schema 3, an older schema-2 app must be blocked before it can open and resave `engine_tune`; pin the v2→v3 and v1→v2→v3 upgrades, rollback to valid v2 after a failed second step, and schema-2-reader rejection of a v3 database in Task 1.
2. An old record or package with no `engine_tune` must still load as blank, while a new value survives save, copy, and same-version sharing; cover Task 1 and Task 2.
3. An outing with no laps and a log with no outings must produce valid typed Parquet files with zero-row schemas; cover Task 4.
4. A batch attachment failure after one successful copy must leave no staged copies or partial list entries; cover Task 3.
5. A failed Parquet table write, existing destination, or managed-path alias must preserve existing data and remove only the new staging directory; cover Task 5.

---

### Task 1: Add the optional tune field and fence older database readers

**Files:** Modify `test_log/vd_test_log/setup_settings.py`, `test_log/vd_test_log/repository.py`, `test_log/tests/test_setup_settings.py`, `test_log/tests/test_repository_migrations.py`, `test_log/tests/test_share_package.py`, and `test_log/tests/test_share_import.py`.

**Interfaces:** Keep `normalise_setup_settings_json(value: str) -> str`. Accept `engine_tune` as trimmed optional text, omit blank values from normalized JSON, and allow legacy JSON with no key. Set `SCHEMA_VERSION = 3`; add `_migrate_schema_two_to_three(connection) -> None`. Keep package `PACKAGE_VERSION` at 1.

- [ ] Add failing normalizer cases for trimmed text, blank omission, wrong value types, and legacy `{}` input. Add sharing cases showing a new tune value affects the semantic fingerprint and a version-1 package without the key still loads.

```python
normalise = normalise_setup_settings_json
assert normalise('{"engine_tune":"  Honda K v3  "}') == '{"engine_tune":"Honda K v3"}'
assert normalise('{"engine_tune":"  "}') == "{}"
assert normalise("{}") == "{}"
```

- [ ] Run `python -m unittest tests.test_setup_settings tests.test_share_package tests.test_share_import -v` from `test_log`; confirm the new assertions fail before implementation.
- [ ] Extend the settings allowlist and text normalization for `engine_tune`. Make the schema-2-to-3 step a transaction that advances `PRAGMA user_version` only; ensure the schema-1-to-2 step writes version 2 explicitly, then route version 1 through both migrations.
- [ ] Add migration cases for v2→v3, v1→v2→v3, preservation of all setup JSON and attachments, rollback to valid v2 if the second step fails, and rejection by a simulated schema-2 reader without changing a v3 database.
- [ ] Run `python -m unittest tests.test_setup_settings tests.test_repository_migrations tests.test_share_package tests.test_share_import -v` from `test_log` and verify legacy and new records pass their respective checks.

### Task 2: Rename the editor and move vehicle fields into the right tabs

**Files:** Modify `test_log/vd_test_log/ui.py`, `test_log/tests/test_ui.py`, and `test_log/README.md`.

**Interfaces:** Keep Python action keys and internal record kinds such as `add_setup`, `duplicate_setup`, and `setup` unchanged. In `_SETUP_TAB_FIELDS`, put `rear_arb_blade_setting` and `rear_arb_motion_ratio_setting` in Suspension; put `diff_ramp_angle`, `diff_preload`, `sprocket_size`, and `engine_tune` in Powertrain. Add `engine_tune` to `_SETUP_TOP_LEVEL_FIELDS` and to `_setup_values()` so a saved value loads and saves.

- [ ] Add UI tests for Outing-facing labels, the unchanged General and Corners tabs, the exact Suspension/Powertrain field membership, and blank display for legacy JSON without `engine_tune`.

```python
from vd_test_log.ui import _SETUP_TAB_FIELDS

fields = {tab: {name for name, _ in entries} for tab, entries in _SETUP_TAB_FIELDS.items()}
assert {"rear_arb_blade_setting", "rear_arb_motion_ratio_setting"} <= fields["Suspension"]
assert {"diff_ramp_angle", "diff_preload", "sprocket_size", "engine_tune"} <= fields["Powertrain"]
```

- [ ] Run `python -m unittest tests.test_ui -v` from `test_log`; confirm the new label and field-location assertions fail.
- [ ] Change only user-facing entity text to Outing, including add/duplicate/copy/delete labels and prompts. Keep Settings and Notes labels, internal action keys, record types, and CSV headers intact.
- [ ] Move the fields and add the optional “Engine tune (name/version)” text input. Preserve every existing field, choice ordering, and unit label; update the README’s user-facing terms and field grouping.
- [ ] Run `python -m unittest tests.test_ui tests.test_setup_settings tests.test_csv_export -v` from `test_log`; verify a saved outing retains every existing setting and `engine_tune` while CSV field names remain unchanged.

### Task 3: Stage multiple opaque attachments as one batch

**Files:** Modify `test_log/vd_test_log/ui.py` and `test_log/tests/test_ui.py`; reuse `test_log/vd_test_log/services.py` and `test_log/vd_test_log/attachments.py` without changing their storage contracts.

**Interfaces:** Keep `TestLogWindow.attach_file() -> None` and `TestLogServices.stage_attachment(source: Path, role: AttachmentRole) -> StagedAttachment`. Change the picker call to `filedialog.askopenfilenames(...)`; append staged values to the editor only after the complete batch succeeds.

- [ ] Add UI tests with two source files, one arbitrary binary and one `.parquet`, and assert the selected paths are staged using the existing `file` role. Add a failure-on-second-file case that expects the first staged copy to be discarded and the visible list to remain unchanged.

```python
with patch("vd_test_log.ui.filedialog.askopenfilenames", return_value=(str(binary), str(parquet))):
    window.attach_file()
assert [item.original_name for item in window._staged] == [binary.name, parquet.name]
```

- [ ] Run `python -m unittest tests.test_ui tests.test_attachments -v` from `test_log`; confirm the batch-selection tests fail against the single-file picker.
- [ ] Collect returned paths into a local batch. On cancel, return without staging. On any staging exception, call `discard_staged()` for only that local batch, report cleanup failures, and do not append a partial batch to `_staged`.
- [ ] Save a record with both files and assert each managed copy has the exact original bytes and display name. Keep the existing single-selection open behavior and do not add a remove-file control.
- [ ] Run `python -m unittest tests.test_ui tests.test_attachments tests.test_services -v` from `test_log`.

### Task 4: Capture one snapshot and define the Parquet table schemas

**Files:** Modify `test_log/vd_test_log/models.py` and `test_log/vd_test_log/repository.py`; create `test_log/vd_test_log/parquet_export.py`; create `test_log/tests/test_parquet_export.py` and extend `test_log/tests/test_repository.py`.

**Interfaces:** Add immutable `ParquetSnapshot(days, setups, event_layouts, laps, attachments)`. Add `SQLiteRepository.capture_parquet_snapshot() -> ParquetSnapshot`, collecting ordered records inside one SQLite read transaction before any output file is written. Add `build_parquet_tables(snapshot: ParquetSnapshot) -> dict[str, object]` in `parquet_export.py`; it imports PyArrow only when called.

- [ ] Create fixture cases for an empty log, an outing with no laps, valid and invalid laps with saved driver/event context, nullable settings, all four corner records, and multiple attachment owners.
- [ ] Pin the `outings` columns to `outing_uuid:string`, `test_day_uuid:string`, `test_date:date32`, `test_location:string`, `weather:string?`, `test_day_notes:string?`, `user_label:string`, `outing_id:string?`, `settings_text:string`, `outing_notes:string?`, `sort_order:int64`, `default_event_layout_uuid:string?`, `default_track_name:string?`, `default_layout_name:string?`, `default_event_name:string?`, `default_event_type:string?`, `default_event_length_m:float64?`, `default_driver:string?`, `created_at:string`, and the structured fields below. Resolve the default event UUID to its details in the same snapshot.

| Structured outing columns | Arrow type | Unit or choices |
|---|---|---|
| `front_wing_height`, `rw_setting`, `front_spring_rate`, `rear_spring_rate`, `rear_arb_blade_setting`, `rear_arb_motion_ratio_setting` | nullable string | Preserve existing values, including `LD`, `OFF`, `MR1`, and `MR2` |
| `front_damping_ratio`, `rear_damping_ratio` | nullable float64 | dimensionless |
| `diff_ramp_angle` | nullable float64 | degrees |
| `diff_preload` | nullable float64 | ft-lb |
| `sprocket_size`, `engine_tune` | nullable string | teeth text; tune name/version text |
| `{FL,FR,RL,RR}_{camber,toe}` | nullable float64 | degrees |
| `{FL,FR,RL,RR}_pressure` | nullable float64 | PSI |
| `{FL,FR,RL,RR}_corner_weight` | nullable float64 | lb |

The `laps` columns are `lap_uuid:string`, `outing_uuid:string`, `event_layout_uuid:string`, `sequence:int64`, `time_ms:int64`, `status:string`, `driver:string?`, `time_of_day:string?`, `notes:string?`, `track_name:string`, `layout_name:string`, `event_name:string?`, `event_type:string?`, `length_m:float64?`, and `created_at:string`. The `attachments` columns are `attachment_uuid:string`, `owner_type:string`, `owner_uuid:string`, `role:string`, `original_name:string`, `relative_path:string`, and `created_at:string`. Map stored `setup` attachment owners to `outing`; retain UUIDs and relative paths as references only. Every table uses its declared schema for empty inputs.

- [ ] Add assertions for the schema and the zero-row behavior:

```python
import pyarrow as pa
from vd_test_log.parquet_export import build_parquet_tables

tables = build_parquet_tables(snapshot)
assert tables["outings"].schema.field("test_date").type == pa.date32()
assert tables["laps"].schema.field("time_ms").type == pa.int64()
assert tables["laps"].num_rows == 0
```

- [ ] Run `python -m unittest tests.test_repository tests.test_parquet_export -v` from `test_log`; verify snapshot ordering and typed empty-table checks fail before implementation.
- [ ] Implement `ParquetSnapshot` and the read transaction. Build rows from a single snapshot: one outing row per `Setup`, one lap row per `Lap`, and metadata-only attachment rows. Keep zero-row tables and `engine_tune` null typed through explicit schemas.
- [ ] Use `pa.Table.from_pylist(rows, schema=schema)` for every table. Verify Parquet readback preserves raw integer lap times, `valid`/`invalid`, nulls, IDs, saved history, units, and zero-row schemas.

### Task 5: Add one guarded export action and publish the dataset atomically

**Files:** Modify `test_log/vd_test_log/services.py` and `test_log/vd_test_log/ui.py`; extend `test_log/vd_test_log/parquet_export.py`, `test_log/tests/test_parquet_export.py`, and `test_log/tests/test_ui.py`; create `test_log/requirements-parquet.txt`.

**Interfaces:** Add `TestLogServices.export_parquet(destination: Path) -> Path`. Add `write_parquet_dataset(snapshot: ParquetSnapshot, destination: Path, *, guard_destination: Callable[[Path], None]) -> Path` and `_write_table(table: object, destination: Path) -> None`; the latter lazily loads PyArrow. Add `_guard_parquet_dataset_destination(paths: DataPaths, destination: Path) -> None` alongside the existing export guards. Name sibling staging directories `.destination-name.staging-<uuid>`. Add one `Export Parquet…` UI action for the whole current log; keep selected-outing CSV export unchanged.

- [ ] Add tests for directory-picker cancellation, a missing PyArrow import with a clear error and no output, an existing destination, and a protected database/attachments/backups path. Simulate a blocked `pyarrow` import in an isolated subprocess; do not uninstall packages or modify the user's interpreter. Add a write-failure injection after one table and assert the destination is absent and the sibling staging directory is cleaned.

```python
with patch("vd_test_log.parquet_export._write_table", side_effect=[None, OSError("disk full")]):
    with self.assertRaises(OSError):
        self.services.export_parquet(destination)
assert not destination.exists()
assert list(destination.parent.glob(f".{destination.name}.staging-*")) == []
```

- [ ] Run `python -m unittest tests.test_parquet_export tests.test_ui tests.test_csv_export -v` from `test_log`; confirm the new service and UI assertions fail.
- [ ] Capture `ParquetSnapshot` under the service mutation/backup coordination path, then write outside that short snapshot section. Lazily load `pyarrow` and `pyarrow.parquet`. Check the protected namespace and aliases for both the final destination and the exporter-owned staging path; reject an existing final destination separately so the newly created staging directory is not rejected as an existing output.
- [ ] Stage the three `.parquet` files and `manifest.json` in a unique sibling directory. Write each explicit schema with `pq.write_table(..., version="1.0", compression="snappy")`. Publish only after all files succeed; reject an existing destination and remove only the exporter-owned staging directory on failure.
- [ ] Add one toolbar action using `filedialog.askdirectory(mustexist=False)`. Report an actionable optional-dependency message without attempting installation. Put `pyarrow` in `test_log/requirements-parquet.txt` only; do not edit the root `pyproject.toml`.
- [ ] Run readback tests when PyArrow is available and skip only those tests when it is absent. Use the isolated blocked-import test for the missing-dependency message, and run app startup plus CSV export to confirm they work without importing PyArrow; do not uninstall or modify the user's interpreter.

### Task 6: Document the dataset and close compatibility regressions

**Files:** Modify `test_log/README.md`; extend `test_log/tests/test_csv_export.py`, `test_log/tests/test_services.py`, `test_log/tests/test_share_package.py`, `test_log/tests/test_share_import.py`, and `test_log/tests/test_lock_backup.py` only where a regression is not already covered.

**Interfaces:** Document the four dataset filenames, table joins, types, units, optional dependency setup, whole-log scope, attachment-path limitation, and unchanged CSV/ZIP behavior. Keep `PACKAGE_VERSION = 1` and existing attachment size limits unchanged.

- [ ] Add or update focused regressions for copying an outing with its tune and file attachments but no laps, ZIP round-trip of opaque binary/Parquet attachments, backup restoration of the v3 database, and the exact legacy CSV column order.
- [ ] Verify the output manifest declares its dataset format version, table filenames, row counts, and units. Document that attachment paths point into the app’s data folder and the Parquet directory does not carry attachment bytes.
- [ ] Run `python -m unittest tests.test_csv_export tests.test_services tests.test_share_package tests.test_share_import tests.test_lock_backup -v` from `test_log`.
- [ ] Run `python -m unittest discover -s tests -v` from `test_log`. Record whether the optional PyArrow readback tests ran or were skipped, and confirm unrelated suspension-geometry files remain untouched.
