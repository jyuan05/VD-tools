# Outings and Parquet Export Design

## Outcome

Make the app present each saved vehicle setup as an **Outing**, retain the current offline record model, and add a whole-log Parquet export for analysis. Keep SQLite and Python record names stable: `Setup`, the `setups` table, `setup_id`, and attachment owner type `setup` remain internal compatibility identifiers. Existing CSV output keeps its exact headers and selected-outing scope.

## Outing editing

Use “Outing” for user-facing setup labels, headings, actions, prompts, and help text, including “Outing ID.” Keep the existing General and Corners tabs and every current optional choice and unit. Move rear ARB blade and motion-ratio choices into Suspension. Put differential ramp angle and preload, sprocket size, and one optional `engine_tune` text field for the tune name/version in Powertrain. Keep all of these inputs optional. Preserve the outing’s default driver and event/layout, freeform Settings and Notes, and the existing lap behavior: time is required; valid/invalid status, driver, time of day, notes, and attachments remain available.

Store `engine_tune` as an optional key in `structured_settings_json`; trim nonblank text, omit it when blank, and display a blank field when older records do not contain it. No SQL column or table rename is needed. Bump the SQLite schema version from 2 to 3 as a compatibility fence because older app versions can otherwise open and resave JSON while dropping this key. Keep the version-1-to-2 migration explicit, then apply 2-to-3. Keep sharing package format version 1; new apps accept older records without `engine_tune`, while older apps reject records they cannot preserve.

## Attachments

Allow selecting multiple files in one action in the existing day, outing, and lap attachment frame. Stage the chosen files as one batch: if staging any file fails, remove copies created by that batch and leave the editor’s attachment list unchanged. Store each file as an opaque byte copy with its original name and safe generated path. Native binary and Parquet files use the existing `file` attachment role; the app does not inspect or convert their contents. Keep event/layout map attachments in their current map flow and keep opening one selected saved attachment at a time. Copying an outing to another day continues to copy its attachments and metadata without copying laps.

## Parquet dataset

Add one **Export Parquet…** action for the whole current log. It writes a new dataset directory containing `outings.parquet`, `laps.parquet`, `attachments.parquet`, and `manifest.json`. `outings.parquet` has one row per existing `Setup`, including outings with zero laps. Its stable `outing_uuid` is the setup ID; its user label is `Setup.name`. `laps.parquet` has one row per lap and links through `outing_uuid`. It preserves `time_ms` as an integer and the saved lap status, driver, and event/layout history. Both Parquet tables are emitted with explicit schemas even when they contain zero rows. There is no row for a test day that has no outing because this export covers outing and lap records.

Use strings for IDs, labels, enum choices, sprocket size, `engine_tune`, freeform text, `created_at`, and unvalidated `time_of_day`; use `date32` for validated test dates, `int64` for lap time and sequence, and nullable `float64` for numeric vehicle values and event length. Preserve existing units in the field mapping and manifest. The optional `engine_tune` column is null when missing or blank. `attachments.parquet` contains metadata references only, with `setup` owners represented as `outing`; it does not include file bytes. Relative paths refer to the source app data folder, so the dataset is not a backup or restore package.

Capture all records in one consistent SQLite snapshot before writing any table. Write into a sibling staging directory and publish the complete directory atomically only after all Parquet files and the manifest succeed. Reject an existing destination and destinations that alias the managed database, sidecars, lock, attachments, or backups. Remove only the exporter-owned staging directory after a failure. Keep the current offline ZIP sharing behavior, UUID identity, same-ID conflict handling, backup behavior, and 512 MiB per-file / 2 GiB aggregate attachment limits.

## Optional dependency

PyArrow is loaded only when the export action runs. The standard-library, Tkinter, and SQLite app remains usable without it; the app never installs packages or connects to a network. Put the optional package in `test_log/requirements-parquet.txt` and document manual installation. Build empty tables with an explicit Arrow schema through [`Table.from_pylist`](https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#pyarrow.Table.from_pylist), then write them with [`pyarrow.parquet.write_table`](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html). Use the official [Apache Arrow installation guidance](https://arrow.apache.org/install/) for the optional local setup.
