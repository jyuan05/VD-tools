# VD Vehicle Test Log

The offline desktop app records test days, outings, event/layouts, lap times, and copied attachments. It uses Python’s standard library, Tkinter, and SQLite; it does not install packages or connect to a network.

## Start the app

On Windows, double-click launch.cmd.

On macOS or Linux, run the portable shell launcher from any working directory:

    sh '/path/to/test_log/launch.sh' --data-dir "$HOME/Vehicle Test Data"
    sh '/path/to/test_log/launch.sh' --help

You can also start the app directly with `python3 -m vd_test_log` from this folder. The app uses `~/VDTestLog` by default on macOS and Linux. The launcher is invoked through `sh`, so it does not depend on an executable file permission.

On macOS, double-click `launch.command` in Finder to start the app in Terminal. It requires Python 3.12 or newer with Tkinter and sqlite3. If an archive extractor removed its executable permission, run this once in Terminal:

    chmod +x '/path/to/test_log/launch.command'

For direct module use, run from this folder with a compatible interpreter:

    python -m vd_test_log
    python -m vd_test_log --data-dir 'D:\Vehicle Test Data\October'
    python -m vd_test_log --help

Help prints usage without creating the data folder.

## Data and attachments

The selected data folder contains test_log.sqlite3, an attachments folder, a backups folder, and the .test_log.lock file held while the app is open. Keep the database and its attachment folder together.

When you attach a file, the app copies it into its data folder under a generated safe name. It leaves the source file untouched. CSV exports contain lap and event details, not attachment contents.

CSV export destinations may be outside the data folder or ordinary files within it. The app protects its database, SQLite sidecars, active lock file, copied attachments, and backup contents from being selected as an export destination.
Parquet exports create a new directory and apply the same protected-path rules to the database, sidecars, lock, attachments, and backups.

## Backups and restore

Choose Backup in the app to create a timestamped folder under the data folder’s backups directory. Each completed backup contains a consistent test_log.sqlite3 snapshot and the attachment copies referenced by that snapshot in an attachments folder.

To reopen a backup, pass its timestamped folder as the data directory:

    & 'C:\VD Tools\test_log\launch.cmd' -DataDir 'C:\Users\you\AppData\Local\VDTestLog\backups\20261005T120000.000000Z-a1b2c3d4'

This opens the snapshot in place. Copy the backup folder first if you want to keep an untouched copy.

## Outing defaults and lap records

An outing can hold an optional event/layout and driver default. Choose an active event/layout on the outing before adding a lap. New laps copy the outing’s event/layout and driver into their saved history; later outing changes apply to future laps only. Historical laps show their saved event/layout and driver as context.

The outing editor keeps the freeform Settings and Notes fields and adds optional structured inputs in General, Aero, Suspension, Powertrain, and Corners. Suspension contains spring and damping values plus the rear anti-roll-bar blade and motion-ratio choices. Powertrain contains differential ramp angle and preload, sprocket size, and the optional engine-tune name/version. Blank vehicle values stay unset. Spring rates use lb/in, damping ratios are dimensionless, differential ramp angle uses degrees, differential preload uses ft-lb, camber and toe use degrees, tire pressure uses PSI, and corner weight uses lb. Sprocket size is tooth-count text. The dropdown guidance identifies the front-wing height order, rear-wing downforce order, and rear anti-roll-bar blade and motion-ratio order.

SQLite schema version 3 is a compatibility fence for `engine_tune`: an older schema-2 app will refuse this database rather than risk dropping that value when it saves. Package format remains version 1.

Lap time is the only required lap entry. New laps start as valid; invalid remains selectable. Time of day, notes, and file attachments are optional. The CSV export keeps its existing columns and order.
## CSV columns

An outing CSV export keeps these existing columns and their order for compatibility:

    test_date, test_location, setup_name, setup_code, lap_sequence, lap_time, status, driver, time_of_day, notes, track_name, event_name, layout_name, event_type, length_m

Lap times use m:ss.sss formatting. Valid and invalid laps are included.

To carry a saved outing to another test day, open the outing and choose Copy Outing to another day. The copy keeps its label, Outing ID, Settings, Notes, driver and event defaults, and engine tune, with independent copies of its attachments; laps stay with the original outing.

## Whole-log Parquet export

Choose **Export Parquet…** to write the current saved log into a new dataset directory. The export includes every outing, including outings with no laps, every lap, and attachment metadata. It does not add a row for a test day that has no outing. Existing directories and files are not overwritten.

The dataset contains four files:

- `outings.parquet`: one row per outing. `outing_uuid` is the existing setup ID; `test_day_uuid` identifies its day. The default event/layout columns describe the outing default at export time.
- `laps.parquet`: one row per lap, joined to its outing with `outing_uuid`. Lap rows retain their saved event/layout ID and descriptive history, driver, status, integer `time_ms`, sequence, and optional time-of-day and notes.
- `attachments.parquet`: attachment IDs, owner IDs, role, original filename, and relative path. Stored `setup` owners are called `outing` in this table. File contents are not included; paths refer to the source app data folder. This dataset is not a backup or a self-contained attachment transfer.
- `manifest.json`: dataset format version, table filenames and row counts, units, and the attachment path/content limitation.

IDs, labels, choices, freeform text, timestamps, sprocket size, and engine-tune name/version are strings. Validated test dates use Arrow `date32`; lap time and sequence use `int64`; numeric vehicle values and event lengths use nullable `float64`. Missing optional values, including engine tune, stay null. Spring rates use lb/in; damping ratios are dimensionless; rear anti-roll-bar choices retain their labels; differential ramp angle uses degrees and preload uses ft-lb; camber and toe use degrees, pressure uses PSI, corner weight uses lb, and event length uses m. Sprocket size is tooth-count text.

PyArrow is an optional dependency used only for Parquet export. From the `test_log` folder, install it with the same Python interpreter that launches the app:

    python -m pip install -r requirements-parquet.txt

The app does not install packages or use the network. Startup, SQLite records, selected-outing CSV export, and ZIP sharing continue to work without PyArrow. CSV headers and selected-outing scope are unchanged; ZIP packages continue to carry attachment bytes.

## Exchange logs between users

Each person should keep their own data folder. To send saved records to another person, choose **Sharing > Export Log Package…** and save the ZIP package. The source label is optional and helps identify who sent it. A package contains saved test days, outings, laps, event/layout definitions and their history, plus copied attachments and map files.

The recipient opens the data folder that will hold the combined log, chooses **Sharing > Import Log Package…**, and selects the ZIP. The preview shows the source label, incoming records, records that will be added or skipped, and any conflicts. A conflict blocks a merge so existing records are not overwritten. The recipient can choose **Import as a separate snapshot** to add a remapped copy of the incoming log; a changed log may duplicate previously imported history. Re-importing an unchanged package is safe and skips records already present. The existing package limits remain 512 MiB per attachment and 2 GiB total per package.

Record IDs distinguish sessions. Separate test days that happen to use the same date and location stay separate; the app does not automatically coalesce them. Imports only add or skip records and files. Deletions from one person's log do not propagate to another person's log. There is no server, account, or automatic sync, and users should not edit the same live data folder at the same time.

To exchange a log created by an older version, open that data folder with the updated app, save any pending edits, and export a package from the **Sharing** menu.
