# VD Vehicle Test Log

The offline desktop app records test days, vehicle setups, event/layouts, lap times, and copied attachments. It uses Python’s standard library, Tkinter, and SQLite; it does not install packages or connect to a network.

## Start the app

On Windows, double-click launch.cmd. To choose a data folder, run the launcher from any working directory:

    & 'C:\VD Tools\test_log\launch.cmd' -DataDir 'D:\Vehicle Test Data\October'

You can call the PowerShell launcher directly with the same option:

    & 'C:\VD Tools\test_log\launch.ps1' -DataDir 'D:\Vehicle Test Data\October'

A relative -DataDir is resolved against the directory from which the launcher was started. Without -DataDir, the app uses %LOCALAPPDATA%\VDTestLog, or VDTestLog under the user’s home directory when LOCALAPPDATA is unavailable.

The launcher checks py -3.12, then python, then the bundled Codex Python runtime. It accepts the first Python 3.12+ interpreter that imports sqlite3 and can create, withdraw, update, and destroy a real Tk window. Install no dependencies; if no candidate works, install or repair Python 3.12+ with Tcl/Tk enabled.

For direct module use, run from this folder with a compatible interpreter:

    python -m vd_test_log
    python -m vd_test_log --data-dir 'D:\Vehicle Test Data\October'
    python -m vd_test_log --help

Help prints usage without creating the data folder.

## Data and attachments

The selected data folder contains test_log.sqlite3, an attachments folder, a backups folder, and the .test_log.lock file held while the app is open. Keep the database and its attachment folder together.

When you attach a file, the app copies it into its data folder under a generated safe name. It leaves the source file untouched. CSV exports contain lap and event details, not attachment contents.

CSV export destinations may be outside the data folder or ordinary files within it. The app protects its database, SQLite sidecars, active lock file, copied attachments, and backup contents from being selected as an export destination.

## Backups and restore

Choose Backup in the app to create a timestamped folder under the data folder’s backups directory. Each completed backup contains a consistent test_log.sqlite3 snapshot and the attachment copies referenced by that snapshot in an attachments folder.

To reopen a backup, pass its timestamped folder as the data directory:

    & 'C:\VD Tools\test_log\launch.cmd' -DataDir 'C:\Users\you\AppData\Local\VDTestLog\backups\20261005T120000.000000Z-a1b2c3d4'

This opens the snapshot in place. Copy the backup folder first if you want to keep an untouched copy.

## Setup defaults and lap records

A setup can hold an optional event/layout and driver default. Choose an active event/layout on the setup before adding a lap. New laps copy the setup’s event/layout and driver into their saved history; later setup changes apply to future laps only. Historical laps show their saved event/layout and driver as context.

The setup editor keeps the freeform Settings and Notes fields and adds optional structured inputs in tabs. Blank vehicle values stay unset. Spring rates use lb/in, damping ratios are dimensionless, differential ramp angle uses degrees, differential preload uses ft-lb, camber and toe use degrees, tyre pressure uses PSI, and corner weight uses lb. Sprocket size is free text with no assumed unit. The dropdown guidance identifies the front-wing height order, rear-wing downforce order, and rear anti-roll-bar blade and motion-ratio order.

Lap time is the only required lap entry. New laps start as valid; invalid remains selectable. Time of day, notes, and file attachments are optional. The CSV export keeps its existing columns and order.
## CSV columns

A setup export uses these columns, in order:

    test_date, test_location, setup_name, setup_code, lap_sequence, lap_time, status, driver, time_of_day, notes, track_name, event_name, layout_name, event_type, length_m

Lap times use m:ss.sss formatting. Valid and invalid laps are included.

To carry a saved setup to another test day, open the setup and choose Copy to another day. The copy keeps its setup label, Setup ID, settings, driver and event defaults, with independent copies of its attachments; laps stay with the original setup.

## Tests

Run the complete test suite from the project folder. Tests use temporary data folders.

    Set-Location 'C:\VD Tools\test_log'
    & 'C:\Users\johny\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest discover -s tests -v