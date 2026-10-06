#!/bin/sh
set -eu

case "$0" in
    */*) script_dir=${0%/*} ;;
    *) script_dir=. ;;
esac
[ -n "$script_dir" ] || script_dir=/
script_dir=$(CDPATH= cd "$script_dir" && pwd)

python_bin=
for candidate in python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sqlite3, sys, tkinter; sys.exit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1; then
        python_bin=$candidate
        break
    fi
done

if [ -z "$python_bin" ]; then
    printf '%s\n' 'Python 3.12 or newer with Tkinter and sqlite3 is required.' >&2
    exit 1
fi

if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH="$script_dir:$PYTHONPATH"
else
    PYTHONPATH="$script_dir"
fi
export PYTHONPATH

exec "$python_bin" -m vd_test_log "$@"
