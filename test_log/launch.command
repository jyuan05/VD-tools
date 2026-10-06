#!/bin/sh
set -eu

case "$0" in
    */*) script_dir=${0%/*} ;;
    *) script_dir=. ;;
esac
[ -n "$script_dir" ] || script_dir=/
script_dir=$(CDPATH= cd "$script_dir" && pwd)

if sh "$script_dir/launch.sh" "$@"; then
    exit 0
else
    status=$?
fi

printf '\nVD Vehicle Test Log could not be started (exit status %s).\n' "$status" >&2
if [ -t 0 ]; then
    printf 'Press Return to close this window...'
    IFS= read -r _ || :
fi
exit "$status"
