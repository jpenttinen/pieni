#!/usr/bin/env bash
# Launcher for pieni.py. The workspace is the directory you launch it from.
set -eu
# Follow symlinks, so an installed ~/.local/bin/pieni still finds pieni.py and
# .venv next to the real file in this checkout. macOS/BSD readlink does not
# support -f, so walk the symlink chain explicitly instead.
launcher="$0"
while [ -L "$launcher" ]; do
    link="$(readlink "$launcher" 2>/dev/null || true)"
    if [ -z "$link" ]; then
        break
    fi
    case "$link" in
        /*) launcher="$link" ;;
        *) 
            cd "$(dirname "$launcher")"
            launcher="$PWD/$link"
            ;;
    esac
done
cd "$(dirname "$launcher")"
here="$PWD"
if [ -x "$here/.venv/bin/python" ]; then
    exec "$here/.venv/bin/python" "$here/pieni.py" "$@"
fi
exec python3 "$here/pieni.py" "$@"
