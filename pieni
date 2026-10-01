#!/usr/bin/env bash
# Launcher for pieni.py. The workspace is the directory you launch it from.
set -eu
# Follow symlinks, so an installed ~/.local/bin/pieni still finds pieni.py and
# .venv next to the real file in this checkout.
launcher="$0"
resolved="$(readlink -f "$launcher" 2>/dev/null || true)"
if [ -n "$resolved" ]; then
    launcher="$resolved"
fi
here="$(cd "$(dirname "$launcher")" && pwd)"
if [ -x "$here/.venv/bin/python" ]; then
    exec "$here/.venv/bin/python" "$here/pieni.py" "$@"
fi
exec python3 "$here/pieni.py" "$@"
