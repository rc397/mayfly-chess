#!/usr/bin/env bash
# MayFly as a UCI engine on Linux, for a GUI, cutechess-cli or a rating list's harness: point it at
# this file. The Linux twin of mayfly_uci.bat. It runs from anywhere (through a symlink too), finds
# the folder it sits in, and starts uci.py with defaults relative to that folder's bundle (see
# make_release.py). Anything already set in the environment wins.
# Nothing here may print to stdout: on this channel stdout is the protocol.
#
#   VERSION         which network plays, a file in VERSION_DIR; unset, uci.py plays the newest there
#   VERSION_DIR     the versions folder (versions.json and the .pt), default <bundle>/versions
#   SYZYGY_DIR      syzygy tables, folders split by ':', default <bundle>/syzygy if it exists
#   PYTHON          the interpreter, default <bundle>/venv/bin/python if it exists, else python3
#   NET_OVERHEAD    seconds a move costs beyond the search, default 0.02, the release tests' value
#                   for a local GUI (set it empty for the Lichess pricing); UCI option Move Overhead
#   PONDER_MEM_GB   the search tree's memory ceiling in GB, default a third of the RAM, 1 to 12
#   AUTOCAST_GRAPH  1 (default) keeps a CUDA graph per batch size on the card: faster, more memory
#   VISITS          visits a move when the GUI gives no clock, time or node limit (default 400)
#   LOG             a file that gets every UCI line in and out, default none

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
BUNDLE="$(dirname "$HERE")"

# the versions folder: the bundle's, else one beside this file
if [ -z "${VERSION_DIR:-}" ]; then
    for d in "$BUNDLE/versions" "$HERE/versions"; do
        if [ -f "$d/versions.json" ]; then
            export VERSION_DIR="$d"
            break
        fi
    done
fi

# tables only if the bundle has them; without, the network plays the endings
if [ -z "${SYZYGY_DIR+x}" ] && [ -d "$BUNDLE/syzygy" ]; then
    export SYZYGY_DIR="$BUNDLE/syzygy"
fi

# the interpreter: a venv made inside the bundle, else the system's
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$BUNDLE/venv/bin/python" ]; then
        PYTHON="$BUNDLE/venv/bin/python"
    else
        PYTHON=python3
    fi
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "mayfly_uci.sh: no Python at '$PYTHON' (set PYTHON, or make <bundle>/venv)" >&2
    exit 1
fi

# a local GUI pays almost nothing to pass a move on, unlike a Lichess bot
export NET_OVERHEAD="${NET_OVERHEAD-0.02}"
[ -z "$NET_OVERHEAD" ] && unset NET_OVERHEAD

# the tree may use a third of the RAM, never more than uci.py's own 12 GB
if [ -z "${PONDER_MEM_GB:-}" ]; then
    kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)
    PONDER_MEM_GB=$(awk -v kb="${kb:-0}" 'BEGIN { g = kb / 1e6 / 3; if (g > 12) g = 12; if (g < 1) g = 1; printf "%.1f", g }')
    export PONDER_MEM_GB
fi

export AUTOCAST_GRAPH="${AUTOCAST_GRAPH:-1}"

exec "$PYTHON" "$HERE/uci.py" "$@"
