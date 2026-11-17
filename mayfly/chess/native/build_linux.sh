#!/bin/sh
# Build mayfly_pick, mayfly_movegen and mayfly_walk on Linux and put the .so files beside chess/*.py.
#
#   sh chess/native/build_linux.sh                         python3 on the PATH (an active venv's)
#   PYTHON=~/venv/bin/python sh chess/native/build_linux.sh
#
# The files are built for that Python only: its version is in their names (cpython-312, ...), and
# any other CPython ignores them. Flags come from setup.py: gcc -O2 -ffp-contract=off -fno-fast-math,
# no -march, so they run on any x86-64 machine. Needs gcc, the Python headers (python3-dev) and
# numpy and setuptools in that Python. mcts.py checks the modules when it loads them.
set -eu

PY="${PYTHON:-python3}"
cd "$(dirname "$0")"

# the file name ending this Python loads, e.g. .cpython-312-x86_64-linux-gnu.so
SUFFIX=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

# a fresh build each time, so a stale object never reaches the .so
rm -rf build
"$PY" setup.py build_ext --inplace

for m in mayfly_pick mayfly_movegen mayfly_walk; do
    mv -f "$m$SUFFIX" "../$m$SUFFIX"
    echo "built chess/$m$SUFFIX"
done
rm -rf build
