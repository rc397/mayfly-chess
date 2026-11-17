"""Build mayfly_cppsearch without LibTorch (the tree only, the network then runs through Python) with MSVC and
CPython 3.13, or with gcc on Linux.

  cd chess/cpp && python setup.py build_ext --inplace

/O2 /fp:precise as in chess/native (no fused multiply-add, no fast math, no /arch), C++17. On gcc the same
comes from -O2 -ffp-contract=off -fno-fast-math, with no -march. The build with the network in C++ as well is
build_rig.bat on Windows and build_linux.sh on Linux.
"""
import sys

from setuptools import Extension, setup

if sys.platform == "win32":
    FLAGS = ["/O2", "/fp:precise", "/std:c++17", "/EHsc"]
else:
    FLAGS = ["-O2", "-ffp-contract=off", "-fno-fast-math", "-std=c++17"]

setup(name="mayfly_cppsearch", ext_modules=[
    Extension("mayfly_cppsearch", ["pyext.cpp", "search2.cpp", "chess_core.cpp"], extra_compile_args=FLAGS),
])
