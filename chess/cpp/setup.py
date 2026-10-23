"""Build mayfly_cppsearch without LibTorch (the tree only, the network then runs through Python) with MSVC and
CPython 3.13.

  cd chess/cpp && python setup.py build_ext --inplace

/O2 /fp:precise as in chess/native (no fused multiply-add, no fast math, no /arch), C++17. The build with the
network in C++ as well is build_rig.bat.
"""
from setuptools import Extension, setup

setup(name="mayfly_cppsearch", ext_modules=[
    Extension("mayfly_cppsearch", ["pyext.cpp", "search2.cpp", "chess_core.cpp"],
              extra_compile_args=["/O2", "/fp:precise", "/std:c++17", "/EHsc"]),
])
