"""Build the compiled parts of the search beside chess/.

  mayfly_pick     (pick.c)     the compiled pick, sealed native_pick (v5)
  mayfly_movegen  (movegen.c)  legal moves, index rows, encoding and the forcing mask, under timing_fixes (v5)
  mayfly_walk     (walk.c)     the descent, unwind and backup over the Python nodes, under timing_fixes (v5)

  cd chess/native && python setup.py build_ext --inplace && move mayfly_walk*.pyd ..

MSVC at /fp:precise, which never contracts a multiply and an add into one fused operation. Bit-identity
with numpy's float32 depends on it. No /arch flag, so plain x86-64 SSE2 and the files run on any x86-64
machine. The same CPython (3.13, x64) must load them, so a .pyd built on one machine can be copied to
another without a compiler. A rebuilt .pyd is not byte-identical to the last (the linker stamps it), so a
module already proven is only replaced when its source changes.
"""
import numpy
from setuptools import Extension, setup

setup(name="mayfly_native", ext_modules=[
    Extension("mayfly_pick", ["pick.c"], extra_compile_args=["/O2", "/fp:precise"]),
    Extension("mayfly_movegen", ["movegen.c"], include_dirs=[numpy.get_include()],
              extra_compile_args=["/O2", "/fp:precise"]),
    Extension("mayfly_walk", ["walk.c"], include_dirs=[numpy.get_include()],
              extra_compile_args=["/O2", "/fp:precise"]),
])
