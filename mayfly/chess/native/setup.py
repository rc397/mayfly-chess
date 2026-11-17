"""Build the compiled parts of the search beside chess/.

  mayfly_pick     (pick.c)     the compiled pick, sealed native_pick (v5)
  mayfly_movegen  (movegen.c)  legal moves, index rows, encoding and the forcing mask, under timing_fixes (v5)
  mayfly_walk     (walk.c)     the descent, unwind and backup over the Python nodes, under timing_fixes (v5)

  Windows:  cd chess/native && python setup.py build_ext --inplace && move mayfly_walk*.pyd ..
  Linux:    sh chess/native/build_linux.sh     (PYTHON=/path/to/python picks the interpreter)

MSVC at /fp:precise, which never contracts a multiply and an add into one fused operation. Bit-identity
with numpy's float32 depends on it. No /arch flag, so plain x86-64 SSE2 and the files run on any x86-64
machine. The same CPython (3.13, x64) must load them, so a .pyd built on one machine can be copied to
another without a compiler. A rebuilt .pyd is not byte-identical to the last (the linker stamps it), so a
module already proven is only replaced when its source changes.

gcc and clang get the same guarantees from -ffp-contract=off (never fuse a multiply and an add) and
-fno-fast-math (IEEE rounding, NaNs and the order of operations kept). No -march, so a Linux build runs on
any x86-64 machine with the same CPython. The .so is named for that CPython.
"""
import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# the flags follow the compiler that builds, not the platform name
MSVC_FLAGS = ["/O2", "/fp:precise"]
GCC_FLAGS = ["-O2", "-ffp-contract=off", "-fno-fast-math"]


class BuildExt(build_ext):
    def build_extensions(self):
        flags = MSVC_FLAGS if self.compiler.compiler_type == "msvc" else GCC_FLAGS
        for ext in self.extensions:
            ext.extra_compile_args = list(flags)
        super().build_extensions()


setup(name="mayfly_native", cmdclass={"build_ext": BuildExt}, ext_modules=[
    Extension("mayfly_pick", ["pick.c"]),
    Extension("mayfly_movegen", ["movegen.c"], include_dirs=[numpy.get_include()]),
    Extension("mayfly_walk", ["walk.c"], include_dirs=[numpy.get_include()]),
])
