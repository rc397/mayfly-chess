#!/usr/bin/env bash
# build_linux.sh is build_rig.bat for Linux: mayfly_cppsearch with the network in C++,
# pyext + search2 + chess_core + torchnet + net_backend2 (LibTorch, CUDA graphs: NET_GRAPH), for the Python that runs
# this script and linked against that Python's torch. No CUDA net (MAYFLY_CUDANET): v6 seals cuda_net false.
# -O2 -ffp-contract=off -fno-fast-math and no -march, as /O2 /fp:precise: no fused multiply-add, no fast math.
# The module loads only beside the torch it was built against, after `import torch` (mcts_cpp imports torch first).
# Needs g++, the Python headers and a CUDA build of torch from pip; no CUDA toolkit.
#   PYTHON    the Python to build for (default python3)
#   OUT       the build folder (default build, beside this script)
#   DEST      where the module is copied for mcts_cpp to import (default .., chess/); empty to leave it in OUT
#   JOBS      compiles at once (default 4)
#   CUDA_INC  the CUDA runtime headers NET_GRAPH needs, CUDART the libcudart to link (a path). Default: the
#             nvidia-cuda-runtime wheel torch came with (the CUDA torch was built with, the libcudart torch loads),
#             with cuda_shim_gcc for the crt folder that only nvcc ships; without that wheel, CUDA_HOME's toolkit
# Usage: PYTHON=~/venv/bin/python bash chess/cpp/build_linux.sh
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
OUT=${OUT:-build}
DEST=${DEST-..}
JOBS=${JOBS:-4}
CXX=${CXX:-g++}
mkdir -p "$OUT"

# every path from the Python and its torch: torch.utils.cpp_extension knows LibTorch's own
VARS=$(CUDA_INC="${CUDA_INC:-}" CUDART="${CUDART:-}" "$PYTHON" - <<'EOF'
import os, shlex, sysconfig
import torch
from torch.utils import cpp_extension as ce

if torch.version.cuda is None:
    raise SystemExit("this torch has no CUDA; the C++ network needs a CUDA build of torch")
inc, lib = os.environ.get("CUDA_INC"), os.environ.get("CUDART")
if not inc or not lib:
    try:   # the CUDA runtime wheel torch's pip install brought
        import nvidia.cuda_runtime as m
        base = list(m.__path__)[0]
        libs = sorted(f for f in os.listdir(os.path.join(base, "lib")) if f.startswith("libcudart.so"))
        if os.path.exists(os.path.join(base, "include", "cuda_runtime_api.h")) and libs:
            inc, lib = inc or os.path.join(base, "include"), lib or os.path.join(base, "lib", libs[0])
    except ImportError:
        pass
if (not inc or not lib) and ce.CUDA_HOME:   # else the CUDA toolkit's
    inc = inc or os.path.join(ce.CUDA_HOME, "include")
    lib = lib or os.path.join(ce.CUDA_HOME, "lib64", "libcudart.so")
if not inc or not os.path.exists(os.path.join(inc, "cuda_runtime_api.h")):
    raise SystemExit(f"no CUDA runtime headers (cuda_runtime_api.h) in {inc}; set CUDA_INC")
if not lib or not os.path.exists(lib):
    raise SystemExit(f"no libcudart at {lib}; set CUDART")
v = {"PY_INC": sysconfig.get_paths()["include"], "EXT": sysconfig.get_config_var("EXT_SUFFIX"),
     "TORCH_INC": " ".join(ce.include_paths()), "TORCH_LIB": ce.library_paths()[0],
     "ABI": str(int(torch._C._GLIBCXX_USE_CXX11_ABI)), "CUDA_INC": inc, "CUDART": lib,
     "SHIM": "0" if os.path.exists(os.path.join(inc, "crt", "host_config.h")) else "1",
     "TORCH_VER": torch.__version__, "TORCH_CUDA": torch.version.cuda}
for k, x in v.items():
    print(f"{k}={shlex.quote(x)}")
EOF
)
eval "$VARS"
TI=()
for d in $TORCH_INC; do TI+=(-I"$d"); done
TI+=(-I"$CUDA_INC")
[ "$SHIM" = 1 ] && TI+=(-I"$(pwd)/cuda_shim_gcc")

# C++17 and the torch's string ABI everywhere (pyext hands torchnet a std::string)
COMMON=(-std=c++17 -O2 -ffp-contract=off -fno-fast-math -fPIC -D_GLIBCXX_USE_CXX11_ABI="$ABI")
CF=("${COMMON[@]}" -Wall -DMAYFLY_TORCH -I"$PY_INC")
TF=("${COMMON[@]}" -w -DNET_GRAPH "${TI[@]}")
echo "torch $TORCH_VER (CUDA $TORCH_CUDA); CUDA headers $CUDA_INC$([ "$SHIM" = 1 ] && echo " + cuda_shim_gcc"); $CUDART"

# the five compiles, JOBS at a time
pids=()
fail=0
cc() {
    "$CXX" "$@" &
    pids+=($!)
    if (( ${#pids[@]} >= JOBS )); then
        wait "${pids[0]}" || fail=1
        pids=("${pids[@]:1}")
    fi
}
cc "${TF[@]}" -c torchnet.cpp -o "$OUT/torchnet.o"
cc "${TF[@]}" -c net_backend2.cpp -o "$OUT/net_backend2.o"
cc "${CF[@]}" -c chess_core.cpp -o "$OUT/chess_core.o"
cc "${CF[@]}" -c search2.cpp -o "$OUT/search2.o"
cc "${CF[@]}" -c pyext.cpp -o "$OUT/pyext.o"
for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) || { echo "a compile failed" >&2; exit 1; }

# torch_cuda kept even if unreferenced (build_rig.bat's /INCLUDE does the same); Python's own symbols resolve at load
SO="$(cd "$OUT" && pwd)/mayfly_cppsearch$EXT"
"$CXX" -shared -o "$SO" "$OUT/pyext.o" "$OUT/search2.o" "$OUT/chess_core.o" "$OUT/torchnet.o" "$OUT/net_backend2.o" \
    -L"$TORCH_LIB" -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -Wl,--as-needed \
    -L"$(dirname "$CUDART")" -l:"$(basename "$CUDART")"
echo "built $SO"
if [ -n "$DEST" ]; then
    cp -f "$SO" "$DEST/"
    echo "copied to $(cd "$DEST" && pwd)/$(basename "$SO")"
fi
