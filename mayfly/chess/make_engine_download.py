r"""make_engine_download builds the Windows download of MayFly as a UCI engine: one folder, no PyTorch, no set-up.

What goes into the folder:
  mayfly.exe, _internal\   the engine (uci.py and the search code it imports) with Python inside, made by
                           PyInstaller. PyTorch is left out: the search and the network run in the C++ module.
  versions\                ONE version: its row in versions.json (the sealed search, its name, its fingerprint)
                           and its .cnw weight file for the hand-written CUDA net (chess\cudanet)
  cuda\                    the four NVIDIA libraries the CUDA net loads and the two headers its kernels include,
                           copied unchanged
  README.txt, LICENSE.txt, THIRD_PARTY.txt

Only a version sealed with engine "cpp" and cuda_net can be packed, because nothing else runs without PyTorch.

  python chess/make_engine_download.py --version mayfly_v7_beta_2026-11-14.pt --versions-dir D:\versions ^
      --module-dir chess\cpp\build_notorch --cuda-bin D:\cuda_kit\bin --cuda-include D:\cuda_kit\include --out D:\out

  --module-dir   the folder cpp\build_notorch.bat wrote (the .pyd and the kernels_*.cu files)
  --cuda-bin     a folder with cublasLt64_12.dll, cudart64_12.dll, nvrtc64_120_0.dll and nvrtc-builtins64_*.dll
                 (NVIDIA's pip wheels nvidia-cublas-cu12, nvidia-cuda-runtime-cu12 and nvidia-cuda-nvrtc-cu12 hold them)
  --cuda-include a folder with cuda_fp16.h and cuda_fp16.hpp
  --cuda-version what to call that CUDA release in THIRD_PARTY.txt (default 12.8)
  --no-zip       leave the folder and write no zip
Needs PyInstaller, numpy, scipy and python-chess in the Python that runs it.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NVIDIA_DLLS = ("cublasLt64_12.dll", "cudart64_12.dll", "nvrtc64_120_0.dll", "nvrtc-builtins64_*.dll")
NVIDIA_HEADERS = ("cuda_fp16.h", "cuda_fp16.hpp")
EULA = "https://docs.nvidia.com/cuda/eula/index.html"

README = """\
{name}
{line}

MayFly is a chess engine: a transformer network of about 146 million parameters searched with a batched PUCT
tree. This folder is MayFly as a UCI engine for Windows, with the search this version was sealed with. It is a
beta of version 7, not an official release.

Source code, results and the licence: https://github.com/rc397/mayfly-chess

What you need
- Windows 10 or 11, 64-bit.
- An NVIDIA graphics card with a current driver. The network runs on the card through MayFly's own CUDA code.
  There is no fallback: without an NVIDIA card the engine does not start a search.
- Nothing to install. Python and PyTorch are not needed.

Using it
- Point a chess GUI or a match runner at mayfly.exe (Cute Chess, Banksia, Arena, cutechess-cli).
- The first search takes a few seconds longer: the network's GPU code is compiled once and then kept in
  %LOCALAPPDATA%\\mayfly_cudanet.
- UCI options: Visits (used only when the host gives no clock and no node count), Ponder, SyzygyPath (one or more
  tablebase folders; a folder named syzygy beside mayfly.exe is used by itself), Move Overhead (milliseconds a
  move costs beyond the search; -1 prices it by the gap between moves, as on Lichess), Version (this folder
  holds one).
- go with a clock, go movetime, go nodes (visits), go infinite and pondering are supported.

What is in the folder
- mayfly.exe and _internal: the engine. versions: this version's sealed search and its weights (.cnw).
- cuda: four NVIDIA CUDA libraries and two NVIDIA headers, unchanged. They are NVIDIA's, under NVIDIA's licence,
  and are here only so MayFly can run. See THIRD_PARTY.txt.

Licence
MayFly is free software under the GNU General Public License, version 3 or later (LICENSE.txt), with the
additional permission under section 7 to combine it with NVIDIA's CUDA libraries, as stated in the repository's
README. The NVIDIA files in cuda are not under the GPL.
"""

THIRD_PARTY = """\
Third-party files in this folder
================================

The folder "cuda" holds files that are not part of MayFly and are not under MayFly's licence:

{files}

They are NVIDIA Corporation's. They come from the NVIDIA CUDA Toolkit {cuda} as published in NVIDIA's own pip
packages (nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12 and the toolkit's headers) and are
copied here unmodified. They are distributed under the NVIDIA CUDA Toolkit End User License Agreement:
{eula}
which lists these libraries and the CUDA floating point headers as distributable with an application.
They are used only by MayFly, from MayFly's own folder: MayFly's network code loads the libraries at run time
and compiles its GPU kernels with them. They are not covered by the GNU General Public License.

MayFly itself is free software under the GNU General Public License, version 3 or later (LICENSE.txt). Its
copyright holders give this additional permission under section 7 of the GPL version 3: if you modify this
program, or any covered work, by linking or combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
Toolkit, including cuBLAS, cuBLASLt and NVRTC (or a modified version of those libraries), containing parts covered
by the terms of the respective license agreement, the licensors of this program grant you additional permission
to convey the resulting work.

The engine folder also contains Python and the Python packages numpy, scipy and python-chess, each under its own
licence (the Python Software Foundation licence, BSD licences and the GPL version 3). Their licence texts are in
their source distributions.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--versions-dir", required=True)
    ap.add_argument("--module-dir", default=os.path.join(HERE, "cpp", "build_notorch"))
    ap.add_argument("--cuda-bin", required=True)
    ap.add_argument("--cuda-include", required=True)
    ap.add_argument("--cuda-version", default="12.8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    rows = json.load(open(os.path.join(a.versions_dir, "versions.json"), encoding="utf-8-sig"))
    row = next((r for r in rows if r.get("file") == a.version), None)
    if row is None:
        raise SystemExit(f"no row for {a.version} in {a.versions_dir}")
    seal = row.get("search") or {}
    if seal.get("engine") != "cpp" or not seal.get("cuda_net"):
        raise SystemExit(f"{a.version} is not sealed for the CUDA net, so it cannot run without PyTorch")
    if not row.get("fingerprint") or not row.get("name"):
        raise SystemExit("the row needs a name and the weights' fingerprint (mcts_cpp.fingerprint)")
    stem = os.path.splitext(a.version)[0]
    cnw = os.path.join(a.versions_dir, stem + ".cnw")
    sys.path.insert(0, HERE)
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        from mcts_cpp import cnw_fingerprint
    if cnw_fingerprint(cnw) != row["fingerprint"]:
        raise SystemExit(f"{cnw} does not carry the row's fingerprint")
    pyd = glob.glob(os.path.join(a.module_dir, "mayfly_cppsearch*.pyd"))
    kernels = sorted(glob.glob(os.path.join(a.module_dir, "kernels_*.cu")))
    if len(pyd) != 1 or not kernels:
        raise SystemExit(f"{a.module_dir} needs one mayfly_cppsearch .pyd and its kernels_*.cu (cpp/build_notorch.bat)")
    dlls = [p for pat in NVIDIA_DLLS for p in sorted(glob.glob(os.path.join(a.cuda_bin, pat)))]
    heads = [os.path.join(a.cuda_include, h) for h in NVIDIA_HEADERS]
    if len(dlls) != len(NVIDIA_DLLS) or not all(os.path.exists(h) for h in heads):
        raise SystemExit("the four CUDA libraries or the two headers were not found")

    name = row["name"]
    folder = name.replace(" ", "-")
    work = os.path.join(a.out, "work")
    dist = os.path.join(a.out, "dist")
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--console", "--name", "mayfly",
            "--distpath", dist, "--workpath", work, "--specpath", work, "--paths", HERE,
            "--collect-submodules", "numpy", "--add-binary", f"{pyd[0]};."]
    for k in kernels:
        args += ["--add-data", f"{k};."]
    for p in sorted(glob.glob(os.path.join(HERE, "mayfly_*.cp313-win_amd64.pyd"))):
        if "cppsearch" not in p:                       # the compiled helpers of the Python side, when they are built
            args += ["--add-binary", f"{p};."]
    for mod in ("mcts", "mcts_cpp", "chess_net", "encode", "moves", "lc0_moves", "book", "timectl", "paths",
                "volunteer.torch_stub", "scipy.stats", "scipy._cyutility", "chess.syzygy"):
        args += ["--hidden-import", mod]
    for mod in ("torch", "torchvision", "jax", "jaxlib", "chessnet_jax", "matplotlib", "tkinter", "IPython", "pandas",
                "sympy", "PIL", "tensorflow", "cv2", "cupy", "cupyx", "cupy_backends", "numba", "llvmlite",
                "onnxruntime", "webview", "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy", "gi", "triton"):
        args += ["--exclude-module", mod]
    args.append(os.path.join(HERE, "uci.py"))
    if subprocess.call(args) != 0:
        raise SystemExit("PyInstaller failed")

    pkg = os.path.join(a.out, folder)
    if os.path.exists(pkg):
        raise SystemExit(f"{pkg} exists already; move it away first")
    shutil.copytree(os.path.join(dist, "mayfly"), pkg)
    os.makedirs(os.path.join(pkg, "versions"))
    os.makedirs(os.path.join(pkg, "cuda"))
    json.dump([row], open(os.path.join(pkg, "versions", "versions.json"), "w", encoding="utf-8"), indent=1)
    shutil.copy2(cnw, os.path.join(pkg, "versions", stem + ".cnw"))
    for p in dlls + heads:
        shutil.copy2(p, os.path.join(pkg, "cuda", os.path.basename(p)))
    files = "\n".join(f"  cuda\\{os.path.basename(p)}" for p in dlls + heads)
    with open(os.path.join(pkg, "README.txt"), "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write(README.format(name=name, line="=" * len(name)))
    with open(os.path.join(pkg, "THIRD_PARTY.txt"), "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write(THIRD_PARTY.format(files=files, cuda=a.cuda_version, eula=EULA))
    shutil.copy2(os.path.join(REPO, "LICENSE"), os.path.join(pkg, "LICENSE.txt"))
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(pkg) for f in fs)
    print(f"folder {pkg}: {size / 1e6:,.0f} MB")
    if not a.no_zip:
        z = os.path.join(a.out, folder + ".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for r, _, fs in os.walk(pkg):
                for f in sorted(fs):
                    p = os.path.join(r, f)
                    zf.write(p, os.path.join(folder, os.path.relpath(p, pkg)))
        print(f"zip {z}: {os.path.getsize(z) / 1e6:,.0f} MB")


if __name__ == "__main__":
    main()
