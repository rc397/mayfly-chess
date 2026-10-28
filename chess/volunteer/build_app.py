"""build_app -- MayFlyVolunteer.exe, one file, with PyInstaller.

What goes in: app.py and its window (ui/), the kit (kit/: versions.json with the sealed searches, corpus_starts.txt),
the search code selfplay_remote imports, ONNX Runtime with DirectML, pywebview, scipy (v6.1's sealed search uses the
Student-t lower confidence bound) and the compiled search helpers (mayfly_pick, mayfly_movegen, mayfly_walk; the
numpy lines run without them). What stays out: PyTorch (the workers use torch_stub) and JAX. The network itself is
not in the exe: the app downloads it from the release the first time.

  python chess/volunteer/build_app.py
  Run it with a Python that has only the app's packages (numpy, scipy, chess, onnxruntime-directml, pywebview,
  pyinstaller): anything else installed, PyTorch above all, can end up inside the exe.
  env: BUILD_DIR=data/volunteer_app/build  LIBS= (an extra package folder)  NATIVE=chess (the compiled helpers)
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHESS = os.path.dirname(HERE)
sys.path.insert(0, CHESS)
import paths  # noqa: E402

E = os.environ.get
BUILD = E("BUILD_DIR", paths.under("volunteer_app", "build"))
LIBS = E("LIBS", "")
NATIVE = E("NATIVE", CHESS)

# --disable-windowed-traceback: a crash in a background process must never leave an error box waiting for a click
# (the process would hang alive); errors go to the engine's pipe and the app shows them
args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed", "--disable-windowed-traceback",
        "--name", "MayFlyVolunteer",
        "--distpath", os.path.join(BUILD, "dist"), "--workpath", os.path.join(BUILD, "work"),
        "--specpath", BUILD, "--paths", CHESS,
        # numpy 2's internals are imported in ways the analysis misses (numpy._core._exceptions on 28 Oct)
        "--collect-submodules", "numpy",
        "--add-data", f"{os.path.join(HERE, 'ui')};ui", "--add-data", f"{os.path.join(HERE, 'kit')};kit"]
if LIBS and os.path.isdir(LIBS):
    args += ["--paths", LIBS]
for pyd in sorted(glob.glob(os.path.join(NATIVE, "mayfly_*.cp313-win_amd64.pyd"))):
    if "cppsearch" not in pyd:        # the C++ tree needs libtorch; remote workers search in Python anyway
        args += ["--add-binary", f"{pyd};."]
for mod in ("selfplay_remote", "selfplay", "mcts", "chess_net", "encode", "moves", "lc0_policy_index", "book",
            "volunteer.torch_stub", "volunteer.gpus", "scipy.stats", "scipy._cyutility", "onnxruntime", "webview"):
    args += ["--hidden-import", mod]
# cupy would bring CUDA's DLLs (found in torch's lib folder, so PyTorch's too) and pywebview's hook adds Qt, which
# Windows doesn't use: pywebview runs on Edge WebView2 there
for mod in ("torch", "torchvision", "jax", "jaxlib", "chessnet_jax", "matplotlib", "tkinter", "IPython", "pandas",
            "sympy", "PIL", "tensorflow", "cv2", "cupy", "cupyx", "cupy_backends", "numba", "llvmlite",
            "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy", "gi"):
    args += ["--exclude-module", mod]
args.append(os.path.join(HERE, "app.py"))

print(" ".join(a if " " not in a else f'"{a}"' for a in args[3:]), flush=True)
rc = subprocess.call(args)
exe = os.path.join(BUILD, "dist", "MayFlyVolunteer.exe")
if rc == 0 and os.path.exists(exe):
    print(f"built {exe}: {os.path.getsize(exe) / 1e6:.0f} MB")
sys.exit(rc)
