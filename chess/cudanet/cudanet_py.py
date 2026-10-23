"""cudanet.dll from Python through ctypes, per cudanet_capi.h.

    net = CudaNet("mayfly_v6.cnw")
    pol, val, qv = net.evaluate(ids, cast, hm, ep, I, M)

The inputs are what MCTS._prepare builds (export_ts.encode), ids (B, 64) int64, cast (B, 4) float32, hm (B,) int64,
ep (B,) int64, I (B, wide) int64 and M (B, wide) uint8 (bool is taken too). The outputs are what _collect hands on
(cpp/netbackend.h), pol (B, wide) the raw policy logits at each row's legal moves (0 on padding), val (B,) and
qv (B, wide), all float32. B is 1 to net.max_batch (48, or env MAYFLY_CUDANET_MAX_BATCH when the net was made), one
batch a call, synchronous.

The DLL is MAYFLY_CUDANET_DLL, else chess/cudanet/build/cudanet.dll. MAYFLY_CUDA_BIN, else the bin folder of
CUDA_PATH, goes on the DLL search path first, for the CUDA libraries the DLL loads at runtime.
"""
import ctypes
import os
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_BIN = os.path.join(os.environ["CUDA_PATH"], "bin") if os.environ.get("CUDA_PATH") else ""
MAX_BATCH = 48          # cudanet_spec.h kMaxBatch (the default, builds with cudanet_max_batch report their own)
ERR_LEN = 2048

_dll_dirs = []          # add_dll_directory handles, kept alive for the process
_libs = {}              # path -> loaded CDLL, one load per path


def dll_path(dll=None):
    return dll or os.environ.get("MAYFLY_CUDANET_DLL") or os.path.join(HERE, "build", "cudanet.dll")


def _search_dirs(path):
    out = []
    for d in (os.environ.get("MAYFLY_CUDA_BIN"), KIT_BIN, os.path.dirname(os.path.abspath(path))):
        if d and os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def _load(path):
    path = os.path.abspath(path)
    if path in _libs:
        return _libs[path]
    if not os.path.exists(path):
        raise FileNotFoundError(f"no cudanet.dll at {path} (set MAYFLY_CUDANET_DLL or build chess/cudanet)")
    dirs = _search_dirs(path)
    for d in dirs:
        if hasattr(os, "add_dll_directory"):
            _dll_dirs.append(os.add_dll_directory(d))
    # PATH as well, for a LoadLibrary by bare name inside the DLL
    os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])
    lib = ctypes.CDLL(path)
    vp, i32, cp = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    lib.cudanet_create.argtypes = [cp, i32, ctypes.c_char_p, i32]
    lib.cudanet_create.restype = vp
    lib.cudanet_evaluate.argtypes = [vp, i32, i32, vp, vp, vp, vp, vp, vp, vp, vp, vp, ctypes.c_char_p, i32]
    lib.cudanet_evaluate.restype = i32
    lib.cudanet_fingerprint.argtypes = [vp, ctypes.c_char_p, i32]
    lib.cudanet_fingerprint.restype = i32
    lib.cudanet_destroy.argtypes = [vp]
    lib.cudanet_destroy.restype = None
    if hasattr(lib, "cudanet_max_batch"):        # older builds take MAX_BATCH only
        lib.cudanet_max_batch.argtypes = [vp]
        lib.cudanet_max_batch.restype = i32
    if hasattr(lib, "cudanet_profile"):          # older builds have no profile
        lib.cudanet_profile.argtypes = [vp, i32, i32, ctypes.POINTER(ctypes.c_double), i32, ctypes.c_char_p, i32]
        lib.cudanet_profile.restype = i32
        lib.cudanet_profile_names.argtypes = [ctypes.c_char_p, i32]
        lib.cudanet_profile_names.restype = i32
    _libs[path] = lib
    return lib


def _arr(a, dtype, shape, name):
    """A C-contiguous array of exactly this dtype and shape (copied only when it has to be)."""
    a = np.asarray(a)
    if dtype is np.uint8 and a.dtype == np.bool_:
        a = a.view(np.uint8)
    a = np.ascontiguousarray(a, dtype=dtype)
    if a.shape != shape:
        raise ValueError(f"{name}: shape {a.shape}, expected {shape}")
    return a


class CudaNet:
    """One cudanet handle, with the weights on one card and a CUDA graph per batch size. One call at a time."""

    def __init__(self, weights_path, device=0, dll=None):
        self.path = os.path.abspath(weights_path)
        self.device = int(device)
        self.dll = dll_path(dll)
        self._lib = _load(self.dll)
        self._lock = threading.Lock()
        err = ctypes.create_string_buffer(ERR_LEN)
        h = self._lib.cudanet_create(self.path.encode("utf-8"), self.device, err, ERR_LEN)
        if not h:
            raise RuntimeError(f"cudanet_create({self.path}): {err.value.decode('utf-8', 'replace')}")
        self._h = ctypes.c_void_p(h)
        self.max_batch = MAX_BATCH
        if hasattr(self._lib, "cudanet_max_batch"):
            self.max_batch = int(self._lib.cudanet_max_batch(self._h))

    def evaluate(self, ids, cast, hm, ep, I, M):
        """One batch. Returns (pol (B, wide), val (B,), qv (B, wide)) float32."""
        if self._h is None:
            raise RuntimeError("CudaNet is closed")
        I = np.asarray(I)
        if I.ndim != 2:
            raise ValueError(f"I: shape {I.shape}, expected (B, wide)")
        B, wide = I.shape
        if not 1 <= B <= self.max_batch:
            raise ValueError(f"batch {B}: cudanet takes 1 to {self.max_batch} (env MAYFLY_CUDANET_MAX_BATCH)")
        if wide < 1:
            raise ValueError("wide must be at least 1")
        ids = _arr(ids, np.int64, (B, 64), "ids")
        cast = _arr(cast, np.float32, (B, 4), "cast")
        hm = _arr(hm, np.int64, (B,), "hm")
        ep = _arr(ep, np.int64, (B,), "ep")
        I = _arr(I, np.int64, (B, wide), "I")
        M = _arr(M, np.uint8, (B, wide), "M")
        pol = np.zeros((B, wide), np.float32)
        val = np.zeros(B, np.float32)
        qv = np.zeros((B, wide), np.float32)
        err = ctypes.create_string_buffer(ERR_LEN)
        p = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        with self._lock:
            rc = self._lib.cudanet_evaluate(self._h, B, wide, p(ids), p(cast), p(hm), p(ep), p(I), p(M),
                                            p(pol), p(val), p(qv), err, ERR_LEN)
        if rc != 0:
            raise RuntimeError(f"cudanet_evaluate: {err.value.decode('utf-8', 'replace')}")
        return pol, val, qv

    def fingerprint(self):
        """The weight file's fingerprint, as mcts_cpp.fingerprint computes it for the checkpoint."""
        if self._h is None:
            raise RuntimeError("CudaNet is closed")
        buf = ctypes.create_string_buffer(256)
        rc = self._lib.cudanet_fingerprint(self._h, buf, len(buf))
        if rc < 0:
            raise RuntimeError("cudanet_fingerprint failed")
        return buf.value.decode("utf-8", "replace")

    def profile(self, B, reps=20):
        """Milliseconds a pass by step at batch B (an empty board, eager, an event after each step), the mean of
        reps passes, and last graph_pass, the whole pass as the search runs it. A dict in pass order."""
        if self._h is None:
            raise RuntimeError("CudaNet is closed")
        if not hasattr(self._lib, "cudanet_profile"):
            raise RuntimeError(f"{self.dll} has no cudanet_profile (an older build)")
        buf = ctypes.create_string_buffer(512)
        self._lib.cudanet_profile_names(buf, len(buf))
        names = buf.value.decode("utf-8").split(",")
        ms = (ctypes.c_double * len(names))()
        err = ctypes.create_string_buffer(ERR_LEN)
        with self._lock:
            k = self._lib.cudanet_profile(self._h, int(B), int(reps), ms, len(names), err, ERR_LEN)
        if k < 0:
            raise RuntimeError(f"cudanet_profile: {err.value.decode('utf-8', 'replace')}")
        return dict(zip(names[:k], list(ms)[:k]))

    def close(self):
        if getattr(self, "_h", None) is not None:
            self._lib.cudanet_destroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
