"""A TensorRT engine of the search's forward pass (build_engine.py) behind cudanet_py.CudaNet's interface, so
test_cudanet's numerics and timings run on it unchanged.

    net = TrtNet("mayfly_v5_typed.engine")
    pol, val, qv = net.evaluate(ids, cast, hm, ep, I, M)

Runs in the TensorRT venv (setup_venv.bat). Memory, copies and CUDA graphs go through cudart64_12.dll by ctypes
(MAYFLY_CUDA_BIN, else the bin folder of CUDA_PATH, else the DLL search path), so no torch with CUDA is needed. As
in cudanet, one pinned upload of the inputs, the pass, one pinned download, synchronous. I and M are padded to the
engine's widest row (256), so the whole pass, _collect's arithmetic included, is one CUDA graph a batch size.
graph=False runs enqueueV3 directly (and with pad=False at each call's own width).
"""
import ctypes
import os
import threading

import numpy as np
import tensorrt as trt

KIT_BIN = os.path.join(os.environ["CUDA_PATH"], "bin") if os.environ.get("CUDA_PATH") else ""
MAX_BATCH = 48
ALIGN = 256

_rt = None


def _cudart():
    global _rt
    if _rt is not None:
        return _rt
    d = os.environ.get("MAYFLY_CUDA_BIN") or KIT_BIN
    lib = ctypes.CDLL(os.path.join(d, "cudart64_12.dll"))
    vp, sz, i32, u32, u64 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_uint, ctypes.c_ulonglong
    pv = ctypes.POINTER(vp)
    sig = {"cudaSetDevice": [i32], "cudaMalloc": [pv, sz], "cudaFree": [vp], "cudaHostAlloc": [pv, sz, u32],
           "cudaFreeHost": [vp], "cudaStreamCreateWithFlags": [pv, u32], "cudaStreamDestroy": [vp],
           "cudaMemcpyAsync": [vp, vp, sz, i32, vp], "cudaStreamSynchronize": [vp],
           "cudaStreamBeginCapture": [vp, i32], "cudaStreamEndCapture": [vp, pv],
           "cudaGraphInstantiateWithFlags": [pv, vp, u64], "cudaGraphLaunch": [vp, vp], "cudaGraphDestroy": [vp],
           "cudaGraphExecDestroy": [vp], "cudaGetLastError": [], "cudaMemGetInfo": [ctypes.POINTER(sz)] * 2}
    for name, args in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, i32
    lib.cudaGetErrorString.argtypes, lib.cudaGetErrorString.restype = [i32], ctypes.c_char_p
    _rt = lib
    return lib


def _ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: CUDA error {rc} ({_cudart().cudaGetErrorString(rc).decode()})")


def _align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def mem_info():
    """(free, total) bytes on the current device."""
    f, t = ctypes.c_size_t(), ctypes.c_size_t()
    _ck(_cudart().cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t)), "cudaMemGetInfo")
    return f.value, t.value


# (name, numpy dtype, elements per row given W), inputs then outputs, in staging order
INPUTS = (("ids", np.int64, lambda w: 64), ("cast", np.float32, lambda w: 4), ("hm", np.int64, lambda w: 1),
          ("ep", np.int64, lambda w: 1), ("I", np.int64, lambda w: w), ("M", np.bool_, lambda w: w))
OUTPUTS = (("pol", np.float32, lambda w: w), ("val", np.float32, lambda w: 1), ("qv", np.float32, lambda w: w))


def layout(spec, B, W):
    """{name: (byte offset, count)} back to back, each 256-aligned, and the total bytes."""
    out, off = {}, 0
    for name, dt, per in spec:
        n = B * per(W)
        out[name] = (off, n)
        off = _align(off + n * np.dtype(dt).itemsize)
    return out, off


class TrtNet:
    """One engine on one card, one execution context and a CUDA graph per batch size. One call at a time."""

    def __init__(self, engine_path, device=0, graph=True, pad=True):
        self.path = os.path.abspath(engine_path)
        self.graph, self.pad = graph, pad
        rt = _cudart()
        _ck(rt.cudaSetDevice(int(device)), "cudaSetDevice")
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(self.path, "rb") as f:
            self.engine = self._runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"could not load the engine {self.path}")
        self.ctx = self.engine.create_execution_context()
        self.wmax = int(self.engine.get_tensor_profile_shape("I", 0)[2][1])
        self.bmax = min(MAX_BATCH, int(self.engine.get_tensor_profile_shape("ids", 0)[2][0]))
        self._lock = threading.Lock()
        self._shapes = None
        self._graphs = {}
        self.stream = ctypes.c_void_p()
        _ck(rt.cudaStreamCreateWithFlags(ctypes.byref(self.stream), 1), "cudaStreamCreate")
        _, nin = layout(INPUTS, self.bmax, self.wmax)
        _, nout = layout(OUTPUTS, self.bmax, self.wmax)
        self._bufs = []
        self.d_in, self.d_out = self._dev(nin), self._dev(nout)
        self.h_in, self.h_out = self._host(nin), self._host(nout)

    def _dev(self, n):
        p = ctypes.c_void_p()
        _ck(_cudart().cudaMalloc(ctypes.byref(p), n), "cudaMalloc")
        self._bufs.append(("d", p))
        return p.value

    def _host(self, n):
        p = ctypes.c_void_p()
        _ck(_cudart().cudaHostAlloc(ctypes.byref(p), n, 0), "cudaHostAlloc")
        self._bufs.append(("h", p))
        return p.value

    def _view(self, base, off, dt, n):
        return np.ctypeslib.as_array((ctypes.c_ubyte * (n * np.dtype(dt).itemsize)).from_address(base + off)).view(dt)

    def _bind(self, B, W):
        """Shapes and device addresses for (B, W)."""
        lin, nin = layout(INPUTS, B, W)
        lout, nout = layout(OUTPUTS, B, W)
        if self._shapes != (B, W):
            for name, _, per in INPUTS:
                shape = (B,) if name in ("hm", "ep") else (B, per(W))
                self.ctx.set_input_shape(name, shape)
            self._shapes = (B, W)
        for name, _, _ in INPUTS:
            self.ctx.set_tensor_address(name, self.d_in + lin[name][0])
        for name, _, _ in OUTPUTS:
            self.ctx.set_tensor_address(name, self.d_out + lout[name][0])
        return lin, nin, lout, nout

    def _enqueue(self):
        if not self.ctx.execute_async_v3(self.stream.value):
            raise RuntimeError("TensorRT execute_async_v3 failed")

    def _graph_for(self, B, W):
        g = self._graphs.get((B, W))
        if g is not None:
            return g
        rt = _cudart()
        self._bind(B, W)
        self._enqueue()             # TensorRT wants one plain run after a shape change, before capture
        _ck(rt.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        _ck(rt.cudaStreamBeginCapture(self.stream, 1), "cudaStreamBeginCapture")   # thread local
        self._enqueue()
        graph = ctypes.c_void_p()
        _ck(rt.cudaStreamEndCapture(self.stream, ctypes.byref(graph)), "cudaStreamEndCapture")
        ex = ctypes.c_void_p()
        _ck(rt.cudaGraphInstantiateWithFlags(ctypes.byref(ex), graph, 0), "cudaGraphInstantiate")
        self._graphs[(B, W)] = (graph, ex)
        return self._graphs[(B, W)]

    def evaluate(self, ids, cast, hm, ep, I, M):
        """One batch. Returns (pol (B, wide), val (B,), qv (B, wide)) float32, as CudaNet.evaluate."""
        I = np.asarray(I)
        B, wide = I.shape
        if not 1 <= B <= self.bmax or not 1 <= wide <= self.wmax:
            raise ValueError(f"batch {B} x {wide}: the engine takes 1..{self.bmax} x 1..{self.wmax}")
        W = self.wmax if self.pad else wide
        rt = _cudart()
        with self._lock:
            lin, nin = layout(INPUTS, B, W)
            src = {"ids": ids, "cast": cast, "hm": hm, "ep": ep}
            for name, dt, per in INPUTS[:4]:
                off, n = lin[name]
                self._view(self.h_in, off, dt, n)[:] = np.asarray(src[name], dt).reshape(-1)
            for name, a in (("I", I), ("M", M)):
                off, n = lin[name]
                dt = INPUTS[4][1] if name == "I" else np.uint8
                v = self._view(self.h_in, off, dt, n).reshape(B, W)
                v[:, :wide] = np.asarray(a).astype(dt, copy=False)
                if W > wide:
                    v[:, wide:] = 0
            _ck(rt.cudaMemcpyAsync(self.d_in, self.h_in, nin, 1, self.stream), "upload")
            if self.graph:
                _, ex = self._graph_for(B, W)
                _ck(rt.cudaGraphLaunch(ex, self.stream), "cudaGraphLaunch")
                lout, nout = layout(OUTPUTS, B, W)
            else:
                lin, nin, lout, nout = self._bind(B, W)
                self._enqueue()
            _ck(rt.cudaMemcpyAsync(self.h_out, self.d_out, nout, 2, self.stream), "download")
            _ck(rt.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
            o = {name: self._view(self.h_out, lout[name][0], dt, lout[name][1]) for name, dt, _ in OUTPUTS}
            pol = o["pol"].reshape(B, W)[:, :wide].copy()
            qv = o["qv"].reshape(B, W)[:, :wide].copy()
            return pol, o["val"].copy(), qv

    def close(self):
        rt = _cudart()
        for graph, ex in self._graphs.values():
            rt.cudaGraphExecDestroy(ex)
            rt.cudaGraphDestroy(graph)
        self._graphs = {}
        self.ctx = None
        self.engine = None
        for kind, p in getattr(self, "_bufs", []):
            (rt.cudaFree if kind == "d" else rt.cudaFreeHost)(p)
        self._bufs = []
        if getattr(self, "stream", None) is not None and self.stream.value:
            rt.cudaStreamDestroy(self.stream)
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
