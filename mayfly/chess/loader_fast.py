r"""loader_fast removes two ways the training loaders held up the trainer's step. Each is off unless asked for.

Profiled on a training run on an RTX 5090, the policy filler thread spent 46% of its time inside
torch.as_tensor(a).to("cuda", non_blocking=True). From pageable memory that copy is not asynchronous. The driver
waits for the stream to drain up to it before it can stage the bytes, and the trainer's own launches into the same
stream wait behind it. So the main thread sat 41% of its time in the torch.cat right after pulling a batch. The
legality mask (python-chess over every policy row) also held the interpreter lock for about 3 ms a step on the
filler threads.

  Shipper       LOADER_PIN: a batch's arrays copied into one pinned buffer, sent in one copy on a stream of its own,
                handed out as typed views of it. The consumer's stream waits for that copy on the GPU, not the CPU.
                The same arrays, byte for byte.
  LegalWorker   LEGAL_WORKER: pack.legal_mask itself, run in a small process of its own (numpy and python-chess, no
                torch), so the filler thread waits on a pipe with the lock released. The same function on the same
                fields of the same rows, so the same mask. If the process fails the mask is computed here as before.

  python loader_fast.py legal     the worker, which reads rows on stdin and writes packed masks on stdout
"""
import os
import struct
import subprocess
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = 64                  # every array starts on a 64-byte boundary, so any dtype can view it
# the only fields legal_mask reads, in a fixed layout, so a validation array saved under an older record works too
DT_LEGAL = np.dtype([("ids", np.uint8, 64), ("cast", np.uint8, 4), ("ep", np.uint8), ("flags", np.uint8)])
_NPD = {}


def _np_dtype(d):
    """The numpy dtype a torch dtype holds."""
    if d not in _NPD:
        import torch
        _NPD[d] = torch.empty(0, dtype=d).numpy().dtype
    return _NPD[d]


class Shipper:
    """Pinned batches over a CUDA stream of their own. One per loader, used by its filler thread."""

    def __init__(self, device):
        import torch
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)

    def send(self, arrs):
        """[(numpy array, torch dtype)] to (device tensors, the copy's event, their buffer), for the queue."""
        import torch
        nps = [np.ascontiguousarray(a, dtype=_np_dtype(d)) for a, d in arrs]
        offs, n = [], 0
        for a in nps:
            offs.append(n)
            n += -(-a.nbytes // ALIGN) * ALIGN
        # from the caching host allocator: a block is not reused until the copy that read it has finished
        host = torch.empty(max(n, ALIGN), dtype=torch.uint8, pin_memory=True)
        hv = host.numpy()
        for a, o in zip(nps, offs):
            hv[o:o + a.nbytes] = a.reshape(-1).view(np.uint8)
        with torch.cuda.stream(self.stream):
            buf = host.to(self.device, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.stream)
        out = tuple(buf[o:o + a.nbytes].view(d).view(a.shape) for a, o, (_, d) in zip(nps, offs, arrs))
        return out, ev, buf

    @staticmethod
    def receive(item):
        """The batch, made safe to use on the caller's stream: it waits for the copy there, on the GPU."""
        import torch
        out, ev, buf = item
        cur = torch.cuda.current_stream(buf.device)
        cur.wait_event(ev)
        buf.record_stream(cur)          # the buffer is not handed to another copy until this stream is done with it
        return out


def _read(f, n):
    """Exactly n bytes from a pipe, or EOFError."""
    chunks, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            raise EOFError(f"pipe closed after {got} of {n} bytes")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


class LegalWorker:
    """pack.legal_mask in a process of its own. Called as legal_mask is, mask = worker(rows, n_moves)."""

    def __init__(self, tag="loader"):
        self.tag, self.p, self.dead = tag, None, False
        self.lock = threading.Lock()
        try:
            self.p = subprocess.Popen([sys.executable, os.path.join(HERE, "loader_fast.py"), "legal"],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=HERE,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            self._give_up(e)

    def _give_up(self, e, quiet=False):
        self.dead = True
        if not quiet:
            print(f"[legal worker {self.tag}] {type(e).__name__}: {e}; masks are computed on the filler thread again",
                  flush=True)
        self.close()

    def __call__(self, rows, n_moves):
        from pack import legal_mask
        if not self.dead:
            try:
                with self.lock:
                    return self._ask(rows, n_moves)
            except Exception as e:
                self._give_up(e, quiet=self.p is None)       # quiet when close() took it away on purpose
        return legal_mask(rows, n_moves)

    def _ask(self, rows, n_moves):
        n = len(rows)
        x = np.zeros(n, DT_LEGAL)
        for k in DT_LEGAL.names:
            x[k] = rows[k]
        self.p.stdin.write(struct.pack("<II", n, n_moves) + x.tobytes())
        self.p.stdin.flush()
        got = _read(self.p.stdout, n * ((n_moves + 7) // 8))
        bits = np.frombuffer(got, np.uint8).reshape(n, -1)
        return np.unpackbits(bits, axis=1, count=n_moves).view(np.bool_)

    def close(self):
        p, self.p = self.p, None
        if p is None:
            return
        try:
            p.stdin.close()             # the worker exits at end of input
            p.wait(timeout=2)
        except Exception:
            p.kill()


def _legal_server():
    """The worker: (n, n_moves) and n DT_LEGAL rows in, n packed mask rows out, until the pipe closes."""
    sys.path.insert(0, HERE)
    from pack import legal_mask
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while True:
        try:
            n, n_moves = struct.unpack("<II", _read(inp, 8))
            rows = np.frombuffer(_read(inp, n * DT_LEGAL.itemsize), DT_LEGAL)
        except EOFError:
            return
        out.write(np.packbits(legal_mask(rows, n_moves), axis=1).tobytes())
        out.flush()


if __name__ == "__main__" and sys.argv[1:2] == ["legal"]:
    _legal_server()
