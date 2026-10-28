"""A stand-in `torch` for the volunteer app's search workers, so the app needs no PyTorch (about 2.5 GB).

The workers run selfplay_remote's RemoteMCTS: the network is on the app's ONNX server, and the tree builds its
batches with numpy (device "cpu" skips the one torch buffer _prepare can make). mcts.py, selfplay.py and chess_net.py
still import torch at the top and use it at import time for decorators (@torch.inference_mode(), @torch.no_grad()),
base classes (nn.Module) and constants (torch.bfloat16). Those get harmless placeholders here.

Any real use fails loudly. A placeholder can be called, decorated with, compared and read from, but arithmetic,
indexing, iteration, len() or turning it into a number or array raises StubUsed. So a worker that ever computes with
torch crashes at once instead of quietly playing bad games.

  install() before importing mcts, selfplay or chess_net, and only when the real torch is missing.
"""
import sys
import types


class StubUsed(RuntimeError):
    pass


def _refuse(name):
    def f(self, *a, **k):
        raise StubUsed(f"torch placeholder {self._path!r} used for {name}: this code path needs the real PyTorch")
    return f


class Placeholder:
    __slots__ = ("_path",)

    def __init__(self, path="torch"):
        self._path = path

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Placeholder(f"{self._path}.{name}")

    def __call__(self, *a, **k):
        if len(a) == 1 and not k and callable(a[0]) and not isinstance(a[0], Placeholder):
            return a[0]                       # used as a decorator: leave the function as it is
        return Placeholder(f"{self._path}()")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __bool__(self):
        return False                          # torch.cuda.is_available() and the like read as "no"

    def __repr__(self):
        return f"<torch placeholder {self._path}>"

    def __hash__(self):
        return hash(self._path)

    def __eq__(self, other):
        return isinstance(other, Placeholder) and other._path == self._path


for _name in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
              "__matmul__", "__neg__", "__lt__", "__le__", "__gt__", "__ge__", "__getitem__", "__setitem__",
              "__iter__", "__len__", "__float__", "__int__", "__index__", "__array__"):
    setattr(Placeholder, _name, _refuse(_name))


class Module:
    """nn.Module as a base class only: the network classes are defined, never built, in a worker."""

    def __init__(self, *a, **k):
        raise StubUsed("an nn.Module was built in a stub worker: this code path needs the real PyTorch")


def _module(path, **attrs):
    m = types.ModuleType(path)
    m.__getattr__ = lambda name: Placeholder(f"{path}.{name}")
    m.__dict__.update(attrs)
    return m


def install(force=False):
    """Put the stand-in in sys.modules unless the real torch imports (force: use it anyway, for tests)."""
    if not force:
        try:
            import torch  # noqa: F401
            return False
        except ImportError:
            pass
    functional = _module("torch.nn.functional")
    nn = _module("torch.nn", Module=Module, functional=functional)
    torch = _module("torch", nn=nn, __version__="0.0-stub", __mayfly_stub__=True)
    sys.modules.update({"torch": torch, "torch.nn": nn, "torch.nn.functional": functional})
    return True
