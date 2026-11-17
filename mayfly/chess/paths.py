"""Where MayFly keeps its data.

Everything lives under one root, MAYFLY_DATA, which defaults to data/ next to the repository. A script that used
its own environment variable for a path (CKPT_DIR, SHARDS, VERSION_DIR and so on) still reads it first, with
under(...) as the default.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("MAYFLY_DATA", os.path.join(ROOT, "data"))


def under(*parts):
    """A path under the data root."""
    return os.path.join(DATA, *parts)


CKPT = os.environ.get("CKPT_DIR", under("ckpt"))
SHARDS = os.environ.get("SHARDS", under("shards"))
VERSIONS = os.environ.get("VERSION_DIR", under("versions"))
SELFPLAY = os.environ.get("SELFPLAY_DIR", under("selfplay"))
