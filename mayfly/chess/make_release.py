r"""make_release packs MayFly as a self-contained Linux release for testers and rating lists, one tar.gz.

What goes in:
  chess/       the engine code uci.py imports, found by reading the imports (so a new module comes along
               by itself and nothing from training does), the launcher mayfly_uci.sh, the compliance
               test, any compiled modules built beside it (mayfly_*.so) and, where their build_linux.sh
               is present, the sources of the compiled modules in native/ and cpp/
  versions/    ONE network and its own versions.json row, the search it was sealed with, and the C++
               network's TorchScript trace (<name>.cuda.ts and its fingerprint) when one is beside it
  mayfly       a link to chess/mayfly_uci.sh, what a GUI is pointed at
  README.md, requirements.txt

  python3 chess/make_release.py                          the newest version in VERSION_DIR, into chess/release
  python3 chess/make_release.py --version mayfly_v6.pt --versions-dir data/versions --out ~/release
  options: --name mayfly-v6-linux  --trace auto|yes|no  --no-binaries

Text files go in with LF line endings whatever the checkout has, and the scripts executable.
"""
import argparse, ast, datetime, hashlib, io, json, os, sys, tarfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402
TEXT = {".py", ".sh", ".md", ".txt", ".json", ".c", ".h", ".cpp", ".cu", ".cfg", ".toml"}
# build leftovers and Windows-only files, never shipped from native/ and cpp/
SKIP_DIRS = {"build", "build_tk", "__pycache__", ".git"}
SKIP_EXT = {".o", ".obj", ".so", ".pyd", ".dll", ".lib", ".exp", ".pdb", ".bat", ".ps1", ".log", ".pyc"}


def imports_of(start):
    """Every module in chess/ that `start` imports, at any depth, by reading the source."""
    local = {f[:-3] for f in os.listdir(HERE) if f.endswith(".py")}
    seen, todo = set(), [start]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        with open(os.path.join(HERE, m + ".py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for n in ast.walk(tree):
            names = ([a.name for a in n.names] if isinstance(n, ast.Import) else
                     [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else [])
            todo += [x.split(".")[0] for x in names if x.split(".")[0] in local]
    return sorted(seen)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def default_versions_dir():
    for d in (os.environ.get("VERSION_DIR"), os.path.join(HERE, "..", "versions"), paths.under("versions")):
        if d and os.path.exists(os.path.join(d, "versions.json")):
            return os.path.abspath(d)
    return None


class Bundle:
    """A tar.gz written as it is filled: files from disk, text from memory, folders and one link."""

    def __init__(self, path, top):
        self.tar = tarfile.open(path, "w:gz", compresslevel=6)
        self.top, self.dirs, self.listing, self.now = top, set(), [], time.time()

    def _info(self, arc, size, mode):
        ti = tarfile.TarInfo(f"{self.top}/{arc}")
        ti.size, ti.mode, ti.mtime = size, mode, self.now
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        return ti

    def _dirs(self, arc):
        """The folders above arc, each written once, before anything in them."""
        parts = arc.split("/")[:-1]
        for i in range(len(parts) + 1):
            d = "/".join(parts[:i])
            if d not in self.dirs:
                self.dirs.add(d)
                ti = self._info(d, 0, 0o755)
                ti.name = f"{self.top}/{d}" if d else self.top
                ti.type = tarfile.DIRTYPE
                self.tar.addfile(ti)

    def text(self, arc, data, mode=0o644):
        raw = data.encode("utf-8") if isinstance(data, str) else data
        raw = raw.replace(b"\r\n", b"\n")
        self._dirs(arc)
        self.tar.addfile(self._info(arc, len(raw), mode), io.BytesIO(raw))
        self.listing.append((arc, len(raw)))

    def file(self, arc, src, mode=None):
        ext = os.path.splitext(src)[1].lower()
        if mode is None:
            mode = 0o755 if ext == ".sh" else 0o644
        if ext in TEXT:
            with open(src, "rb") as fh:
                return self.text(arc, fh.read(), mode)
        self._dirs(arc)
        size = os.path.getsize(src)
        with open(src, "rb") as fh:
            self.tar.addfile(self._info(arc, size, mode), fh)
        self.listing.append((arc, size))

    def link(self, arc, target):
        self._dirs(arc)
        ti = self._info(arc, 0, 0o777)
        ti.type, ti.linkname = tarfile.SYMTYPE, target
        self.tar.addfile(ti)
        self.listing.append((f"{arc} -> {target}", 0))

    def close(self):
        self.tar.close()


def source_folder(b, name):
    """chess/<name>/ for the compiled modules, only when its Linux build script is there."""
    root = os.path.join(HERE, name)
    if not os.path.exists(os.path.join(root, "build_linux.sh")):
        return False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            if os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            src = os.path.join(dirpath, f)
            b.file("chess/" + os.path.relpath(src, HERE).replace(os.sep, "/"), src)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default="", help="the network's file, e.g. mayfly_v6.pt (default: the newest)")
    ap.add_argument("--versions-dir", default="", help="versions.json and the .pt files")
    ap.add_argument("--out", default=os.path.join(HERE, "release"), help="default chess/release (ignored by git)")
    ap.add_argument("--name", default="")
    ap.add_argument("--trace", choices=("auto", "yes", "no"), default="auto",
                    help="ship the C++ network's TorchScript trace (auto: when it is beside the version)")
    ap.add_argument("--no-binaries", action="store_true", help="leave out compiled modules built beside the code")
    a = ap.parse_args()

    vdir = os.path.abspath(a.versions_dir) if a.versions_dir else default_versions_dir()
    if not vdir or not os.path.exists(os.path.join(vdir, "versions.json")):
        sys.exit("no versions.json: pass --versions-dir")
    with open(os.path.join(vdir, "versions.json"), encoding="utf-8-sig") as fh:
        rows = json.load(fh)
    rows = [r for r in rows if r.get("file") and os.path.exists(os.path.join(vdir, r["file"]))]
    if a.version:
        want = a.version if a.version.endswith(".pt") else a.version + ".pt"
        rows = [r for r in rows if r["file"] == want]
    if not rows:
        sys.exit(f"no version {a.version or ''} on disk in {vdir}")
    row = max(rows, key=lambda r: float(r.get("version", 0)))
    if not row.get("search"):
        sys.exit(f"{row['file']} has no sealed search in versions.json: not a release")
    ver, pt = row["file"], os.path.join(vdir, row["file"])
    num = row.get("version")
    name = a.name or f"mayfly-v{num}-linux"
    stem = os.path.splitext(ver)[0]
    trace = os.path.join(vdir, stem + ".cuda.ts")
    has_trace = os.path.exists(trace) and os.path.exists(trace + ".json")
    if a.trace == "yes" and not has_trace:
        sys.exit(f"--trace yes but no {trace} (and .json) beside the version")
    ship_trace = has_trace and a.trace != "no"

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, name + ".tar.gz")
    b = Bundle(path + ".part", name)

    # the engine: what uci.py imports, the launcher and the compliance test
    mods = imports_of("uci")
    for m in mods + ["test_uci_compliance"]:
        b.file(f"chess/{m}.py", os.path.join(HERE, m + ".py"))
    b.file("chess/mayfly_uci.sh", os.path.join(HERE, "mayfly_uci.sh"))
    b.link("mayfly", "chess/mayfly_uci.sh")
    binaries = []
    if not a.no_binaries:
        binaries = sorted(f for f in os.listdir(HERE) if f.startswith("mayfly_") and f.endswith(".so"))
        for f in binaries:
            b.file(f"chess/{f}", os.path.join(HERE, f), 0o755)
    built = {n: source_folder(b, n) for n in ("native", "cpp")}

    # the network, its sealed search, and the C++ network's trace
    b.text("versions/versions.json", json.dumps([row], indent=1) + "\n")
    b.file(f"versions/{ver}", pt)
    if ship_trace:
        b.file(f"versions/{stem}.cuda.ts", trace)
        b.file(f"versions/{stem}.cuda.ts.json", trace + ".json")

    b.text("requirements.txt", REQUIREMENTS)
    b.text("README.md", readme(row, name, ver, sha256(pt), mods, binaries, built, ship_trace))
    b.close()
    os.replace(path + ".part", path)
    with open(path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{sha256(path)}  {os.path.basename(path)}\n")

    for arc, size in b.listing:
        print(f"{size / 1e6:10.1f} MB  {name}/{arc}" if size >= 1e5 else f"{size:10d} B   {name}/{arc}")
    print(f"\n{path}: {os.path.getsize(path) / 1e6:.1f} MB, {len(b.listing)} files")


REQUIREMENTS = """\
# MayFly's Python dependencies. Install a CUDA build of torch first, from https://pytorch.org
# (for example: pip install torch --index-url https://download.pytorch.org/whl/cu128), then:
#   pip install -r requirements.txt
torch>=2.1
numpy>=1.24
scipy>=1.10
chess>=1.10
# only to build the compiled modules (chess/native/build_linux.sh)
setuptools
"""


def readme(row, name, ver, digest, mods, binaries, built, trace):
    num = row.get("version")
    stem = os.path.splitext(ver)[0]
    day = str(row.get("at", ""))[:10] or "unknown"
    so_note = ("\n".join(f"    chess/{f}" for f in binaries) if binaries else "    (none in this bundle)")
    build = []
    if built.get("native"):
        build.append("    PYTHON=venv/bin/python sh chess/native/build_linux.sh     "
                     "# mayfly_pick, mayfly_movegen, mayfly_walk (gcc)")
    if built.get("cpp"):
        build.append("    PYTHON=venv/bin/python bash chess/cpp/build_linux.sh      "
                     "# mayfly_cppsearch, the search and network in C++ (g++)")
    build = "\n".join(build) or "    (this bundle has no build scripts: the engine runs on its Python search)"
    trace_note = (f"This bundle includes it (versions/{stem}.cuda.ts). To make it again, for example after\n"
                  f"changing torch:" if trace else
                  "This bundle does not include it. Make it once, on the machine with the GPU:")
    return README.format(num=num, name=name, ver=ver, stem=stem, day=day, digest=digest,
                         nmods=len(mods), so_note=so_note, build=build, trace_note=trace_note,
                         made=datetime.date.today().isoformat())


README = """\
# MayFly v{num} for Linux

MayFly is a chess engine by Ryan Chen: a transformer network (146 million parameters) that scores
moves and positions, searched by Monte Carlo tree search, in the family of Leela Chess Zero. This
bundle is version {num} (`{ver}`, released {day}) with the search settings it was sealed with at
release, so it plays exactly as that version does everywhere else. Bundle made {made}.

## Requirements

- Linux on x86-64 (tested on Ubuntu 24.04).
- An NVIDIA GPU with about 3 GB of free memory and a driver for CUDA 12 (tested on an RTX 5090).
  Without a GPU the engine runs on the processor, far too slowly for testing.
- Python 3.10 or newer with a CUDA build of PyTorch, numpy, scipy and python-chess
  (tested: Python 3.12.3, torch 2.11.0+cu128, numpy 2.5.4, scipy 1.18.1, python-chess 1.11.2).
- Memory: the search tree is capped at PONDER_MEM_GB (by default a third of the RAM, at most
  12 GB), plus about 3 GB for Python, PyTorch and the network.
- Disk: about 1 GB.

## Install

    tar xzf {name}.tar.gz
    cd {name}
    python3 -m venv venv
    venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
    venv/bin/pip install -r requirements.txt

The launcher uses `venv/bin/python` when that folder exists, otherwise `python3` from the PATH, or
whatever the environment variable PYTHON names.

Check the install (about two minutes; it loads the network and runs every UCI command):

    venv/bin/python chess/test_uci_compliance.py --quick

Without `--quick` it then plays six short games of MayFly against itself (about half an hour).

## Optional: the compiled modules (faster)

The engine is complete without them: each one that is missing is replaced by the same arithmetic
in Python, with the same moves, only slower. Build them for the Python that runs the engine:

    sudo apt install build-essential python3-dev
{build}

They need gcc and g++ and the Python headers; the C++ search also needs the CUDA build of torch
from pip (it links against that torch's LibTorch and CUDA runtime) but not the CUDA toolkit.
Modules built for this bundle's test machine (Python 3.12, torch 2.11.0+cu128) are included:

{so_note}

Another Python ignores them, and the C++ search loads only beside the torch it was built with, so
on any other setup build them again. The C++ search runs the network in C++ too when it finds the
network's TorchScript trace beside the version. {trace_note}

    venv/bin/python -c "import sys; sys.path.insert(0, 'chess'); import mcts_cpp; mcts_cpp.export_trace('versions/{ver}')"

What actually runs is printed on stderr at the first search: a line starting `[mcts]` or
`[mcts_cpp]` names each module that is missing and what replaces it. No such line means all of them
are in use.

## Run

Point the GUI at `{name}/mayfly` (a link to `chess/mayfly_uci.sh`) as a UCI engine. It runs from
any folder. For example with cutechess-cli:

    cutechess-cli -engine name=MayFly cmd=/path/to/{name}/mayfly proto=uci \\
                  -engine name=Other cmd=/path/to/other proto=uci -each tc=60+1 -games 2

At start the network loads in the background (3 to 5 seconds), so `uci` and `isready` answer at
once and the first `go` waits for it. Only if the network is still not loaded 8 seconds after a
game's first `go` does a standard first move go out without it.

## UCI options

| Option        | Default        | What it does |
|---------------|----------------|--------------|
| Version       | {ver}   | The network, a file in `versions/`. Each version plays the search it was sealed with. |
| Visits        | 400            | Visits a move when `go` gives no clock, movetime or nodes (`go depth` too). |
| Ponder        | false          | Think on the opponent's time; the GUI sets it. |
| SyzygyPath    | (empty)        | Syzygy tablebase folders, separated by `:`. Probed up to 6 pieces (TB_PIECES). |
| Move Overhead | 20             | Milliseconds each move costs beyond the search. Raise it if MayFly loses on time. -1 is the pricing the Lichess bot uses. |

There is no Threads or Hash: one search thread drives the GPU, and the tree's memory ceiling is
PONDER_MEM_GB.

## Environment variables

Read by the launcher and the engine; anything set before starting it wins.

| Variable       | Default | |
|----------------|---------|-|
| VERSION        | the newest in VERSION_DIR | the network to play |
| VERSION_DIR    | `versions/` | versions.json and the networks |
| SYZYGY_DIR     | `syzygy/` if it exists | tablebases, as SyzygyPath |
| PYTHON         | `venv/bin/python`, else `python3` | the interpreter |
| NET_OVERHEAD   | 0.02 | seconds, as Move Overhead; empty for the Lichess pricing |
| PONDER_MEM_GB  | a third of the RAM, 1 to 12 | the tree's memory ceiling |
| AUTOCAST_GRAPH | 1 | CUDA graphs for the Python network path (0 saves GPU memory) |
| MCTS_NATIVE    | 1 | 0 switches the compiled pick, move generator and walk off |
| TB_PIECES      | 6 | the most pieces probed in the tablebases |
| LOG            | none | a file that gets every UCI line in and out, time-stamped |

## How it uses the clock and the commands

- `go wtime btime winc binc`: MayFly's own allocator, which reads both clocks. `movestogo` is read
  but not used, so a repeating time control is planned as one long game and spends less than it could.
- `go movetime`: searches for that long and stops, except a game's first move, capped at 3 s.
- `go nodes`: that many visits. `go depth`: a tree search has no depth limit, so it plays Visits.
- `go infinite`: searches until `stop` (or until the tree reaches its memory ceiling, then waits).
- `go ponder` and `ponderhit`: the ponder searches the opponent's position, so on a miss the tree
  under the move they actually played is kept too.
- One `info` line a move, as the search ends: `depth` (the length of the line it expects), `nodes`
  (visits), `score cp` (from the tree's win probability p, as -400 log10(1/p - 1), capped at 3000)
  and `pv`. Nothing is printed while it thinks.

## Expected speed

{speed}

## Files

- `mayfly`: the launcher (a link to `chess/mayfly_uci.sh`).
- `chess/`: the engine ({nmods} Python modules, from `uci.py` down), the compliance test, and the
  compiled modules and their sources.
- `versions/{ver}` (sha256 {digest}) and `versions/versions.json`, its sealed search.
"""

# measured with `test_uci_compliance.py --bench 5` (the average of its five positions)
SPEED = """\
Measured on an RTX 5090 (an AMD Ryzen 7 7800X3D host) while a training run shared the GPU, so a
card to itself is faster. Visits a second, the average of five positions from the opening to an
ending at `go movetime 5000` (`venv/bin/python chess/test_uci_compliance.py --bench 5` measures
yours the same way):

| What is built | Visits a second |
|---------------|-----------------|
| nothing (the Python search) | about 2,400 |
| chess/native (the compiled pick, move generator and walk) | about 3,500 |
| chess/native and chess/cpp (search and network in C++) | about 3,750 |

With more jobs on the card it fell to about 1,900, 2,700 and 2,850. The search often stops before the movetime once no other move can catch the best one. One process
uses one GPU and about 2 GB of its memory; there is no thread count."""
README = README.replace("{speed}", SPEED)


if __name__ == "__main__":
    main()
