r"""uci speaks the UCI protocol, so any chess GUI, cutechess-cli or lichess-bot can play MayFly.

`setoption name Version value mayfly_v3.pt` picks which frozen version answers, and each plays the search it
was sealed with. `go wtime ... btime ... winc ... binc ...` feeds timectl, including the opponent's clock.
With `TM_PARITY` on, a position that will not resolve may spend a clock lead over the opponent down to level.
`go movetime` and `go nodes` are honoured exactly, and `go infinite` searches until `stop`.

  python chess/uci.py                       then type uci, isready, position startpos, go nodes 400
  env: VERSION=mayfly_v3.pt  VISITS=400  TM_PARITY=0  LOG=uci.log
"""
import os, sys, threading, time
import chess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

E = os.environ.get
DEFAULT_VERSION = E("VERSION", "mayfly_v3.pt")
DEFAULT_VISITS = int(E("VISITS", 400))
LOG = E("LOG", "")
# the same file the board touches, watched by the trainer, which yields while it is fresh
PLAYING_FILE = E("PLAYING_FILE", os.path.join(E("CKPT_DIR", paths.under("ckpt")), "playing.flag"))


# The tree's memory is the limit, not its visits. Every search, pondering or not, may grow the tree to
# TREE_GB and no further, so a ponder runs until the opponent moves or the tree reaches that size.
# Measured with mem_probe.py, a middlegame node costs about 8.5 KB of process memory with the real model,
# so 12 GB is about 1.4 million nodes. Endgame nodes are smaller, so the ceiling errs low there. The
# network, CUDA and the eval cache sit outside this budget.
TREE_GB = float(os.environ.get("PONDER_MEM_GB", 12))
NODE_BYTES = int(os.environ.get("NODE_BYTES", 8500))
MAX_NODES = int(TREE_GB * 1e9 / NODE_BYTES)


def raise_priority():
    """Ask Windows for ABOVE_NORMAL priority. On a busy machine a search that is descheduled mid-move
    checks its deadline late, and a deadline checked late is missed. Best effort, and silent when it
    is not Windows."""
    try:
        import ctypes
        ABOVE_NORMAL = 0x00008000
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), ABOVE_NORMAL)
    except Exception:
        pass


_LOG_LOCK = threading.Lock()


def log(line):
    """Every line in and out, when LOG is set. A GUI's complaint is unreproducible without it.

    One writer at a time, with a retry. The search thread logs its `bestmove` while the main thread logs the next
    command, often in the same millisecond, and on Windows one of two simultaneous opens can fail. So the writes take
    turns, and a failed open is tried again a few times before the line is given up."""
    if not LOG:
        return
    text = f"{time.strftime('%H:%M:%S')} {line}\n"
    with _LOG_LOCK:
        for attempt in range(5):
            try:
                with open(LOG, "a", encoding="utf-8") as fh:
                    fh.write(text)
                return
            except OSError:
                time.sleep(0.002 * (attempt + 1))


# The protocol stream, captured once at import. `ready` sends the loader's chatter to stderr with
# contextlib.redirect_stdout, which rebinds sys.stdout for the whole process, so a background load would
# otherwise send the main thread's uciok, readyok and bestmove to stderr, where the GUI never reads them.
PROTO = sys.stdout
FIRST_MOVE_CAP = float(E("FIRST_MOVE_CAP", 3.0))   # seconds, see Engine.budget
# The first move may not wait for a slow load. lichess-bot starts this process when the game starts, and
# on a busy GPU the load can take 20 to 30 s, long enough for Lichess to abort the game. So on the game's
# first move only, if the network is not loaded FIRST_WAIT seconds after go, a standard opening move goes
# out at once (1.d4 as White, the usual reply as Black) and the load continues behind it.
FIRST_WAIT = float(E("FIRST_WAIT", 8.0))
FIRST_REPLIES = {"": "d2d4", "e2e4": "e7e5", "d2d4": "g8f6", "c2c4": "e7e5", "g1f3": "g8f6"}
INSTANT = float(E("INSTANT", 0.25))          # an allowance below this plays the tree's move at once
EASY_SLICE = float(E("EASY_SLICE", 0.15))    # share of the allowance for the first look
EASY_FLOOR = float(E("EASY_FLOOR", 0.15))    # but never less than this, in seconds
EASY_SHARE = float(E("EASY_SHARE", 0.90))    # of the root's visits, for a move to be easy
EASY_VISITS = float(E("EASY_VISITS", 400))   # and the root must have at least this many
# What a move costs the clock beyond the search, and how long a game to plan for. Online, a move costs
# about 0.5 s between the opponent's move and ours outside the engine, more on a high-latency route.
# Each move is priced by its own gap (the time since the last bestmove plus its plan), because an idle
# pooled HTTP connection can go cold and cost a new handshake. The clock is planned on the mean of this
# game's moves so far. NET_OVERHEAD, when set, is one fixed price instead. With lichess-bot's keep-warm
# patch a cold move costs about what a warm one does, so both are priced the same. Raise NET_COLD
# (about 1.9) without that patch. NET_MOVES_FLOOR keeps long endings from running out of clock, since
# this engine plays 60 to 160 move games.
NET_OVERHEAD = float(E("NET_OVERHEAD")) if E("NET_OVERHEAD") else None
NET_WARM = float(E("NET_WARM", 0.55))
NET_COLD = float(E("NET_COLD", 0.55))
NET_COLD_AFTER = float(E("NET_COLD_AFTER", 9.0))
# What a move costs inside the engine beyond its planned search (the bridge, stopping a ponder,
# re-rooting the tree, and evaluating the root even for an instant move). Without it the reserve for the
# moves still to come is priced below what the cheapest move costs, and long 5+0 games flag.
NET_INSIDE = float(E("NET_INSIDE", 0.15))
# Without an increment the flat price stays. Nothing pays a move back and a fast game can slow down, so
# the reserve for every move still to come is what keeps the flag away. The gap price is for games where
# the increment pays it.
NET_SUDDEN = float(E("NET_SUDDEN", 1.25))
NET_MOVES_FLOOR = float(E("NET_MOVES_FLOOR", 25))
# A game that started at a minute or less (1+0) is planned at the tree's tm_short_sudden where a version
# has it (v6, while v1 to v5 use 0, the flat price). At 1+0 the flat price reserves almost the whole clock,
# so every allowance falls under INSTANT. The game's start is the larger clock at its first timed move
# within SHORT_PLIES, so a longer game is never planned this way once its clocks fall under a minute.
SHORT_BASE = float(E("SHORT_BASE", 60.0))
SHORT_PLIES = int(E("SHORT_PLIES", 8))
# How far a critical move may run. A close but stable decision keeps the sealed 1.6x. One that is still
# moving (the most visited move is not the best valued, or the leader just changed) may run to CRIT_X
# times its share, the ceiling of Baier and Winands' best MCTS time rule, and never past CRIT_SHARE of the
# clock left (Lc0 caps a move at 0.42). Only where the increment pays for a move. In sudden death and
# under LOW_CLOCK the sealed cap stands.
CRIT_X = float(E("CRIT_X", 2.5))
CRIT_SHARE = float(E("CRIT_SHARE", 0.25))
# Where the increment does not pay for a move but the clock is not being spent either (for example 3+2
# on cold connections), an unstable decision may still run to CRIT_X times its share, but never past
# CRIT_SHARE_SUDDEN of the clock and never under LOW_CLOCK. With no increment it stays off.
CRIT_SHARE_SUDDEN = float(E("CRIT_SHARE_SUDDEN", 0.05))


def say(line):
    PROTO.write(line + "\n")
    PROTO.flush()
    log("< " + line)


class Engine:
    def __init__(self):
        self.version = DEFAULT_VERSION
        self.visits = DEFAULT_VISITS
        self.board = chess.Board()
        self.tree = None
        self.model = None
        self.label = ""
        self.searching = None
        self.pondering = None      # the thread thinking on the opponent's time
        self.ponder_opts = None    # the clock the GUI sent with `go ponder`, used on `ponderhit`
        self.last_spent = 0.0      # seconds the last real search took, to correct that clock
        self._load_lock = threading.RLock()   # one load at a time, shared by the warm-up thread and the GUI
        self._go_started = 0.0     # when the current `go` arrived, for the first-move cap
        self._bm_at = None         # when the last real bestmove went out, for the network price
        self._ov_seen = []         # this game's per-move network prices, planned on their mean
        self._start_clock = None   # the larger clock at this game's first timed move (SHORT_BASE)

    # ------------------------------------------------------------------ the network and the tree
    def ready(self):
        """Load on demand, because a GUI sends `uci` long before it sends `go` and a four hundred
        megabyte load inside the handshake looks like a hung engine."""
        with self._load_lock:
            self._load()

    def _load(self):
        if self.tree is not None:
            return
        # Everything the loader prints goes to stderr. Here stdout is the protocol, and one stray line
        # from `load_version` would reach the GUI as a command.
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            import torch                               # noqa: F401, imported for its side effects
            from mcts import MCTS, pinned_search
            from play_web import load_version
            ver = self._loading = self.version
            self.model, self.label, _ = load_version(ver)
            dev = str(next(self.model.parameters()).device)
            self.tree = MCTS(self.model, dev, batch=48, cache_size=200000).apply(
                pinned_search(ver))
        log(f"# loaded {self.label} on {dev}")

    def reload(self):
        """A new version, under the load lock, so the warm-up can never publish the old one after
        this has cleared it. A load of this version already running or done is kept."""
        if getattr(self, "_loading", None) == self.version:
            return
        with self._load_lock:
            if getattr(self, "_loading", None) == self.version and self.tree is not None:
                return
            self.tree = self.model = None
            self._load()

    # ------------------------------------------------------------------ the protocol
    def uci(self):
        say("id name MayFly " + (self.label or self.version.replace(".pt", "")))
        say("id author the MayFly developers")
        say(f"option name Version type string default {DEFAULT_VERSION}")
        say(f"option name Visits type spin default {DEFAULT_VISITS} min 1 max 10000000")
        say("option name Ponder type check default false")
        say("uciok")

    def setoption(self, words):
        """`setoption name X value Y`, where either part may contain spaces."""
        try:
            i, j = words.index("name"), (words.index("value") if "value" in words else len(words))
        except ValueError:
            return
        name = " ".join(words[i + 1:j]).lower()
        value = " ".join(words[j + 1:]) if j < len(words) else ""
        if name == "version" and value:
            self.version = value if value.endswith(".pt") else value + ".pt"
            self.reload()
        elif name == "visits" and value.isdigit():
            self.visits = max(1, int(value))

    def position(self, words):
        # a new position while pondering means the GUI moved on without `stop`, so end it quietly
        self._end_ponder()
        if "startpos" in words:
            self.board = chess.Board()
            rest = words[words.index("startpos") + 1:]
        elif "fen" in words:
            i = words.index("fen") + 1
            j = words.index("moves") if "moves" in words else len(words)
            try:
                board = chess.Board(" ".join(words[i:j]))
            except ValueError as exc:
                # UCI ignores what it cannot read, and raising here would end the main loop
                log(f"# position ignored, bad FEN: {exc}")
                return
            self.board = board
            rest = words[j:]
        else:
            return
        if rest and rest[0] == "moves":
            for u in rest[1:]:
                try:
                    mv = chess.Move.from_uci(u)
                except ValueError:
                    break
                if mv not in self.board.legal_moves:
                    break
                self.board.push(mv)

    def budget(self, opts):
        """Seconds for this move, and the furthest it may be stretched, from both clocks.

        Returns (seconds, ceiling, visits). A GUI that sends `movetime` or `nodes` gets exactly that.
        A GUI that sends a clock gets the allocator.
        """
        import timectl
        self.crit = None                  # set below for a clock that allows it, never carried over
        if "movetime" in opts:
            secs = opts["movetime"] / 1000.0
            # The game's first move is capped and pays for any wait on the load. lichess-bot asks for
            # ten seconds on it and Lichess aborts a game whose first move has not arrived after about
            # twenty, so a slow load plus the full ten would abort the game.
            if len(self.board.move_stack) <= 1:
                secs = max(0.2, min(secs, FIRST_MOVE_CAP) - (time.time() - self._go_started))
            return secs, None, 10 ** 7
        if "nodes" in opts:
            # A node limit beside a clock is a cap, not a licence. UCI stops at whichever comes first.
            clock = {k: v for k, v in opts.items() if k != "nodes"}
            if ("wtime" if self.board.turn == chess.WHITE else "btime") in clock:
                secs, hard, visits = self.budget(clock)
                return secs, hard, min(visits, max(1, int(opts["nodes"])))
            return None, None, int(opts["nodes"])
        if "infinite" in opts:
            return None, None, 10 ** 9
        mine = opts.get("wtime" if self.board.turn == chess.WHITE else "btime")
        if mine is None:
            return None, None, self.visits
        inc = opts.get("winc" if self.board.turn == chess.WHITE else "binc", 0) / 1000.0
        theirs = opts.get("btime" if self.board.turn == chess.WHITE else "wtime")
        parity = getattr(self.tree, "tm_parity", 0.0)
        # The game's ply for the clock plan, from the FEN's move number, so a game set up from a book
        # position is not planned as move one. The search's own opening rule reads the same count.
        game_ply = 2 * (self.board.fullmove_number - 1) + int(self.board.turn == chess.BLACK)
        if self._start_clock is None and game_ply <= SHORT_PLIES:
            self._start_clock = max(mine, theirs or 0) / 1000.0
        open_spend = getattr(self.tree, "tm_open_spend", 1.0)
        open_plies = getattr(self.tree, "tm_open_plies", 16)
        def plan(ov):
            return timectl.allocate(mine / 1000.0, inc, game_ply,
                                    opp=(theirs / 1000.0 if theirs is not None else None),
                                    parity=parity,
                                    open_spend=open_spend,
                                    open_plies=open_plies,
                                    overhead=ov, floor_moves=NET_MOVES_FLOOR)
        ov = self._net_price(plan, inc)
        secs = plan(ov)
        # The search stretches its allowance by up to 1 + tm_extend_max on its own when the top two
        # moves are close. With an increment that is paid back, in sudden death it is not, so there
        # the search is handed the share whose worst case is the plan.
        stretch = (1.0 + max(0.0, float(getattr(self.tree, "tm_extend_max", 0.6)))
                   if getattr(self.tree, "tm_extend", False) else 1.0)
        # and with an increment the AVERAGE stretch is planned for, except in the opening, where
        # the search never extends (mcts has no extensions while the opening ramp is on)
        opening = open_spend < 1.0 and game_ply < open_plies
        extends = stretch > 1.0 and not opening
        avg = timectl.EXT_AVG if extends else 1.0
        secs = timectl.search_share(secs, mine / 1000.0, inc, stretch, avg=avg,
                                    overhead=ov,
                                    avg_sudden=timectl.EXT_AVG_SUDDEN if extends else 1.0)
        # Too little for one batch means move now. A batch of 48 leaves takes 0.1 to 0.2 s on a busy
        # GPU, so an allowance under INSTANT cannot be honoured by searching. Play what the tree already
        # holds (the reused subtree, or the root's prior after its one evaluation) and bank the rest.
        if secs is not None and secs < INSTANT:
            # a version sealed without prior_fallback (v1 to v3) played root.moves[0] when the
            # deadline passed before its first visit, so it gets its one visit with no deadline
            return (secs if getattr(self.tree, "prior_fallback", False) else None), None, 1
        hard = (timectl.ceiling(mine / 1000.0, (theirs / 1000.0) if theirs is not None else None,
                                inc, game_ply, parity=parity,
                                deep=getattr(self.tree, "tm_deep", 0.5), base=secs * stretch)
                if parity > 0 else None)
        # No ceiling in time trouble either. The clock-based ceiling can only raise the search's hard
        # stop (it is how a clock lead buys a long think). Where the stretch is off, so is the ceiling.
        if timectl.sudden(inc, ov) or mine / 1000.0 < timectl.LOW_CLOCK:
            hard = None
        if hard is not None and extends:
            self.crit = min(CRIT_X * secs, CRIT_SHARE * mine / 1000.0)
        elif extends and inc > 0 and mine / 1000.0 >= timectl.LOW_CLOCK:
            self.crit = min(CRIT_X * secs, CRIT_SHARE_SUDDEN * mine / 1000.0)
        return secs, hard, 10 ** 7

    def _net_price(self, plan, inc):
        """The network price per move to plan this clock on (NET_WARM above). This move is warm or
        cold by its gap, the time since the last bestmove went out plus what it plans to think at this
        game's mean price so far. The clock is planned on the mean of the last twenty moves including
        this one, so one slow opponent reply does not swing the plan. Without an increment the flat
        NET_SUDDEN is kept (the gap is still logged), or the tree's tm_short_sudden in a game that
        started at SHORT_BASE or less."""
        if NET_OVERHEAD is not None:
            return NET_OVERHEAD
        flat = NET_SUDDEN
        short = float(getattr(self.tree, "tm_short_sudden", 0.0) or 0.0)
        if short > 0.0 and self._start_clock is not None and self._start_clock <= SHORT_BASE:
            flat = short
        # A game's first moves are planned as if the missing ones were warm (ten in all), so one cold
        # gap early, such as the opponent's untimed first move on Lichess, cannot price the plan alone.
        def mean_of(seen):
            seen = seen + [NET_WARM] * max(0, 10 - len(seen))
            return sum(seen) / len(seen) + NET_INSIDE
        prior = flat if inc <= 0 else mean_of(self._ov_seen)
        since = 0.0 if self._bm_at is None else max(0.0, time.time() - self._bm_at)
        now = NET_COLD if since + plan(prior) > NET_COLD_AFTER else NET_WARM
        self._ov_seen = (self._ov_seen + [now])[-20:]
        mean = mean_of(self._ov_seen)
        price = flat + NET_INSIDE if inc <= 0 else mean
        log(f"# net {now:.2f}s after a {since:.1f}s gap, planned at {price:.2f}")
        return price

    def warm(self):
        """Load and run one small search in the background the moment the process starts.

        lichess-bot starts the engine when the game starts and asks for a fixed ten-second first move,
        so a load that only began then could get the game aborted. The first search after a load is
        slow on its own too (kernel selection and the autocast weight cache), so one throwaway search
        pays for that here and reset() then discards every trace of it. The GUI's own commands wait on
        the same lock instead of loading twice."""
        try:
            with self._load_lock:
                self._load()
                self.tree.search(chess.Board(), visits=96, reuse=False)
                self.tree.reset()
            log("# warmed")
        except Exception as e:                       # a failed warm-up must never stop the engine
            log(f"# warm-up failed: {type(e).__name__}: {e}")

    def _first_move_fallback(self):
        """The first move without the network, when the load has outrun FIRST_WAIT (see there).
        Returns True if a move went out."""
        prev = self.board.move_stack[0].uci() if self.board.move_stack else ""
        legal = list(self.board.legal_moves)
        for u in (FIRST_REPLIES.get(prev), "g8f6", "d7d5", "d2d4"):
            if u and chess.Move.from_uci(u) in legal:
                log(f"# first move without the network: not loaded {FIRST_WAIT:.0f} s after go, "
                    f"playing {u} and loading behind it")
                say(f"bestmove {u}")
                return True
        return False

    def go(self, words):
        self._go_started = time.time()
        if (self.tree is None and len(self.board.move_stack) <= 1 and "ponder" not in words
                and FIRST_WAIT > 0):
            # the warm-up holds the load lock, so wait for it, but not past FIRST_WAIT (see there)
            if self._load_lock.acquire(timeout=FIRST_WAIT):
                self._load_lock.release()
            elif self._first_move_fallback():
                return
        self.ready()
        # Tell the trainer somebody is playing, with the same flag the board raises. The trainer watches
        # its mtime and hands over YIELD_MS between steps while it is fresh, so a live game gets priority
        # over a training run on the same GPU.
        self._touch_playing()
        opts = self._opts(words)
        # The clock ran while go waited. isready does not wait for the warm-up, so a GUI that starts its
        # clock at go pays for the load here, and the plan is made on what is left of the clock it sent.
        side = "wtime" if self.board.turn == chess.WHITE else "btime"
        if side in opts and "ponder" not in words:
            opts[side] = max(1, opts[side] - int(1000 * max(0.0, time.time() - self._go_started)))
        if "ponder" in words:
            self.ponder(opts)
            return
        self._search(opts)

    @staticmethod
    def _opts(words):
        opts, i = {}, 1
        while i < len(words):
            w = words[i]
            if w == "infinite":
                opts["infinite"] = 1
            elif w in ("wtime", "btime", "winc", "binc", "movetime", "nodes", "depth", "movestogo"):
                try:
                    opts[w] = int(words[i + 1])
                except (IndexError, ValueError):
                    pass
                i += 1
            i += 1
        return opts

    # ------------------------------------------------------------------ thinking on their time
    def ponder(self, opts):
        """Think on the opponent's time. The GUI sends `position ... X Y` and `go ponder`, Y being the
        reply this engine predicted. The search runs from the position before Y, the opponent's own, so
        every reply is explored in proportion to how likely the tree finds it. On `ponderhit` the tree
        under Y is kept, and on a miss the reply actually played still has its own subtree, which a
        search rooted after Y would have thrown away. Nothing is printed while pondering."""
        self.ponder_opts = dict(opts)
        root = self.board.copy()
        if root.move_stack:
            root.pop()
        log("# pondering on the opponent's time")
        # The abort flag is cleared here, before the thread exists, and never inside it. If the search
        # cleared it as it started, a ponderhit landing between `go ponder` and that line would be
        # silently undone and the ponder would run on beside the real search. MCTS.ponder has the same
        # rule for the board.
        self.tree.abort.clear()

        def run():
            try:
                _, info = self.tree.search(root, visits=10 ** 9, reuse=True, clear_abort=False,
                                           max_nodes=MAX_NODES)
                if info.get("at_ceiling") and not self.tree.abort.is_set():
                    log(f"# ponder stopped at the memory ceiling: {info['tree_nodes']} nodes, "
                        f"about {info['tree_nodes'] * NODE_BYTES / 1e9:.1f} GB")
            except Exception as exc:
                log(f"# ponder failed: {exc!r}")

        self.pondering = threading.Thread(target=run, daemon=True)
        self.pondering.start()

    def _end_ponder(self):
        if self.pondering is None:
            return False
        if self.tree is not None:
            self.tree.abort.set()
        self.pondering.join(timeout=10)
        if self.pondering.is_alive():
            # A ponder that has not stopped still owns the tree. Clearing the flag now would leave two
            # threads on one tree, so the flag stays set until it has gone.
            log("# ponder slow to stop, still waiting")
            self.pondering.join()
        self.pondering = None
        if self.tree is not None:
            self.tree.abort.clear()
        return True

    def ponderhit(self):
        """They played the predicted move. Search for real from the position after it, reusing the
        tree, on the clock the GUI sent with `go ponder`, corrected as below before the budget is
        worked out."""
        opts = dict(self.ponder_opts or {})
        self._end_ponder()
        mine = "wtime" if self.board.turn == chess.WHITE else "btime"
        inc = "winc" if self.board.turn == chess.WHITE else "binc"
        # python-chess sends `go ponder` with the clock as it stands after this engine's move, the
        # increment included, so the search time and increment must not be counted again. But the
        # network cost of posting the last move is not in it, so this game's price for that move
        # comes off. With a fixed NET_OVERHEAD (clock_gate) nothing does.
        if mine in opts:
            back = self._ov_seen[-1] if (self._ov_seen and NET_OVERHEAD is None) else 0.0
            opts[mine] = max(1, opts[mine] - int(1000 * back))
        # and the opponent's clock ran while they thought, since python-chess sends the one it had
        # when pondering began
        theirs = "btime" if self.board.turn == chess.WHITE else "wtime"
        if theirs in opts and self._bm_at is not None:
            opts[theirs] = max(1, opts[theirs] - int(1000 * max(0.0, time.time() - self._bm_at)))
        self._touch_playing()
        log("# ponderhit")
        self._search(opts)

    def _touch_playing(self):
        """Tell the trainer somebody is playing (it yields the card while this file is fresh)."""
        try:
            with open(PLAYING_FILE, "w") as fh:
                fh.write(str(int(time.time())))
        except OSError:
            pass

    def _keepalive(self):
        """Refresh the playing flag every 20 s while a search or a ponder runs, so a long ponder
        followed by a long search does not let the flag go stale and the trainer take the GPU back."""
        while True:
            time.sleep(20)
            if any(th is not None and th.is_alive() for th in (self.searching, self.pondering)):
                self._touch_playing()

    def _easy(self, mv):
        """True when the move just found holds EASY_SHARE of the root's visits and the root has
        at least EASY_VISITS, a decision the rest of the allowance would not change."""
        r = getattr(self.tree, "root", None)
        if mv is None or r is None or getattr(r, "N", None) is None or not len(r.moves):
            return False
        if len(r.moves) == 1:
            return True                     # forced, nothing to search
        n = r.N
        tot = float(n.sum())
        if tot < EASY_VISITS:
            return False
        i = int(n.argmax())
        return r.moves[i] == mv and float(n[i]) >= EASY_SHARE * tot

    def _search(self, opts):
        secs, hard, visits = self.budget(opts)
        # `go movetime` is exact. The search's own stretch (tm_extend, up to 1.6x) is for clock play, and
        # a controller that fixes the time per move scores an overrun as a loss. So a movetime search runs
        # with the stretch off, and the sealed value is restored after it.
        exact = "movetime" in opts
        # cleared before the thread starts, for the same reason as in ponder(), so a `stop` that
        # arrives before the search begins is not undone by it
        self.tree.abort.clear()

        def run():
            sealed_ext = getattr(self.tree, "tm_extend", False)
            if exact:
                self.tree.tm_extend = False
            try:
                # An easy move is played early. tm_stop only fires after a third of the allowance and
                # on a clear value lead, which this network's flat evaluations rarely show. So the
                # search first gets a short slice. If one move then holds nearly all of a well-visited
                # tree (visits carried from pondering included) it is played, otherwise the search
                # continues on the same tree with the rest of the allowance. The sealed search is
                # untouched.
                t_first = time.time()
                mv, info, easy = None, {}, False
                looked = secs is not None and secs > 2.0 * EASY_FLOOR
                if looked:
                    mv, info = self.tree.search(self.board, visits=visits,
                                                seconds=max(EASY_FLOOR, EASY_SLICE * secs),
                                                reuse=True, clear_abort=False, max_nodes=MAX_NODES)
                    easy = self._easy(mv)
                    if easy:
                        log("# easy move")
                if not easy and not self.tree.abort.is_set():
                    used = time.time() - t_first
                    # The same position continued, not a new one. Reuse discounts carried visits by
                    # ponder_discount, which is meant for a tree carried across a move and here would
                    # throw away most of the first look. So the continuation runs undiscounted and the
                    # sealed value is put back at once.
                    sealed = getattr(self.tree, "ponder_discount", 1.0)
                    # Only after a first look. Without one this is the move's first search, and the
                    # carried tree must be discounted as the sealed search intends.
                    if looked:
                        self.tree.ponder_discount = 1.0
                    crit = getattr(self, "crit", None)
                    first = info
                    try:
                        mv, info = self.tree.search(
                            self.board, visits=visits,
                            seconds=None if secs is None else max(0.001, secs - used),
                            hard_seconds=None if hard is None else max(0.001, hard - used),
                            reuse=True, clear_abort=False, max_nodes=MAX_NODES,
                            crit_seconds=None if crit is None else max(0.001, crit - used))
                    finally:
                        self.tree.ponder_discount = sealed
                    if first:
                        # the log's `carried` is what came into this move (the ponder, the last
                        # search), not the first look this move has just taken
                        info = dict(info, carried=first.get("carried", 0))
                        # and its time is the whole move's, first look included. Only the log
                        # changes, not the search.
                        for k in ("spent", "allowed"):
                            if info.get(k) is not None:
                                info[k] = round(float(info[k]) + used, 3)
            except Exception as exc:                   # a crash must not hang the GUI forever
                log(f"# search failed: {exc!r}")
                mv, info = None, {}
            finally:
                if exact:
                    self.tree.tm_extend = sealed_ext
            if "infinite" in opts:
                # In UCI `go infinite` answers only after `stop`, even when the tree is full or the root
                # has no move. stop() and quit set the flag before they join this thread.
                self.tree.abort.wait()
            if mv is None:
                # A search that returned nothing still owes a move. The bare policy head answers in
                # milliseconds and sensibly, better than the first legal move.
                try:
                    from mcts import pinned_search
                    from play_web import policy_move
                    dev = str(next(self.model.parameters()).device)
                    # the seal's castle_prior too, since v4 keeps O-O at e1h1
                    seal = pinned_search(self.version)
                    mv = policy_move(self.model, self.board.copy(), dev,
                                     seal.get("policy_head", "cls"),
                                     castle=seal.get("castle_prior", "king"))
                except Exception:
                    legal = list(self.board.legal_moves)
                    mv = legal[0] if legal else None
            if mv is None:
                say("bestmove 0000")
                return
            # A GUI wants a score in centipawns from the side to move, and the tree speaks in a
            # win probability. The mapping is the usual logistic one, capped so a proven win
            # prints as a large number rather than an infinity the GUI has to parse.
            q = float(info.get("q", 0.0))
            p = min(max((q + 1.0) * 0.5, 1e-4), 1 - 1e-4)
            cp = int(max(-3000, min(3000, -400.0 * __import__("math").log10(1 / p - 1))))
            n = int(info.get("visits", 0))
            pv = info.get("pv") or mv.uci()
            # Every move's time goes in the log, so a missed deadline leaves evidence.
            spent, allowed = info.get("spent"), info.get("allowed")
            if spent is not None:
                self.last_spent = float(spent)
                # An extension is not an overrun. v4 seals tm_extend with tm_extend_max 0.6, so a move
                # whose top two will not separate may run to 1.6x its allocation by design.
                ext = int(info.get("extended", 0) or 0)
                over = (f" extended x{ext}" if ext else
                        (" OVERRUN" if allowed and spent > allowed * 1.25 else ""))
                log(f"# spent {spent}s of {allowed}s, {n} visits, {info.get('carried', 0)} carried, "
                    f"tree {info.get('tree_nodes', 0)} nodes{over}, stop: {info.get('stop', '?')}")
            say(f"info depth {self._depth(mv, info)} nodes {n} score cp {cp} pv {pv}")
            # The reply to ponder on is the second move of the principal line, when it is legal. A GUI
            # that does not ponder ignores it, and one that does sends it back with `go ponder`.
            reply = ""
            words_pv = pv.split()
            if len(words_pv) >= 2:
                try:
                    after = self.board.copy()
                    after.push(mv)
                    guess = chess.Move.from_uci(words_pv[1])
                    # Don't offer a reply that ends the game, since there is nothing to ponder after it.
                    if guess in after.legal_moves:
                        after.push(guess)
                        if not after.is_game_over():
                            reply = f" ponder {guess.uci()}"
                except ValueError:
                    pass
            self._bm_at = time.time()
            say(f"bestmove {mv.uci()}{reply}")

        self.searching = threading.Thread(target=run, daemon=True)
        self.searching.start()

    def _depth(self, mv, info):
        """Depth is the length of the line the tree expects, in plies. A tree search has no iteration
        depth, so this follows the chosen move and then the most visited reply for as long as the
        replies have visits. lc0 prints the tree's mean depth instead, which needs a walk of every
        node. The printed line stays at twelve plies, and this one is not capped."""
        root = getattr(self.tree, "root", None)
        if not info.get("visits") or root is None or not root.expanded:
            return 1
        try:
            first = root.moves.index(mv)
        except ValueError:
            return 1
        return max(1, len(self.tree._pv(root, depth=10 ** 4, first=first).split()))

    def stop(self):
        if self._end_ponder():
            # A miss, they played something else. UCI owes a bestmove after `stop` and the GUI
            # throws this one away. The real search follows with the position they reached and
            # reuses whatever the ponder built under their move.
            legal = list(self.board.legal_moves)
            log("# ponder miss")
            say(f"bestmove {legal[0].uci()}" if legal else "bestmove 0000")
            return
        if self.tree is not None:
            self.tree.abort.set()
        if self.searching is not None:
            self.searching.join(timeout=10)
            if self.searching.is_alive():
                # the flag stays set until the search has gone, or a search with no deadline would
                # run on and never answer, and the next go would share its tree
                log("# search slow to stop, still waiting")
                self.searching.join()
            self.searching = None
        if self.tree is not None:
            self.tree.abort.clear()


_STACKS = None


def stall_dump():
    """Every thread's traceback into <LOG>.stacks every STALL_DUMP seconds. A search that never
    returns prints nothing by itself, the engine just goes quiet and loses on time. Off unless asked
    for."""
    global _STACKS
    every = E("STALL_DUMP", "")
    if not every:
        return
    import faulthandler
    _STACKS = open((LOG or "uci") + ".stacks", "a", encoding="utf-8", buffering=1)
    faulthandler.dump_traceback_later(float(every), repeat=True, exit=False, file=_STACKS)


def main():
    raise_priority()
    stall_dump()
    eng = Engine()
    threading.Thread(target=eng.warm, daemon=True).start()
    threading.Thread(target=eng._keepalive, daemon=True).start()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        log("> " + line)
        words = line.split()
        cmd = words[0]
        if cmd == "uci":
            eng.uci()
        elif cmd == "isready":
            # A warm-up in progress is not waited for here. python-chess holds go back until readyok,
            # so a wait here would land before go, where the first-move cap cannot count it. go waits
            # on the same lock and counts it.
            if eng._load_lock.acquire(blocking=False):
                try:
                    eng.ready()
                finally:
                    eng._load_lock.release()
            say("readyok")
        elif cmd == "setoption":
            eng.setoption(words)
        elif cmd == "ucinewgame":
            eng._end_ponder()
            # the same here, a warm-up holding the lock resets the tree itself when it ends
            if eng._load_lock.acquire(blocking=False):
                try:
                    eng.ready()
                    eng.tree.reset()
                finally:
                    eng._load_lock.release()
            eng.board = chess.Board()
            eng._bm_at, eng._ov_seen, eng._start_clock = None, [], None
        elif cmd == "position":
            eng.position(words)
        elif cmd == "go":
            eng.go(words)
        elif cmd == "stop":
            eng.stop()
        elif cmd == "ponderhit":
            eng.ponderhit()
        elif cmd == "quit":
            eng.stop()
            break
    # EOF or quit. A search running in the background still owes a bestmove (a closing pipe is how
    # this gets tested from a shell), so wait for it rather than exiting mid-think.
    if eng.searching is not None:
        eng.searching.join(timeout=600)


if __name__ == "__main__":
    main()
