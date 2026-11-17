r"""timectl decides how many seconds to give a move under a clock.

Under a clock the question is where the time should go, not how good a search is per visit. A constant
multiplier on the budget got most of what an entropy-scaled allocator got, so the allocation here is the
plain one, made safe.

    a slice of what remains, plus most of the increment, capped so one move cannot flag the game

Where the thinking goes lives in the search. `tm_extend` keeps thinking while the top two moves are too
close to separate, and `tm_stop` stops early when the leader is clearly ahead and banks the time.

  from timectl import allocate
  allocate(remaining=180.0, inc=2.0, ply=12)
"""
import os

E = os.environ.get
# How many of this side's moves are still to come. It falls with the ply but never below a floor,
# since a long game is not about to end just because the estimate says so.
MOVES_EARLY = float(E("TM_MOVES_EARLY", 44.0))
MOVES_DECAY = float(E("TM_MOVES_DECAY", 0.4))
MOVES_FLOOR = float(E("TM_MOVES_FLOOR", 12.0))
# Most of the increment is free to spend because it comes back after the move. Not all of it, or a
# long game drifts down to the increment and every move is played at the same shallow depth.
INC_SHARE = float(E("TM_INC_SHARE", 0.85))
# No single move may take more than this fraction of what is left. This is the whole flag guard.
MAX_SHARE = float(E("TM_MAX_SHARE", 0.25))
# Below this the engine is in time trouble and plays on the increment alone.
PANIC = float(E("TM_PANIC", 1.0))


# A move costs the clock more than its search. Stopping the ponder, the bridge and the network to the
# server add about 0.4 to 0.6 s a move online. Without counting it, no-increment games past about 50
# moves flagged. Where the increment pays for it, nothing changes.
OVERHEAD = float(E("TM_OVERHEAD", 0.6))
# With the clock only draining, plan for a long game, since a won ending can take thirty moves to
# convert. Forty keeps a 130-move 3+0 game on the board in simulation for 0.3 s less thought a move early.
SUDDEN_FLOOR = float(E("TM_SUDDEN_FLOOR", 40.0))


def sudden(inc, overhead=None):
    """True when the increment does not pay for a move's overhead, so the clock only drains.
    `overhead` is the caller's measured cost of a move beyond its search, and None means OVERHEAD."""
    return INC_SHARE * float(inc or 0.0) < (OVERHEAD if overhead is None else float(overhead))


def moves_left(ply, inc=None, floor=None, overhead=None):
    n = max(MOVES_FLOOR if floor is None else float(floor), MOVES_EARLY - MOVES_DECAY * ply)
    if inc is not None and sudden(inc, overhead):
        n = max(n, SUDDEN_FLOOR)
    return n


# Under this much clock the stretch is off whatever the increment. At 1+1 with a second or two left
# the search could stretch its allowance several times over while each move also paid the overhead.
LOW_CLOCK = float(E("TM_LOW_CLOCK", 15.0))


# The stretch is paid for on average, not only in the worst case, and the increment does not pay it
# back. When the search extended on most moves without this, games spent about 1.4 times the plan and
# ran short of time. A caller that passes `avg` hands the search the plan divided by the expected
# stretch, so the clock pays the plan on average and a close decision can still run to the full stretch.
#
# Measured over 2,742 searches the search used 1.16 of its share. Close but stable moves keep the sealed
# 1.6x and an unstable one may run to uci.CRIT_X (2.5x). 1.25 is the measured 1.16 plus an estimate for
# those longer runs.
EXT_AVG = float(E("TM_EXT_AVG", 1.25))
EXT_AVG_SUDDEN = float(E("TM_EXT_AVG_SUDDEN", 1.16))


def search_share(secs, remaining, inc, ext=1.6, avg=1.0, overhead=None, avg_sudden=None):
    """The allowance to hand the search, which may stretch it by `ext` on its own when the top two
    moves are close. With an increment the average stretch `avg` is planned for (EXT_AVG above, and
    1.0 leaves the allowance as it is). Under LOW_CLOCK the worst case must be the plan, so the
    allowance is divided by the whole stretch. In sudden death above LOW_CLOCK the caller may pass
    `avg_sudden`, the stretch it measured there, and None keeps the worst case."""
    if secs is None:
        return secs
    if float(remaining) < LOW_CLOCK:
        return max(MOVE_NOW, secs / max(1.0, ext))
    if sudden(inc, overhead):
        return max(MOVE_NOW, secs / max(1.0, ext if avg_sudden is None else avg_sudden))
    if avg <= 1.0:
        return secs
    return max(MOVE_NOW, secs / float(avg))


OPEN_SPEND = float(E("TM_OPEN_SPEND", 0.4))   # share of a normal allocation on move one
OPEN_PLIES = int(E("TM_OPEN_PLIES", 16))     # ramped back to full by here


def opening_factor(ply, open_spend=1.0, open_plies=OPEN_PLIES):
    """How much of a normal allocation this ply deserves, ramping up out of the opening.

    `moves_left` falls with the ply, so the plain formula gives move one almost as much time as the
    middlegame. The opening is where the policy is most confident and search buys least, so time saved
    there goes where the game is decided. With open_spend 1.0 this is a no-op, which keeps v1 to v3 as
    they were released.
    """
    if open_spend >= 1.0 or open_plies <= 0:
        return 1.0
    t = min(1.0, max(0, ply) / float(open_plies))
    return open_spend + (1.0 - open_spend) * t


# A zero allowance must mean "move now", never "no clock". The search reads seconds=0.0 as no deadline
# (`if seconds`), so a nearly empty clock would start an unlimited search. The smallest positive
# allowance is a deadline of now, so the search stops at its first check and plays.
MOVE_NOW = 0.001


def allocate(remaining, inc=0.0, ply=0, floor=0.05, spend=1.0, ext=1.6, opp=None, parity=0.0,
             open_spend=1.0, open_plies=OPEN_PLIES, overhead=None, floor_moves=None):
    """Seconds to give this move.

    `remaining` is this side's clock, `inc` the increment, `ply` the plies played so far, `spend` a
    multiplier for a side that wants to use its clock harder, and `ext` the most the search may stretch
    this by, which must be known here for the guarantee below to hold.

    `opp` and `parity` are the opponent's clock and how much of a lead over it to convert into depth.
    Time not spent is thrown away at the end of the game, so the lead is spread over the moves still
    expected. The quarter-of-the-clock cap still applies, so converting a lead can never flag.

    The invariant is that after this move, extended as far as the search may, the clock still holds
    `keep`. The increment is not counted, because it only arrives once the move is made.

    `overhead` and `floor_moves` are for a caller that has measured what a move costs beyond its search
    and plays long games. None for both gives the plain allocation. Given an overhead, the increment pays
    it first and only the rest is thinking time. Online a move can cost about 1.25 s more than its search.
    """
    remaining = max(0.0, float(remaining))
    keep = max(0.15, 0.05 * remaining)
    room = max(0.0, remaining - keep) / max(1.0, ext)
    if overhead is None:
        free = INC_SHARE * inc                                   # the increment spent on thinking
        owed = max(0.0, OVERHEAD - INC_SHARE * inc)              # the overhead it leaves unpaid
    else:
        free = min(INC_SHARE * inc, max(0.0, inc - float(overhead)))
        owed = max(0.0, float(overhead) - inc)
    if remaining <= PANIC:
        # no time to think, so move on what the increment will refund, or a tenth of the rest
        return max(MOVE_NOW, min(room, remaining * 0.10 + free))
    n = moves_left(ply, inc, floor_moves, overhead)
    # No cap on the reserve. In simulation, capping it at half the clock searched more of a 1+0 or 2+0
    # game but flagged more of them within 200 plies. With 0.7 s of overhead a move, a long 1+0 game is
    # short of time whatever this does.
    reserve = owed * n                                           # the overhead every move still owes
    want = (max(0.0, remaining - PANIC - reserve) / n) * opening_factor(ply, open_spend, open_plies)
    want += free
    if parity > 0.0 and opp is not None:
        want += parity * max(0.0, remaining - max(0.0, float(opp))) / n
    return max(MOVE_NOW, min(floor, room), min(want * spend, remaining * MAX_SHARE, room))


# How much of a lead over the opponent's clock one move may spend, and the hardest cap on a single
# think. A side ahead on the clock may spend a critical move's time down to parity.
SURPLUS = float(E("TM_SURPLUS", 1.0))       # all of the lead, down to parity
DEEP_SHARE = float(E("TM_DEEP_SHARE", 0.5))  # and never more than half its own clock on one move


def ceiling(remaining, opp=None, inc=0.0, ply=0, floor=0.05, spend=1.0, parity=1.0, deep=DEEP_SHARE,
            base=None):
    """The furthest a single move may be stretched when the position will not resolve.

    `allocate` caps a normal move at a quarter of the clock. For the move the game turns on, a side
    that is ahead on the clock can spend its lead down to parity and be no worse off in the time race.
    So this is the ordinary allocation plus `parity` of the lead, capped at `deep` of its own clock and
    at what the flag guard leaves. With `parity` at zero nothing changes, which keeps v1 to v3 as
    released.

    It is a ceiling, not a plan. The search walks out to it in small steps and stops as soon as the
    decision resolves.

    `base` is the search's own furthest stretch when the caller knows it. Recomputing it here from the
    full allocation would sit above what search_share hands the search, so close moves would overrun.
    """
    remaining = max(0.0, float(remaining))
    keep = max(0.15, 0.05 * remaining)
    if base is None:
        base = allocate(remaining, inc, ply, floor, spend, ext=1.0) * 1.6
    room = max(0.0, remaining - keep)
    if parity <= 0.0 or opp is None:
        return max(MOVE_NOW, min(base, room))
    lead = max(0.0, remaining - max(0.0, float(opp)))
    return max(MOVE_NOW, min(max(base, base + parity * lead), deep * remaining, room))


def describe(base, inc):
    """The name a person would use for this time control."""
    total = base + 40 * inc
    kind = ("bullet" if total < 180 else "blitz" if total < 600
            else "rapid" if total < 3600 else "classical")
    mins = base / 60
    label = f"{mins:g}+{inc:g}" if base % 60 == 0 else f"{base:g}s+{inc:g}"
    return f"{label} {kind}"
