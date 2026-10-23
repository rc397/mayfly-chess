# The A/B queue

One search experiment at a time, from a frozen copy of the code, so an edit to the working tree
cannot change an experiment halfway through a run.

    queue.txt      one experiment a line, name | ENV=value;ENV=value | opts | games | visits [| script]
    done.txt       a finished name, so a restart resumes rather than repeats
    results.jsonl  one row per finished run
    logs/          per run console output and a per game log, so a crash costs one game
    snap/chess     the frozen code, a copy of chess/ refreshed by hand

`snap/` is not tracked. Copy `chess/` there when you start a queue and note the commit it came from
in `snap/TAKEN_FROM.txt`. A row can name its script by absolute path, which picks the frozen copy it
runs from, so several snapshots can serve one queue. A snapshot's code is never changed under a row
that is running.

    python _abq/ab_queue.py

The runner follows a few rules.

- It never stops on its own, so rows queued later by a scheduled job still run. Set IDLE_HOURS to
  have it leave after that many idle hours.
- A failed run gets one retry, and everything it runs resumes game by game. A run that lasted an
  hour before failing made progress, so its failure counts as a fresh one.
- Jobs start only at night unless their env has DAY_OK=1. The night hours come from
  NIGHT_HOURS_FILE (see the top of `ab_queue.py`). A night job still running when the day starts
  stops and resumes the next night.
- The openings come from PGN, `chess/twic` by default.
