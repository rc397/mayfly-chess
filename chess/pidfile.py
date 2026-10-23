r"""Pid files for long running jobs, so a supervisor can find them.

On Windows a process's command line can be unreadable across session boundaries (for example from
a scheduled task), while a process id is visible from anywhere. So every long running job writes its
own id down and the supervisor reads the file.

  from pidfile import claim
  claim("train_av")        # writes .pid_train_av.txt beside the code
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def path_for(name):
    # PID_SUFFIX keeps a throwaway copy, such as selftest.py's own play_web on a spare port, from
    # overwriting the real job's pid file.
    return os.path.join(HERE, ".pid_" + name + os.environ.get("PID_SUFFIX", "") + ".txt")


def claim(name):
    """Record this process's id.

    Best effort, since a job that cannot write its pid file should still run."""
    try:
        with open(path_for(name), "w") as f:
            f.write(str(os.getpid()))
    except OSError:
        pass
    return os.getpid()


def read(name):
    """The id last claimed under this name, or None."""
    try:
        return int(open(path_for(name)).read().strip())
    except Exception:
        return None
