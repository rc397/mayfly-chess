"""Run search_match.py with its three arguments taken from the environment.

The queue runner only passes arguments to a row whose script is search_match.py, so a row that runs
from a second code snapshot uses this file. AB_OPTS holds B's settings, AB_GAMES and AB_VISITS the
match size. BASE, VERSION and the rest are read by search_match as usual. Imports come from this
file's own folder, which is the snapshot's.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["search_match.py", os.environ.get("AB_OPTS", ""), os.environ.get("AB_GAMES", "200"),
            os.environ.get("AB_VISITS", "200")]
runpy.run_path(os.path.join(HERE, "search_match.py"), run_name="__main__")
