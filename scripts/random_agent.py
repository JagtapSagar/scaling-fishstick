"""Run Panda task with random actions and the native MuJoCo viewer."""

import panda_mjlab.tasks  # noqa: F401 — registers Mjlab-Lift-Cube-Panda*

from mjlab.scripts.play import main

if __name__ == "__main__":
  # Equivalent to: play.py <task> --agent random
  import sys

  sys.argv.insert(2, "--agent")
  sys.argv.insert(3, "random")
  main()
