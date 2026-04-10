"""Play script that registers Panda tasks then delegates to mjlab."""

import panda_mjlab.tasks  # noqa: F401 — registers Mjlab-Lift-Cube-Panda*

from mjlab.scripts.play import main

if __name__ == "__main__":
  main()
