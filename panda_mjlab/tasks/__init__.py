"""Panda manipulation task registration."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from panda_mjlab.tasks.env_cfgs import (
  panda_lift_cube_depth_env_cfg,
  panda_lift_cube_env_cfg,
)
from panda_mjlab.tasks.rl_cfg import (
  panda_lift_cube_depth_ppo_runner_cfg,
  panda_lift_cube_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Panda",
  env_cfg=panda_lift_cube_env_cfg(),
  play_env_cfg=panda_lift_cube_env_cfg(play=True),
  rl_cfg=panda_lift_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Panda-Depth",
  env_cfg=panda_lift_cube_depth_env_cfg(),
  play_env_cfg=panda_lift_cube_depth_env_cfg(play=True),
  rl_cfg=panda_lift_cube_depth_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
