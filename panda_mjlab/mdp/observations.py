"""Custom observation functions for Panda cube lifting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_position_in_robot_root_frame(
  env: ManagerBasedRlEnv,
  object_name: str = "cube",
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Object position expressed in the robot's root (base) frame.

  Matches Isaac Lab's object_position_in_robot_root_frame.
  """
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_name]

  obj_pos_w = obj.data.root_link_pos_w  # (num_envs, 3)
  robot_pos_w = robot.data.root_link_pos_w  # (num_envs, 3)
  robot_quat_w = robot.data.root_link_quat_w  # (num_envs, 4)

  # Transform object position into robot base frame.
  obj_pos_rel_w = obj_pos_w - robot_pos_w
  obj_pos_b = quat_apply(quat_inv(robot_quat_w), obj_pos_rel_w)
  return obj_pos_b


def object_root_lin_vel_w(
  env: ManagerBasedRlEnv,
  object_name: str = "cube",
) -> torch.Tensor:
  """Object root linear velocity in world frame.

  Matches Isaac Lab's critic obs: root_lin_vel_w for the object.
  """
  obj: Entity = env.scene[object_name]
  return obj.data.root_link_lin_vel_w  # (num_envs, 3)


def object_root_quat_w(
  env: ManagerBasedRlEnv,
  object_name: str = "cube",
) -> torch.Tensor:
  """Object root quaternion in world frame.

  Matches Isaac Lab's critic obs: root_quat_w for the object.
  """
  obj: Entity = env.scene[object_name]
  return obj.data.root_link_quat_w  # (num_envs, 4)
