"""Custom reward functions for Panda cube lifting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_goal_distance(
  env: ManagerBasedRlEnv,
  std: float = 0.3,
  minimal_height: float = 0.04,
  command_name: str = "lift_height",
  object_name: str = "cube",
) -> torch.Tensor:
  """Reward for tracking the goal position using tanh-kernel, gated by lift height.

  Matches Isaac Lab's object_goal_distance:
    (obj_z > minimal_height) * (1 - tanh(dist_to_goal / std))

  Note: MJLab's LiftingCommand.target_pos is already in world frame,
  unlike Isaac Lab which stores it in robot base frame.
  """
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  goal_pos_w = command.target_pos  # (num_envs, 3), world frame

  obj_pos_w = obj.data.root_link_pos_w
  distance = torch.norm(goal_pos_w - obj_pos_w, dim=1)

  lifted = (obj_pos_w[:, 2] > minimal_height).float()
  return lifted * (1.0 - torch.tanh(distance / std))


def object_is_lifted(
  env: ManagerBasedRlEnv,
  minimal_height: float = 0.04,
  object_name: str = "cube",
) -> torch.Tensor:
  """Binary reward: 1.0 if object is above minimal_height, else 0.0.

  Matches Isaac Lab's object_is_lifted.
  """
  obj: Entity = env.scene[object_name]
  return torch.where(obj.data.root_link_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
  env: ManagerBasedRlEnv,
  std: float = 0.1,
  object_name: str = "cube",
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", site_names=("grasp_site",)
  ),
) -> torch.Tensor:
  """Reward for reaching the object using tanh-kernel.

  Matches Isaac Lab's object_ee_distance: 1 - tanh(dist / std).
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]

  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)  # (num_envs, 3)
  obj_pos_w = obj.data.root_link_pos_w  # (num_envs, 3)

  distance = torch.norm(obj_pos_w - ee_pos_w, dim=1)
  return 1.0 - torch.tanh(distance / std)


def finger_object_distance(
  env: ManagerBasedRlEnv,
  std: float = 0.1,
  object_name: str = "cube",
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot",
    body_names=("left_finger", "right_finger"),
    joint_names=("finger_joint1", "finger_joint2"),
  ),
) -> torch.Tensor:
  """Proximity reward for fingers near object, plus gripper closure bonus."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]

  # Finger body positions via body_link_pose_w (num_envs, num_bodies, 7).
  # asset_cfg.body_ids resolves the named bodies to indices.
  body_poses = robot.data.body_link_pose_w[:, asset_cfg.body_ids]
  left_pos = body_poses[:, 0, :3]   # left_finger
  right_pos = body_poses[:, 1, :3]  # right_finger

  # Object root position.
  obj_pos = obj.data.root_link_pos_w

  # Exponential proximity reward.
  left_dist = torch.norm(left_pos - obj_pos, dim=-1)
  right_dist = torch.norm(right_pos - obj_pos, dim=-1)
  proximity = torch.exp(-left_dist / std) + torch.exp(-right_dist / std)

  # Gripper closure: finger_joint range is [0, 0.04], 0=closed, 0.04=open.
  finger_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
  gripper_closed = 1.0 - finger_pos.mean(dim=-1) / 0.04  # 0=open, 1=closed

  # Only reward closing when fingers are near the object.
  near_object = (proximity > 1.5).float()
  closure_reward = near_object * gripper_closed

  return proximity + closure_reward


def ee_height_penalty(
  env: ManagerBasedRlEnv,
  min_height: float = 0.13,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("grasp_site",)),
) -> torch.Tensor:
  """Penalize end-effector dropping below a minimum height."""
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids]  # (num_envs, 1, 3)
  ee_height = ee_pos[:, 0, 2]
  penalty = torch.clamp(min_height - ee_height, min=0.0)
  return -penalty
