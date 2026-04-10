"""Binary gripper action for Franka Panda."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BinaryJointPositionActionCfg(ActionTermCfg):
  """Binary open/close gripper action.

  Takes a 1D action from the policy: >0 opens, <=0 closes.
  Maps to joint position targets for the specified finger joint.
  """

  actuator_names: tuple[str, ...] = ("finger_joint1",)
  """Finger joint actuator(s) to control."""

  open_position: float = 0.04
  """Joint position when gripper is open."""

  close_position: float = 0.0
  """Joint position when gripper is closed."""

  def build(self, env: ManagerBasedRlEnv) -> BinaryJointPositionAction:
    return BinaryJointPositionAction(self, env)


class BinaryJointPositionAction(ActionTerm):
  """Binary open/close gripper action term."""

  cfg: BinaryJointPositionActionCfg
  _entity: Entity

  def __init__(self, cfg: BinaryJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    target_ids, target_names = self._entity.find_joints_by_actuator_names(
      cfg.actuator_names
    )
    self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
    self._target_names = target_names
    self._num_targets = len(target_ids)

    # Policy outputs 1D action per env.
    self._action_dim = 1
    self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
    self._processed_actions = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    # Binary threshold: >0 → open, <=0 → close.
    open_mask = (actions[:, 0] > 0.0).unsqueeze(-1).expand(-1, self._num_targets)
    self._processed_actions = torch.where(
      open_mask,
      torch.full_like(self._processed_actions, self.cfg.open_position),
      torch.full_like(self._processed_actions, self.cfg.close_position),
    )

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(
      self._processed_actions, joint_ids=self._target_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
