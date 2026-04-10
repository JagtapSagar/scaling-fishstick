"""Custom domain randomization events for Panda cube lifting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def randomize_cube_size(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("cube", geom_names=("cube_geom",)),
  scale_range: tuple[float, float] = (0.7, 1.0),
) -> None:
  """Randomize cube geom size on reset.

  Samples a uniform scale factor and multiplies the default cube half-sizes.
  """
  entity = env.scene[asset_cfg.name]
  geom_ids = entity.find_geoms(asset_cfg.geom_names)[0]

  num_reset = len(env_ids)
  scales = (
    torch.rand(num_reset, 1, device=env.device) * (scale_range[1] - scale_range[0])
    + scale_range[0]
  )

  # Default geom sizes are stored per-env; scale uniformly in all 3 dims.
  default_sizes = entity.data.default_geom_size[env_ids][:, geom_ids, :]
  entity.data.geom_size[env_ids, geom_ids[0], :] = default_sizes[:, 0, :] * scales
