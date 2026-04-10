"""Sweep each action dimension across its full range one at a time, with viewer.

For each action dimension, holds all others at zero and linearly sweeps
from -1 to +1 (the normalized action range) over a configurable number
of steps, then returns to zero before moving to the next dimension.

Usage:
  python scripts/sweep_actions.py Mjlab-Lift-Cube-Panda
  python scripts/sweep_actions.py Mjlab-Lift-Cube-Panda --steps-per-dim 100 --pause-steps 20
"""

from __future__ import annotations

import argparse

import torch

import panda_mjlab.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import NativeMujocoViewer


class SweepPolicy:
  """Policy that sweeps one action dimension at a time from -1 to +1."""

  def __init__(
    self,
    action_dim: int,
    device: str,
    steps_per_dim: int = 60,
    pause_steps: int = 10,
    labels: list[str] | None = None,
  ):
    self.action_dim = action_dim
    self.device = device
    self.steps_per_dim = steps_per_dim
    self.pause_steps = pause_steps
    self.labels = labels or [f"dim_{i}" for i in range(action_dim)]

    self._current_dim = 0
    self._step_in_dim = 0
    self._pausing = False
    self._done = False

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    num_envs = obs.shape[0]
    action = torch.zeros(num_envs, self.action_dim, device=self.device)

    if self._done:
      return action

    if self._pausing:
      self._step_in_dim += 1
      if self._step_in_dim >= self.pause_steps:
        self._current_dim += 1
        self._step_in_dim = 0
        self._pausing = False
        if self._current_dim >= self.action_dim:
          print("\nSweep complete. Sending zeros.")
          self._done = True
      return action

    # Sweep current dim from -1 to +1.
    t = self._step_in_dim / max(self.steps_per_dim - 1, 1)
    value = -1.0 + 2.0 * t

    action[:, self._current_dim] = value

    if self._step_in_dim == 0:
      label = self.labels[self._current_dim]
      print(f"--- Sweeping dim {self._current_dim}: {label} (-1 -> +1) ---")

    self._step_in_dim += 1
    if self._step_in_dim >= self.steps_per_dim:
      self._step_in_dim = 0
      self._pausing = True
      print(f"    pause ({self.pause_steps} steps at zero)")

    return action


def main():
  parser = argparse.ArgumentParser(description="Sweep actions one dimension at a time with viewer.")
  parser.add_argument("task", type=str, help="Registered task ID")
  parser.add_argument("--steps-per-dim", type=int, default=60)
  parser.add_argument("--pause-steps", type=int, default=10)
  parser.add_argument("--device", type=str, default=None)
  args = parser.parse_args()

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Build action labels from action manager terms.
  action_dim = env.unwrapped.action_space.shape[-1]
  labels: list[str] = []
  for term_name in env.unwrapped.action_manager.active_terms:
    term = env.unwrapped.action_manager.get_term(term_name)
    for i in range(term.action_dim):
      if hasattr(term, "_target_names") and i < len(term._target_names):
        labels.append(f"{term_name}/{term._target_names[i]}")
      else:
        labels.append(f"{term_name}[{i}]")

  print(f"Action space: {action_dim} dims")
  for i, label in enumerate(labels):
    print(f"  dim {i}: {label}")
  print()

  policy = SweepPolicy(
    action_dim=action_dim,
    device=device,
    steps_per_dim=args.steps_per_dim,
    pause_steps=args.pause_steps,
    labels=labels,
  )

  NativeMujocoViewer(env, policy).run()
  env.close()


if __name__ == "__main__":
  main()
