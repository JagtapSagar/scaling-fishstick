"""Franka Panda cube-lifting environment configurations."""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.envs.mdp.observations import generated_commands
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg

from panda_mjlab.mdp.actions import BinaryJointPositionActionCfg
from mjlab.envs.mdp import dr

from panda_mjlab.mdp.observations import (
  object_position_in_robot_root_frame,
  object_root_lin_vel_w,
  object_root_quat_w,
)
from panda_mjlab.mdp.rewards import (
  ee_height_penalty,
  object_ee_distance,
  object_goal_distance,
  object_is_lifted,
)
from panda_mjlab.robot import get_panda_robot_cfg


def get_cube_spec(
  cube_size: float = 0.02,
  mass: float = 0.05,
  rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
  )
  return spec


def panda_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Proprioceptive Panda cube-lifting environment."""
  cfg = make_lift_cube_env_cfg()

  # --- Scene: Panda robot + cube ---
  cfg.scene.entities = {
    "robot": get_panda_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  # --- Actions: arm (joint position) + gripper (binary open/close) ---
  # Remove the default single joint_pos action and replace with two terms.
  # Flat 0.5 scale matching Isaac Lab's JointPositionActionCfg for panda_joint.*.
  cfg.actions = {
    "arm": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
      scale=0.5,
      use_default_offset=True,
    ),
    "gripper": BinaryJointPositionActionCfg(
      entity_name="robot",
      actuator_names=("finger_joint1",),
      open_position=0.04,
      close_position=0.0,
    ),
  }

  # --- Observations: match Isaac Lab's absolute object pos + goal pos ---
  # Replace relative distance obs (ee_to_cube, cube_to_goal) with absolute position obs.
  actor_terms = cfg.observations["actor"].terms
  actor_terms.pop("ee_to_cube", None)
  actor_terms.pop("cube_to_goal", None)
  actor_terms["object_pos"] = ObservationTermCfg(
    func=object_position_in_robot_root_frame,
    params={"object_name": "cube", "robot_cfg": SceneEntityCfg("robot")},
  )
  actor_terms["target_object_position"] = ObservationTermCfg(
    func=generated_commands,
    params={"command_name": "lift_height"},
  )

  # Do the same for critic (mirrors actor in base config).
  critic_terms = cfg.observations["critic"].terms
  critic_terms.pop("ee_to_cube", None)
  critic_terms.pop("cube_to_goal", None)
  critic_terms["object_pos"] = ObservationTermCfg(
    func=object_position_in_robot_root_frame,
    params={"object_name": "cube", "robot_cfg": SceneEntityCfg("robot")},
  )
  critic_terms["target_object_position"] = ObservationTermCfg(
    func=generated_commands,
    params={"command_name": "lift_height"},
  )
  # Privileged critic obs matching Isaac Lab: object velocity + quaternion.
  critic_terms["object_vel"] = ObservationTermCfg(
    func=object_root_lin_vel_w,
    params={"object_name": "cube"},
  )
  critic_terms["object_quat"] = ObservationTermCfg(
    func=object_root_quat_w,
    params={"object_name": "cube"},
  )

  # --- Rewards: match Isaac Lab reward structure ---
  # Remove MJLab-specific rewards.
  cfg.rewards.pop("lift", None)
  cfg.rewards.pop("lift_precise", None)

  # Reaching object: 1 - tanh(dist / std), matching Isaac Lab.
  cfg.rewards["reaching_object"] = RewardTermCfg(
    func=object_ee_distance,
    weight=10.0,
    params={
      "std": 0.1,
      "object_name": "cube",
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
  )

  # Lifting object: binary reward when object above threshold.
  cfg.rewards["lifting_object"] = RewardTermCfg(
    func=object_is_lifted,
    weight=20.0,
    params={"minimal_height": 0.04, "object_name": "cube"},
  )

  # Object goal tracking: tanh reward gated by lift height.
  cfg.rewards["object_goal_tracking"] = RewardTermCfg(
    func=object_goal_distance,
    weight=10.0,
    params={
      "std": 0.3,
      "minimal_height": 0.04,
      "command_name": "lift_height",
      "object_name": "cube",
    },
  )

  # Fine-grained goal tracking: tighter std for precision.
  cfg.rewards["object_goal_tracking_fine_grained"] = RewardTermCfg(
    func=object_goal_distance,
    weight=5.0,
    params={
      "std": 0.05,
      "minimal_height": 0.04,
      "command_name": "lift_height",
      "object_name": "cube",
    },
  )

  # Replace MJLab-specific penalty rewards with Isaac Lab equivalents.
  cfg.rewards.pop("joint_pos_limits", None)
  cfg.rewards.pop("joint_vel_hinge", None)

  # Action rate: L2 penalty matching Isaac Lab weight.
  cfg.rewards["action_rate_l2"].weight = -1e-4

  # Joint velocity: L2 penalty matching Isaac Lab.
  from mjlab.envs.mdp.rewards import joint_vel_l2
  cfg.rewards["joint_vel"] = RewardTermCfg(
    func=joint_vel_l2,
    weight=-1e-4,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  cfg.rewards["ee_height_penalty"] = RewardTermCfg(
    func=ee_height_penalty,
    weight=1.0,
    params={
      "min_height": 0.13,
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
  )

  # --- Commands: widen object spawn range to match Isaac Lab ---
  assert cfg.commands is not None
  from mjlab.tasks.manipulation.mdp.commands import LiftingCommandCfg
  cfg.commands["lift_height"].object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(-0.45, 0.45),
    y=(-0.4, 0.4),
    z=(0.02, 0.05),
    yaw=(-3.14, 3.14),
  )
  # Resample time matching Isaac Lab (5.0s fixed).
  cfg.commands["lift_height"].resampling_time_range = (5.0, 5.0)

  # --- Events: remove MJLab-specific friction DR (not in Isaac Lab) ---
  cfg.events.pop("fingertip_friction_slide", None)
  cfg.events.pop("fingertip_friction_spin", None)
  cfg.events.pop("fingertip_friction_roll", None)

  # Cube size randomization.
  cfg.events["randomize_cube_size"] = EventTermCfg(
    func=dr.geom_size,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("cube", geom_names=("cube_geom",)),
      "operation": "scale",
      "distribution": "uniform",
      "ranges": (0.7, 1.0),
    },
  )

  # --- Curriculum: remove base curriculum referencing removed rewards ---
  cfg.curriculum = {}

  # --- Collision sensor: hand contacts ground ---
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "hand"

  # --- Terminations: add object dropping (from Isaac Lab base) ---
  from mjlab.envs.mdp.terminations import root_height_below_minimum
  from mjlab.managers.termination_manager import TerminationTermCfg
  cfg.terminations["object_dropping"] = TerminationTermCfg(
    func=root_height_below_minimum,
    params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube")},
  )

  # --- Simulation: match Isaac Lab timing ---
  cfg.episode_length_s = 5.0

  # --- Viewer ---
  cfg.viewer.body_name = "link0"

  # --- Play mode ---
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg


def panda_lift_cube_depth_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Panda cube-lifting with wrist-mounted depth camera."""
  cfg = panda_lift_cube_env_cfg(play=play)

  # --- Depth camera on wrist ---
  cam_cfg = CameraSensorCfg(
    name="wrist_camera",
    camera_name="robot/wrist_camera",
    height=128,
    width=128,
    data_types=("depth",),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  # --- Camera observation group ---
  cam_terms: dict[str, ObservationTermCfg] = {
    "depth": ObservationTermCfg(
      func=manipulation_mdp.camera_depth,
      params={"sensor_name": "wrist_camera", "cutoff_distance": 2.0},
    ),
  }
  cfg.observations["camera"] = ObservationGroupCfg(
    terms=cam_terms, enable_corruption=False, concatenate_terms=True
  )

  # --- Remove privileged object position from actor (learned from depth instead) ---
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("object_pos", None)

  # --- Camera pose DR matching Isaac Lab: ±5mm position, ±3° rotation ---
  cfg.events["randomize_camera_pos"] = EventTermCfg(
    func=dr.cam_pos,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", camera_names=("wrist_camera",)),
      "operation": "add",
      "distribution": "uniform",
      "ranges": (-0.005, 0.005),
    },
  )
  cfg.events["randomize_camera_quat"] = EventTermCfg(
    func=dr.cam_quat,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", camera_names=("wrist_camera",)),
      "roll_range": (-0.052, 0.052),
      "pitch_range": (-0.052, 0.052),
      "yaw_range": (-0.052, 0.052),
    },
  )

  return cfg
