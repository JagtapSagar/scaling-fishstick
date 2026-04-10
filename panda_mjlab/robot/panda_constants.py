"""Franka Emika Panda constants for MJLab."""

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PANDA_XML: Path = Path(__file__).parent / "xmls" / "panda.xml"
assert PANDA_XML.exists(), f"Panda XML not found at {PANDA_XML}"


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(PANDA_XML))


##
# Actuator config.
#
# The Menagerie Panda XML defines 8 ``general`` actuators with affine-bias PD
# control.  We wrap them with XmlActuatorCfg so gains/limits stay in sync with
# the XML.
##

ARM_ACTUATORS = XmlActuatorCfg(
  target_names_expr=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
)

GRIPPER_ACTUATOR = XmlActuatorCfg(
  target_names_expr=("finger_joint1",),
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": -1.57079,
    "joint5": 0.0,
    "joint6": 1.57079,
    "joint7": -0.7853,
    "finger_joint1": 0.04,
    "finger_joint2": 0.04,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Only enable collision on the hand and finger geoms (link7 onward) for
# lifting tasks.  Arm link collisions are disabled to avoid self-collision
# overhead.
GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=("hand_collision", "[lr]f_collision", "[lr]f_pad_.*"),
  contype={
    "[lr]f_pad_.*": 1,
    "[lr]f_collision": 1,
    "hand_collision": 1,
  },
  conaffinity={
    "[lr]f_pad_.*": 1,
    "[lr]f_collision": 1,
    "hand_collision": 1,
  },
  condim={
    "[lr]f_pad_.*": 6,
    ".*": 3,
  },
  friction={
    "[lr]f_pad_.*": (1.0, 5e-3, 5e-4),
    ".*": (0.6,),
  },
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ARM_ACTUATORS, GRIPPER_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)


def get_panda_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(GRIPPER_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


# Action scale: 0.25 * effort_limit / stiffness per actuator.
# Menagerie Panda actuator gains (from XML):
#   joints 1-4: gainprm=4500/3500, forcerange=87 (default)
#   joints 5-7: gainprm=2000, forcerange=12
#   finger:     gainprm=100, forcerange=20
_ACTUATOR_PARAMS = {
  "joint1": (87.0, 4500.0),
  "joint2": (87.0, 4500.0),
  "joint3": (87.0, 3500.0),
  "joint4": (87.0, 3500.0),
  "joint5": (12.0, 2000.0),
  "joint6": (12.0, 2000.0),
  "joint7": (12.0, 2000.0),
  "finger_joint1": (20.0, 100.0),
}

PANDA_ACTION_SCALE: dict[str, float] = {
  name: 0.25 * effort / stiffness
  for name, (effort, stiffness) in _ACTUATOR_PARAMS.items()
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_panda_robot_cfg())
  viewer.launch(robot.spec.compile())
