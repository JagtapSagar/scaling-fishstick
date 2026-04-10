"""Interactive joint control for the Panda robot via Viser (browser GUI).

Each joint gets a slider. Move sliders to pose the robot in real time.
Opens at http://localhost:8080.

Usage:
  python scripts/joint_control.py
"""

from __future__ import annotations

import time

import mujoco
import numpy as np
import viser
import viser.transforms as vtf

from panda_mjlab.robot.panda_constants import PANDA_XML


def main():
  spec = mujoco.MjSpec.from_file(str(PANDA_XML))

  # Add ground plane and lighting.
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(2, 2, 0.01),
    rgba=(0.9, 0.9, 0.9, 1),
  )

  # Add a cube to see grasping.
  cube_body = spec.worldbody.add_body(name="cube")
  cube_body.pos = (0.5, 0.0, 0.02)
  cube_body.add_freejoint(name="cube_joint")
  cube_body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.02, 0.02, 0.02),
    mass=0.05,
    rgba=(0.8, 0.2, 0.2, 1),
  )

  model = spec.compile()
  data = mujoco.MjData(model)

  # Load home keyframe.
  key_id = model.keyframe("home").id
  mujoco.mj_resetDataKeyframe(model, data, key_id)
  mujoco.mj_forward(model, data)

  # Collect controlled joints (skip freejoint).
  joint_info: list[tuple[str, int, float, float]] = []  # (name, qposadr, lo, hi)
  for i in range(model.njnt):
    jnt = model.jnt(i)
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joint_info.append((
      jnt.name,
      jnt.qposadr,
      float(model.jnt_range[i, 0]),
      float(model.jnt_range[i, 1]),
    ))

  # Map actuator index -> joint name for ctrl.
  actuator_to_joint: dict[int, str] = {}
  for i in range(model.nu):
    act = model.actuator(i)
    trntype = model.actuator_trntype[i]
    if trntype == mujoco.mjtTrn.mjTRN_JOINT:
      jnt_id = model.actuator_trnid[i, 0]
      actuator_to_joint[i] = model.jnt(jnt_id).name

  # Start Viser server.
  server = viser.ViserServer(port=8080)
  print("Joint Control GUI at http://localhost:8080")

  # Create MuJoCo scene renderer.
  renderer = mujoco.Renderer(model, height=480, width=640)

  # Build GUI sliders.
  sliders: dict[str, viser.GuiInputHandle] = {}
  with server.gui.add_folder("Joint Controls"):
    for name, adr, lo, hi in joint_info:
      initial = float(data.qpos[adr].item())
      sliders[name] = server.gui.add_slider(
        label=name,
        min=lo,
        max=hi,
        step=0.001,
        initial_value=initial,
      )

    server.gui.add_markdown("---")
    reset_btn = server.gui.add_button("Reset to Home")

  @reset_btn.on_click
  def _(_):
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    for name, adr, _, _ in joint_info:
      sliders[name].value = float(data.qpos[adr].item())

  print("\nControlled joints:")
  for name, adr, lo, hi in joint_info:
    print(f"  {name:20s}  [{lo:+.3f}, {hi:+.3f}]  home={data.qpos[adr].item():+.4f}")

  # Main loop.
  try:
    while True:
      # Read slider values → set actuator ctrl targets.
      for act_idx, jnt_name in actuator_to_joint.items():
        if jnt_name in sliders:
          data.ctrl[act_idx] = sliders[jnt_name].value

      # Step physics.
      mujoco.mj_step(model, data)

      # Render and send to Viser.
      renderer.update_scene(data)
      img = renderer.render()
      server.scene.set_background_image(img)

      time.sleep(1.0 / 60.0)
  except KeyboardInterrupt:
    print("\nDone.")


if __name__ == "__main__":
  main()
