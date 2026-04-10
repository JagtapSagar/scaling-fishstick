# Migration Plan: Isaac Lab Panda Lift → MJLab

## Context

The Isaac Lab repo (`Robto_2_ARM`) has a **Franka Panda cube-lifting environment** with wrist-mounted depth camera, domain randomization, and two-phase training (ground-truth object pos → depth-only). The goal is to port this to MJLab's manager-based framework, following the same pattern as the existing YAM lift-cube task.

MJLab already has a complete lift-cube task infrastructure (`make_lift_cube_env_cfg()`) — we just need to plug in a Panda robot and adapt the Isaac Lab-specific features.

**Project location:** New standalone package at `/home/sagar/proj/COMPSENSATE/panda_mjlab/` — lives alongside `mjlab/` and `Robto_2_ARM/`, imports from `mjlab` as a dependency.

---

## Step 1: Project Scaffold + Franka Panda Robot Asset

**Create:** `/home/sagar/proj/COMPSENSATE/panda_mjlab/`

```
panda_mjlab/
├── pyproject.toml              # Package metadata, dependency on mjlab
├── panda_mjlab/
│   ├── __init__.py
│   ├── robot/                  # Panda robot asset
│   │   ├── __init__.py
│   │   ├── panda_constants.py  # Robot config (mirrors yam_constants.py pattern)
│   │   └── xmls/
│   │       ├── panda.xml       # From MuJoCo Menagerie
│   │       └── meshes/         # Mesh files from Menagerie
│   ├── tasks/                  # Task configs
│   │   ├── __init__.py
│   │   ├── env_cfgs.py
│   │   └── rl_cfg.py
│   └── mdp/                   # Custom MDP terms
│       ├── __init__.py
│       ├── rewards.py
│       ├── actions.py          # BinaryJointPositionActionCfg
│       └── events.py
```

### Robot Asset (`panda_mjlab/robot/`)

- **`xmls/panda.xml`** — Franka Panda MJCF model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda). Includes:
  - 7 revolute arm joints (`joint1`–`joint7`)
  - 2 linear finger joints (`finger_joint1`, `finger_joint2`) coupled via equality constraint
  - A `grasp_site` on the end-effector (add if not present)
  - A wrist camera site on `panda_hand` (add to XML: pos `0.05 0 -0.065`)

- **`panda_constants.py`** — Robot config module (follow `mjlab/asset_zoo/robots/i2rt_yam/yam_constants.py` pattern):
  - `PANDA_XML` path + `get_spec()` loader
  - Actuator configs: 7 arm joints with PD gains (Panda default stiffness/damping from Menagerie), 1 finger actuator (left finger, right coupled via equality)
  - `HOME_KEYFRAME` — Panda home position (from Menagerie keyframe)
  - `COLLISION` config — gripper-only collision for lifting (contype/conaffinity)
  - `ARTICULATION` — combines arm + gripper actuators
  - `get_panda_robot_cfg() -> EntityCfg`
  - `PANDA_ACTION_SCALE` dict — `0.25 * effort_limit / stiffness` per joint

---

## Step 2: Create Panda Lift-Cube Task Config

**Create:** `panda_mjlab/tasks/`

### `env_cfgs.py`
Follows the YAM `env_cfgs.py` pattern (`mjlab/src/mjlab/tasks/manipulation/config/yam/env_cfgs.py`):

**`panda_lift_cube_env_cfg(play=False)`:**
1. Start from `make_lift_cube_env_cfg()` (base lift task)
2. Set `cfg.scene.entities` = `{"robot": get_panda_robot_cfg(), "cube": EntityCfg(spec_fn=get_cube_spec)}`
3. Override action scale: `cfg.actions["joint_pos"].scale = PANDA_ACTION_SCALE`
4. Set EE site: `site_names = ("grasp_site",)` for observations + rewards
5. Set fingertip geom names for friction DR (Panda finger geom names from XML)
6. Set collision sensor pattern (e.g., `"panda_hand"`)
7. Set viewer body_name
8. Play mode: disable corruption, disable curriculum, infinite episode

**`panda_lift_cube_depth_env_cfg(play=False)`:**
1. Start from `panda_lift_cube_env_cfg()`
2. Add `CameraSensorCfg` for wrist depth camera (128x128 or 32x32, attached to `robot/wrist_camera`)
3. Add depth observation group using `manipulation_mdp.camera_depth`
4. Pop `ee_to_cube` and `cube_to_goal` from actor (privileged info — learned from depth)
5. Add `goal_position` to actor observations

### `rl_cfg.py`
PPO runner config for Panda:
- Actor/Critic: `[256, 128, 64]` hidden dims, ELU
- Learning rate: `3e-4`, gamma: `0.99`, lambda: `0.95`
- `num_steps_per_env: 24`, `max_iterations: 3000`
- Observation normalization: off
- For depth variant: add CNN feature extractor config (SpatialSoftmaxCNN or custom)

### `__init__.py`
Register tasks:
- `"Mjlab-Lift-Cube-Panda"` — proprioceptive
- `"Mjlab-Lift-Cube-Panda-Depth"` — depth vision

---

## Step 3: Port Custom Reward Functions

**Create:** `panda_mjlab/mdp/rewards.py`

### Rewards to port from Isaac Lab:

1. **`finger_object_distance`** — Proximity reward for fingers near cube + gripper closure bonus
   - Map `robot.find_bodies("panda_leftfinger")` → MJLab's `env.scene["robot"].data` body access
   - Access finger joint positions for closure computation

2. **`ee_height_penalty`** — Penalize EE dropping below min height (0.13m)
   - Map `robot.data.body_pos_w[:, ee_idx, 2]` → MJLab equivalent site/body position

**Note:** The base `make_lift_cube_env_cfg()` already provides `staged_position_reward`, `bring_object_reward`, `action_rate_l2`, `joint_pos_limits`, and `joint_vel_hinge` — these cover most of the Isaac Lab base rewards (`reaching_object`, `lifting_object`, `object_goal_tracking`). The custom rewards above are additions.

---

## Step 4: Port Domain Randomization Events

The Isaac Lab env has these custom events to migrate:

| Isaac Lab Event | MJLab Equivalent | Notes |
|---|---|---|
| `randomize_camera_pose` | `dr.site_pos` / custom event | Randomize camera site position/rotation on reset. MJLab uses MjSpec-based DR — check if existing `dr.*` functions support site randomization, otherwise write a custom event |
| `randomize_object_scale` | `dr.geom_size` or custom | Randomize cube geom size on reset |
| `randomize_table_scale` | Skip initially | Not critical for core task |
| `reset_object_position` | Already in base config via `LiftingCommandCfg.ObjectPoseRangeCfg` | Object spawn range already handled by command generator |
| Fingertip friction DR | Already in base `make_lift_cube_env_cfg()` | Just need correct Panda geom names |

---

## Step 5: Handle Isaac Lab-Specific Features

### Two-Phase Training (PHASE 1/2)
The Isaac Lab env uses a `PHASE` global to switch between:
- **Phase 1**: Object position in obs, depth encoder outputs zeros
- **Phase 2**: Gradually anneal out object position, use depth

**MJLab approach**: Use the existing vision env pattern — the `panda_lift_cube_depth_env_cfg` naturally removes privileged object position from actor obs and adds depth camera. Phase 1 = proprioceptive task (`Mjlab-Lift-Cube-Panda`), Phase 2 = depth task (`Mjlab-Lift-Cube-Panda-Depth`). Transfer weights between phases.

### Depth Encoder CNN
- Isaac Lab uses a custom `DepthEncoderModifier` CNN (module-level singleton)
- MJLab handles this via the RL runner's CNN feature extractor config (`SpatialSoftmaxCNN` or custom). The observation manager outputs raw depth; the RL framework handles encoding.

### Binary Gripper Action
- Isaac Lab uses `BinaryJointPositionActionCfg` (open=0.04, close=0.0)
- **Decision**: Implement a `BinaryJointPositionActionCfg` in `panda_mjlab/mdp/actions.py`
- Takes 1D action (>0 = open, <=0 = close), maps to finger joint targets
- Open: `finger_joint = 0.04`, Close: `finger_joint = 0.0`
- Only actuate `finger_joint1`; `finger_joint2` coupled via equality constraint in MJCF

---

## Files to Create

All files live in `/home/sagar/proj/COMPSENSATE/panda_mjlab/` — **no modifications to the `mjlab` repo**.

### Package setup:
1. `pyproject.toml` — package metadata with `mjlab` dependency
2. `panda_mjlab/__init__.py`

### Robot asset:
3. `panda_mjlab/robot/__init__.py`
4. `panda_mjlab/robot/panda_constants.py` — actuators, keyframe, collision, `get_panda_robot_cfg()`
5. `panda_mjlab/robot/xmls/panda.xml` + `meshes/` — from MuJoCo Menagerie

### Task configs:
6. `panda_mjlab/tasks/__init__.py` — task registration
7. `panda_mjlab/tasks/env_cfgs.py` — `panda_lift_cube_env_cfg()`, `panda_lift_cube_depth_env_cfg()`
8. `panda_mjlab/tasks/rl_cfg.py` — PPO runner configs

### Custom MDP terms:
9. `panda_mjlab/mdp/__init__.py`
10. `panda_mjlab/mdp/rewards.py` — `finger_object_distance`, `ee_height_penalty`
11. `panda_mjlab/mdp/actions.py` — `BinaryJointPositionActionCfg`
12. `panda_mjlab/mdp/events.py` — camera randomization (if needed beyond mjlab's built-in DR)

---

## Verification

1. **Asset validation**: Run `python panda_constants.py` (viewer launch) to verify MJCF loads correctly
2. **Env creation**: Instantiate `panda_lift_cube_env_cfg()` and call `env.reset()` + `env.step()` with random actions
3. **Training smoke test**: `python scripts/train.py --task Mjlab-Lift-Cube-Panda --max_iterations 100`
4. **Depth variant**: Same with `Mjlab-Lift-Cube-Panda-Depth`
5. **Reward sanity**: Log individual reward terms, verify reaching/lifting/precision rewards activate in correct sequence
