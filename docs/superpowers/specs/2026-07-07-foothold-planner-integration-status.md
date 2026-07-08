# Foothold Planner Integration Status

Date: 2026-07-08

## Purpose

This note records the current integration state of the foothold planner in the
parkour training stack. The planner is now connected across the IsaacLab sensor,
command, observation, reward, visualization, terrain-clearance, smoke-test, and
short-training paths.

The current version should be understood as an integrated foothold-planner
prototype: it is usable inside training and exposes terrain-clearance signals,
but final reward weights, long-horizon terrain evaluation, and planner failure
policy are still intentionally conservative.

## Integrated Layers

### Standalone foothold modules

The simulator-independent foothold modules live under `instinctlab_foothold`.

Implemented pieces:

- flat foothold target sampling;
- gait state machine;
- swing reference trajectory;
- terrain query abstractions;
- terrain provider utilities;
- edge-clearance apex adjustment;
- shared foothold data types and exports.

These modules are PyTorch-based and do not depend on Isaac Sim, Omniverse, or
the InstinctLab task runtime. They can be tested with ordinary unit tests and
used by visualization scripts without booting the full training stack.

### Planner sensor

The planner is mounted as an IsaacLab sensor:

- `instinctlab.sensors.foothold_planner.FootholdPlannerCfg`
- `instinctlab.sensors.foothold_planner.FootholdPlanner`
- `FootholdPlannerData`

The sensor reads:

- robot body poses;
- ankle-link poses;
- sole-center offsets;
- contact sensor forces;
- base velocity command;
- registered virtual terrain obstacles for edge-clearance checks.

It exposes:

- gait mode;
- swing side;
- phase;
- target foothold in world and stance-foot frames;
- feasible velocity;
- default swing reference position;
- adjusted swing reference position;
- default and adjusted swing apex height;
- swing clearance safety flag;
- swing clearance penetration depth;
- actual stance and swing foot positions;
- foot contact;
- touchdown acceptance;
- planner validity.

### Terrain and edge-clearance integration

The planner can receive registered virtual obstacles from the parkour terrain
stack. The current clearance path focuses on edge obstacles and adjusts the
swing apex height when the default centerline trajectory would penetrate the
registered edge obstacle representation.

Current behavior:

1. Generate the default swing trajectory from frozen swing start to target
   foothold.
2. Query registered edge obstacles along sampled swing-centerline points.
3. If the default trajectory penetrates an edge obstacle, increase the apex
   height in configured increments.
4. Stop once the trajectory is safe or the configured maximum apex height is
   reached.
5. Expose both the adjusted apex and the residual penetration through
   `FootholdPlannerData`.

The clearance parameters are configurable through `FootholdPlannerCfg`:

- `enable_edge_clearance`;
- `clearance_max_apex_height_m`;
- `clearance_apex_step_m`;
- `clearance_sample_spacing_m`.

This makes the feature easy to disable or retune without removing code.

### Parkour environment mount

The planner is mounted in the parkour scene configuration as
`scene.foothold_planner`.

It uses:

- `robot_name="robot"`;
- `contact_sensor_name="contact_forces"`;
- ankle/contact body names from the G1 ankle roll links;
- the robot prim path for initialization.

The sensor update period is synchronized with the simulation timestep in the
environment configuration.

Terrain virtual obstacles are registered with the planner during startup so the
planner can consume the same obstacle representation already used by the
parkour stack.

### Command synchronization

The planner and reward/observation functions synchronize the planner desired
velocity from:

```python
env.command_manager.get_command("base_velocity")
```

This keeps the foothold target generator aligned with the same command that the
locomotion policy is trying to follow.

### Observation integration

The policy and critic receive a compact foothold planner observation.

Current single-frame observation layout:

```text
target_foothold_f              3
feasible_velocity_f            3
phase                          1
swing_side_sign                1
swing_apex_height              1
swing_apex_delta               1
swing_clearance_safe           1
swing_clearance_penetration    1
```

Total single-frame size: `12`.

The parkour config uses:

```python
history_length=8
flatten_history_dim=True
```

Therefore the policy and critic see `96` foothold-planner observation values.

Smoke testing confirmed that both policy and critic observation groups contain
the `foothold_planner` term with shape `(num_envs, 96)`.

### Reward integration

The foothold reward module is connected through `instinctlab.envs.mdp`.

Current reward terms:

- `foothold_swing_tracking_exp`;
- `foothold_touchdown_tracking_exp`.

Current diagnostic terms:

- swing mode;
- reset mode;
- left swing;
- right swing;
- touchdown confirm;
- early contact;
- overdue;
- stance lost;
- touchdown accepted;
- plan invalid;
- clearance-safe indicator;
- clearance penetration depth.

The clearance reward terms are currently intended for logging and diagnosis.
They are wired into the reward manager with conservative weights so the
training loop can observe them without letting them dominate the existing
parkour objective.

### Visualization and smoke tooling

Dedicated foothold smoke and visualization scripts live under
`tests/parkour/foothold`.

Current tooling covers:

- full Isaac smoke test for planner sensor creation and data access;
- policy/critic observation shape checks;
- reward-manager term visibility;
- base-command synchronization;
- flat foothold visualization with two idealized foot spheres;
- raw target vs feasible target visualization;
- terrain-height target adjustment visualization;
- edge-obstacle clearance visualization showing default and adjusted swing
  trajectories.

The idealized visualization is not a physics validation. Its purpose is to make
planner geometry, target projection, terrain height adjustment, and swing
clearance behavior visually inspectable without requiring a full robot policy.

## Verified Behavior

Current verification covers:

- standalone flat-provider tests;
- gait state-machine tests;
- swing trajectory tests;
- terrain query tests;
- terrain provider tests;
- clearance adjustment tests;
- planner data tests;
- reward function tests;
- foothold observation tests;
- smoke test environment creation;
- planner sensor creation;
- planner data querying;
- policy/critic observation manager integration;
- reward manager integration;
- short training run with the foothold planner, clearance observation, and
  clearance reward logs enabled.

The current integration has reached the point where training can run end-to-end
with the foothold planner included in the network architecture.

## Current Limitations

### Flat target provider remains the main target generator

The target provider still begins from a flat-ground target generator. Terrain
height and edge-clearance adjustment are layered on top, but the target search
itself is not yet a full terrain candidate selection or traversability planner.

### Edge-clearance checks cover the swing centerline

The current edge-clearance logic checks sampled swing-centerline points. This
helps avoid asking the policy to track a centerline trajectory that intersects
known edge obstacles, but it does not fully replace foot-volume collision
handling.

Foot toe/heel/side collisions are still expected to be handled by the existing
parkour collision/volume-point machinery and reinforcement learning objective.

### Clearance failure is diagnostic by default

If the apex reaches the configured maximum and the trajectory is still unsafe,
the planner exposes the unsafe flag and penetration value. It does not yet force
a different target, terminate the episode, or override the policy.

This is deliberate: the current stage prioritizes observability and stable
training integration before adding hard planner decisions.

### Conservative reward design

The current foothold and clearance rewards are intentionally minimal. Early
training is not expected to produce strong foothold tracking or frequent
clearance penalties, because the policy may not yet stand, swing, or land
reliably.

Final reward weights and additional penalties should be tuned only after the
integrated policy can produce stable locomotion behavior.

### Long-horizon training evaluation is still pending

Short training can complete, but the current branch has not yet established
long-horizon performance improvements from the foothold planner. The next
training phase should compare baseline and planner-enabled runs using the same
seeds, terrain setup, and checkpoint policy.

## Recommended Next Steps

1. Keep clearance reward weights conservative while observing their logs during
   short and medium training runs.
2. Run a longer planner-enabled training job and compare stability, reward
   terms, contact behavior, and clearance statistics against the baseline.
3. Decide whether `swing_clearance_safe=False` should remain diagnostic, become
   a soft penalty, or mark a plan as invalid.
4. Add more terrain-aware target selection only after the current edge-clearance
   observation/reward path is stable.
5. Tune foothold reward weights only after swing and touchdown events appear
   regularly in training.
6. Keep the idealized visualization script as a geometric regression tool for
   future planner changes.

## Non-Goals for the Current Stage

- Do not replace the existing locomotion controller.
- Do not make the state machine terminate episodes by default.
- Do not tune foothold reward weights based only on very early unstable
  training logs.
- Do not require touchdown tracking reward to be large before the policy can
  reliably stand and step.
- Do not treat centerline clearance as a complete foot-volume collision
  solution.
