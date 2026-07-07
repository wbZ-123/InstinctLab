# Foothold Planner Integration Status

Date: 2026-07-07

## Purpose

This note records the current integration state of the foothold planner in the
parkour training stack. The planner is now connected to the IsaacLab
environment, reward system, command system, and observation pipeline, but it is
still an early functional version rather than a final terrain-aware foothold
planner.

## Integrated Layers

### Standalone foothold modules

The simulator-independent foothold modules live under
`instinctlab_foothold`.

Implemented pieces:

- flat foothold target sampling;
- gait state machine;
- swing reference trajectory;
- shared foothold data types and exports.

These modules are PyTorch-based and do not depend on Isaac Sim, Omniverse, or
the InstinctLab task runtime.

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
- base velocity command.

It exposes:

- gait mode;
- swing side;
- phase;
- target foothold in world and stance-foot frames;
- feasible velocity;
- swing reference position;
- actual stance and swing foot positions;
- foot contact;
- touchdown acceptance;
- planner validity.

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
- plan invalid.

The reward terms are intentionally low-weight and conservative. Their current
purpose is to prove the integration path and provide weak guidance, not to
dominate the original parkour objective.

### Command synchronization

The planner and reward/observation functions synchronize the planner desired
velocity from:

```python
env.command_manager.get_command("base_velocity")
```

This keeps the foothold target generator aligned with the same command that the
locomotion policy is trying to follow.

### Observation integration

The policy and critic now receive a compact foothold planner observation.

Single-frame observation layout:

```text
target_foothold_f      3
feasible_velocity_f    3
phase                  1
swing_side_sign        1
```

Total single-frame size: `8`.

The parkour config uses:

```python
history_length=8
flatten_history_dim=True
```

Therefore the policy and critic see `64` foothold-planner observation values.

Smoke testing confirmed that both policy and critic observation groups contain
the `foothold_planner` term with shape `(num_envs, 64)`.

## Verified Behavior

Current verification covers:

- standalone flat-provider tests;
- gait state-machine tests;
- swing trajectory tests;
- reward function tests;
- foothold observation tests;
- smoke test environment creation;
- planner sensor creation;
- planner data querying;
- policy/critic observation manager integration;
- short training startup with the new observation dimension.

The current integration has reached the point where training can start with the
foothold planner included in the network architecture.

## Current Limitations

### Flat target provider only

The current target provider is still a flat-ground provider. It does not yet
choose footholds from terrain height maps, local traversability maps, or
candidate foothold patches.

### Simplified swing clearance

The swing reference uses a simple apex-clearance trajectory. It does not yet
adapt the foot height to steps, obstacles, or terrain between the start and
target foothold.

### Diagnostic state machine

The state machine is integrated as a diagnostic/planning component. It does not
override policy actions, reset the simulator, or directly control the robot.

Failure states such as `EARLY_CONTACT`, `OVERDUE`, `STANCE_LOST`, and
`PLAN_INVALID` are exposed for logging and optional reward shaping, not hard
terminations by default.

### Conservative reward design

The current foothold rewards are intentionally minimal. Early training is not
expected to produce strong foothold tracking or frequent touchdown rewards,
because the policy may not yet stand, swing, or land reliably.

The final reward weights and additional penalties should be tuned only after
the integrated policy can produce stable locomotion behavior.

### No terrain-aware visual validation yet

The current smoke test prints planner and observation tensors. It does not yet
render target footholds or swing references as visual markers in the simulator.

## Recommended Next Steps

1. Keep the current foothold rewards low-weight while the policy learns basic
   stability with the new observations.
2. Add visual markers for target footholds and swing references so planner
   outputs can be inspected directly in Isaac Sim.
3. After visual validation, add terrain-aware target selection and
   terrain-aware swing clearance.
4. Tune foothold reward weights only after swing and touchdown events appear
   regularly in training.
5. Consider adding optional failure penalties or curriculum scheduling later,
   but only behind explicit configuration choices.

## Non-Goals for the Current Stage

- Do not replace the existing locomotion controller.
- Do not make the state machine terminate episodes by default.
- Do not tune foothold reward weights based only on very early unstable
  training logs.
- Do not require touchdown tracking reward to be large before the policy can
  reliably stand and step.
