# G1 Flat Foothold Tracking and Program Interface Design

**Date:** 2026-07-01

**Status:** Revised Sensor architecture ready for user review

## 1. Purpose

This document defines the first independently testable sub-project of the G1 foothold program: a reusable `FootholdPlannerSensor` that produces continuous left/right footholds and swing references, plus the observation and reward adapters needed to train a flat-ground tracker. It also freezes the interfaces that later oracle-terrain, depth-map, multi-step planning, and AMP curriculum work must use.

The mandatory engineering deliverable is a tested Sensor, trainable environment, reward set, visualization, integration documentation, and at least one resumable baseline checkpoint. The performance criteria in this document are research targets, not a condition for handing off the working Sensor integration.

## 2. Scope

### 2.1 Included in this sub-project

- A new task, `Instinct-Parkour-Foothold-G1-v0`, without changing the existing parkour task.
- Explicit left/right sole frames derived from the existing ankle links and shoe collision geometry.
- A new independent `FootholdPlannerSensor`, configuration class, and structured data object.
- A fixed actor command and observation contract.
- A flat-ground target provider that produces coherent continuous footsteps.
- A time-driven gait state machine corrected by debounced contact and kinematic evidence.
- A soft two-segment quintic swing-foot reference.
- Phase-gated foothold, timing, clearance, support, impact, and slip rewards.
- Per-environment L0-L2 curriculum with a global difficulty frontier.
- Four-expert MoE instrumentation and progressive AMP scheduling hooks.
- Curriculum and schedule state that survives checkpoint resume.
- Play-mode visualization for footholds, sole frames, swing references, and planner state.
- Real-time Weights & Biases monitoring with an offline fallback.
- Unit, visualization, vectorized smoke, short-training, and resume tests.
- A handoff API that lets an external navigation stack write desired velocity while consumers read only `sensor.data`.

### 2.2 Deferred to later sub-projects

- Oracle terrain support-region extraction and multi-step terrain planning.
- Depth-image fusion into a local terrain map.
- Full L3-L7 terrain curriculum.
- Deployment integration with a physical depth camera.

These are deferred implementations, not deferred interfaces. Their command fields, validity flags, terrain summaries, provider boundaries, and checkpoint requirements are fixed here.

## 3. System Architecture

The complete program uses the following data flow:

```text
Navigation target
    -> set_navigation_command(v_des)
    -> FootholdPlannerSensor
       -> TerrainProvider (Flat / Oracle / Depth)
       -> LocalTerrainMap
       -> reachability and full-sole safety filtering
       -> gait state machine and frozen stance frame
       -> current/next footholds and soft swing reference
       -> structured FootholdPlannerData
    -> Observation adapter -> 44-D foothold observation -> four-expert MoE actor
    -> 29 joint-position targets

 FootholdPlannerData
    -> phase-gated rewards
    -> critic observations
    -> metrics and play visualization
```

The planner core remains a pure, simulator-independent component. `FootholdPlannerSensor` is the Isaac Lab `SensorBase` adapter that reads robot state, caches one update per simulation timestamp, owns per-environment reset/state buffers, and exposes results through `data`. It does not bind its structured output to one policy-network shape.

The first sub-project uses a `FlatProvider`. The later oracle provider crops a local elevation map from simulation truth and combines reachability with sole-point support tests. The future depth provider replaces only the map source.

The Sensor automatically reads base and sole state from simulation. Desired navigation velocity is supplied through `set_navigation_command(vx_vy_wz)`. A fixed deterministic command source is used by the virtual demonstration; the training environment forwards its navigation command through the same API.

The actor receives only deployment-available quantities. The critic, rewards, and evaluation metrics may use privileged simulation truth.

## 4. Coordinate Frames and Foot Geometry

### 4.1 Sole frames

The planner and tracker use virtual left and right `sole_frame` frames. Each frame is a fixed transform from the corresponding `ankle_roll_link`:

- The origin is the geometric center of the bottom contact surface.
- The frame represents the sole contact plane, not the ankle-link origin.
- The transform is derived from the actual shoe collision geometry and verified by visualization and contact-height tests.

The ankle link remains the articulation body used to read simulation state. All target, reward, support-polygon, and swing calculations convert ankle state to sole state first.

### 4.2 Frozen stance frame

At the start of a swing, the Sensor snapshots a gravity-aligned stance frame:

- Origin: current stance sole center.
- Z axis: gravity-up.
- X/Y orientation: stance-foot yaw.

The snapshot remains fixed for the swing. A target is transformed to a fixed world pose once and is not dragged by later base motion or stance-foot jitter.

If stance slip exceeds the configured translational or yaw tolerance, the current plan is invalidated and replanned.

### 4.3 Full-foot support

A point is not safe merely because the ankle or sole center lies on a surface. Safety requires the transformed sole support polygon to fit inside the support region.

The support polygon is derived from the union of the shoe collision primitives. Candidate poses are evaluated with a sole-point template transformed into the local elevation map. A candidate is safe only when valid coverage, plane-fit residual, slope, edge margin, and unknown-area thresholds pass.

The existing `VolumePoints` point generator may be reused to construct the template, but `FootholdPlannerSensor` remains independent from `VolumePoints`. The current volume-penetration term remains an auxiliary collision penalty and is not treated as proof of safe support.

## 5. Sensor and Actor Interfaces

### 5.1 Structured Sensor data

`FootholdPlannerSensor.data` is the authoritative structured result. It contains:

- Measured left/right sole pose and velocity.
- Frozen stance-frame pose.
- Current and preview foothold pose, normal, timing, and validity.
- Swing reference position, velocity, acceleration, orientation, and yaw rate.
- Gait state, swing side, phase, phase masks, and one-shot touchdown event.
- Requested and feasible navigation velocity.
- Planner validity, recovery state, and failure reason.
- Support ratio, edge margin/risk, terrain confidence, and unknown fraction.
- Position, velocity, orientation, and touchdown errors.
- Optional local-map and candidate diagnostics for the critic and visualization.

Observation, reward, critic, metrics, and visualization functions read this object. They do not recompute planning state. A lazy Sensor read may update buffers at most once for a simulation timestamp, so multiple consumers cannot advance the state machine more than once.

### 5.2 Actor foothold observation

An observation adapter packs deployment-available Sensor fields into a fixed 44-dimensional actor input:

| Field | Size | Meaning |
|---|---:|---|
| `swing_foot_one_hot` | 2 | Left/right swing identity |
| `phase_sin_cos` | 2 | Continuous gait phase encoding |
| `normalized_time_to_touchdown` | 1 | Remaining planned swing time |
| `swing_reference_pos_stance` | 3 | Current reference sole position |
| `swing_reference_vel_stance` | 3 | Current reference sole velocity |
| `current_target_pos_stance` | 3 | Current target sole position |
| `current_target_yaw_sin_cos` | 2 | Current target heading |
| `current_target_normal_stance` | 3 | Current target surface normal |
| `next_target_pos_stance` | 3 | Preview target sole position |
| `next_target_yaw_sin_cos` | 2 | Preview target heading |
| `next_target_normal_stance` | 3 | Preview target surface normal |
| `feasible_velocity` | 3 | Velocity induced by the accepted plan |
| `swing_position_error_stance` | 3 | Actual minus reference sole position |
| `swing_velocity_error_stance` | 3 | Actual minus reference sole velocity |
| `swing_apex_height` | 1 | Required apex above the frozen stance origin |
| `planner_valid` | 1 | Current plan validity |
| `next_target_valid` | 1 | Preview target validity |
| `terrain_confidence` | 1 | Aggregate terrain confidence |
| `support_margin` | 1 | Conservative full-sole support margin |
| `edge_risk` | 1 | Normalized edge risk |
| `unknown_fraction` | 1 | Fraction of the candidate sole over unknown terrain |
| `recovery_state` | 1 | Planner/gait recovery indicator |
| **Total** | **44** | |

Position, velocity, height, and time fields remain in SI units with observation scale `1.0`; unit vectors, sine/cosine pairs, validity flags, and confidences are dimensionless. Invalid optional fields are zero-filled and accompanied by their validity or confidence values.

The actor does not receive raw binary contact flags or raw depth pixels.

### 5.3 Other actor observations

The new policy retains:

- Base angular-velocity history.
- Projected-gravity history.
- Joint-position history.
- Joint-velocity history.
- Previous-action history.

The 44-dimensional foothold observation is current-state data and is not placed in the generic eight-frame observation history. Targets are already constant within a swing, and phase and touchdown time provide temporal context.

Raw depth is consumed by the future `DepthTerrainProvider`. The actor receives only the resulting targets, instantaneous reference, tracking errors, and aggregate safety/confidence values.

### 5.4 Critic observations

The asymmetric critic additionally receives:

- True base linear velocity.
- Debounced and raw contact state.
- Full privileged fields from `FootholdPlannerSensor.data`.
- Oracle terrain agreement metrics.

Privileged critic fields never enter actor observations or exported inference inputs.

### 5.5 Action and network structure

- Action remains 29 joint-position targets.
- The actor and critic use four MoE experts from the first baseline checkpoint.
- Expert gate utilization, mean gate probabilities, maximum occupancy, and gate entropy are logged.
- A load-balancing regularizer is enabled only if short-run evidence shows persistent expert collapse.

## 6. Flat Target Provider

The flat provider generates a coherent nominal foothold from navigation intent, nominal stance width, swing-foot identity, yaw intent, and step duration. It then samples a curriculum-controlled residual around that nominal target.

Three constraint layers are distinct:

1. A fixed, conservative outer operating envelope representing G1 geometric and collision limits.
2. A curriculum sampling envelope that expands from L0 to L2 but never exceeds the outer envelope.
3. State-dependent filtering based on current kinematics and a lightweight capture-point/DCM viability check.

The provider alternates feet, prevents leg crossing, limits step-to-step target jumps, and supports standing, stopping, and restarting. A navigation velocity that cannot be realized is reduced to `feasible_velocity`; safety constraints are never relaxed to preserve `v_des`.

The fixed-root virtual demonstration uses a deterministic 18-step sequence: six forward steps, six forward-left-turning steps, and six lateral steps. In this mode, a virtual stance pose advances to each accepted target and a virtual touchdown event advances the gait state. Formal training disables virtual mode and uses measured sole poses and debounced contact.

## 7. Gait State Machine

Each environment owns an independent state machine:

```text
HOLD
    -> LEFT_SWING or RIGHT_SWING
    -> TOUCHDOWN_CONFIRM
    -> swap stance and swing roles
    -> next swing
```

Exceptional states are:

- `EARLY_CONTACT`
- `OVERDUE`
- `STANCE_LOST`
- `PLAN_INVALID`
- `RECOVERY`

### 7.1 Reset and startup

After reset, an environment enters `HOLD` for a default settling interval of 0.4 seconds. It does not infer the first stance foot from contact buffers at the reset instant. The first swing foot is selected only after the robot satisfies double-support and posture conditions.

### 7.2 Phase

The phase clock advances using the environment control period. Contact does not directly overwrite phase.

A touchdown transition requires:

- Entry into the configured late-swing touchdown window.
- Debounced contact force with hysteresis, or a kinematic fallback based on relative sole height and vertical velocity.
- Sole position inside the touchdown acceptance region.

Early contact is a collision event and does not immediately swap feet. A missed deadline enters a bounded grace interval. It cannot extend indefinitely.

### 7.3 Failure fallback

Planning failure follows this order:

1. Search alternate candidates without relaxing hard safety constraints.
2. Change step duration, shorten the step, and adjust step width.
3. Reduce `feasible_velocity`.
4. Permit a bounded heading deviation.
5. Take a verified short recovery step.
6. Enter stable double support and wait for new observations.
7. Ask navigation to reroute.
8. On timeout, terminate/reset in training or remain in safe stand on hardware.

Unknown terrain, incomplete sole support, and hard dynamic infeasibility are never relaxed.

## 8. Swing Reference

The reference position is a two-segment quintic trajectory:

```text
lift-off -> apex -> touchdown
```

It enforces continuous position, velocity, and acceleration at segment boundaries. Flat-ground training starts with a 0.10 m sole apex. Future terrain providers derive the apex from the maximum valid corridor height plus a default 0.05 m clearance margin.

Target orientation interpolates smoothly toward target yaw and surface normal.

The swing reference is:

- A soft training signal.
- A clearance and swept-volume checking reference.
- A debug visualization.

It is not a hard whole-body trajectory. The actor may deviate to preserve balance and recover from disturbances.

## 9. Reward Design

Reward terms are independently phase-gated.

Every foothold reward obtains the same authoritative object:

```text
data = env.scene.sensors["foothold_planner"].data
```

Reward code does not resample a target, advance phase, or reconstruct a swing reference.

### 9.1 Swing terms

- Soft sole-position and orientation tracking to the swing reference.
- Clearance relative to the terrain corridor.
- Swept toe, heel, and foot-volume collision penalty.
- Early-contact penalty.

### 9.2 Touchdown terms

- Sole XYZ error at the first valid touchdown.
- Yaw and surface-normal error.
- Touchdown-time error.
- Landing vertical velocity and impulse penalty.

Touchdown rewards are latched and paid once per step. Hovering over a target or repeatedly bouncing cannot farm reward.

### 9.3 Support terms

- Effective supported-sole ratio.
- Minimum edge margin.
- Stance-foot tangential slip.
- Correct support sequence.

Toe-only or heel-only contact is not a successful touchdown.

### 9.4 Navigation and regularization

- Track `feasible_velocity`, not an infeasible requested velocity.
- Reward progress toward the navigation target.
- Retain joint-limit, torque, energy, joint-velocity, joint-acceleration, action-rate, body-orientation, and undesired-contact regularizers.

The existing alive reward is reduced or removed if it permits standing still to dominate task return. Reward groups are scaled so no auxiliary regularizer overwhelms touchdown and support objectives.

Reward exists only during training. At deployment, `FootholdPlannerSensor -> observation adapter -> actor` remains and all reward terms are removed.

## 10. Curriculum and AMP

### 10.1 Curriculum

- L0: flat ground, fixed short steps, fixed timing, no yaw.
- L1: continuous left/right targets.
- L2: expanded step length, width, duration, direction, and yaw.

Each environment has an independent level. A global frontier limits the maximum unlocked level, and minimum occupancy quotas retain easy and intermediate samples.

Promotion and demotion use rolling touchdown error, timing error, early-contact rate, support ratio, consecutive-step success, and fall rate. An environment changes by at most one level per episode.

### 10.2 AMP

The Wasabi discriminator trains from the beginning, while its reward coefficient follows:

```text
0 -> 0.02 -> 0.05 -> 0.1
```

Transitions are performance-gated. A mature checkpoint may be cloned into 256-environment validation runs to compare `0.1`, `0.15`, and `0.25`; the main lineage then resumes with the highest value that does not degrade safety or foothold accuracy.

Target footholds and raw contact flags are not added to discriminator observations. Phase-aware AMP is deferred unless reference contacts can be derived reliably and symmetrically.

### 10.3 Resume state

A production checkpoint must restore:

- Actor, critic, discriminator, and optimizer state.
- Observation normalizers.
- Current learning iteration.
- Global curriculum frontier.
- Per-level occupancy or per-environment curriculum state.
- AMP schedule state.
- Curriculum rolling statistics.
- Stateful Sensor buffers required to resume phase, virtual/real stance ownership, and accepted targets consistently.

The current runner does not save all environment state, so the foothold task requires an explicit environment-training-state save/load integration.

## 11. Performance Scheduling

- Physics simulation: 200 Hz.
- Actor and `FootholdPlannerSensor` state/reference updates: 50 Hz.
- Future depth-map updates: selected by the depth-provider spec from the 10-20 Hz operating range after profiling.
- Future planner: event-driven on touchdown, material map change, navigation change, or plan invalidation.

The current 24-step PPO rollout spans 0.48 seconds. Validation compares 32 and 48 steps and selects the shortest rollout that covers the configured maximum flat-ground step duration with margin while respecting GPU memory.

## 12. Validation

### 12.1 Unit tests

- Ankle-to-sole transform and visual ground contact.
- Coordinate transform round trips.
- Quintic endpoint, velocity, acceleration, and continuity properties.
- All state-machine transitions and debounce behavior.
- Left/right symmetry and leg-crossing rejection.
- Curriculum envelope containment in the fixed outer envelope.
- Reward phase masks and touchdown latching.
- Sensor reset, lazy-update idempotence, navigation-command injection, and structured-data shape/device consistency.
- Structured Sensor data to 44-D observation packing.
- Checkpoint training-state round trip.

### 12.2 Single-environment visualization

The first visualization gate uses a play-only fixed-root configuration. Physics time and Sensor updates continue, but the robot remains in its default standing pose. It validates geometry and planning without requiring a trained policy.

Visualize:

- Ankle and sole frames.
- Sole support polygon.
- Frozen stance frame.
- Current and preview targets.
- Swing trajectory, apex, and corridor.
- State-machine state and touchdown gate.
- Capture point/DCM diagnostic.
- Deterministic forward, turning, and lateral virtual-step sequences.
- Measured contact state separately from the virtual touchdown used by demonstration mode.

The Play task enables these markers by default and the training task disables them by default. Visualization can be toggled at runtime without changing observations or Sensor state.

Marker conventions are stable across runs:

- Left-foot targets and sole outlines: blue.
- Right-foot targets and sole outlines: red.
- Current executable target: opaque.
- Preview target: translucent.
- Valid swing reference: green.
- Invalid or overdue reference: orange.
- Safe support region: green.
- Unknown or rejected region: gray.
- Edge exclusion region: yellow.

The Play overlay also reports swing-foot identity, gait state, phase, time to touchdown, planner validity, terrain confidence, and current curriculum level.

### 12.3 Vectorized validation

- 64 environments force all L0-L2 levels and exceptional states.
- 256 environments validate reward signs, rollout length, MoE utilization, AMP scheduling, throughput, and checkpoint resume.
- No 4096-environment baseline training starts until these pass. Later oracle, depth, and L3-L7 work has its own validation gates.
- At least one resumable baseline checkpoint is produced. Failure to reach the research exit criteria does not invalidate an otherwise correct Sensor handoff.

### 12.4 Flat-tracker exit criteria

- Touchdown XY error: median at most 2.5 cm and P95 at most 5 cm.
- Touchdown Z error: P95 at most 2 cm.
- Touchdown yaw error: P95 at most 8 degrees.
- Touchdown timing error: P95 at most 80 ms.
- Early-contact rate below 1%.
- At least 90% success over 20-step sequences.
- Episode fall rate below 2%.
- Resume restores curriculum and AMP progress without reverting to L0.

## 13. Training Observability

Training supports real-time Weights & Biases monitoring. A lost network connection must not terminate training: metrics continue to local logs and are buffered for later synchronization.

Every run records:

- Git commit and dirty-state summary.
- Environment and agent configuration.
- Parent checkpoint and checkpoint iteration.
- Random seed, environment count, device, and effective rollout length.

The live dashboard includes:

- PPO losses, KL, entropy, learning rate, action standard deviation, and throughput.
- Wasabi discriminator loss, actor/reference scores, gradient penalty, and current AMP reward coefficient.
- Total reward and each foothold reward group.
- Touchdown XYZ, yaw, and timing median/P95.
- Early-contact, overdue, planner-failure, fall, and recovery rates.
- Sole support ratio, minimum edge margin, landing impact, and stance slip.
- Per-level curriculum occupancy and global frontier.
- MoE mean gate probability, expert occupancy, maximum occupancy, and gate entropy.
- Simulation FPS, policy FPS, map/planner update time, and GPU memory.

Metrics are aggregated across environments before logging. Raw per-environment streams are not uploaded. Play or evaluation video upload is optional and rate-limited.

## 14. Later Provider Contracts

Future oracle and depth sub-projects must preserve the structured Sensor contract and the versioned 44-D observation adapter above.

`LocalTerrainMap` contains:

- Metric height.
- Surface normal.
- Roughness.
- Confidence.
- Observed mask.
- Observation age.

The depth provider preserves invalid returns before clipping and never conflates a missing ray with a valid maximum-range measurement. Unknown or low-confidence regions are not support regions.

The terrain planner first intersects the fixed reachability envelope, curriculum envelope, and dynamic-feasibility region. It then evaluates candidate sole poses against the local elevation map using the sole-point template. Validity requires sufficient observed coverage, bounded plane-fit residual and slope, adequate edge margin, and bounded unknown fraction.

The planner returns the first executable step, one preview step, timing, `feasible_velocity`, swing reference, aggregate safety values, and confidence through `FootholdPlannerData`. Raw map and candidate diagnostics remain optional structured fields and are not embedded in the fixed actor vector.

## 15. Git and Delivery

- Existing parkour task registrations and behavior remain unchanged.
- The new task lives on the dedicated `feat/foothold-01-flat-tracking` branch/worktree.
- User changes in the original worktree are not overwritten or bundled.
- Design, implementation plan, tests, implementation, and tuning changes use separate focused commits.
- Training artifacts are not committed. Reproducible configs, checkpoint lineage, evaluation summaries, and Git commit identifiers are recorded.
- Handoff includes `FootholdPlannerSensor`, its config and data classes, the pure planner core, observation adapter, reward terms, provider interfaces, play visualization, tests, and integration documentation.
