# Learned Explicit Foothold Planner Design

## Goal

Build a learning-based explicit foothold planner that outputs the next swing-foot foothold as an explicit point, while preserving the existing analytic swing trajectory generator and low-level action policy. The planner should learn where to step from terrain/state observations instead of relying on increasingly complex hand-written candidate search.

## Non-goals

- Do not make the low-level action policy implicitly solve foothold planning by itself.
- Do not make the network output a full swing trajectory curve.
- Do not use danger-cylinder positions, radii, or simulator-only obstacle internals as deployment-time network inputs.
- Do not remove the current explicit planner immediately; it remains as nominal-prior, debug, and fallback infrastructure.
- Do not introduce per-step expensive candidate enumeration as the final planner interface.

## Final Architecture

The system is a hierarchical policy with explicit foothold output:

```text
current robot state + terrain observation + nominal foothold prior
        ↓
learned high-level foothold head
        ↓
final foothold x_f, y_f in support-foot planner frame
        ↓
world-frame terrain height query gives z_w
        ↓
final foothold x_f, y_f, z_f
        ↓
analytic swing trajectory generator
        ↓
low-level action head tracks the generated reference
```

The current mathematical planner is not deleted. It is reduced to a nominal-prior generator:

```text
math planner → nominal foothold prior
learned planner head → final explicit foothold
```

The nominal prior carries walking intent: speed direction, nominal step length, step width, and left/right alternation. It is not treated as the final safe foothold.

## Network Outputs

The learned high-level foothold head directly outputs the final foothold horizontal coordinates in the support-foot planner frame:

```text
final_foothold_x_f
final_foothold_y_f
```

It does not output `dx, dy` as the primary interface. The nominal foothold is provided as an input prior, and a reward term may penalize excessive deviation from that prior, but the semantic output is the final foothold.

The output is bounded before use:

```text
x_f: limited to the robot's configured reachable step-length interval
y_f: limited according to swing side and configured step-width interval
```

The network never predicts terrain height directly.

## Coordinate-System Contract

This is a hard implementation contract.

1. The learned foothold head outputs `x_f, y_f` in the support-foot planner frame.
2. Terrain height query always uses world-frame horizontal coordinates.
3. Before querying terrain height, the system converts the local foothold to world coordinates:

```text
support-foot world position + planner yaw + x_f, y_f → x_w, y_w
```

4. The terrain module returns world height:

```text
terrain_height_query(x_w, y_w) → z_w
```

5. The final local height is computed from the support-foot world height:

```text
z_f = z_w - support_foot_z_w
```

6. Reward, debug, and visualization must compare quantities in the same coordinate frame.
7. Tests must cover non-zero support-foot world position and non-zero planner yaw, so local/world mix-ups are caught.

## Timing Contract

The high-level foothold output is event-triggered, not continuously applied during swing.

```text
HOLD / new-swing preparation:
    read high-level foothold output
    convert to final 3D foothold
    generate swing reference
    cache and lock the final foothold

SWING:
    ignore new high-level foothold outputs
    track the locked foothold and trajectory

TOUCHDOWN / next HOLD:
    evaluate touchdown and prepare the next planning event
```

This prevents the target from moving while the swing leg is already trying to track a trajectory.

## Inputs

The learned foothold head can use inputs available in simulation and eventually reproducible on hardware:

- nominal foothold prior in the support-foot planner frame;
- local terrain representation, initially privileged height map or terrain-derived features;
- current support-foot and swing-foot positions;
- swing side;
- gait mode and phase / planning-event flag;
- velocity command;
- base orientation and velocity;
- proprioceptive state needed by the existing policy.

Training-time privileged labels/rewards may use simulator-only information, but simulator-only information must not be fed as deployment-time observation.

Allowed training-only privileged signals:

- danger-cylinder penetration depth;
- number or ratio of penetrating sole perimeter points;
- exact terrain mesh height;
- exact collision/contact diagnostics.

Forbidden deployment-time inputs:

- danger-cylinder center/radius;
- candidate safety masks;
- simulator mesh internal identifiers;
- future terrain not observable from the robot's sensor setup.

## Height and Safety Checks

After the learned head outputs `x_f, y_f`, the system queries height and forms a 3D foothold. The 3D foothold must pass hard checks before being used:

- terrain height query is valid;
- step height difference from current support foot is at most `0.25 m`;
- foothold is inside the configured reachable support-foot region;
- foot support area is physically plausible for the current terrain representation;
- during simulation, danger-cylinder penetration diagnostics are computed for training reward and debug.

During training, a dangerous foothold is penalized. During execution, the planner must not knowingly output a foothold that fails the hard safety gate. If the gate fails, the environment should continue HOLD/recovery instead of producing a dangerous swing reference.

## Swing Trajectory Contract

The network does not output a complete swing trajectory. The existing analytic trajectory generator remains responsible for connecting current swing-foot start and final learned foothold.

The trajectory generator may adapt parameters from terrain and safety diagnostics:

- apex height;
- clearance above obstacle/step face;
- touchdown shaping;
- final target position.

But the trajectory remains an analytic curve, not a directly learned point sequence.

## Joint Training Strategy

The learned foothold head and the low-level action head are trained in the same rollout process, but their responsibilities stay separated.

The policy action is expanded conceptually into two parts:

```text
high-level foothold action: final x_f, y_f
low-level motor action: joint targets / current existing action vector
```

The high-level foothold action is only consumed during planning events. The low-level motor action is consumed every control step.

Rewards are separated by role:

### Foothold-planning rewards

- danger-cylinder penetration penalty during simulation;
- penetrating sole-point ratio penalty;
- support-area / edge-safety reward;
- reachability reward or hard gate;
- step-height hard gate at `0.25 m`;
- nominal-prior deviation penalty;
- velocity-intent consistency.

### Execution rewards

- swing trajectory tracking;
- touchdown xy/z error;
- touchdown within tolerance;
- velocity tracking;
- stability and posture;
- contact quality and slip penalties.

The high-level head should not be blamed for low-level tracking errors alone, and the low-level action head should not be expected to repair unsafe high-level footholds.

## Curriculum

Training starts with small high-level foothold output bounds and easier terrain, then gradually expands:

```text
small reachable interval around nominal walking footholds
→ larger step-length/step-width range
→ more terrain variation
→ stronger edge/danger-cylinder penalties
```

The curriculum should be based on rollout readiness metrics such as episode survival, velocity tracking, and touchdown quality, not only global iteration count.

## Existing Planner Migration

The current explicit foothold planner remains useful and should not be deleted in the first implementation.

It will serve as:

- nominal foothold prior generator;
- visualization/debug reference;
- baseline for ablation;
- fallback when learned planner is disabled;
- source of coordinate-frame utilities and terrain height query code.

The current hand-written safe-target candidate search should not be further expanded as the final intelligence source. It may remain as debug/fallback while the learned planner is being introduced.

## Testing Requirements

The implementation must include tests for:

1. Local-to-world terrain query conversion with non-zero support-foot origin.
2. Local-to-world terrain query conversion with non-zero planner yaw.
3. `z_f = z_w - support_foot_z_w` conversion.
4. High-level foothold output ignored during SWING after target lock.
5. High-level foothold output consumed during HOLD/new-swing planning event.
6. Action bounds prevent impossible local footholds.
7. Hard safety gate rejects height differences above `0.25 m`.
8. Danger-cylinder penetration reward uses training-only diagnostics and is not part of deployment observation.
9. Existing planner path still works when learned planner is disabled.
10. Play/debug output reports both local and world target coordinates.

## Performance Constraints

- No per-step dense candidate enumeration in the final learned planner path.
- High-level foothold head forward pass should be lightweight relative to the existing action policy.
- Terrain height query happens only at planning events, not every simulation step for every possible candidate.
- Any rollout-time feature extraction must be benchmarked with 4096 environments before a long training run.

## Open Implementation Choice

The preferred first implementation is a separate high-level output head sharing the main policy observation pipeline where practical. It can be implemented either as:

1. an expanded action vector from the current policy; or
2. a small high-level head attached beside the current action head.

The implementation plan should choose the option that fits the current PPO runner with the smallest stable change.

The design decision is fixed at the behavioral level: the learned planner directly outputs final explicit foothold `x_f, y_f` in the support-foot planner frame.
