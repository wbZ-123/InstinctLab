# Foothold planner metric semantics

This note defines the foothold planner diagnostic fields and TensorBoard tags.
It exists because some planner values are cached state, while others are sparse
planning events. Mixing those denominators can make curves look contradictory.

## Core planner validity

### `planner_valid`

Runtime field. True means the currently exposed planner output is executable by
the gait state machine.

Safe-target search failure must propagate to this field. If no safe target is
found, `planner_valid` should become false so the state machine can enter
`PLAN_INVALID` or recovery behavior.

### `foothold_planner_plan_invalid_fraction`

TensorBoard tag:

```text
Step_Monitor/foothold_planner_plan_invalid_fraction
Episode_Monitor/foothold_planner_plan_invalid_fraction
```

Meaning:

```text
steps where planner_valid is false / total steps
```

Expected value is usually zero or near zero. If it rises, the planner is often
unable to produce an executable target.

## Safe-target search fields

Safe-target search only runs when a new swing target is planned. These fields
are therefore event diagnostics, not continuous per-step state.

### `safe_target_search_performed`

Runtime field. True only on update steps where a new safe-target search was
executed.

This should be used as the denominator mask for all safe-target search success,
fallback, and score metrics.

### `safe_target_final_valid`

Runtime field. Result of the current safe-target search event.

True means either:

- the nominal target was already safe; or
- the nominal target was unsafe, but a safe fallback candidate was found.

False means no safe executable target was found for that search event.

### `safe_target_used_fallback`

Runtime field. True for search events where the nominal target was unsafe and
the planner selected a nearby safe candidate.

False means one of:

- the nominal target was used;
- no search occurred on this step; or
- no valid fallback was found.

### `safe_target_score`

Runtime field. XY distance between the nominal target and selected fallback
target, in meters.

It is zero when:

- the nominal target was used;
- no search occurred on this step; or
- no valid fallback was selected.

### `safe_target_nominal_inside_ellipse`

Runtime field. True for a search event when the nominal target is inside the
reachable foothold ellipse.

If this is often false, the flat target generator or velocity command is asking
for steps beyond the configured reachable region.

### `safe_target_nominal_obstacle_safe`

Runtime field. True for a search event when the nominal target's sole perimeter
points do not penetrate edge-obstacle cylinders.

If this is often false, the nominal target is geometrically reachable but lands
too close to unsafe terrain edges.

### `safe_target_nominal_valid`

Runtime field. True only when the nominal target passes both checks:

```text
inside reachable ellipse AND obstacle safe
```

If this is low while `safe_target_final_valid` is high, fallback search is doing
useful correction work.

### Candidate count fields

Runtime fields:

```text
safe_target_candidate_count
safe_target_candidate_inside_ellipse_count
safe_target_candidate_obstacle_safe_count
safe_target_candidate_valid_count
```

These are per-search-event counts. They describe how many fallback candidates
exist before filtering, how many remain inside the reachable ellipse, how many
are obstacle-safe, and how many pass both checks.

They are the main localization tools when `safe_target_final_valid_fraction`
drops:

- low inside-ellipse count -> candidate grid is mostly outside reach;
- low obstacle-safe count -> terrain-edge constraints are filtering candidates;
- low valid count -> both constraints together leave too little room.

## Safe-target TensorBoard tags

### `foothold_planner_safe_target_search_rate`

TensorBoard tag:

```text
Step_Monitor/foothold_planner_safe_target_search_rate
Episode_Monitor/foothold_planner_safe_target_search_rate
```

Meaning:

```text
safe-target search events / total steps
```

This is not a quality metric by itself. It tells how often new swing target
searches are being planned.

### `foothold_planner_safe_target_final_valid_fraction`

TensorBoard tag:

```text
Step_Monitor/foothold_planner_safe_target_final_valid_fraction
Episode_Monitor/foothold_planner_safe_target_final_valid_fraction
```

Meaning:

```text
valid safe-target searches / safe-target search events
```

Expected value should be close to one. If it drops, candidate search is failing
to find safe executable footholds.

### `foothold_planner_safe_target_fallback_fraction`

TensorBoard tag:

```text
Step_Monitor/foothold_planner_safe_target_fallback_fraction
Episode_Monitor/foothold_planner_safe_target_fallback_fraction
```

Meaning:

```text
search events using fallback target / safe-target search events
```

A nonzero value is expected on edge-obstacle terrain. A rising value means the
nominal foothold planner increasingly asks for targets near unsafe areas.

### `foothold_planner_safe_target_score_mean`

TensorBoard tag:

```text
Step_Monitor/foothold_planner_safe_target_score_mean
Episode_Monitor/foothold_planner_safe_target_score_mean
```

Meaning:

```text
mean fallback XY correction distance over safe-target search events
```

Small values mean fallback is only making local corrections. A rising value
means the nominal target is drifting farther into unsafe regions or the search
needs larger detours.

### `foothold_planner_safe_target_score_max`

TensorBoard tag:

```text
Step_Monitor/foothold_planner_safe_target_score_max
Episode_Monitor/foothold_planner_safe_target_score_max
```

Meaning:

```text
maximum fallback XY correction distance seen in the monitor window
```

This often reaches the configured largest search radius. Interpret it together
with `safe_target_score_mean`; max alone can be triggered by rare edge cases.

### Safe-target localization tags

TensorBoard tags:

```text
Step_Monitor/foothold_planner_safe_target_nominal_inside_ellipse_fraction
Step_Monitor/foothold_planner_safe_target_nominal_obstacle_safe_fraction
Step_Monitor/foothold_planner_safe_target_nominal_valid_fraction
Step_Monitor/foothold_planner_safe_target_candidate_count_mean
Step_Monitor/foothold_planner_safe_target_candidate_inside_ellipse_count_mean
Step_Monitor/foothold_planner_safe_target_candidate_obstacle_safe_count_mean
Step_Monitor/foothold_planner_safe_target_candidate_valid_count_mean
```

Episode tags use the same suffix under `Episode_Monitor/`.

The first three are fractions over safe-target search events. The candidate
metrics are mean candidate counts over safe-target search events. Use them as a
funnel:

```text
nominal target
  -> inside ellipse?
  -> obstacle safe?
fallback candidates
  -> candidate_count
  -> candidate_inside_ellipse_count
  -> candidate_obstacle_safe_count
  -> candidate_valid_count
```

This tells whether failures come from reachability, obstacle filtering, or a
candidate grid that is too sparse or pointed in the wrong directions.

## Swing clearance tags

### `foothold_planner_clearance_safe_fraction`

Meaning:

```text
clearance-safe swing samples / swing samples
```

This is about the generated swing center trajectory clearing edge obstacles. It
does not mean the target foothold itself is safe.

### `foothold_planner_penetration_mean` and `foothold_planner_penetration_max`

Mean and max penetration depth of sampled swing center trajectory points into
edge obstacles. Expected values should stay near zero.

### `foothold_planner_apex_delta_mean` and `foothold_planner_apex_delta_max`

Amount added to the default swing apex height to clear obstacles. Rising values
mean the swing trajectory is being lifted more often or more aggressively.

## Gait-state tags

### `foothold_planner_swing_fraction`

Meaning:

```text
steps in left/right swing / total steps
```

If this collapses toward zero, the planner or state machine is not allowing
regular stepping.

### `foothold_planner_overdue_fraction`

Steps where swing became overdue. Low is expected.

### `foothold_planner_early_contact_fraction`

Steps where swing foot contact was detected too early. Low is expected.

### `foothold_planner_stance_lost_fraction`

Steps where support foot contact was lost. Low is expected; early training can
show some noise here.

### `foothold_planner_touchdown_confirm_step_rate`

Rate of entering touchdown-confirm mode. This can be sparse because touchdown
confirm is a short state. Interpret it together with `swing_fraction`,
`touchdown_accepted_step_rate`, reward, and episode length.

## Legacy tag warning

Older runs may contain:

```text
foothold_planner_safe_target_valid_fraction
```

That legacy tag used step count as denominator for a cached safe-target state.
It should not be interpreted as true search success rate. Prefer new runs with:

```text
foothold_planner_safe_target_final_valid_fraction
```
