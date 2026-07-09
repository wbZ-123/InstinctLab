# Foothold Planner Monitor Design

## Goal

Expose raw foothold-planner health metrics in TensorBoard without changing
reward values, policy inputs, or training behavior. The monitor must distinguish
planner/state-machine failures from poor policy performance and remain cheap
enough to keep enabled during normal training.

## Scope

Add one `FootholdPlannerMonitorTerm` registered in the parkour environment's
existing `MonitorCfg`. It reads the existing `foothold_planner` sensor data and
accumulates per-environment episode statistics.

The first version records:

- swing-mode fraction;
- touchdown-accepted and touchdown-confirmed event rates;
- early-contact, overdue, and stance-lost fractions;
- clearance-safe fraction;
- clearance penetration mean and maximum;
- apex-height adjustment mean and maximum;
- invalid-plan fraction.

This work does not change planner algorithms, reward weights, observations, or
the RL runner.

## Architecture

The implementation follows the repository's existing monitor framework:

1. `FootholdPlannerMonitorTerm` subclasses `MonitorTerm`.
2. Its constructor resolves the configured planner sensor and allocates compact
   per-environment accumulators on the simulation device.
3. `update()` reads `FootholdPlannerData` under `torch.no_grad()` and updates
   counters, sums, and maxima.
4. `get_log(is_episode=False)` exposes inexpensive current/global summaries for
   step logging.
5. On episode reset, the manager calls `reset_idx()`. The term first preserves
   statistics for the completing environments, then clears only their
   accumulators. `get_log(is_episode=True)` returns summaries of those completed
   episodes.
6. `MonitorCfg` registers one term using `MonitorTermCfg`, so TensorBoard tags
   appear under `Step_Monitor/foothold_planner_*` and
   `Episode_Monitor/foothold_planner_*`.

The term must use named gait-state constants or the planner's existing helper
semantics rather than duplicating unexplained numeric states.

## Metric Semantics

Fractions use the number of observed simulation steps as denominator. Event
rates use the number of completed swing cycles when that denominator is
available; otherwise they use observed steps and the TensorBoard name explicitly
ends in `_step_rate`.

Clearance statistics are reported only over valid swing-planning samples. If an
episode has no such samples, its mean penetration and apex delta are zero rather
than NaN. All values passed to TensorBoard are sanitized with
`torch.nan_to_num`.

An invalid plan means the planner's existing validity/safety output declares the
current generated plan unusable. The monitor does not invent a second validity
rule. If no explicit plan-valid field currently exists, the first version uses
`not swing_clearance_safe` and names the metric
`clearance_unsafe_fraction`, postponing a general `plan_invalid_fraction`.

## Performance Constraints

The monitor stores only scalar accumulators per environment and never stores
per-step histories. With roughly twelve `float32` accumulators and 4096
environments, persistent storage is about 192 KiB, excluding minor framework
overhead.

Updates remain on the GPU, do not participate in autograd, and do not perform
per-step `.cpu()` or `.item()` synchronization. Reduction and logging happen
through the existing monitor-manager path. The entire term can be disabled by
removing or setting its config entry to `None` for profiling.

## Error Handling

- Missing planner sensor: fail during monitor construction with a clear sensor
  name in the error.
- Non-finite planner values: sanitize logged accumulations and count a
  `nonfinite_fraction` diagnostic rather than propagating NaN into training.
- Empty denominators: clamp denominators and return zero.
- Partial environment reset: reset only the supplied environment IDs.

## Testing and Acceptance

Unit tests use a minimal fake environment and planner data to verify:

- each gait mode updates the expected counter;
- touchdown events are counted once according to their actual event semantics;
- penetration/apex mean and maximum are correct;
- partial reset preserves other environments;
- empty and non-finite input never produces NaN or Inf;
- episode logging returns pre-reset statistics.

Integration acceptance:

1. Existing foothold unit tests remain green.
2. The foothold smoke test reaches clean environment/application shutdown.
3. Monitor Manager reports one active foothold monitor.
4. A short training run completes without changing reward terms.
5. TensorBoard contains non-weighted `Episode_Monitor/foothold_planner_*` tags.
