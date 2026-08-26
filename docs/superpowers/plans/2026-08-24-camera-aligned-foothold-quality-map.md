# Camera-Aligned Foothold Quality Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the learned foothold planner to move an unsafe nominal foothold to the nearest observable safe foothold without relying on penetration-direction inputs or changing the 29-dimensional motor-policy behavior.

**Architecture:** Register the existing head-camera depth history into a frozen support-foot frame only when a new HOLD planning transaction is created. A small planner-only network predicts a dense safety map over a 2 cm grid centered on the nominal foothold; a nearest-safe spatial selector produces a bounded absolute XY prior, and the existing event-gated PPO head is limited to a one-cell refinement around that prior. Simulation-only terrain and danger cylinders supervise the safety map but never enter the actor observation.

**Tech Stack:** PyTorch, IsaacLab manager observations and sensors, InstinctRL event-gated PPO, pytest, batched CUDA tensor operations.

## Global Constraints

- The head camera remains the only exteroceptive actor sensor; no height scanner, terrain mesh, edge cylinder, or privileged terrain label enters the actor observation.
- Foot positions come from proprioception/state estimation; the camera is only responsible for observing possible support surfaces.
- Final foothold XY remains an absolute point in the frozen support-foot frame; final Z is still queried in world coordinates.
- The existing 29-dimensional motor MoE input and output behavior must remain unchanged.
- Local terrain registration and quality-map evaluation run only for environments whose foothold transaction generation changed.
- Unknown or stale terrain cells are not treated as safe.
- Planning unavailability remains HOLD and must not enter physical RECOVERY.
- Existing trajectory preflight, HOLD transaction locking, SWING frame locking, and physical recovery behavior remain unchanged.
- Do not start a 30,000-iteration run until the camera-coverage and 4096-environment performance gates pass.

---

### Task 1: Pose-synchronized camera history

**Files:**
- Modify: `source/instinctlab/instinctlab/sensors/noisy_camera/noisy_camera_cfg.py`
- Modify: `source/instinctlab/instinctlab/sensors/noisy_camera/noisy_camera.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
- Test: `tests/parkour/foothold/test_camera_pose_history.py`

**Interfaces:**
- Produces: camera outputs named `<depth_type>_pos_w_history` with shape `(N, T, 3)` and `<depth_type>_quat_w_ros_history` with shape `(N, T, 4)`.
- Guarantees: index `t` in the depth, position, and quaternion histories represents the same sensor update.

- [ ] **Step 1: Write failing tests for pose/depth synchronization**

```python
def test_pose_history_uses_the_same_ring_index_as_depth_history():
    history = FakeNoisyCameraHistory(length=3)
    for index in range(4):
        history.append(
            depth=torch.full((1, 2, 2, 1), float(index)),
            pos_w=torch.tensor([[float(index), 0.0, 0.0]]),
            quat_w_ros=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
    assert history.depth[:, :, 0, 0, 0].tolist() == [[1.0, 2.0, 3.0]]
    assert history.pos_w[:, :, 0].tolist() == [[1.0, 2.0, 3.0]]


def test_pose_history_reset_clears_only_requested_environments():
    history = FakeNoisyCameraHistory(length=2, num_envs=2)
    history.reset(torch.tensor([1]))
    assert torch.count_nonzero(history.pos_w[1]) == 0
    assert torch.count_nonzero(history.pos_w[0]) > 0
```

- [ ] **Step 2: Run the tests and confirm they fail because pose history is absent**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q tests/parkour/foothold/test_camera_pose_history.py
```

Expected: FAIL on missing pose-history outputs.

- [ ] **Step 3: Add an opt-in camera configuration flag**

Add to `NoisyCameraCfgMixin`:

```python
record_pose_history: bool = False
```

When enabled, `build_history_buffers()` allocates position and ROS-quaternion buffers with exactly the same configured history length as every depth history. `update_history_buffers()` appends `self.data.pos_w` and `self.data.quat_w_ros` in the same call that appends depth. `reset_history_buffers()` resets all three buffers using the same environment IDs.

- [ ] **Step 4: Enable synchronized pose history only for the parkour camera**

Set:

```python
record_pose_history=True
```

on `SceneCfg.camera`. Do not change camera resolution, crop, FOV, depth noise, or update period in this task.

- [ ] **Step 5: Run focused and existing camera tests**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q \
  tests/parkour/foothold/test_camera_pose_history.py \
  tests/parkour/foothold/test_play_learned_config.py
```

Expected: PASS.

---

### Task 2: Pure support-frame local terrain registration

**Files:**
- Create: `source/instinctlab/instinctlab_foothold/depth_local_map.py`
- Modify: `source/instinctlab/instinctlab_foothold/__init__.py`
- Test: `tests/parkour/foothold/test_depth_local_map.py`

**Interfaces:**
- Produces: `LocalTerrainMap(height_f, observed, confidence, reachable)` where every tensor has shape `(N, 11, 11)`.
- Produces: `build_nominal_centered_local_map(...) -> LocalTerrainMap`.
- Grid: offsets `[-0.10, 0.10]` m in 0.02 m increments in both XY axes, reusing the existing 0.10 m analytic search limit and the existing 2 cm precision requirement.

- [ ] **Step 1: Write failing geometry tests**

```python
def test_flat_plane_registers_to_zero_relative_height():
    result = build_synthetic_local_map(
        plane_height_w=0.4,
        stance_height_w=0.4,
    )
    assert result.observed.any()
    torch.testing.assert_close(
        result.height_f[result.observed],
        torch.zeros_like(result.height_f[result.observed]),
        atol=1.0e-5,
        rtol=0.0,
    )


def test_historical_frame_is_registered_after_camera_translation():
    result = register_two_views_of_one_step()
    assert_step_edge_is_at_same_support_frame_x(result)


def test_unobserved_cells_remain_unknown_instead_of_flat():
    result = build_map_with_partial_frustum()
    assert torch.all(result.confidence[~result.observed] == 0.0)


def test_grid_cells_outside_reachability_ellipse_are_masked():
    result = build_map_near_ellipse_boundary(radius_x=0.42, radius_y=0.25)
    assert not result.reachable[0, 0, 0]
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q tests/parkour/foothold/test_depth_local_map.py
```

Expected: FAIL on import.

- [ ] **Step 3: Implement batched depth back-projection and frame registration**

Implement these exact public types and functions:

```python
@dataclass(frozen=True)
class LocalTerrainMap:
    height_f: torch.Tensor
    observed: torch.Tensor
    confidence: torch.Tensor
    reachable: torch.Tensor


def build_nominal_centered_local_map(
    *,
    depth_history: torch.Tensor,
    camera_pos_w_history: torch.Tensor,
    camera_quat_w_ros_history: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    frozen_origin_w: torch.Tensor,
    frozen_yaw_w: torch.Tensor,
    nominal_xy_f: torch.Tensor,
    radius_x: float,
    radius_y: float,
    half_extent_m: float = 0.10,
    resolution_m: float = 0.02,
) -> LocalTerrainMap:
    ...
```

Rules:

- Back-project only finite depths inside the configured camera range.
- Transform every historical point from its recorded ROS camera frame to world and then to the frozen support-foot frame.
- Bin points into the nominal-centered 11x11 grid with CUDA tensor operations.
- Store height relative to the frozen support-foot origin Z.
- `observed=False` and `confidence=0` for cells with no measurement.
- Confidence is the bounded fraction of selected historical frames that observed the cell; it is not a learned safety score.
- Apply the existing `(x/0.42)^2 + (y/0.25)^2 <= 1` reachability mask to the absolute support-frame cell centers.
- Do not query the terrain mesh or edge cylinders.

- [ ] **Step 4: Run the pure map tests**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q tests/parkour/foothold/test_depth_local_map.py
```

Expected: PASS.

---

### Task 3: Event-gated map observation and camera coverage diagnostics

**Files:**
- Modify: `source/instinctlab/instinctlab/envs/mdp/observations/exteroception.py`
- Modify: `source/instinctlab/instinctlab/envs/mdp/observations/__init__.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
- Modify: `source/instinctlab/instinctlab/monitors/foothold.py`
- Modify: `scripts/instinct_rl/play_debug.py`
- Test: `tests/parkour/foothold/test_local_terrain_observation.py`
- Test: `tests/parkour/foothold/test_foothold_monitor.py`

**Interfaces:**
- Produces observation component `foothold_local_map` with four channels `(height_f, observed, confidence, reachable)` and shape `(4, 11, 11)`.
- Uses `FootholdPlannerData.learned_foothold_event_generation` as the only rebuild trigger.
- Produces coverage metrics for current-frame, historical, nominal-cell, and teacher-nearest-safe-cell visibility.

- [ ] **Step 1: Write failing cache/event tests**

```python
def test_local_map_rebuilds_only_when_transaction_generation_changes():
    term = make_local_map_term(num_envs=2)
    term(env_with_generation([1, 1]))
    first_count = term.rebuild_count.clone()
    term(env_with_generation([1, 2]))
    assert term.rebuild_count.tolist() == [first_count[0], first_count[1] + 1]


def test_reset_invalidates_cached_map_for_selected_environment():
    term = make_local_map_term(num_envs=2)
    term.reset(torch.tensor([1]))
    assert not term.cached_observed[1].any()
```

- [ ] **Step 2: Implement `foothold_local_terrain` as a stateful manager term**

The term must:

- read synchronized depth and pose histories from `camera`;
- read frozen origin/yaw and nominal foothold from `foothold_planner`;
- compare transaction generation against a cached generation;
- build maps only for changed environment IDs;
- return cached maps for all other IDs without rebuilding;
- clear cache on environment reset.

- [ ] **Step 3: Add read-only coverage metrics and Play output**

Publish:

```text
foothold_map_nominal_observed_fraction
foothold_map_current_frame_observed_fraction
foothold_map_history_recovered_fraction
foothold_map_reachable_observed_fraction
foothold_map_no_observed_candidate_fraction
foothold_map_height_error_mean_m
foothold_map_height_error_max_m
```

The height-error metrics may compare against the terrain query only in monitor/debug code and must never be included in policy observations.

- [ ] **Step 4: Run coverage tests and a two-environment step-terrain Play**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q \
  tests/parkour/foothold/test_local_terrain_observation.py \
  tests/parkour/foothold/test_foothold_monitor.py
```

Then run the existing two-environment stair Play with only the new coverage diagnostics enabled. Do not change planner routing yet.

- [ ] **Step 5: Apply the perception go/no-go gate**

Proceed to Task 4 only if, separately for flat, ascent, and descent planning events:

- local-map maximum height error is at most 0.02 m;
- the privileged nearest safe cell is observed in at least 99% of events;
- no-observed-candidate events are at most 1%.

If the gate fails, adjust camera crop/FOV/history and repeat this task. Do not compensate with privileged actor inputs.

---

### Task 4: Privileged dense safety teacher for the local map

**Files:**
- Modify: `source/instinctlab/instinctlab_foothold/target_search.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`
- Test: `tests/parkour/foothold/test_target_search.py`
- Test: `tests/parkour/foothold/test_foothold_planner_data.py`

**Interfaces:**
- Produces `FootholdGridTeacher(safe, safety_score, best_cell_index, available)` for event environments.
- Uses the existing terrain-height query, 0.25 m height-difference bound, 0.42x0.25 m reachability ellipse, 26-point sole perimeter, and danger-cylinder penetration implementation.

- [ ] **Step 1: Write failing nearest-safe teacher tests**

```python
def test_teacher_selects_nominal_center_when_center_is_safe():
    result = evaluate_foothold_grid_teacher(center_safe_scene())
    assert result.best_cell_index.item() == 60


def test_teacher_selects_nearest_safe_cell_when_nominal_is_unsafe():
    result = evaluate_foothold_grid_teacher(edge_crossing_center_scene())
    assert result.safe.flatten()[result.best_cell_index].item()
    assert_selected_cell_has_minimum_distance_among_safe_cells(result)


def test_teacher_never_selects_unobserved_or_unreachable_cell():
    result = evaluate_foothold_grid_teacher(partially_observed_scene())
    assert result.available.item()
    assert result.observed.flatten()[result.best_cell_index].item()
    assert result.reachable.flatten()[result.best_cell_index].item()
```

- [ ] **Step 2: Implement vectorized grid evaluation**

Flatten only event-environment grid cells, query world Z, reject height-invalid and unreachable cells, generate the existing sole perimeter, and call the existing danger-cylinder scorer in one batched operation. Do not loop over 121 cells in Python.

Selection order must be lexicographic:

1. observed and reachable;
2. geometrically valid;
3. zero sole-perimeter penetration;
4. minimum Euclidean XY distance to the nominal center;
5. maximum frozen-command progress to break exact distance ties.

No new weighted sum or penetration-direction term is introduced.

- [ ] **Step 3: Store teacher outputs only at planning events**

Allocate cached `(N, 11, 11)` labels and one best-cell index per environment. Update them only when transaction generation changes and clear them on reset.

- [ ] **Step 4: Run focused safety tests**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q \
  tests/parkour/foothold/test_target_search.py \
  tests/parkour/foothold/test_foothold_planner_data.py
```

Expected: PASS.

---

### Task 5: Planner-only quality-map network with bounded absolute XY output

**Files:**
- Create: `source/instinctlab/instinctlab/learning/foothold_quality_map.py`
- Create: `source/instinctlab/instinctlab/learning/planner_isolated_encoder.py`
- Modify: `source/instinctlab/instinctlab/learning/independent_foothold_actor_critic.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`
- Modify: `source/instinctlab/instinctlab/learning/foothold_checkpoint.py`
- Test: `tests/parkour/foothold/test_foothold_quality_map.py`
- Test: `tests/parkour/foothold/test_independent_foothold_actor_critic.py`
- Test: `tests/parkour/foothold/test_foothold_checkpoint.py`

**Interfaces:**
- Consumes: cached `(4, 11, 11)` local map plus existing nominal foothold and state observations.
- Produces: 121 safety logits, a nearest-safe absolute XY prior, and the existing two-dimensional normalized action mean.
- Preserves: a 31-dimensional environment action `(29 motor + 2 foothold)`.

- [ ] **Step 1: Write failing network behavior tests**

```python
def test_quality_map_prefers_teacher_nearest_safe_cell():
    model = make_quality_map_model_with_fixed_logits(best_index=61)
    output = model(local_map_with_center_nominal())
    torch.testing.assert_close(output.prior_offset_m, torch.tensor([[0.02, 0.0]]))


def test_unknown_cells_receive_no_selection_probability():
    output = make_model()(map_with_high_logit_on_unknown_cell())
    assert output.selection_probability[0, 0, 0] == 0.0


def test_final_mean_is_absolute_and_within_one_cell_of_safe_prior():
    output = make_model()(known_safe_map())
    assert torch.linalg.norm(output.final_xy_f - output.safe_prior_xy_f, dim=-1).max() <= 0.02 + 1.0e-6


def test_motor_mean_is_identical_when_only_local_map_changes():
    actor = make_isolated_actor()
    action_a = actor.act_inference(obs_with_map_a())
    action_b = actor.act_inference(obs_with_map_b())
    torch.testing.assert_close(action_a[:, :29], action_b[:, :29])
```

- [ ] **Step 2: Implement the quality-map module**

Use a small convolutional model:

```python
Conv2d(4, 8, 3, padding=1)
ELU()
Conv2d(8, 8, 3, padding=1)
ELU()
Conv2d(8, 1, 1)
```

Mask unknown and unreachable cells before softmax. Convert the masked distribution to a nominal-centered metric XY prior using the fixed 2 cm grid. The existing planner MLP outputs only a refinement passed through `tanh` and scaled to at most 0.02 m. Add the refinement to the prior, then convert the result to the existing normalized absolute XY action representation.

The externally visible result remains an absolute foothold. The internal one-cell refinement cannot produce the current 0.35 m drift away from the nominal intent.

- [ ] **Step 3: Keep the local map invisible to the motor MoE**

Implement `PlannerIsolatedParallelLayer`, which removes `foothold_local_map` from the general encoded observation. The independent planner extracts the raw component and processes it through `FootholdQualityMap`; the motor actor receives the exact same encoded component set and width as before.

- [ ] **Step 4: Add audited checkpoint migration**

Copy all legacy motor encoder, actor, critic-zero, AMP, and discriminator parameters exactly. Initialize the new quality-map layers and planner refinement output independently. Fail loading if any motor tensor is silently resized or skipped.

- [ ] **Step 5: Run actor and migration tests**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q \
  tests/parkour/foothold/test_foothold_quality_map.py \
  tests/parkour/foothold/test_independent_foothold_actor_critic.py \
  tests/parkour/foothold/test_foothold_checkpoint.py
```

Expected: PASS, including bitwise-equivalent motor means for two observations differing only in the planner map.

---

### Task 6: Dense auxiliary safety loss plus existing event-gated PPO

**Files:**
- Modify: `source/instinctlab/instinctlab/learning/foothold_rollout_storage.py`
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`
- Modify: `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py`
- Test: `tests/parkour/foothold/test_foothold_rollout_storage.py`
- Test: `tests/parkour/foothold/test_event_gated_foothold_ppo.py`

**Interfaces:**
- Stores local-map safety labels and best-cell labels only for planner events.
- Adds `foothold_safety_map_loss` and `foothold_best_cell_loss` to the planner optimizer only.
- Retains the existing balanced safe/unsafe event PPO surrogate and independent planner KL control.

- [ ] **Step 1: Write failing masked-loss tests**

```python
def test_dense_safety_loss_ignores_non_event_and_unknown_cells():
    loss = dense_safety_loss(batch_with_one_event_and_unknown_cells())
    assert_loss_uses_only_known_cells_of_the_event(loss)


def test_best_cell_cross_entropy_is_zero_for_unavailable_teacher():
    loss = best_cell_loss(batch_with_no_available_teacher())
    assert loss.item() == 0.0


def test_dense_losses_do_not_update_motor_parameters():
    algorithm = make_algorithm()
    algorithm.backward_planner_dense_losses()
    assert all(parameter.grad is None for parameter in algorithm.actor_critic.motor_parameters())
```

- [ ] **Step 2: Extend event storage without duplicating every control step**

Store dense labels only for `learned_foothold_action_event=True`. Non-event transitions carry no copied 11x11 teacher tensors. Preserve existing GAE and reward storage.

- [ ] **Step 3: Add normalized dense losses**

Use mean binary cross entropy over known cells for the safety map and mean cross entropy over available best-cell labels. Add both with coefficient `1.0` to the independent planner optimizer; log their unweighted values and gradient norm separately. Do not route either loss into motor parameters.

- [ ] **Step 4: Preserve PPO semantics**

Keep:

- the 0.5 safe / 0.5 unsafe branch-balanced surrogate when both branches exist;
- current planner entropy coefficient;
- current independent planner KL limit and update skipping;
- current motor minibatch-level adaptive learning-rate logic.

- [ ] **Step 5: Run storage and PPO tests**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q \
  tests/parkour/foothold/test_foothold_rollout_storage.py \
  tests/parkour/foothold/test_event_gated_foothold_ppo.py
```

Expected: PASS.

---

### Task 7: Routing, observability gate, diagnostics, and acceptance

**Files:**
- Modify: `source/instinctlab/instinctlab_foothold/learned_target.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`
- Modify: `source/instinctlab/instinctlab/monitors/foothold.py`
- Modify: `scripts/instinct_rl/play_debug.py`
- Modify: `docs/foothold_project_status.md`
- Test: `tests/parkour/foothold/test_learned_foothold_planner.py`
- Test: `tests/parkour/foothold/test_foothold_monitor.py`

**Interfaces:**
- Adds route reason `PERCEPTION_UNAVAILABLE` without mapping it to RECOVERY.
- Preserves endpoint geometry/safety and trajectory-preflight gates.

- [ ] **Step 1: Write failing perception-unavailable routing tests**

```python
def test_no_observed_candidate_holds_without_entering_recovery():
    route = route_quality_map_target(
        map_available=torch.tensor([False]),
        physical_recovery=torch.tensor([False]),
    )
    assert route.keep_hold.item()
    assert not route.enter_recovery.item()


def test_observed_safe_target_still_requires_existing_preflight():
    route = route_quality_map_target(
        map_available=torch.tensor([True]),
        endpoint_safe=torch.tensor([True]),
        preflight_safe=torch.tensor([False]),
    )
    assert route.keep_hold.item()
```

- [ ] **Step 2: Add the observability gate before learned target execution**

An action may enter endpoint and trajectory preflight only when the map has at least one observed, reachable candidate and the quality head selected an observed cell. Otherwise clear the proposal, remain in the same HOLD transaction, and wait for a new perception/planning event. Do not alter physical recovery entry conditions.

- [ ] **Step 3: Add final diagnostics**

Log:

```text
quality_map_teacher_safe_fraction
quality_map_best_cell_accuracy
quality_map_selected_cell_observed_fraction
quality_map_selected_distance_to_nominal_m
quality_map_no_candidate_fraction
quality_map_refinement_saturation_fraction
quality_map_route_success_fraction
```

- [ ] **Step 4: Run the complete foothold test suite**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold
```

Expected: all tests PASS.

- [ ] **Step 5: Run 64-environment functional smoke test**

Run 100 iterations from scratch. Confirm finite losses, nonzero planner events, decreasing dense map loss, zero motor-parameter gradients from dense losses, and no new PLAN_INVALID/RECOVERY loop.

- [ ] **Step 6: Run 4096-environment 100-iteration performance acceptance**

Compare against a same-session baseline with identical terrain, seed, and checkpoint:

- collection time increase <= 10%;
- learning time increase <= 15%;
- no unbounded GPU-memory growth;
- quality-map computation count equals planning-event count, not control-step count.

If the performance gate fails, optimize event gathering and batched grid evaluation before any long training run. Do not reduce environment count to hide the overhead.

- [ ] **Step 7: Run stair Play acceptance before long training**

For both ascent and descent, verify that the displayed local map aligns with the stair surfaces, selected cells are observed, selected distance to nominal remains bounded, and missing perception produces HOLD rather than Recovery. Only then authorize a new 30,000-iteration run.

