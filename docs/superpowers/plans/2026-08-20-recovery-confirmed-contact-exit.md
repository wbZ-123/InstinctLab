# Recovery Confirmed-Contact Exit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exit contact-adaptive Recovery immediately when both feet are already confirmed in contact, without `dont_wait` opposing contact recovery.

**Architecture:** Preserve the existing per-foot contact confirmation logic and replace only the second Recovery dwell gate with the already-computed `both confirmed contacts` predicate. Keep the standalone calibration helper for diagnostics and legacy callers.

**Tech Stack:** Python, PyTorch, pytest.

## Global Constraints

- Do not change `swing_duration_s`, normal HOLD timing, learned foothold routing, or any reward other than Recovery masking for `dont_wait`.
- Raw contact is insufficient; the exit predicate must consume `confirmed_foot_contact`.
- Do not modify legacy non-adaptive Recovery behavior.

---

### Task 1: Confirmed-contact Recovery handoff

**Files:**
- Modify: `tests/parkour/foothold/test_foothold_planner_data.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`

**Interfaces:**
- Consumes: `confirmed_contact: torch.Tensor` with shape `(num_envs, 2)`.
- Produces: `stabilization_ready: torch.Tensor`, true exactly when both confirmed-contact columns are true.

- [ ] **Step 1: Write the failing source-contract test**

Assert that the adaptive planner assigns `stabilization_ready` directly from
the both-confirmed-contact predicate and no longer calls the additional dwell
gate in the runtime path.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_foothold_planner_data.py
```

Expected: the new assertion fails because runtime still calls
`stability_ready(... require_both_contact=True)`.

- [ ] **Step 3: Implement the minimal runtime change**

Use the already-computed `recovery_contact_stable = torch.all(confirmed_contact, dim=-1)` as `stabilization_ready`. Keep motion signals and bounds available for diagnostics.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Run state-machine and contact-adaptation regression tests**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_state_machine.py tests/parkour/foothold/test_contact_adaptation.py
```

Expected: PASS.

---

### Task 2: Mask only `dont_wait` during Recovery

**Files:**
- Modify: `tests/parkour/foothold/test_reward_foothold.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`

**Interfaces:**
- Consumes: `foothold_stabilization_mask(data)` through the existing
  `mdp.dont_wait_recovery_masked` wrapper.
- Produces: the original `dont_wait` value outside Recovery and zero inside
  Recovery; all other locomotion rewards retain their current functions.

- [ ] **Step 1: Write a failing configuration contract test**

Assert that the `dont_wait` reward term uses
`mdp.dont_wait_recovery_masked`, includes `sensor_name="foothold_planner"`,
and retains weight `-2.0`. Also assert that linear and angular velocity terms
continue to use their existing unmasked functions.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_reward_foothold.py
```

Expected: FAIL because `dont_wait` still calls the upstream reward directly.

- [ ] **Step 3: Implement the minimal configuration change**

Change only the reward function and add the sensor parameter:

```python
dont_wait = RewTerm(
    func=mdp.dont_wait_recovery_masked,
    weight=-2.0,
    params={
        "command_name": "base_velocity",
        "sensor_name": "foothold_planner",
    },
)
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Run the complete foothold suite**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold
```

Expected: PASS, followed by a clean `git diff --check`.

- [ ] **Step 6: Run the complete foothold suite**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold
```

Expected: PASS.
