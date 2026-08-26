# No-Fly Penalty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded parkour reward term that contributes a `-1.0` weighted penalty on every control step where both feet are airborne.

**Architecture:** Implement a lightweight contact-history indicator in the existing foothold reward module so it can be tested without initializing Isaac Sim. Register it in `RewardsCfg` with the existing contact-force sensor and the two ankle-roll links. The term is independent of Recovery and leaves the existing single-support `feet_air_time` reward unchanged.

**Tech Stack:** Python, PyTorch, IsaacLab manager reward configuration, pytest.

## Global Constraints

- The unweighted function returns only `0.0` or `1.0` per environment.
- The configured weight is exactly `-1.0`.
- A force magnitude strictly greater than `1.0 N` means contact, matching existing parkour contact rewards.
- Any left or right foot contact suppresses the penalty; only double flight is penalized.
- Reuse `contact_forces`; do not add a sensor, grace-period parameter, command gate, terrain branch, or Recovery mask.
- Do not alter `feet_air_time`.

---

### Task 1: Add and register the no-fly indicator

**Files:**
- Modify: `tests/parkour/foothold/test_reward_foothold.py`
- Modify: `source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`

**Interfaces:**
- Consumes: `env.scene.sensors[sensor_cfg.name].data.net_forces_w_history` with shape `(num_envs, history, bodies, 3)` and `sensor_cfg.body_ids` selecting the two ankle links.
- Produces: `no_fly(env, sensor_cfg, threshold: float = 1.0) -> torch.Tensor` with shape `(num_envs,)`, plus `RewardsCfg.no_fly` configured with weight `-1.0`.

- [ ] **Step 1: Write failing behavioral and configuration tests**

Add the following test to `tests/parkour/foothold/test_reward_foothold.py`:

```python
def test_no_fly_penalizes_only_when_both_feet_are_airborne():
    foothold = _load_foothold_reward_module()
    forces = torch.zeros(4, 3, 2, 3)
    forces[1, -1, 0, 2] = 2.0
    forces[2, -1, 1, 2] = 2.0
    forces[3, -1, :, 2] = 2.0
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "contact_forces": SimpleNamespace(
                    data=SimpleNamespace(net_forces_w_history=forces)
                )
            }
        )
    )
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])

    torch.testing.assert_close(
        foothold.no_fly(env, sensor_cfg=sensor_cfg, threshold=1.0),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )


def test_parkour_config_registers_no_fly_with_minus_one_weight():
    cfg_path = (
        Path(__file__).resolve().parents[3]
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    )
    cfg_text = cfg_path.read_text()
    block = cfg_text.split("    no_fly = RewTerm(", 1)[1].split(
        "    feet_slide = RewTerm(", 1
    )[0]

    assert "func=instinct_mdp.no_fly" in block
    assert "weight=-1.0" in block
    assert 'SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")' in block
    assert '"threshold": 1.0' in block
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_reward_foothold.py \
-k 'no_fly'
```

Expected: FAIL because `no_fly` is not defined and the config term is absent.

- [ ] **Step 3: Implement the minimal contact indicator**

Add to `source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py`:

```python
def no_fly(
    env,
    sensor_cfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return one only when neither selected foot has ground contact."""

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    force_history = env.scene.sensors[
        sensor_cfg.name
    ].data.net_forces_w_history
    selected_forces = force_history[:, :, sensor_cfg.body_ids]
    foot_contact = (
        torch.linalg.vector_norm(selected_forces, dim=-1)
        .amax(dim=1)
        > threshold
    )
    return (~torch.any(foot_contact, dim=-1)).to(force_history.dtype)
```

- [ ] **Step 4: Register the reward term**

Insert immediately after `feet_air_time` and before `feet_slide` in
`source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`:

```python
    no_fly = RewTerm(
        func=instinct_mdp.no_fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_ankle_roll_link",
            ),
            "threshold": 1.0,
        },
    )
```

- [ ] **Step 5: Run focused and regression tests and verify GREEN**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_reward_foothold.py \
-k 'no_fly or recovery_masked_feet_air_time'
```

Expected: PASS.

Then run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold
```

Expected: all foothold tests PASS.

- [ ] **Step 6: Review and commit only no-fly files**

```bash
git diff --check -- \
  source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py \
  source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py \
  tests/parkour/foothold/test_reward_foothold.py
git add \
  source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py \
  source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py \
  tests/parkour/foothold/test_reward_foothold.py
git commit -m "Add no-fly contact penalty"
```
