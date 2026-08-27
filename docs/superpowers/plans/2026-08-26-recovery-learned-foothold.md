# Recovery 学习式落足实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让接触自适应 Recovery 在恰好一只脚确认支撑时执行一次学习式恢复落点，并把命令方向评分改成能惩罚后退点的有符号连续分。

**Architecture:** Recovery 只放宽危险圆柱和轨迹擦边的执行门，将“地形高度有效、目标有限、可达椭圆内、最大步高差不超过 0.27 m、轨迹可构造”保留为几何底线。单支撑恢复进入短 HOLD 事务，使用悬空脚真实位置作为起点；学习落点只采样一次并锁定，支撑丢失立即废弃。学习事件在该事务中正常写入 PPO，不再被 Recovery 稳定掩码清零。

**Tech Stack:** Python, PyTorch, pytest, IsaacLab manager-based environment.

## Global Constraints

- 不修改 AMP、MoE、电机 PPO 主干、观测维度、正常双脚步态或解析名义点生成器。
- 世界坐标地形高度查询和 SWING 后坐标冻结保持不变。
- 不重新引入 32 个候选点搜索，不添加新的米制方向阈值。
- 危险圆柱侵入仍记录点数、比例、总深度和最深相位。
- 分数全部限制在 `[-1, 1]`。

---

### Task 1: 为有符号命令方向评分补充失败测试

**Files:**
- Modify: `tests/parkour/foothold/test_reward_foothold.py`
- Modify: `source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py`

**Interfaces:**
- 保持 `learned_foothold_planning_event_reward(...) -> torch.Tensor` 签名不变。
- 新增私有 `_signed_command_progress_score(learned_velocity, desired_velocity)`，返回形状为 `(N,)`、范围 `[-1, 1]` 的连续分。

- [x] **Step 1: 写测试，验证同向、静止、反向的顺序**

```python
def test_signed_command_progress_penalizes_reverse_foothold():
    data = _make_planner_data_for_reward(
        learned_decoded_f=torch.tensor([[0.20, 0.18, 0.0],
                                        [0.00, 0.18, 0.0],
                                        [-0.20, 0.18, 0.0]]),
        nominal_feasible_velocity_f=torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1),
        nominal_geometric_valid=torch.ones(3, dtype=torch.bool),
        nominal_safety_valid=torch.zeros(3, dtype=torch.bool),
        learned_geometric_valid=torch.ones(3, dtype=torch.bool),
        learned_safety_valid=torch.ones(3, dtype=torch.bool),
    )
    reward = foothold.learned_foothold_planning_event_reward(
        _env_with_data(data), velocity_lookahead_s=0.20,
    )
    assert reward[0] > reward[1] > reward[2]
    assert reward[2] < 0.0
```

- [x] **Step 2: 运行该测试，确认旧的非负命令评分使测试失败**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_reward_foothold.py::test_signed_command_progress_penalizes_reverse_foothold
```

Expected: FAIL because the existing safe correction branch cannot make a reverse point negative.

### Task 2: 实现有符号命令评分并保留安全分支语义

**Files:**
- Modify: `source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py`
- Modify: `tests/parkour/foothold/test_reward_foothold.py`

**Interfaces:**
- `signed_command_progress_score = clamp(dot(v_learned, v_desired) / max(||v_desired||², eps), -1, 1)`。
- 当冻结期望速度接近零时，使用已有名义偏离分作为退化分，避免零速度下产生 NaN。

- [x] **Step 1: 添加最小私有 helper**

```python
def _signed_command_progress_score(
    learned_velocity: torch.Tensor,
    desired_velocity: torch.Tensor,
) -> torch.Tensor:
    desired_norm_sq = torch.sum(torch.square(desired_velocity), dim=-1)
    projection = torch.sum(learned_velocity * desired_velocity, dim=-1)
    eps = torch.finfo(learned_velocity.dtype).eps
    score = (projection / desired_norm_sq.clamp_min(eps)).clamp(-1.0, 1.0)
    return torch.where(
        desired_norm_sq > eps,
        score,
        torch.zeros_like(score),
    )
```

- [x] **Step 2: 在不安全名义点且学习点安全分支中使用符号分**

保留危险圆柱安全分和几何无效分支；只替换当前始终非负的 `command_consistency * (1 - nominal_deviation_cost)`：

```python
signed_command = _signed_command_progress_score(
    learned_velocity,
    data.nominal_feasible_velocity_f[:, :2],
)
safe_learned_score = torch.where(
    nominal_safe,
    nominal_deviation_reward,
    signed_command - nominal_deviation_cost,
)
```

随后统一 `clamp(-1.0, 1.0)`，不改变几何无效和侵入点负分逻辑。

- [x] **Step 3: 运行奖励测试**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_reward_foothold.py
```

Expected: PASS，且安全名义点仍满足精确匹配最高、2 cm 内为正、超过 2 cm 为负。

### Task 3: 为单支撑 Recovery 事务添加状态机失败测试

**Files:**
- Modify: `tests/parkour/foothold/test_state_machine.py`
- Modify: `source/instinctlab/instinctlab_foothold/state_machine.py`

**Interfaces:**
- `advance_gait(...)` 的现有参数保持兼容。
- 接触自适应模式下，Recovery 从单脚确认支撑退出到 `HOLD` 时，将 `recovery_step_pending=True`；双脚确认时直接普通 HOLD，零脚时继续 Recovery。

- [x] **Step 1: 添加单脚、双脚和零脚三条测试**

```python
def test_contact_adaptive_recovery_single_support_opens_one_shot_hold():
    state = initial_gait_state(1, device="cpu")
    state.mode.fill_(GaitState.RECOVERY)
    next_state = advance_gait(
        state, torch.tensor([[True, False]]), torch.tensor([False]),
        torch.tensor([True]), 0.02, GaitMachineConfig(),
        event_response=torch.tensor([EventResponse.STABILIZE]),
        stabilization_ready=torch.tensor([True]),
        stability_current=torch.tensor([True]),
    )
    assert next_state.mode.item() == GaitState.HOLD
    assert next_state.recovery_step_pending.item()
    assert next_state.swing_side.item() == 1
```

同时断言零接触保持 Recovery，双脚接触退出且 `recovery_step_pending=False`。

- [x] **Step 2: 运行新增测试，确认当前实现失败**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_state_machine.py -k \
'single_support_opens_one_shot_hold or recovery_requires_stability'
```

Expected: 单支撑测试 FAIL，因为当前 `stabilization_ready` 只接受双脚接触且退出时清除 recovery step 标志。

### Task 4: 实现单支撑 Recovery 状态转移

**Files:**
- Modify: `source/instinctlab/instinctlab_foothold/state_machine.py`

**Interfaces:**
- `stabilization_ready` 在接触自适应模式中仍用于外部稳定信号，但 Recovery 退出必须区分 `any_confirmed_contact` 和 `both_contacts_confirmed`。

- [x] **Step 1: 仅在 Recovery 中允许单支撑退出到恢复 HOLD**

在 `exited_recovery` 计算前增加：

```python
recovery_single_support = (
    (state.mode == GaitState.RECOVERY)
    & stabilization_ready
    & torch.any(confirmed_contact, dim=-1)
    & ~torch.all(confirmed_contact, dim=-1)
)
recovery_double_support = (
    (state.mode == GaitState.RECOVERY)
    & stabilization_ready
    & torch.all(confirmed_contact, dim=-1)
)
exited_recovery = recovery_single_support | recovery_double_support
```

- [x] **Step 2: 设置支撑/摆动角色和一次性事务标志**

```python
if torch.any(exited_recovery).item():
    _, next_swing = support_roles_from_contacts(
        confirmed_contact, swing_side,
    )
    valid_next_swing = exited_recovery & (next_swing >= 0)
    mode[exited_recovery] = GaitState.HOLD
    swing_side[valid_next_swing] = next_swing[valid_next_swing]
    elapsed_s[exited_recovery] = 0.0
    hold_elapsed_s[exited_recovery] = 0.0
    hold_required_s[exited_recovery] = step_hold_s[exited_recovery]
    recovery_step_pending[recovery_single_support] = True
    recovery_step_active[recovery_single_support] = False
    recovery_step_pending[recovery_double_support] = False
    recovery_step_active[recovery_double_support] = False
    stabilization_elapsed_s[exited_recovery] = 0.0
```

- [x] **Step 3: 运行状态机全量测试**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_state_machine.py
```

Expected: PASS，旧的双脚稳定测试仍通过，单支撑只生成一次恢复 HOLD。

### Task 5: 让恢复 HOLD 调用学习 planner，并把危险安全门改为软诊断

**Files:**
- Modify: `source/instinctlab/instinctlab_foothold/learned_target.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`
- Modify: `tests/parkour/foothold/test_learned_foothold_planner.py`
- Modify: `tests/parkour/foothold/test_foothold_planner_data.py`

**Interfaces:**
- Recovery 单支撑使用与正常 HOLD 相同的学习 target preparation，但 `recovery_step=True` 时仅以 `learned_geometric_valid` 作为可执行门。
- `learned_foothold_evaluated` 在恢复 HOLD 保持一次性锁存；清除仅发生在支撑关系改变、事务完成或支撑丢失。

- [x] **Step 1: 添加 route 测试**

```python
def test_recovery_route_executes_geometric_learned_point_with_negative_safety():
    route = route_nominal_and_learned_footholds(
        nominal_geometric_valid=torch.tensor([True]),
        nominal_safety_valid=torch.tensor([True]),
        learned_prepared=torch.tensor([True]),
        learned_geometric_valid=torch.tensor([True]),
        learned_safety_valid=torch.tensor([False]),
        recovery_step=torch.tensor([True]),
    )
    assert route.use_learned.item()
    assert route.executable.item()
```

- [x] **Step 2: 修改 route 逻辑**

Recovery 时：

```python
use_learned = (
    (~recovery_mask & learned_available_safe)
    | (recovery_mask & learned_prepared.bool() & learned_geometric_valid.bool())
)
```

正常 WALK 继续要求 `learned_safety_valid`；Recovery 只放宽危险圆柱安全门。

- [x] **Step 3: 修改 planner 事件准备和预检**

在接触自适应模式中，`recovery_step_pending` 不再屏蔽 `prepare_learned`；名义/学习目标仍使用冻结支撑坐标系，轨迹起点使用 `actual_swing_foot_pos_w`。预检保留数值与轨迹可构造检查，但 Recovery 时不以危险圆柱碰撞结果清除可执行学习目标；将侵入量写入已有诊断字段。

- [x] **Step 4: 确保 Recovery 事件奖励不被屏蔽**

Recovery 单支撑进入 HOLD 后，将 `stabilization_active=False`，但保留 `recovery_step_pending=True` 标记供路由使用。这样 planner 事件奖励正常进入 PPO，电机策略仍可得到已有恢复奖励。

- [x] **Step 5: 运行路由、planner-data 与状态机测试**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_learned_foothold_planner.py \
tests/parkour/foothold/test_foothold_planner_data.py \
tests/parkour/foothold/test_state_machine.py
```

Expected: PASS，且 Recovery 不会在同一支撑关系下每步重复采样。

### Task 6: 完整验证与轻量运行检查

**Files:**
- No source changes unless a regression test identifies a direct defect in the scoped behavior.

- [x] **Step 1: 运行全部落足模块测试**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold
```

- [x] **Step 2: 运行静态编译和 diff 检查**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m py_compile \
source/instinctlab/instinctlab_foothold/learned_target.py \
source/instinctlab/instinctlab_foothold/state_machine.py \
source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py \
source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py
git diff --check
```

- [ ] **Step 3: 运行 64 环境 30 轮 smoke test**

确认日志满足：Recovery 单支撑能产生学习 planner 事件；危险分为负时仍有学习落点执行；无脚支撑不产生目标；支撑丢失能清除事务；收集时间没有因每步重复采样显著增加。
