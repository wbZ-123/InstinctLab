# Balanced Learned Foothold PPO Loss Design

## Goal

让学习式落足 planner 的 PPO 更新同时关注“名义落点安全”和“名义落点不安全”两类事件，避免后期约 91% 的普通安全事件淹没约 9% 的台阶边缘修正事件。

本次不重写安全奖励，也不改变落足点路由和电机策略行为。目标是验证：不安全分支的训练信号增加后，planner 的危险点修正成功率提高，且平均修正距离从当前约 0.34 m 降低。

## Current evidence

最近完整训练的后期统计约为：

- planner 事件占控制步约 4.7%；
- 名义点不安全事件占 planner 事件约 8.7%；
- 不安全修正成功率约 67%；
- 整体学习点安全率约 97%；
- 不安全事件平均修正距离约 0.34 m。

因此当前主要风险是事件类别不平衡，而不是安全评分完全没有信号。

## Scope

### Included

1. 在 rollout transition 中保存 planner 事件的名义安全分支标签；
2. 在 minibatch 中传递该标签；
3. 分别计算两类事件的 planner PPO surrogate loss；
4. 两类事件都存在时使用等权平均：

   ```text
   L_planner = 0.5 * L_nominal_safe + 0.5 * L_nominal_unsafe
   ```

5. 记录两类事件计数、优势均值和 surrogate loss，便于验证实际更新是否覆盖困难分支；
6. 保持现有 planner KL、熵、优化器和事件门控逻辑。

### Excluded

- 不修改 `learned_foothold_planning_event_reward` 的侵入深度、侵入点数量和命令一致性公式；
- 不新增任意米制距离阈值或安全圆柱参数；
- 不修改电机策略 PPO、AMP、MoE、Recovery、摆动轨迹跟踪或深度编码器；
- 不改变安全/不安全提案的执行路由；
- 不使用负号合并两类损失；
- 不把非 planner 事件伪装成安全分支事件。

## Reward semantics retained

当前奖励语义保持不变：

```text
名义点安全：
    学习点越接近名义点越好

名义点不安全、学习点仍不安全：
    使用脚掌外围点的侵入点数量和总侵入深度得到负安全分

名义点不安全、学习点安全：
    使用命令一致性与名义点接近程度评价安全修正
```

这里的“最近安全点”由安全检查作为前提，再由现有名义距离代价和命令一致性共同选择。当前阶段不再添加第三个距离奖励，以便隔离 PPO 样本平衡的影响。

## Data flow

### Environment to rollout

在 planner 评估事件发生时，环境已经拥有：

- `learned_foothold_evaluated`：当前控制步是否是 planner 事件；
- `nominal_geometric_valid`；
- `nominal_safety_valid`。

定义分支标签：

```text
nominal_safe_event =
    learned_foothold_evaluated
    AND nominal_geometric_valid
    AND nominal_safety_valid

nominal_unsafe_event =
    learned_foothold_evaluated
    AND NOT nominal_safe_event
```

两者必须互斥，且并集严格等于 planner 事件。非 planner 控制步两个标签都为 false。

标签在 transition 写入 rollout storage 时使用 `detach().clone()`，不能引用下一控制步会被清零的传感器 buffer。

### Rollout to minibatch

`FootholdRolloutStorage` 的 transition 和 minibatch 都增加两个布尔字段：

- `foothold_nominal_safe_event`；
- `foothold_nominal_unsafe_event`。

已有 `foothold_action_event` 继续保留，用于 planner 动作、KL、熵和非事件过滤。新增标签不能替代已有事件掩码。

## Loss calculation

对每个 minibatch，先计算每个样本的 planner clipped surrogate 项以及现有 planner value/entropy 项。

### Surrogate branch reduction

```python
safe_loss = masked_mean(per_event_surrogate, safe_event)
unsafe_loss = masked_mean(per_event_surrogate, unsafe_event)

if safe_count > 0 and unsafe_count > 0:
    surrogate_loss = 0.5 * safe_loss + 0.5 * unsafe_loss
elif safe_count > 0:
    surrogate_loss = safe_loss
elif unsafe_count > 0:
    surrogate_loss = unsafe_loss
else:
    surrogate_loss = zero_with_gradient
```

`masked_mean` 的空掩码行为必须返回零张量，不产生 NaN；但训练更新仍由已有 planner event count 和 KL 门控决定。

### Value and entropy terms

value loss 和 entropy 仍只针对所有 planner event 做 event-masked mean，不对 critic 的回报定义做类别重加权。这样本次改动只改变 actor planner surrogate 的类别平衡，不改变 value target 或熵正则的语义。

最终 planner loss 仍为：

```text
planner_loss =
    foothold_surrogate_loss_coef * balanced_surrogate_loss
  + foothold_value_loss_coef * event_masked_value_loss
  + foothold_entropy_coef * event_masked_entropy_loss
```

### Advantage handling

保留现有 PPO advantage 计算和归一化流程。不能用 `-0.5 * unsafe_loss`，因为 surrogate loss 已经包含 `-advantage * ratio`；额外取负会反转不安全分支的策略更新方向。

## Diagnostics

每个 minibatch/训练轮增加：

- `foothold_nominal_safe_event_count`；
- `foothold_nominal_unsafe_event_count`；
- `foothold_nominal_safe_advantage_mean`；
- `foothold_nominal_unsafe_advantage_mean`；
- `foothold_nominal_safe_surrogate_loss`；
- `foothold_nominal_unsafe_surrogate_loss`；
- `foothold_balanced_surrogate_loss`。

已有的以下指标继续保留：

- `foothold_kl`；
- `foothold_kl_skip_count`；
- `foothold_std_m_x/y`；
- `learned_foothold_nominal_unsafe_correction_success_fraction`；
- `learned_foothold_nominal_unsafe_correction_distance_mean`。

## Edge cases

1. 一个 minibatch 只有安全事件：使用安全分支平均损失，不用人为补零的不安全分支；
2. 一个 minibatch 只有不安全事件：使用不安全分支平均损失；
3. minibatch 没有 planner 事件：planner update 继续按现有门控跳过；
4. 一个样本同时被标记为安全和不安全：测试必须失败，运行时应通过互斥定义避免；
5. 非有限优势或 loss：沿用现有非有限检查，不静默更新；
6. 分支标签和 event mask 形状不一致：立即抛出清晰的 `ValueError`。

## Verification criteria

### Unit tests

- 分支标签在安全、不安全、非事件样本上的互斥/完备性；
- storage 写入和 minibatch 展平后标签不丢失；
- 两类都有样本时结果等于两类 masked mean 的算术平均；
- 只有一类样本时不被另一类零填充值稀释；
- 两类都为空时返回有限零张量；
- 负优势不会因为分支合并符号被反转。

### Runtime acceptance

先执行：

1. foothold 全量单元测试；
2. 64 环境短训 100 轮；
3. 4096 环境短训 100 轮。

接受标准不是只看 `reward_1`：

- 两类事件计数非零且比例符合环境实际；
- planner KL 没有持续爆炸；
- 不安全修正成功率不低于修改前短训基线；
- 平均修正距离不继续系统性增大；
- 电机 reward、Recovery 比例和 collection 时间没有明显异常变化。

只有短训通过后，才重新启动 30000 轮长训。
