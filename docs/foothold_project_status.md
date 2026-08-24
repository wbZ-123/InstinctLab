# Foothold 项目交接状态

更新时间：2026-08-24
当前分支：`feat/foothold-01-flat-tracking`

这份文档是换电脑、换 AI 或继续训练时的工作入口。它记录的是当前代码和最近一次完整训练的证据，不把旧聊天中的计划误认为已经实现，也不把训练曲线的改善误认为所有地形都已经解决。

## 1. 项目要解决的问题

在 Hiking in the Wild / InstinctLab 的 G1 parkour AMP 行走任务上，增加显式落足点规划能力：

- 解析规划器给出行走意图和名义落足点；
- 学习式落足 planner 结合机器人状态和深度观测输出最终水平落点；
- 世界坐标地形查询补齐目标高度；
- 摆动轨迹由落足点生成，原有电机策略负责跟踪；
- 仿真训练用脚底外围点、危险圆柱、侵入深度、可达椭圆和 clearance 做安全奖励与诊断；
- 训练和实机部署输入必须区分：危险圆柱等是仿真特权信息，不能假设实机能够直接获得。

当前没有修改 AMP、MoE 的基本结构，也没有把外部 Instinct-RL 仓库复制进来。项目内只提供环境、planner、独立落足网络和 PPO 扩展。

## 2. 已经完成并在当前代码中的工作

### 2.1 落足规划主链路

当前主链路是：

```text
稳定支撑/HOLD
  → 冻结支撑脚世界坐标和身体朝向
  → 生成解析名义落点
  → 学习式 planner 输出水平 XY
  → 局部 XY 转世界 XY
  → 用世界坐标查询地形 Z
  → 检查高度、可达性和脚底安全
  → 在 HOLD 内生成并检查完整摆动轨迹
  → 锁定支撑坐标系、起点、目标和轨迹
  → 进入 SWING，由电机策略跟踪
```

进入 SWING 后，不再用新的支撑脚坐标系重新解释旧目标。

### 2.2 坐标和脚底几何

- 学习式 planner 的输出是支撑脚 planner frame 下的水平坐标；
- 地形高度查询使用世界坐标；
- 最终目标是世界坐标三维点 `(x_w, y_w, z_w)`；
- 脚底 footprint 已按当前 G1 鞋底 URDF 外包络接入，约为 `0.186 m × 0.072 m`；
- 危险圆柱检查使用真实世界坐标脚底点；
- 学习式落点不再使用解析名义点生成器中的 6 cm 横向硬拒绝，解析名义点自身的 18 cm 脚宽意图仍保留。

### 2.3 学习式 planner

- 动作输出为最终水平 XY，不是增量；
- Z 由世界地形查询补齐；
- 使用独立的 planner 深度编码器和落足 MLP；
- 电机动作和 planner 动作分开使用独立标准差、优化器、KL 和事件门控；
- planner 奖励只在落足规划事件发生时产生；
- 名义点安全时奖励学习点接近名义意图；
- 名义点不安全时，先要求学习点安全，再评价命令一致性和偏离代价；
- 不安全提案不会直接执行危险摆动，但会保留负奖励用于 PPO 学习；
- Recovery 阶段暂停学习式 planner，恢复后建立全新的 HOLD 规划事务。

### 2.4 状态机、接触和 Recovery

已经覆盖或修复的重点包括：

- planner 未评估完成不能进入 SWING；
- 一个 HOLD 事务不会因为每个控制周期而重复采样、重复扣 planner 奖励；
- 左右脚角色依据真实接触关系更新，避免切换后认错支撑脚；
- 触地验收要求摆动脚此前确实离地并重新确认接触；
- 规划失败与物理异常 Recovery 分开；
- Recovery 不再生成另一套危险恢复落点；
- Recovery 的恢复奖励/`dont_wait` 处理已按当前设计接入；
- Recovery 退出使用确认接触，恢复后回到正常 HOLD 再开始新规划。

实现细节和历史设计见：

- `docs/foothold_planner_implementation.md`
- `docs/superpowers/specs/2026-08-18-foothold-hold-transaction-lifecycle-design.md`
- `docs/superpowers/specs/2026-08-20-recovery-confirmed-contact-exit-design.md`

### 2.5 诊断和测试

已经补充：

- planner 提案、名义安全/不安全、学习点安全、路由和失败阶段统计；
- play 中的事件级落足、轨迹、接触、可达椭圆和脚宽诊断；
- planner、reward、geometry、state machine、contact adaptation、play debug 和 PPO 扩展测试。

完整测试入口：

```bash
cd ~/InstinctLab-foothold
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q tests/parkour/foothold
```

## 3. 最近完整训练的客观结果

最近一次完整训练目录：

```text
logs/instinct_rl/g1_parkour/20260821_124341_planner_branch_reward_from0_30000it
```

最近 play 使用的是 `model_28000.pt`。从训练事件和 play 日志得到的结论如下：

- planner 评估约占控制步 `4.7%`；
- 其中名义点不安全约占评估事件 `8.7%`；
- 不安全修正样本因此约占全部控制步 `0.4%`；
- 不安全名义点的学习修正成功率后期约 `67%`，不是完全不会学习，但仍有约三分之一失败；
- 学习点整体安全率后期约 `97%`，路由成功率约 `99.9%`；
- `mean_reward_1` 从训练初期接近 `0` 上升到约 `0.33` 后进入平台；
- 不安全事件的平均修正距离约 `0.34 m`，偏大，说明网络有时倾向于“跳到较远的安全区域”，还没有稳定学会台阶边缘附近的最小、准确修正；
- planner KL 后期约在目标 `0.01` 附近，偶尔有 KL 超限跳过的 minibatch，需要继续监控。

因此，最近 play 中“机器人仍然向上走”与“某些阶段没有规划小球”并不矛盾：planner 是间歇性成功/失败，不是全程 HOLD。成功的已锁定摆动和低层策略仍会让机器人继续运动；失败的台阶边缘提案会在 HOLD 或 Recovery 中暂时没有可执行的新轨迹。

## 4. 当前版本的主要问题

### 4.1 planner 分支样本严重不平衡

当前 PPO 对 planner 事件统一求平均。后期约 91% 是安全名义点，只有约 9% 是不安全名义点；困难的“台阶边缘修正”对总梯度贡献太小。`reward_1` 平台更像是“普通安全点已经学会，困难分支没有完全学会”的平均效果，而不是网络完全失效。

### 4.2 不安全修正仍然偏大且不够稳定

当前安全分包含侵入点数量和总侵入深度，确实能告诉 PPO 哪个试探点更安全，但它是标量结果，不是直接的 XY 方向梯度。危险区域边界、地形高度变化和脚底外围点判断还会造成离散跳变。深度图虽然已经输入 planner，但它必须通过稀疏事件奖励自己学会“图像位置 → 足端 XY”的对应关系。

### 4.3 reward_1 不能单独代表 planner 质量

`reward_1` 混合了非事件控制步、安全名义点事件和不安全修正事件。必须同时查看：

- `nominal_safe_fraction`
- `nominal_unsafe_fraction`
- `nominal_unsafe_correction_success_fraction`
- `nominal_unsafe_safety_score_mean`
- `nominal_unsafe_correction_distance_mean`
- `foothold_event_count`
- `foothold_kl` / `foothold_kl_skip_count`

不能只凭 `reward_1` 一条曲线判断 planner 是否学会了楼梯落点。

### 4.4 参数仍有需要标定的项目

可达椭圆、摆动时序、触地阶段、清障上限、名义脚宽和 Recovery 步长仍应结合 URDF、运动学扫描、air-time 和 play 统计继续标定。详见 [foothold_parameter_audit.md](foothold_parameter_audit.md)。

## 5. 已确定但尚未实施的下一步

下一步只处理 planner PPO 的样本分支平衡，不同时修改轨迹跟踪、Recovery、AMP、MoE、深度编码器或安全奖励公式。

目标是：

```text
planner 损失 =
    0.5 × 名义点安全事件平均损失
  + 0.5 × 名义点不安全事件平均损失
```

这里必须是加号。PPO 损失已经包含优势符号；不安全输出的负奖励会通过优势进入正确的更新方向，再减去不安全损失会把方向反过来。

实施时需要：

1. 在 rollout 中保留每个 planner 事件的“名义安全/不安全”分支标签；
2. 分别计算两类事件的 PPO surrogate/value/entropy 统计；
3. 两类都存在时等权平均，某一类为空时只使用存在的分支；
4. 增加分支计数、分支优势和分支损失日志；
5. 先跑 64 环境/100 轮和 4096 环境/100 轮，再比较修正成功率和平均修正距离；
6. 只有确认困难分支改善后，才重新启动长训。

这个修改不是在奖励中额外增加一个拍脑袋权重，而是避免普通安全事件在 PPO 汇总时淹没困难事件。当前尚未修改这部分代码。

## 6. 换电脑复现清单

Git 只保存源码、测试和文档，不保存 checkpoint、TensorBoard 日志、运动参考数据或 IsaacLab。

必须另外准备：

- IsaacLab 对应提交 `f73c331738`；
- Isaac Sim 5.1；
- 外部 Instinct-RL 对应提交 `f870ead0953fa0e3c3da3349b0aece1c74bfb421`；
- Python 3.11、匹配的 PyTorch/CUDA；
- `parkour_motion_reference` 和 `parkour_motion_without_run.yaml`；
- 通过 `python -m pip install -e source/instinctlab` 安装当前仓库。

复现后先运行：

```bash
cd ~/InstinctLab-foothold
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q tests/parkour/foothold
```

再做短训和短 play，不要直接把旧日志目录名当作新代码版本。
