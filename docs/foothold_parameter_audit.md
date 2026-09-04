# Foothold planner parameter audit

这份文档专门记录落足点规划器里“参数从哪里来”。它的目标不是解释功能，而是避免后续换机器人、换模型或继续调参时继续依赖拍脑袋常数。

当前结论很直接：步态规划器主链路已经接入训练，但 flat target、可达椭圆、时序和脚底几何里仍有一批临时参数。尤其是 `FlatProviderConfig` 里已经明确写着“临时保守值，后续由 G1 运动学扫描标定”。所以后续工作应该围绕这些参数做来源对齐和标定。

## 1. 参数来源分类

后续所有 foothold planner 参数都应该归到下面四类之一。

### 1.1 机器人结构参数

这些参数应该来自机器人模型、初始站姿或脚底几何。换机器人时必须重新读取或重新测量。

- 左右脚默认宽度；
- 脚底长度、宽度；
- 脚底中心相对 ankle/sole link 的偏移；
- 安全落足检查时使用的脚底 footprint。

### 1.2 训练/策略行为统计参数

这些参数应该来自训练出来的策略、接触传感器统计或 TensorBoard monitor。它们描述“策略实际怎么走”，不应该孤立手设。

- 实际摆动脚离地时间；
- 实际 swing 持续步数；
- 提前接触分布；
- 落足点超出规划点的分布；
- recovery 后真实能否重新稳定。

### 1.3 运动学可达参数

这些参数应该来自机器人腿部运动学扫描或 IK 可达范围。它们决定落足点不能超过哪里。

- 支撑脚坐标系下的最大前后可达范围；
- 最大横向可达范围；
- recovery 最大步长；
- 目标搜索候选点是否还在可达区域内。

### 1.4 训练课程设计参数

这些参数可以保留为训练设计，但必须清楚说明它们不是机器人物理参数。

- flat target 的课程扰动半径；
- yaw 课程限制；
- foothold reward scale 的启动曲线；
- 课程 level 的数量和映射规则。

## 2. 当前参数审计表

| 参数 | 当前值 | 位置 | 当前来源 | 应该对齐到 | 状态 |
| --- | --- | --- | --- | --- | --- |
| `outer_radius_x` | `1.00` | `instinctlab_foothold/flat_provider.py` | 固定前后可达半轴 | G1 腿部运动学可达范围扫描 | 临时固定，仍需实机标定 |
| `outer_radius_y` | `0.50` | `instinctlab_foothold/flat_provider.py` | 2026-09-03 放宽的实验动作/搜索边界，不代表已标定的 G1 物理极限 | G1 腿部运动学可达范围扫描 | 需要标定 |
| `min_lateral_separation` | `0.06` | `instinctlab_foothold/flat_provider.py` | 临时保守值 | 左右脚防交叉安全间距，至少要结合脚宽和脚间距 | 需要标定 |
| `nominal_step_width` | `0.26` | `instinctlab_foothold/flat_provider.py` | 当前实验期望步宽；active G1 shoe URDF 静态链路粗算左右 sole 中心距约 `0.237 m`，当前值略高于静态几何值 | reset 初始站姿左右脚中心距离，或 gait/reference 中的默认脚宽 | 需要标定 |

当前边界：`min_lateral_separation=0.06` 仍只属于解析平地名义目标生成器的临时参数；学习式落点的几何有效性不再使用它作左右脚硬拒绝。学习式目标只检查有限性、世界地形高度、最大步高和可达椭圆，横向偏离通过名义点距离代价和真实安全评分学习。
| `flat_target_lookahead_phase` | `0.8` | `sensors/foothold_planner/foothold_planner_cfg.py` | 临时标定比例：预计在 swing phase 的 80% 处触地 | 真实 touchdown phase 分布、`last_air_time / swing_duration_s` 分布 | 需要标定 |
| `velocity_lookahead_s` | `flat_target_lookahead_phase * swing_duration_s = 0.256` | `sensors/foothold_planner/foothold_planner.py` | 由 planner 标称摆动时间推导，不再使用独立 `0.10` 临时值 | `flat_target_lookahead_phase` 的标定结果 | 临时已对齐，仍需验证 |
| `curriculum_radius_x` | `(0.0, 0.0, 0.0)` | `instinctlab_foothold/flat_provider.py` | 已取消 XY 随机课程，仅保留兼容字段 | 名义落点不再随等级随机偏移 | 已停用 |
| `curriculum_radius_y` | `(0.0, 0.0, 0.0)` | `instinctlab_foothold/flat_provider.py` | 已取消 XY 随机课程，仅保留兼容字段 | 名义落点不再随等级随机偏移 | 已停用 |
| `curriculum_yaw_limit_rad` | `(0.0, 0.10, 0.20)` | `instinctlab_foothold/flat_provider.py` | 训练课程设计 | yaw 命令范围、策略转向能力和落足误差 | 可保留，但需监控 |
| `swing_duration_s` | `0.32` | `sensors/foothold_planner/foothold_planner_cfg.py` | planner 设定 | TensorBoard 的 `mean_swing_duration_steps` × `step_dt`，以及接触传感器 `last_air_time` | 需要对齐 |
| `contact_confirm_s` | `0.04` | `sensors/foothold_planner/foothold_planner_cfg.py` | 接触去抖设定 | 环境控制步长和 contact sensor 噪声 | 基本合理，但需记录 dt |
| `early_contact_phase` | `0.65` | `sensors/foothold_planner/foothold_planner_cfg.py` | planner 设定 | 真实 touchdown phase 分布 | 需要统计验证 |
| `overdue_s` | `0.12` | `sensors/foothold_planner/foothold_planner_cfg.py` | planner 设定 | swing duration 和实际 touchdown 延迟分布 | 需要统计验证 |
| `step_hold_s` | `0.04` | `sensors/foothold_planner/foothold_planner_cfg.py` | 双支撑缓冲设定 | 控制步长、速度和触地稳定时间 | 需要和速度动态规则一起验证 |
| `step_hold_velocity_scale_s_per_mps` | `0.02` | `sensors/foothold_planner/foothold_planner_cfg.py` | 速度相关双支撑启发式 | 速度越高双支撑越短的实际需求 | 需要统计验证 |
| `recovery_step_length_m` | `0.04` | `sensors/foothold_planner/foothold_planner_cfg.py` | 保守恢复步设定 | recovery 后真实稳定率和速度偏差 | 需要统计验证 |
| `recovery_step_velocity_lookahead_s` | `0.10` | `sensors/foothold_planner/foothold_planner_cfg.py` | 保守恢复步设定 | recovery 阶段实际可控时间窗 | 需要标定 |
| `recovery_step_max_length_m` | `0.12` | `sensors/foothold_planner/foothold_planner_cfg.py` | 保守恢复步设定 | 运动学可达范围和稳定性 | 需要标定 |
| `recovery_step_width_m` | `0.26` | `sensors/foothold_planner/foothold_planner_cfg.py` | 与当前实验 `nominal_step_width` 保持一致；尚未标定 | reset 初始站姿左右脚中心距离，或 recovery 阶段更保守的同源缩放值 | 需要和 `nominal_step_width` 合并来源 |
| `sole_center_offset_b` | `(0.039, 0.0, -0.058)` | `sensors/foothold_planner/foothold_planner_cfg.py` | active G1 shoe URDF 脚底接触圆柱外包络：x 中心 `0.039`，底面 z `-0.058` | 换机器人/换鞋版 URDF 时重新解析 ankle roll link 下的 foot contact collisions | 已对齐 |
| `sole_half_length` | `0.093` | `sensors/foothold_planner/foothold_planner_cfg.py` | active G1 shoe URDF 外包络 x 范围 `[-0.054, 0.132]`，长度 `0.186` | 换机器人/换鞋版 URDF 时重新解析 | 已对齐 |
| `sole_half_width` | `0.036` | `sensors/foothold_planner/foothold_planner_cfg.py` | active G1 shoe URDF 外包络 y 范围 `[-0.036, 0.036]`，宽度 `0.072` | 换机器人/换鞋版 URDF 时重新解析 | 已对齐 |
| `safe_target_search_radii_m` | `(0.025, 0.05, 0.075, 0.10)` | `sensors/foothold_planner/foothold_planner_cfg.py` | 候选搜索设计 | 障碍尺寸、脚底 footprint、可达椭圆 | 可保留，但需用可达范围约束 |
| `safe_target_foot_length_m` | `0.186` | `sensors/foothold_planner/foothold_planner_cfg.py` | 和 active G1 shoe URDF 脚底外包络长度一致 | 若后续要加安全余量，应单独记录 margin，不混进物理脚长 | 已对齐 |
| `safe_target_foot_width_m` | `0.072` | `sensors/foothold_planner/foothold_planner_cfg.py` | 和 active G1 shoe URDF 脚底外包络宽度一致 | 若后续要加安全余量，应单独记录 margin，不混进物理脚宽 | 已对齐 |
| `swing_apex_height_m` | `0.08` | `sensors/foothold_planner/foothold_planner_cfg.py` | planner 默认抬脚高度 | 原始 gait/策略脚高、地形高度和 clearance 需求 | 需要统计验证 |
| `clearance_max_apex_height_m` | `0.14` | `sensors/foothold_planner/foothold_planner_cfg.py` | clearance 上限 | 机器人抬脚运动学能力和训练稳定性 | 临时固定，仍需标定 |
| `clearance_apex_step_m` | `0.03` | `sensors/foothold_planner/foothold_planner_cfg.py` | clearance 搜索分辨率 | 地形高度分辨率和控制平滑性 | 可保留，但需说明 |
| `clearance_sample_spacing_m` | `0.03` | `sensors/foothold_planner/foothold_planner_cfg.py` | clearance 采样分辨率 | 障碍尺寸、footprint 和地形网格分辨率 | 可保留，但需说明 |
| `touchdown_xy_tolerance_m` | `0.08` | `sensors/foothold_planner/foothold_planner_cfg.py` | 监控/奖励容忍范围 | 训练后落足误差分布 | 需要统计验证 |
| `touchdown_z_tolerance_m` | `0.06` | `sensors/foothold_planner/foothold_planner_cfg.py` | 监控/奖励容忍范围 | 地形高度误差和接触传感器噪声 | 需要统计验证 |

## 3. 当前已有可用证据

### 3.1 命令速度范围

`parkour_env_cfg.py` 里 `base_velocity` 的训练速度范围已经定义：

- 平地/粗糙地形前向速度通常在 `0.45 ~ 1.0 m/s` 或 `0.45 ~ 0.8 m/s`；
- 侧向速度当前为 `0.0`；
- yaw 速度通常在 `-1.0 ~ 1.0 rad/s`；
- 站立地形 `perlin_rough_stand` 是零速度命令。

这说明 flat target 的前向落点应该主要由前向速度和 swing 时间决定，侧向目标不应该因为速度命令产生大偏移。

### 3.2 接触传感器支持 air time

`contact_forces` 已经开启 `track_air_time=True`。原始 reward 里也有 `feet_air_time`，它使用 `last_air_time` 在首次触地时奖励脚在空中的时间。

因此 `velocity_lookahead_s` 不应该长期使用 `0.10` 这种独立临时值。当前已经改为从 planner 的摆动时序推导：

```text
velocity_lookahead_s = flat_target_lookahead_phase * swing_duration_s
                     = 0.8 * 0.32
                     = 0.256 s
```

其中 `flat_target_lookahead_phase=0.8` 仍然是临时标定比例，表示 planner 假设脚通常在 swing phase 的 80% 附近触地。它不是物理常数，后续应由：

1. touchdown 发生时的 phase 分布；
2. contact sensor 的 `last_air_time / swing_duration_s` 分布；
3. TensorBoard monitor 里的 `foothold_planner_mean_swing_duration_steps`;

共同确认。

### 3.3 monitor 已经有部分标定指标

当前 TensorBoard/inspect 脚本已经包含：

- `foothold_planner_mean_swing_duration_steps`；
- `foothold_planner_reward_curriculum_scale`；
- 左右脚 swing entry rate；
- early contact / overdue / stance lost / recovery per swing entry；
- safe target candidate inside ellipse / valid / fallback 相关指标。

这些可以支持两件事：

1. 判断 planner 时序是否和策略真实接触节奏一致；
2. 判断可达椭圆是不是过小或过大。

### 3.4 play debug 已经补充即时诊断

当前 play debug 会打印：

- `air_time_s`；
- `last_air_time_s`；
- `contact_time_s`；
- `swing_air_time_s`；
- `flat_level`；
- `lookahead_s`；
- `target_delta_f`；
- `ellipse_max_x`；
- `ellipse_usage`。
- `left_sole_w` / `right_sole_w`；
- `sole_width_y_w` / `sole_width_xy_w`；
- `planned_width_f`。
- `actual_delta_f`；
- `actual_width_f`；
- `actual_minus_planned_width_f`。

这些字段可以直接用来判断：

- planner 认为的 swing 和真实离地时间是否一致；
- 当前落足点是否经常靠近或超过可达椭圆边界；
- 规划点偏短是否来自 `velocity_lookahead_s` 太小；
- 策略真实落点是否系统性超出规划点。
- 当前策略真实脚宽和 planner 规划脚宽是否系统性不一致。

其中 `sole_width_y_w` 是世界坐标 y 方向宽度，只适合粗看；机器人 yaw 或前后错步都会污染这个数。真正用于标定 `nominal_step_width` / `recovery_step_width_m` 的主字段应该是 `actual_width_f` 和 `actual_minus_planned_width_f`，它们会把真实 swing-minus-stance 脚间向量投影回 planner 生成 `target_f` 时使用的支撑脚局部 frame。

### 3.5 active G1 shoe URDF 的脚底外包络

当前 parkour G1 配置通过 `ShoeConfigMixin` 使用：

```text
source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf
```

这个 URDF 里，左右 `ankle_roll_link` 下的鞋底由 7 个横向圆柱 collision 组成。解析这些 foot-contact collisions 后得到相对 ankle roll link 的外包络：

```text
x range = [-0.054, 0.132] m, length = 0.186 m, center x = 0.039 m
y range = [-0.036, 0.036] m, width  = 0.072 m, center y = 0.000 m
z range = [-0.058, -0.038] m, bottom z = -0.058 m
```

因此当前 foothold planner 的脚底中心和 footprint 已改为：

```text
sole_center_offset_b = (0.039, 0.0, -0.058)
sole_half_length = 0.093
sole_half_width = 0.036
safe_target_foot_length_m = 0.186
safe_target_foot_width_m = 0.072
```

注意：这里的 `safe_target_foot_*` 现在表示真实鞋底外包络，不包含额外安全余量。如果后续需要更保守的避障，应新增或使用单独的 margin 参数，而不是把物理脚长/脚宽偷偷放大。

### 3.6 active G1 shoe URDF 的静态脚宽粗算

从 active G1 shoe URDF 的腿部关节 origin 粗算，左右 ankle/sole 中心横向位置大约是：

```text
left sole center y  ≈ +0.1185 m
right sole center y ≈ -0.1185 m
left-right width    ≈  0.2370 m
```

这不是最终 `nominal_step_width` 的标定值，因为实际 reset 后的关节姿态、motion reference 和策略站姿都会影响真实脚宽。当前实验值 `nominal_step_width=0.26` 和 `recovery_step_width_m=0.26` 略高于该静态几何粗算值，必须通过正常 touchdown 数据继续验证，不能视为已经完成标定。

下一步应该用运行时统计确认：

1. reset 后两脚 sole center 的横向距离分布；
2. 正常行走 touchdown 后左右脚中心距离分布；
3. recovery touchdown 后左右脚中心距离分布。

确认后再决定 `nominal_step_width` 和 `recovery_step_width_m` 是否同值，或者 recovery 是否需要在同源脚宽基础上做保守缩放。

当前 analyzer 已经会统计：

```text
sole_width_y_w
sole_width_xy_w
planned_width_f
actual_minus_planned_width_y_w
actual_width_f
actual_minus_planned_width_f
```

因此可以先用 play log 判断：如果 `actual_minus_planned_width_f` 长期为正，说明策略在 planner frame 下的真实脚宽比目标宽；如果长期为负，说明 planner 目标比策略真实脚宽更宽。`actual_minus_planned_width_y_w` 仅保留为世界坐标粗略参考，不作为最终标定依据。

注意：完整 play log 会混入 reset、HOLD、目标未初始化、recovery 和摔倒前异常样本。参数标定时优先看 analyzer 输出中的 `calibration_subset`，它只保留 planner 有效、非 zero-action、非 recovery、正常 swing、规划目标非零且 touchdown 误差不过大的样本。全量统计用于排查稳定性，`calibration_subset` 才用于决定是否改 `nominal_step_width`、`flat_target_lookahead_phase` 和椭圆/课程半径。

`analyze_foothold_play_log.py` 还会输出课程残差使用率：

```text
curriculum_residual_x_f / curriculum_residual_y_f
curriculum_usage_x / curriculum_usage_y / curriculum_usage_norm
```

新日志会直接打印 planner 生成 flat target 同一帧的真值：

```text
curriculum_residual_f
curriculum_radius_f
curriculum_usage
```

因此新日志里的课程使用率不再依赖当前 `command` 反推，可以直接用于判断课程半径是否过大、过小或被长期闲置。老日志没有这些 direct 字段时，analyzer 仍会用当前 `command` 做近似反推；这类结果只适合粗看趋势，不作为最终标定依据。

## 4. 标定优先级

### 4.1 第一优先级：时间尺度

先标定：

- `velocity_lookahead_s`；
- `swing_duration_s`；
- `flat_target_lookahead_phase`；
- `early_contact_phase`；
- `overdue_s`。

原因：如果时间尺度错了，落足点前向距离会系统性偏短或偏长。你现在观察到“机器人真实脚步比规划点更远”，最可能首先要查这一组参数。

建议流程：

1. 用当前最好的 checkpoint play；
2. 收集 `last_air_time_s`、touchdown 时的 `phase`、`mean_swing_duration_steps`；
3. 计算 p50 / p75 / p90；
4. 用 touchdown phase 分布来决定 `flat_target_lookahead_phase`；
5. 再根据 touchdown phase 分布调整 `early_contact_phase` 和 `overdue_s`。

### 4.2 第二优先级：脚宽和 footprint

然后标定：

- `nominal_step_width`；
- `recovery_step_width_m`;
- `sole_center_offset_b`;
- `sole_half_length`;
- `sole_half_width`;
- `safe_target_foot_length_m`;
- `safe_target_foot_width_m`。

原因：这些决定左右脚目标是否过宽、过窄，以及危险圆柱/边缘检查是否和真实脚掌对齐。

建议来源：

1. reset 后左右 ankle/sole 中心的世界坐标；
2. G1 机器人模型的脚底尺寸；
3. play debug 中脚掌 marker 和实际脚掌的视觉对齐。

### 4.3 第三优先级：可达椭圆

再标定：

- `outer_radius_x`;
- `outer_radius_y`;
- `recovery_step_max_length_m`。

原因：这些是物理边界。它们不应该由训练课程决定，而应该来自机器人腿部可达能力。

建议来源：

1. 离线运动学扫描；
2. 关节限制下的足端可达点云；
3. 保守取可达点云内部区域，不取边界极限。

### 4.4 第四优先级：课程扰动半径

最后调整：

- `curriculum_radius_x`;
- `curriculum_radius_y`;
- `curriculum_yaw_limit_rad`。

原因：这些是训练设计参数，不是机器人物理参数。它们应该在上面三类参数标定后，再根据训练效果微调。

原则：

- 最大课程档不能让目标越过可达椭圆；
- 低档应该明显比完整难度容易；
- yaw 限制应该和命令 yaw 范围、策略转向能力一致；
- 不应该用课程扰动去补偿错误的 `velocity_lookahead_s`。

## 5. 换机器人模型时必须重新检查的清单

换机器人后，至少要重新确认：

1. 左右 ankle/contact body 名称；
2. base body 名称；
3. contact sensor body 选择；
4. 脚底中心 offset；
5. 脚底长度和宽度；
6. 默认左右脚宽；
7. 支撑脚坐标系下的腿部可达椭圆；
8. swing air time 分布；
9. touchdown phase 分布；
10. recovery 最大安全步长；
11. safe target footprint 是否覆盖真实脚掌；
12. clearance 最大抬脚高度是否在运动学可达范围内。

## 6. 下一步建议

下一步不要再直接改独立 `velocity_lookahead_s`。当前 `velocity_lookahead_s` 已经由 `flat_target_lookahead_phase * swing_duration_s` 推导。外椭圆横向半轴已经作为实验边界放宽到 `0.50 m`，接下来应该继续用 play debug 或 TensorBoard 标定以下内容：

1. touchdown 时 `phase` 的 p50、p75、p90，用于确认 `flat_target_lookahead_phase=0.8`；
2. `last_air_time_s / swing_duration_s` 的分布，用于交叉验证 lookahead phase；
3. `mean_swing_duration_steps × step_dt`；
4. `target_delta_f` 和实际落足误差的分布；
5. `ellipse_usage` 分布；
6. 左右脚分别的统计，防止左右腿再次失衡。

拿到这些数之后，再更新参数，并在这个文档里把“当前值”和“来源”从临时值改成标定值。

当前已有 play debug 日志分析工具：

```bash
python tests/parkour/foothold/analyze_foothold_play_log.py \
  logs/foothold_play_debug_fullgate_30000it.txt
```

如果要保存 play 日志，可以这样运行：

```bash
./play.sh 2>&1 | tee logs/foothold_play_debug_fullgate_30000it.txt
```

工具会输出：

- 当前日志里的 `lookahead_s`；
- 根据完整 `last_air_time_s` 中位数给出的参考 lookahead；
- `swing_air_time_s`、`last_air_time_s`、`target_delta_f`、`ellipse_usage`、`ref_xy_err`、`td_xy_err` 的统计；
- 左右脚分开统计，用于检查左右腿是否再次失衡。

## 7. Event-SAC planner 基线（2026-09-04）

本节只描述学习式落足 planner 的 SAC，不改变电机 PPO、AMP、MoE 或状态机。
planner 的动作仍是名义落足点周围的二维归一化残差，物理解码范围为
`x=0.12 m`、`y=0.10 m`。

| 项目 | 基线值 | 说明 |
| --- | --- | --- |
| replay 容量 | `100000` | 只保存完整落足事件转移 |
| batch | `256` | 事件级 SAC 小批量 |
| warmup | `10000` 个事件 | 总事件数达到后才更新 |
| unsafe warmup | `512` 个事件 | 防止只用平地安全事件启动 |
| 事件采样比 | `0.5` | 每个新事件平均贡献约 0.5 个梯度更新 |
| 单轮更新上限 | `24` | 超出部分保留为 update credit，不丢失 |
| actor/target 更新频率 | 每 `2` 次 critic 更新 | 降低稀疏事件下的 actor 抖动 |
| actor/critic/alpha 学习率 | `1e-4` | planner 专用，不影响电机 PPO |
| 折扣因子 | `0.95` | 落足事件的短期后果更重要 |
| 目标网络软更新 | `tau=0.005` | SAC 常用稳定值 |
| 初始 alpha | `0.05` | 二维落点动作的初始探索温度 |
| 目标熵 | `-0.5` | 不按电机 29 维动作数设置 |
| 安全名义锚定 | `0.25` | 只对“名义点安全”事件约束残差接近零 |

SAC 回放额外保存“名义点安全/不安全”分支，并在两类样本都存在时按约
`1:1` 抽样。事件奖励先按环境事件掩码清零，再进入回放；奖励必须保持在
`[-1,1]`，否则立即报错，避免把别的环境的分数或重复加权带入 planner。

planner 的高斯标准差改用残差解码范围归一化，而不是可达椭圆半径：

```text
初始物理标准差 = (0.025 m, 0.020 m)
最小物理标准差 = (0.005 m, 0.005 m)
最大物理标准差 = (0.040 m, 0.040 m)
```

因此初始归一化标准差约为 `(0.208, 0.200)`。标准差由 planner 特征给出，
并在上述物理范围对应的归一化上下界内裁剪；电机动作分布仍使用原有 PPO
路径。新增的 `foothold_log_std_actor` 只属于 planner actor 参数组。

旧的 SAC 回放/critic 状态不能直接续接到这版：事件奖励掩码、分支标签、
动作探索尺度都已改变。加载版本低于 3 的 checkpoint 时只保留 planner
actor 权重，并重新建立 SAC replay、critic、温度和优化器状态。
