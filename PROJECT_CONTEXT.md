# InstinctLab-Foothold 项目上下文

这份文件是给第一次接触本项目的开发者或 AI 的入口说明。开始修改代码前，先阅读本文件，再阅读：

1. `README.md`
2. `docs/foothold_planner_implementation.md`
3. `docs/foothold_parameter_audit.md`
4. 与当前任务相关的 `docs/superpowers/specs/` 设计文档

本文件描述当前 Git 分支的实现，不是历史聊天记录，也不代表所有旧实验都仍然有效。

## 1. 项目目标

本项目是 Project-Instinct 生态中的 IsaacLab 环境侧代码，当前重点是 Unitree G1 parkour/AMP 行走任务：

- 在平地、台阶和其他 parkour 地形上训练全身动作策略；
- 保留显式落足点规划器，明确输出下一只摆动脚的落足目标；
- 使用解析规划器提供行走意图和名义落点；
- 使用学习式落足 planner 根据机器人状态和深度观测输出最终水平落点；
- 使用原有动作策略跟踪由落足点生成的摆动轨迹；
- 在训练阶段用危险圆柱、脚掌外围侵入、可达性和地形高度等特权信息提供奖励/诊断；这些仿真特权量不应直接作为实机部署输入。

当前目标不是重新设计 AMP、MoE 或整个 locomotion 框架，而是在其上增加并验证显式落足规划能力。

## 2. 当前代码版本和分支

当前分支：

```text
feat/foothold-01-flat-tracking
```

开始工作前执行：

```bash
git status -sb
git log -3 --oneline
```

不要根据某个旧日志目录名推断当前代码版本。代码、训练脚本、测试和文档的实际状态以 Git 为准。

## 3. 外部依赖

本仓库不是完整的 IsaacLab 和 Instinct-RL 源码镜像。运行训练至少需要：

- Isaac Sim 5.1.0；
- IsaacLab 对应提交：`f73c331738`；
- Python 3.11、匹配的 PyTorch/CUDA 环境；
- 外部 Instinct-RL；当前复现实验使用的提交：
  `f870ead0953fa0e3c3da3349b0aece1c74bfb421`；
- parkour motion reference 数据集及其筛选 YAML。

Instinct-RL 通常位于本仓库旁边：

```text
~/IsaacLab
~/instinct_rl
~/InstinctLab-foothold
```

Instinct-RL 没有被本项目复制进来，也没有修改外部仓库。项目内的学习扩展通过继承外部 Instinct-RL 的 PPO、MoE、RolloutStorage 等基础类实现。换电脑时必须安装与上述提交兼容的 Instinct-RL；只克隆本仓库而不安装外部依赖不能运行训练。

安装外部 Instinct-RL：

```bash
git clone https://github.com/project-instinct/instinct_rl.git
cd instinct_rl
git checkout f870ead0953fa0e3c3da3349b0aece1c74bfb421
python -m pip install -e .
```

安装本项目：

```bash
cd ~/InstinctLab-foothold
python -m pip install -e source/instinctlab
```

## 4. 数据准备

运动参考数据不在 Git 仓库中，需要准备：

```text
<运动参考目录>/parkour_motion_without_run.yaml
```

推荐通过环境变量指定：

```bash
export PARKOUR_MOTION_REFERENCE_DIR="/实际路径/parkour_motion_reference"
export PARKOUR_MOTION_SELECTION_FILE="\${PARKOUR_MOTION_REFERENCE_DIR}/parkour_motion_without_run.yaml"
```

路径中可能包含 `&`。不要把它作为未加引号的 Hydra override 直接传入；`scripts/foothold_train.sh` 和 `scripts/foothold_play_step.sh` 已经负责安全地传递路径。

## 5. 当前架构

### 5.1 解析规划器

主要文件：

- `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`
- `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_cfg.py`
- `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`
- `source/instinctlab/instinctlab_foothold/target_search.py`
- `source/instinctlab/instinctlab_foothold/state_machine.py`
- `source/instinctlab/instinctlab_foothold/clearance.py`

解析规划器负责支撑脚/摆动脚关系、名义落点、世界坐标地形高度、摆动轨迹、状态机事务和安全/轨迹诊断。正常流程是：

```text
稳定支撑/HOLD
  → 冻结支撑脚世界位置和身体朝向
  → 生成名义落足点
  → 学习式 planner 提案最终水平落点
  → 世界坐标查询地形高度得到 z
  → 进行几何/可达性/轨迹检查
  → 锁定落点和支撑坐标系
  → 生成解析摆动轨迹
  → 低层动作策略跟踪轨迹
```

坐标约定是硬约束：

1. 学习式 planner 输出支撑脚 planner frame 下的 `x_f, y_f`；
2. 先把局部水平点变换到世界坐标 `x_w, y_w`；
3. 地形查询只使用世界坐标；
4. 查询结果给出世界高度 `z_w`；
5. 最终世界落点是 `(x_w, y_w, z_w)`，需要局部表示时再转换回支撑脚坐标系。

进入 SWING 后，支撑脚世界坐标、身体朝向快照、轨迹起点和落足终点都锁定；不能用新的支撑坐标系重新解释旧目标。

### 5.2 学习式 planner 网络

主要文件：

- `source/instinctlab/instinctlab/learning/independent_foothold_actor_critic.py`
- `source/instinctlab/instinctlab/learning/foothold_depth_encoder.py`
- `source/instinctlab/instinctlab/learning/foothold_rollout_storage.py`
- `source/instinctlab/instinctlab/learning/foothold_checkpoint.py`
- `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`

开启学习式 planner 后，动作分成：

```text
29维：原有电机/关节动作
  +
2维：学习式落足水平输出
```

学习式 planner 的输入包括名义落点、机器人状态和 planner 专用深度特征；它直接表达最终水平落足意图，不直接预测地形高度。高度仍由世界坐标地形查询模块补齐。

当前 planner 结构大致为：

```text
深度图 → planner 专用小型编码器 → 64维特征
机器人状态/名义点 + 64维深度特征
  → [128, 64] 落足 MLP
  → 2维水平落足输出
```

正常行走时，准备好且几何有效的学习式提案可以被执行；名义点主要承担行走意图/先验和不可用时的 fallback。Recovery 阶段禁用学习式落点，保留保守解析恢复路径。

危险圆柱侵入、侵入点数量、总侵入深度等主要是训练和仿真诊断信号；它们不能被假设为实机可直接获得的传感器输入。几何有效性、地形高度、可达性和轨迹 clearance 是执行侧硬检查，安全分本身不应被误解为“网络一定已经学会安全”。

### 5.3 MoE 电机策略

MoE 是原有 Instinct-RL 电机策略的结构，不是新增 planner 的专用网络。配置位置：

`source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`

当前默认配置包含 4 个专家、隐藏层 `[256, 128, 64]`。门控网络根据当前观测计算专家权重，再将专家输出加权混合为 29 维电机动作。专家没有人工固定的“平地/上楼/下楼”编号，具体分化由训练形成。

MoE 不是时间记忆网络，也不等同于 LSTM/GRU。它增加的是行为表达和条件分工能力；是否比单一 MLP 更好，需要公平对照实验验证，不能仅凭专家编号或“记忆性”下结论。

当前项目没有直接修改外部 Instinct-RL 的 MoE 实现；项目内只通过配置和独立 planner 扩展使用它。

## 6. 状态机和异常边界

状态机实现见 `source/instinctlab/instinctlab_foothold/state_machine.py`，完整说明见 `docs/foothold_planner_implementation.md`。

核心原则：

- startup/reset HOLD 期间不提前执行摆动；
- 规划未准备完成不能进入 SWING；
- 已进入 SWING 后不替换锁定落点；
- 触地验收结合实际接触、此前确实离地以及现有 XY/Z 容差；
- 单脚支撑时保留真实支撑脚关系，不把另一只脚误当支撑；
- 双脚都不稳定时进入恢复/稳定流程；
- Recovery 不使用学习式 planner，恢复稳定后重新建立新 HOLD 和新坐标系；
- 不要通过随意缩短/放大异常阈值来掩盖跟踪或接触问题。

当前配置中的重要示例值可在配置文件中核对，不能只相信本文件：摆动时长约 `0.32 s`，训练任务中 startup/reset hold 为 `0.15 s`，最大落足高度差为 `0.25 m`，触地监控容差为 XY `0.08 m`、Z `0.06 m`。2 cm 是高精度奖励目标，不等于状态机唯一触地判定门槛。

## 7. 训练入口

### 7.1 先做短训验证

长训前先确认依赖、任务注册和显存：

```bash
cd ~/InstinctLab-foothold

ENABLE_LEARNED_FOOTHOLD_PLANNER=1 \
RUN_NAME=planner_depth_encoder_4096env_100it \
NUM_ENVS=4096 \
MAX_ITERATIONS=100 \
SAVE_INTERVAL=50 \
./scripts/foothold_train.sh 2>&1 | tee logs/planner_depth_encoder_4096env_100it.txt
```

### 7.2 当前学习式 planner 长训命令

确认短训和测试正常后：

```bash
cd ~/InstinctLab-foothold

ENABLE_LEARNED_FOOTHOLD_PLANNER=1 \
RUN_NAME=planner_depth_encoder_4096env_30000it \
NUM_ENVS=4096 \
MAX_ITERATIONS=30000 \
SAVE_INTERVAL=2000 \
./scripts/foothold_train.sh 2>&1 | tee logs/planner_depth_encoder_4096env_30000it.txt
```

这条命令从新的训练初始化开始，不继承旧 checkpoint。需要续训时，使用 `--resume --load_run ... --checkpoint model_....pt`，不要把“初始化 planner 权重”和“恢复完整 PPO 状态”混为一谈。

训练脚本会自动：

- 把当前仓库的 `source/instinctlab` 放到 `PYTHONPATH` 前面，避免误 import 另一个 checkout；
- 设置运动参考文件路径；
- 启用学习式 planner 时注册 `EventGatedWasabiPPO`；
- 检查 IsaacLab launcher 和运动参考 YAML 是否存在；
- 按 `SAVE_INTERVAL` 保存 checkpoint。

## 8. Play 和诊断

普通 play 使用 `scripts/instinct_rl/play.py`。台阶专用诊断入口是：

```bash
cd ~/InstinctLab-foothold

LOAD_RUN=<run_name> \
FOOTHOLD_DEBUG_INTERVAL=20 \
FOOTHOLD_DEBUG_ENV_IDS=all \
./scripts/foothold_play_step.sh \
  --checkpoint model_12000.pt \
  --video_length 3000 \
  2>&1 | tee logs/foothold_play_step.txt
```

`scripts/foothold_play_step.sh` 默认只使用台阶地形族，并打开落足调试标记和规划事件打印。它不会修改训练配置。可通过环境变量调整 `STEP_TERRAIN_NAME`、`STEP_TERRAIN_LEVEL`、`NUM_ENVS` 和 `FOOTHOLD_DEBUG_INTERVAL`。

play 时注意：

- `--load_run` 必须匹配实际日志目录名；
- `--checkpoint` 必须是目录中存在的文件，例如 `model_12000.pt`；
- `FOOTHOLD_CURRICULUM_SCALE_OVERRIDE=1.0` 只表示按满课程可视化/评估，不会改变 checkpoint 内策略；
- 不要把低频调试输出误认为规划器没有更新；规划事件和普通控制步不是同一频率；
- 旧 marker 残留、Recovery、HOLD 和计划失败必须结合日志字段区分，不能只看一张画面。

常用日志分析工具：

```bash
python tests/parkour/foothold/analyze_foothold_play_log.py logs/foothold_play_step.txt
python tests/parkour/foothold/inspect_foothold_tensorboard.py --latest-pattern <run_pattern>
```

## 9. 测试和验证

换机器或修改代码后，先运行 foothold 测试：

```bash
cd ~/InstinctLab-foothold
PYTHONPATH="\$PWD/source/instinctlab:\$PYTHONPATH" \
python -m pytest -q tests/parkour/foothold
```

重点测试范围包括：

- 坐标变换、世界高度查询和地形 provider；
- 脚掌外围安全/危险圆柱侵入和 clearance；
- planner 数据、学习式落足路由、checkpoint 迁移；
- 状态机、接触适应、Recovery；
- reward/observation、play 调试和训练保存间隔。

还应至少检查：

```bash
python -m py_compile \
  source/instinctlab/instinctlab/learning/foothold_depth_encoder.py \
  source/instinctlab/instinctlab/learning/independent_foothold_actor_critic.py \
  source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py \
  source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py
git diff --check
```

## 10. 参数来源和当前限制

参数审计表在 `docs/foothold_parameter_audit.md`。不要把所有数字都当作机器人物理常数：

- 脚底尺寸、脚底中心偏移、运动学可达范围：应来自当前机器人 URDF/运动学；换机器人必须重新检查；
- 摆动时间、接触确认和 Recovery 时序：应通过策略实际接触/air-time 日志标定；
- 课程半径和课程 episode 门槛：是训练设计参数，不是物理边界；
- 深度图分辨率、归一化和特征维度：是感知/网络接口参数，修改会影响 checkpoint 兼容性；
- 危险圆柱和 penetration 量：是仿真训练特权诊断，不能假设实机有同样的几何真值。

当前训练迭代包含 collection 和 PPO learning 两部分，总耗时受 IsaacLab/物理仿真、重置、planner 事件、深度编码器和 GPU 状态影响。不要为了追求更短迭代时间随意删除安全检查、状态机锁定或坐标转换；任何性能优化都应先做 4096 环境 A/B 测试并确认逻辑等价。

当前版本仍需要通过 play 检查上下楼梯、平地、Recovery 比例、摆动轨迹跟踪和学习式落点采用率。训练曲线变好不等于落足点已经在所有地形上安全。

## 11. 给新 AI 的工作规则

新 AI 开始工作时应遵循：

1. 先读本文件、实现文档和参数审计，再看最近 Git 提交；
2. 先运行 foothold 测试和最小 smoke test，再提出修改；
3. 不要直接修改 `/home/zhangweibo/instinct_rl`；需要扩展时优先放在本仓库 `source/instinctlab/instinctlab/learning/`；
4. 不要修改 AMP、locomotion、原始 MoE 或外部 IsaacLab 参数来掩盖 planner 问题；
5. 任何坐标、时序、接触、Recovery 修改都要同时检查训练、play、日志和测试；
6. 不要把旧实验命令或旧 checkpoint 当作当前默认方案；
7. 修改前说明原因和影响范围，修改后提供验证命令和实际输出；
8. 长训前先做短训，保存 checkpoint，并记录运行目录、依赖提交和运动数据配置；
9. 未经明确授权不要执行 `git push`，也不要把日志、模型、个人脚本和数据集提交进仓库。

## 12. 当前已知的非阻塞问题

- 学习式 planner 事件天然比普通电机控制步稀疏，planner 指标不能按控制步频率直接解释；
- 4096 环境长训单次迭代可能约 8 秒，必须分别看 collection 和 learning 时间；
- Recovery、提前触地、轨迹 clearance 失败需要通过 play 日志分相位诊断；
- MoE 是否优于单一 MLP 尚未通过严格对照实验确认；
- 运动参考数据、IsaacLab、Instinct-RL 和训练 checkpoint 不由本仓库自动提供。

如果这些约束或依赖发生变化，先更新本文件和对应实现文档，再开始新的长训。


