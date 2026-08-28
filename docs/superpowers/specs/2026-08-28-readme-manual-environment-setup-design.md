# README 手动环境复现说明设计

日期：2026-08-28

## 目标

为 `feat/foothold-01-flat-tracking` 分支补齐面向 Ubuntu 22.04 与 NVIDIA GPU 新电脑的手动环境安装说明。接手人仅依赖 README 中可复制的命令，即可创建 `hiking` Conda 环境、安装固定版本运行依赖、配置外部运动数据并完成最小验收。

本文档只设计 README 修改，不新增自动安装脚本、Conda YAML 或 pip 锁文件。

## 适用范围

- Ubuntu 22.04 x86_64；
- NVIDIA GPU 与可用驱动；
- 已安装 Conda；
- 项目分支 `feat/foothold-01-flat-tracking`；
- Isaac Sim 5.1 pip 安装方式；
- Python 3.11；
- PyTorch 2.7.0 CUDA 12.8 wheel。

不承诺支持 Windows、无 NVIDIA GPU、其他 Ubuntu 版本或其他 Isaac Sim 主版本。

## 固定版本

| 组件 | 版本或提交 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.7.0 + cu128 |
| torchvision | 0.22.0 + cu128 |
| torchaudio | 2.7.0 + cu128 |
| Isaac Sim | 5.1.0 |
| IsaacLab | `f73c33173801f5f8afea4142482e47b7710c2b75` |
| Instinct-RL | `f870ead0953fa0e3c3da3349b0aece1c74bfb421` |

IsaacLab 与 Instinct-RL 由主仓库的 Git submodule 固定，不要求使用者另行寻找或克隆不同版本。

## README 结构

在 Installation 开头增加“Foothold 分支：Ubuntu 22.04 新电脑完整复现”章节。该章节优先于上游 InstinctLab 的通用安装说明，明确两套说明的适用范围，避免接手人重复安装或克隆依赖。

章节顺序如下：

1. 外部内容与前置条件；
2. 驱动、系统和 Conda 检查；
3. 使用 `--recurse-submodules` 克隆；
4. 创建 `hiking` Python 3.11 环境；
5. 安装 PyTorch 2.7.0 CUDA 12.8；
6. 安装 Isaac Sim 5.1；
7. 使用固定子模块安装 IsaacLab；
8. editable 安装 Instinct-RL 与本项目；
9. 配置 Parkour motion reference；
10. 运行环境检查和测试；
11. 执行一轮最小训练和台阶 Play 验证；
12. 常见错误处理。

## 命令设计

### 前置检查

README 给出以下只读命令：

```bash
lsb_release -a
nvidia-smi
conda --version
git --version
```

若 `nvidia-smi` 或 `conda` 不可用，使用者必须先处理驱动或 Conda，不能继续执行项目安装步骤。

### 递归克隆

```bash
git clone --recurse-submodules \
  --branch feat/foothold-01-flat-tracking \
  git@github.com:wbZ-123/InstinctLab.git \
  InstinctLab-foothold
cd InstinctLab-foothold
git submodule status
```

如果已使用普通 clone，则补充：

```bash
git submodule update --init --recursive
```

### Conda 环境

```bash
conda create -n hiking python=3.11 pip -y
conda activate hiking
python -m pip install --upgrade pip "setuptools<82.0.0" wheel
```

README 明确：若同名环境已存在，不执行删除命令；先通过 `python --version` 和包版本检查决定是否复用。

### PyTorch

```bash
python -m pip install \
  torch==2.7.0 \
  torchvision==0.22.0 \
  torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

安装后打印 `torch.__version__`、`torch.version.cuda` 和 `torch.cuda.is_available()`。

### Isaac Sim

根据 NVIDIA Isaac Sim 5.1 官方 Python 环境文档：

```bash
python -m pip install \
  "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
```

README 在命令前提醒使用者阅读并接受 NVIDIA 对应许可，不替使用者自动接受。

官方依据：

- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html
- https://pytorch.org/get-started/previous-versions/

### IsaacLab、Instinct-RL 与项目

在 `hiking` 已激活时：

```bash
./third_party/IsaacLab/isaaclab.sh -i all
python -m pip install -e third_party/instinct_rl
python -m pip install -e source/instinctlab
```

使用固定子模块自身安装器，避免在 README 重复维护 IsaacLab 内部扩展依赖。`all` 与当前环境的完整学习框架安装最接近，虽然耗时更长，但优先保证复现稳定性。

### 运动数据

Git 仓库不包含 Parkour motion reference、筛选 YAML、checkpoint 和训练日志。README 使用明显占位符并注明必须替换：

```bash
export PARKOUR_MOTION_REFERENCE_DIR="/替换为真实路径/parkour_motion_reference"
export PARKOUR_MOTION_SELECTION_FILE="${PARKOUR_MOTION_REFERENCE_DIR}/parkour_motion_without_run.yaml"
```

路径可能包含 `&`，所有变量赋值必须保留双引号。

### 验收

依次执行：

```bash
./scripts/bootstrap_foothold.sh --check-only
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q tests/parkour/foothold
```

README 记录当前基准为 444 项落足测试通过，但不把固定测试数量作为永久成功条件；成功标准是命令退出码为零。

最小训练使用 1 个环境、1 次迭代和独立运行名；其目的仅验证启动链路，不评价策略性能。台阶 Play 必须使用已有兼容 checkpoint，因此 README 明确 checkpoint 不随 Git 提供，没有 checkpoint 时跳过 Play 验收。

## 常见错误

README 至少覆盖：

- 子模块目录为空：运行 `git submodule update --init --recursive`；
- `/实际路径/` 或 `/替换为真实路径/` 被原样使用：改成数据实际位置；
- 路径含 `&`：使用环境变量并保留双引号；
- `ModuleNotFoundError: instinctlab.learning`：确认在仓库根目录、当前分支正确，并通过封装脚本运行；
- Isaac Sim import 失败：确认 `hiking` 已激活且版本为 5.1；
- CUDA 不可用：先检查 `nvidia-smi`，再检查 PyTorch 是否为 cu128 wheel；
- 第一次启动慢：Isaac Sim 首次加载扩展和 Shader 属于预期现象；
- checkpoint 找不到：`--load_run` 使用运行目录名而非随意拼接路径，且 checkpoint 必须单独迁移。

## 不做的工作

- 不新增 `create_hiking_env.sh`；
- 不新增 `environment-hiking.yml`；
- 不新增或维护 318 项 pip freeze 锁文件；
- 不自动安装 Miniconda 或 NVIDIA 驱动；
- 不自动接受 NVIDIA 许可；
- 不删除或重建现有 Conda 环境；
- 不把数据集、checkpoint 或日志加入 Git；
- 不修改训练、Planner、Recovery、奖励或 PPO 逻辑。

## 验证方式

README 修改完成后执行：

1. Markdown 代码块与 shell 命令静态检查；
2. 核对 `.gitmodules` 和固定提交；
3. 在当前 `hiking` 环境运行 `bootstrap_foothold.sh --check-only`；
4. 运行完整落足测试；
5. 检查 README 不再要求为 foothold 分支另行克隆 IsaacLab 或 Instinct-RL；
6. 检查所有示例路径均明确标记为占位符。
