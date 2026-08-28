# Project Instinct

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Overview

This repository is the environment side of [Project-Instinct](https://project-instinct.github.io/).

We aim at industralize Reinforcement Learning for Humanoid (legged robots) whole-body control.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.
- `Unified Ecosystem` This repository is a part of the Project-Instinct ecosystem, which includes the [instinct_rl](https://github.com/project-instinct/instinct_rl) and [instinct_onboard](https://github.com/project-instinct/instinct_onboard) repositories.
    - The core design of this ecosystem is to treat each experiment as a standalone structured folder, which start with a timestamp as a unique identifier.
    - Adding `--exportonnx` flag to the `play.py` script will export the policy as an ONNX model. After that, you should directly copy the logdir to the robot computer and use the `instinct_onboard` workflow to run the policy on the real robot.

**Keywords:** extension, template, isaaclab

## Warning
This codebase is under [CC BY-NC 4.0 license](LICENSE), with inherited license in IsaacLab. You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products or wrap the code for your own commercial purposes.

## Contributing
See our [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md) for contribution guidelines. By contributing or submitting a pull request, you agree to transfer copyright ownership of your contributions to the project maintainers.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of acknowledged contributors.

## Installation

### Foothold branch: complete Ubuntu 22.04 setup

This workflow targets Ubuntu 22.04 x86_64 with an NVIDIA GPU. It reproduces
the environment audited for `feat/foothold-01-flat-tracking`; it is not a
Windows or CPU-only installation guide.

The Git repository includes the pinned IsaacLab and Instinct-RL source trees
as submodules. It does **not** include Isaac Sim, the Conda environment,
Parkour motion data, checkpoints, or logs.

#### 1. Check prerequisites

Install Conda before continuing. Confirm the operating system, NVIDIA driver,
Conda, and Git are visible:

```bash
lsb_release -a
nvidia-smi
conda --version
git --version
```

Stop here if `nvidia-smi` or `conda` fails. NVIDIA publishes the Isaac Sim 5.1
[system and installation requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html).

#### 2. Clone the project and pinned source dependencies

```bash
git clone --recurse-submodules \
  --branch feat/foothold-01-flat-tracking \
  git@github.com:wbZ-123/InstinctLab.git \
  InstinctLab-foothold
cd InstinctLab-foothold
git submodule status
```

Expected submodule commits:

```text
f73c33173801f5f8afea4142482e47b7710c2b75 third_party/IsaacLab
f870ead0953fa0e3c3da3349b0aece1c74bfb421 third_party/instinct_rl
```

If the project was cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

#### 3. Create the `hiking` Conda environment

```bash
conda create -n hiking python=3.11 pip -y
conda activate hiking
python --version
python -m pip install --upgrade pip "setuptools<82.0.0" wheel
```

If `hiking` already exists, do not delete it automatically. Activate it and
confirm that `python --version` reports Python 3.11 before reusing it.

#### 4. Install the audited PyTorch CUDA build

```bash
python -m pip install \
  torch==2.7.0 \
  torchvision==0.22.0 \
  torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify the package and GPU runtime:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("wheel CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

Expected package versions are PyTorch `2.7.0+cu128`, torchvision
`0.22.0+cu128`, and torchaudio `2.7.0+cu128`. Resolve the driver or wheel
installation before continuing if `CUDA available` is `False` on the target
GPU machine.

#### 5. Install Isaac Sim 5.1

Read and accept the applicable NVIDIA license terms before installing. Follow
the official [Isaac Sim 5.1 Python environment instructions](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html):

```bash
python -m pip install \
  "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
```

Verify the installed version:

```bash
python -c 'from importlib.metadata import version; print(version("isaacsim"))'
```

Expected: `5.1.0.0`.

#### 6. Install pinned IsaacLab, Instinct-RL, and this project

Keep `hiking` activated and run from the repository root:

```bash
./third_party/IsaacLab/isaaclab.sh -i all
python -m pip install -e third_party/instinct_rl
python -m pip install -e source/instinctlab
python -m pip check
```

The training and stair-Play wrappers use `third_party/IsaacLab` and
`third_party/instinct_rl` by default. `ISAACLAB_ROOT` and `INSTINCT_RL_ROOT`
remain available only as explicit overrides.

#### 7. Configure external Parkour motion data

Copy the motion-reference directory to the new machine. Replace the example
below with its real location; do not use `/replace/with/real/path` literally:

```bash
export PARKOUR_MOTION_REFERENCE_DIR="/replace/with/real/path/parkour_motion_reference"
export PARKOUR_MOTION_SELECTION_FILE="${PARKOUR_MOTION_REFERENCE_DIR}/parkour_motion_without_run.yaml"
```

Paths may contain `&`, so keep the double quotes. The motion data, selection
YAML, checkpoints, logs, and TensorBoard runs are external assets and must be
migrated separately.

#### 8. Verify the installation

```bash
./scripts/bootstrap_foothold.sh --check-only

PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q tests/parkour/foothold
```

Success means both commands return exit code zero. The audited baseline had
444 passing foothold tests, but the exact count may grow with later commits.

#### 9. Run a minimal training startup check

This verifies the launch path only; one iteration does not evaluate policy
quality:

```bash
ENABLE_LEARNED_FOOTHOLD_PLANNER=1 \
RUN_NAME=foothold_install_smoke_1env_1it \
NUM_ENVS=1 \
MAX_ITERATIONS=1 \
SAVE_INTERVAL=1 \
./scripts/foothold_train.sh
```

Play requires a compatible checkpoint copied from the original machine. After
placing its run directory below `logs/instinct_rl/g1_parkour/`, use:

```bash
LOAD_RUN=<run-directory-name> \
./scripts/foothold_play_step.sh --checkpoint model_<iteration>.pt
```

Do not run the Play command with the angle-bracket placeholders unchanged.

#### Troubleshooting

- Empty `third_party/` directories: run `git submodule update --init --recursive`.
- `motion selection file not found`: replace the example motion path with the
  real directory and export both motion variables.
- A motion path containing `&`: keep the complete assignment inside double
  quotes; do not pass the path as an unquoted Hydra override.
- `ModuleNotFoundError: instinctlab.learning`: confirm the current directory is
  this repository, the foothold branch is checked out, submodules are ready,
  and use the provided train/Play wrappers.
- Isaac Sim import failure: activate `hiking` and confirm Isaac Sim reports
  version `5.1.0.0`.
- `torch.cuda.is_available()` is false: check `nvidia-smi` and reinstall the
  cu128 PyTorch wheels rather than CPU wheels.
- First launch is slow: Isaac Sim may populate extension and shader caches.
- Checkpoint not found: `--load_run` takes a run directory name, not a guessed
  arbitrary path, and checkpoints are not stored in Git.

## Documentation of Critical Components

- [Instinct-RL Documentation](https://github.com/project-instinct/instinct_rl/blob/main/README.md)
- [InstinctLab Documentation](https://github.com/project-instinct/instinctlab/blob/main/DOCS.md)

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

To make the `pre-commit` run automatically on every commit, you can use the following command in your repository:

```bash
pre-commit install
```

## Train your own projects

***To preserve your code development and progress. PLEASE create your own repository as an individual project by referring to https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/index.html***

And copy `scripts/instinct_rl` to your own repository.

### Or you are just to stubborn and want to fork and directly modify the code in this repo.

- Please create a new folder in the `source/instinctlab/instinctlab/tasks` directory. The name of the folder should be your project name. Inside the folder, DO add `__init__.py` in each level of the subfolders. (Many people tend to forget this step and could not find the supposely registered tasks.)

- We inherit the manager based RL env from IsaacLab to add new features. DO use `instinctlab.envs:InstinctRlEnv` as the entry_point in the `gym.register` call. For example, if you want to add a new task, you can use the following code:

```python
import gymnasium as gym
from . import agents
task_entry = "instinctlab.tasks.shadowing.perceptive.config.g1"
gym.register(
    id="Instinct-Perceptive-Shadowing-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perceptive_shadowing_cfg:G1PerceptiveShadowingEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/instinctlab"
    ]
}
```
