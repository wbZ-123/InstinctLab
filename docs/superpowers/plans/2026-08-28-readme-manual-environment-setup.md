# README Manual Hiking Environment Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incomplete foothold installation notes with a copy-pasteable Ubuntu 22.04 manual setup that creates the `hiking` Conda environment and installs the pinned simulation and learning dependencies.

**Architecture:** Keep environment provisioning documentation-only. A README contract test locks the required versions, submodule workflow, external-data warning, verification commands, and removal of obsolete separate-clone instructions. The existing `bootstrap_foothold.sh` remains a checker and is not expanded into an installer.

**Tech Stack:** Markdown, Bash command examples, pytest static contract tests, Conda, pip, PyTorch CUDA wheels, NVIDIA Isaac Sim pip packages, Git submodules.

## Global Constraints

- Target Ubuntu 22.04 x86_64 with an NVIDIA GPU and an existing Conda installation.
- Use Python 3.11.
- Use PyTorch 2.7.0, torchvision 0.22.0, and torchaudio 2.7.0 from the CUDA 12.8 wheel index.
- Use Isaac Sim 5.1.0 from NVIDIA's pip index.
- Use IsaacLab commit `f73c33173801f5f8afea4142482e47b7710c2b75` through `third_party/IsaacLab`.
- Use Instinct-RL commit `f870ead0953fa0e3c3da3349b0aece1c74bfb421` through `third_party/instinct_rl`.
- Do not add an environment creation script, Conda environment YAML, or pip lock file.
- Do not install Miniconda, NVIDIA drivers, motion data, checkpoints, or logs.
- Do not modify training, Planner, Recovery, reward, or PPO logic.
- Do not silently accept NVIDIA license terms.

---

### Task 1: Replace the foothold installation section with a complete manual workflow

**Files:**
- Create: `tests/parkour/foothold/test_readme_environment_setup.py`
- Modify: `README.md:34-101`

**Interfaces:**
- Consumes: `.gitmodules`, `scripts/bootstrap_foothold.sh`, `scripts/foothold_train.sh`, and `scripts/foothold_play_step.sh`.
- Produces: A stable README contract for new-machine setup; no runtime API changes.

- [ ] **Step 1: Write the failing README contract test**

Create `tests/parkour/foothold/test_readme_environment_setup.py` with:

```python
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_pins_the_verified_hiking_runtime():
    required_fragments = [
        "Ubuntu 22.04",
        "conda create -n hiking python=3.11 pip -y",
        "torch==2.7.0",
        "torchvision==0.22.0",
        "torchaudio==2.7.0",
        "https://download.pytorch.org/whl/cu128",
        'isaacsim[all,extscache]==5.1.0',
        "https://pypi.nvidia.com",
        "./third_party/IsaacLab/isaaclab.sh -i all",
        "python -m pip install -e third_party/instinct_rl",
        "python -m pip install -e source/instinctlab",
    ]
    for fragment in required_fragments:
        assert fragment in README


def test_readme_uses_pinned_submodules_instead_of_separate_dependency_clones():
    assert "git clone --recurse-submodules" in README
    assert "git submodule update --init --recursive" in README
    assert "f73c33173801f5f8afea4142482e47b7710c2b75" in README
    assert "f870ead0953fa0e3c3da3349b0aece1c74bfb421" in README
    assert "git clone https://github.com/project-instinct/instinct_rl.git" not in README


def test_readme_marks_external_assets_and_verification_steps():
    required_fragments = [
        "PARKOUR_MOTION_REFERENCE_DIR",
        "parkour_motion_without_run.yaml",
        "./scripts/bootstrap_foothold.sh --check-only",
        "python -m pytest -q tests/parkour/foothold",
        "checkpoint",
        "logs",
        "&",
    ]
    for fragment in required_fragments:
        assert fragment in README
```

- [ ] **Step 2: Run the contract test and verify it fails**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q -p no:cacheprovider \
tests/parkour/foothold/test_readme_environment_setup.py
```

Expected: FAIL because the current README does not contain the Conda creation command, pinned PyTorch/Isaac Sim install commands, full submodule commit hashes, or IsaacLab submodule installation command.

- [ ] **Step 3: Replace the current Installation section**

Replace `README.md` from `## Installation` through the command immediately before `## Documentation of Critical Components` with the following Markdown:

````markdown
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
````

- [ ] **Step 4: Run the README contract test and verify it passes**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q -p no:cacheprovider \
tests/parkour/foothold/test_readme_environment_setup.py
```

Expected: `3 passed`.

- [ ] **Step 5: Run static checks and the full foothold suite**

Run:

```bash
git diff --check
rg -n "git clone https://github.com/project-instinct/instinct_rl.git" README.md
./scripts/bootstrap_foothold.sh --check-only
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q tests/parkour/foothold
```

Expected:

- `git diff --check` prints nothing;
- the obsolete separate Instinct-RL clone search prints nothing and exits 1;
- bootstrap reports `Foothold project dependencies are ready.` on the current configured machine;
- the complete foothold suite passes (444 tests before adding the three README tests, approximately 447 afterwards).

- [ ] **Step 6: Review the rendered README diff**

Run:

```bash
GIT_PAGER=cat git diff -- README.md tests/parkour/foothold/test_readme_environment_setup.py
```

Confirm:

- every shell block is copy-pasteable from the documented working directory;
- placeholder paths are explicitly marked and never presented as real paths;
- IsaacLab and Instinct-RL are installed from `third_party/` rather than cloned separately;
- external data, checkpoints, and logs are not claimed to be in Git;
- no Planner, reward, Recovery, or PPO files changed.

- [ ] **Step 7: Commit the implementation**

```bash
git add README.md tests/parkour/foothold/test_readme_environment_setup.py
git commit -m "Document complete hiking environment setup"
```

Expected: one commit containing only the README and its static contract test.
