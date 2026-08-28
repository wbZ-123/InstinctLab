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
