from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_bootstrap_initializes_and_verifies_pinned_submodules():
    script = (REPO_ROOT / "scripts" / "bootstrap_foothold.sh").read_text()

    assert "git submodule update --init --recursive" in script
    assert "f73c33173801f5f8afea4142482e47b7710c2b75" in script
    assert "f870ead0953fa0e3c3da3349b0aece1c74bfb421" in script
    assert "third_party/IsaacLab" in script
    assert "third_party/instinct_rl" in script


def test_bootstrap_checks_external_runtime_and_motion_data():
    script = (REPO_ROOT / "scripts" / "bootstrap_foothold.sh").read_text()

    assert "Isaac Sim 5.1" in script
    assert "PARKOUR_MOTION_SELECTION_FILE" in script
    assert "--check-only" in script
