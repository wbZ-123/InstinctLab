"""Configuration contract for selecting planner SAC vs legacy planner PPO."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_training_wrapper_defaults_learned_planner_to_sac_and_allows_ppo():
    wrapper = (REPO_ROOT / "scripts" / "foothold_train.sh").read_text()
    assert 'LEARNED_FOOTHOLD_ALGORITHM="${LEARNED_FOOTHOLD_ALGORITHM:-sac}"' in wrapper
    assert '"${LEARNED_FOOTHOLD_ALGORITHM}" != "sac"' in wrapper
    assert '"${LEARNED_FOOTHOLD_ALGORITHM}" != "ppo"' in wrapper
    assert '--learned_foothold_algorithm' in wrapper


def test_train_script_dispatches_to_sac_config_method():
    source = (REPO_ROOT / "scripts" / "instinct_rl" / "train.py").read_text()
    assert "--learned_foothold_algorithm" in source
    assert "enable_event_gated_foothold_sac" in source
    assert "enable_event_gated_foothold_ppo" in source


def test_sac_defaults_are_planner_only():
    source = (
        REPO_ROOT
        / "source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py"
    ).read_text()
    assert "self.algorithm.class_name = algorithm_class_name" in source
    assert "sac_batch_size = 256" in source
    assert "sac_warmup_events = 1024" in source
    assert "sac_updates_per_rollout = 2" in source
    assert "sac_target_entropy = -2.0" in source
