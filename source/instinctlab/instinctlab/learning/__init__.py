"""Project-local learning extensions for Instinct-RL."""

from .foothold_rollout_storage import (
    FootholdMiniBatch,
    FootholdRolloutStorage,
    FootholdTransition,
)
from .event_gated_foothold_ppo import (
    EventGatedWasabiPPO,
    register_event_gated_foothold_algorithm,
)
from .foothold_checkpoint import (
    build_legacy_input_column_map,
    FootholdCheckpointMigrationReport,
    initialize_runner_from_legacy_checkpoint,
    learned_foothold_policy_input_expansion,
    migrate_foothold_model_state,
)
from .independent_foothold_actor_critic import (
    IndependentFootholdEncoderMoEActorCritic,
    IndependentFootholdMoEActorCritic,
)
from .foothold_depth_encoder import FootholdDepthEncoder

__all__ = [
    "EventGatedWasabiPPO",
    "build_legacy_input_column_map",
    "FootholdCheckpointMigrationReport",
    "FootholdMiniBatch",
    "FootholdRolloutStorage",
    "FootholdTransition",
    "FootholdDepthEncoder",
    "IndependentFootholdEncoderMoEActorCritic",
    "IndependentFootholdMoEActorCritic",
    "initialize_runner_from_legacy_checkpoint",
    "learned_foothold_policy_input_expansion",
    "migrate_foothold_model_state",
    "register_event_gated_foothold_algorithm",
]
