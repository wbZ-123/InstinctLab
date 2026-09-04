from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    output_size = 128
    channels = [4]
    kernel_sizes = [3]
    strides = [1]
    hidden_sizes = [256, 256]
    paddings = [1]
    nonlinearity = "ReLU"
    use_maxpool = True
    component_names = [
        "depth_image",
    ]


@configclass
class EncoderConfigs:
    depth_encoder = DepthEncoderConv2dCfg()


@configclass
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std = 1.0
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    encoder_configs = EncoderConfigs()
    critic_encoder_configs = EncoderConfigs()


@configclass
class AmpAlgoCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "WasabiPPO"
    # With the legacy single reward this is equivalent to the upstream scalar
    # default, while making the intended routing explicit.
    auxiliary_reward_per_env_reward_coefs = [1.0]
    discriminator_kwargs = {
        "hidden_sizes": [1024, 512],
        "nonlinearity": "ReLU",
    }

    discriminator_reward_coef = 0.25
    discriminator_reward_type = "quad"
    discriminator_loss_func = "MSELoss"
    discriminator_gradient_penalty_coef = 5.0
    discriminator_optimizer_class_name = "AdamW"
    discriminator_weight_decay_coef = 3e-4
    discriminator_logit_weight_decay_coef = 0.04
    discriminator_optimizer_kwargs = {
        "lr": 1.0e-4,
        "betas": [0.9, 0.999],
    }
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.006
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
    # Planner-only SAC defaults.  Motor PPO/AMP values above are unchanged.
    sac_hidden_dims = [128, 128]
    sac_replay_capacity = 100000
    sac_batch_size = 256
    sac_warmup_events = 10000
    sac_min_unsafe_events = 512
    sac_target_sample_ratio = 0.5
    sac_max_updates_per_rollout = 24
    sac_actor_update_frequency = 2
    sac_target_update_frequency = 2
    sac_actor_learning_rate = 1.0e-4
    sac_critic_learning_rate = 1.0e-4
    sac_alpha_learning_rate = 1.0e-4
    sac_gamma = 0.95
    sac_tau = 0.005
    sac_target_entropy = -0.5
    sac_initial_alpha = 0.05
    sac_nominal_anchor_coef = 0.25
    sac_max_grad_norm = 1.0
    # The planner action is a normalized residual around the analytic
    # nominal point; these are the decoder's physical XY limits, not the
    # larger reachability ellipse used for geometric validation.
    foothold_residual_limits_m = (0.12, 0.10)


@configclass
class G1ParkourPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    experiment_name = "g1_parkour"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = MoEPolicyCfg()
    algorithm = AmpAlgoCfg()

    def enable_event_gated_foothold_ppo(
        self, reachability_radii_m: tuple[float, float]
    ) -> None:
        """Select the legacy sparse planner PPO comparison mode."""
        self._enable_event_gated_foothold(
            reachability_radii_m,
            algorithm_class_name="EventGatedWasabiPPO",
        )

    def enable_event_gated_foothold_sac(
        self, reachability_radii_m: tuple[float, float]
    ) -> None:
        """Select planner SAC while preserving motor PPO/AMP behavior."""
        self._enable_event_gated_foothold(
            reachability_radii_m,
            algorithm_class_name="EventGatedWasabiSAC",
        )

    def _enable_event_gated_foothold(
        self,
        reachability_radii_m: tuple[float, float],
        *,
        algorithm_class_name: str,
    ) -> None:
        """Configure the shared 29-D motor + 2-D planner architecture."""

        if (
            len(reachability_radii_m) != 2
            or any(radius <= 0.0 for radius in reachability_radii_m)
        ):
            raise ValueError(
                "reachability_radii_m must contain positive XY radii."
            )
        if algorithm_class_name not in {
            "EventGatedWasabiPPO",
            "EventGatedWasabiSAC",
        }:
            raise ValueError(
                "algorithm_class_name must select EventGatedWasabiPPO or "
                "EventGatedWasabiSAC."
            )
        self.algorithm.class_name = algorithm_class_name
        self.policy.class_name = (
            "instinctlab.learning.independent_foothold_actor_critic:"
            "IndependentFootholdEncoderMoEActorCritic"
        )
        self.policy.motor_action_dim = 29
        self.policy.foothold_hidden_dims = [128, 64]
        self.policy.foothold_depth_output_size = 64
        self.policy.foothold_depth_hidden_channels = 8
        self.algorithm.auxiliary_reward_per_env_reward_coefs = [1.0, 0.0]
        self.algorithm.motor_action_dim = 29
        self.algorithm.execution_reward_index = 0
        self.algorithm.foothold_reward_index = 1
        self.algorithm.foothold_initial_std_m = (0.025, 0.020)
        self.algorithm.foothold_min_std_m = (0.005, 0.005)
        self.algorithm.foothold_max_std_m = (0.040, 0.040)
        self.algorithm.foothold_residual_limits_m = (0.12, 0.10)
        self.algorithm.foothold_reachability_radii_m = reachability_radii_m
        # A 64-env acceptance run showed that one 1e-3 planner update creates
        # O(1e2) KL.  Start at PPO's existing adaptive lower bound; the
        # independent KL schedule can increase it when updates are genuinely
        # small instead of accepting one destructive bootstrap step.
        self.algorithm.foothold_learning_rate = 1.0e-5
        self.algorithm.foothold_desired_kl = self.algorithm.desired_kl
        self.algorithm.foothold_kl_stop_multiplier = 2.0
        self.algorithm.foothold_surrogate_coef = 1.0
        self.algorithm.foothold_entropy_coef = self.algorithm.entropy_coef
        self.algorithm.full_finite_check_interval = 100
