"""PPO helpers for separating dense motor and sparse foothold actions."""

from collections import defaultdict
from collections.abc import Sequence
import importlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import instinct_rl.modules as instinct_modules
from instinct_rl.algorithms.ppo import PPO
from instinct_rl.algorithms.wasabi import WasabiAlgoMixin
from instinct_rl.storage.amp_storage import AmpStorage
from instinct_rl.utils.utils import get_subobs_size

from .foothold_rollout_storage import (
    FootholdMiniBatch,
    FootholdRolloutStorage,
    FootholdTransition,
)
from .foothold_sac import (
    FootholdSAC,
    FootholdSACConfig,
    PlannerEventAccumulator,
    radial_squash,
)


def normalized_foothold_std(
    std_m: Sequence[float],
    radii_m: Sequence[float],
) -> torch.Tensor:
    """Convert physical XY exploration scales into normalized ellipse units."""

    std = torch.as_tensor(std_m, dtype=torch.float32)
    radii = torch.as_tensor(radii_m, dtype=torch.float32)
    if std.shape != (2,) or radii.shape != (2,):
        raise ValueError("Foothold standard deviations and radii must be XY pairs.")
    if torch.any(std <= 0.0):
        raise ValueError("Foothold physical standard deviations must be positive.")
    if torch.any(radii <= 0.0):
        raise ValueError("Foothold reachability radii must be positive.")
    return std / radii


def should_run_full_finite_check(
    learning_iteration: int,
    interval: int,
) -> bool:
    """Return whether expensive whole-tensor diagnostics run this iteration."""

    if interval <= 0:
        raise ValueError("full finite-check interval must be positive.")
    return int(learning_iteration) % int(interval) == 0


def grouped_log_prob(
    distribution: torch.distributions.Distribution,
    actions: torch.Tensor,
    motor_action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum per-dimension likelihoods independently for motor and foothold."""

    if actions.ndim < 2:
        raise ValueError("Actions must have a batch and action dimension.")
    if not 0 < motor_action_dim < actions.shape[-1]:
        raise ValueError(
            "motor_action_dim must split motor and foothold dimensions."
        )
    log_prob_each = distribution.log_prob(actions)
    if log_prob_each.shape != actions.shape:
        raise ValueError(
            "Action distribution must return one log probability per action "
            "dimension."
        )
    motor = log_prob_each[..., :motor_action_dim].sum(dim=-1)
    foothold = log_prob_each[..., motor_action_dim:].sum(dim=-1)
    return motor, foothold


def event_masked_mean(
    values: torch.Tensor,
    event_mask: torch.Tensor,
) -> torch.Tensor:
    """Average only causal planner events while preserving a zero grad path."""

    if event_mask.dtype != torch.bool:
        raise TypeError("Foothold event mask must use torch.bool.")
    if event_mask.shape != values.shape:
        raise ValueError("Foothold event mask and values shapes must match.")
    selected = values[event_mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def balanced_event_masked_mean(
    values: torch.Tensor,
    event_mask: torch.Tensor,
    nominal_safe_event: torch.Tensor,
    nominal_unsafe_event: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average planner values with equal weight for the two event branches.

    The branch masks are a partition of ``event_mask``.  When a minibatch only
    contains one branch, its mean is used without dilution by an empty branch.
    """

    masks = (event_mask, nominal_safe_event, nominal_unsafe_event)
    if any(mask.shape != values.shape for mask in masks):
        raise ValueError("Balanced event masks and values shapes must match.")
    if any(mask.dtype != torch.bool for mask in masks):
        raise TypeError("Balanced event masks must use torch.bool.")
    if torch.any(nominal_safe_event & nominal_unsafe_event).item():
        raise ValueError("Balanced nominal event masks overlap.")
    if not torch.equal(
        nominal_safe_event | nominal_unsafe_event,
        event_mask,
    ):
        raise ValueError("Balanced nominal event masks must union to event mask.")

    safe_mean = event_masked_mean(values, nominal_safe_event)
    unsafe_mean = event_masked_mean(values, nominal_unsafe_event)
    safe_count = int(nominal_safe_event.sum().item())
    unsafe_count = int(nominal_unsafe_event.sum().item())
    if safe_count > 0 and unsafe_count > 0:
        balanced = 0.5 * (safe_mean + unsafe_mean)
    elif safe_count > 0:
        balanced = safe_mean
    elif unsafe_count > 0:
        balanced = unsafe_mean
    else:
        balanced = values.sum() * 0.0
    return balanced, safe_mean, unsafe_mean


def _clipped_surrogate_terms(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip_param: float,
) -> torch.Tensor:
    """Return per-sample PPO clipped surrogate loss terms."""

    ratio = torch.exp(new_log_prob - old_log_prob)
    surrogate = -advantage * ratio
    surrogate_clipped = -advantage * torch.clamp(
        ratio,
        1.0 - clip_param,
        1.0 + clip_param,
    )
    return torch.maximum(surrogate, surrogate_clipped)


def grouped_clipped_surrogates(
    *,
    new_motor_log_prob: torch.Tensor,
    old_motor_log_prob: torch.Tensor,
    new_foothold_log_prob: torch.Tensor,
    old_foothold_log_prob: torch.Tensor,
    execution_advantage: torch.Tensor,
    foothold_advantage: torch.Tensor,
    event_mask: torch.Tensor,
    clip_param: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute independent PPO losses for dense and event-gated actions."""

    expected_shape = new_motor_log_prob.shape
    tensors = (
        old_motor_log_prob,
        new_foothold_log_prob,
        old_foothold_log_prob,
        execution_advantage,
        foothold_advantage,
        event_mask,
    )
    if any(value.shape != expected_shape for value in tensors):
        raise ValueError("Grouped PPO inputs must have matching shapes.")
    if event_mask.dtype != torch.bool:
        raise TypeError("Foothold event mask must use torch.bool.")
    if clip_param <= 0.0:
        raise ValueError("PPO clip_param must be positive.")

    motor_loss = _clipped_surrogate_terms(
        new_motor_log_prob,
        old_motor_log_prob,
        execution_advantage,
        clip_param,
    ).mean()
    foothold_loss = event_masked_mean(
        _clipped_surrogate_terms(
            new_foothold_log_prob,
            old_foothold_log_prob,
            foothold_advantage,
            clip_param,
        ),
        event_mask,
    )
    return motor_loss, foothold_loss


def _require_finite(name: str, tensors) -> None:
    """Raise with a precise boundary name when any tensor is non-finite."""

    if hasattr(tensors, "items"):
        named_tensors = tensors.items()
    else:
        named_tensors = (
            (str(index), tensor)
            for index, tensor in enumerate(tensors)
        )
    tensors_by_device: dict[
        torch.device,
        list[tuple[str, torch.Tensor, torch.Tensor]],
    ] = {}
    for tensor_name, tensor in named_tensors:
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if tensor.is_floating_point() or tensor.is_complex():
            finite = torch.isfinite(tensor)
            check = finite.all()
            tensors_by_device.setdefault(check.device, []).append(
                (str(tensor_name), finite, check)
            )
    for entries in tensors_by_device.values():
        if torch.stack([entry[2] for entry in entries]).all().item():
            continue
        failures: list[str] = []
        for tensor_name, finite, check in entries:
            if check.item():
                continue
            invalid = (~finite).nonzero(as_tuple=False)
            first_index = tuple(invalid[0].tolist())
            failures.append(
                f"{name}.{tensor_name}: "
                f"nonfinite_count={invalid.shape[0]} "
                f"first_index={first_index}"
            )
        raise FloatingPointError("; ".join(failures))


def _optimizer_state_tensors(optimizer):
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                yield value


class EventGatedWasabiPPO(WasabiAlgoMixin, PPO):
    """WASABI PPO with dense motor and sparse foothold policy objectives."""

    def __init__(
        self,
        *args,
        motor_action_dim: int,
        execution_reward_index: int,
        foothold_reward_index: int,
        foothold_initial_std_m: Sequence[float],
        foothold_min_std_m: Sequence[float],
        foothold_max_std_m: Sequence[float],
        foothold_reachability_radii_m: Sequence[float],
        foothold_learning_rate: float,
        foothold_desired_kl: float,
        foothold_kl_stop_multiplier: float = 2.0,
        foothold_surrogate_coef: float = 1.0,
        foothold_entropy_coef: float | None = None,
        full_finite_check_interval: int = 100,
        **kwargs,
    ):
        optimizer_class_name = kwargs.get("optimizer_class_name", "Adam")
        super().__init__(*args, **kwargs)
        if motor_action_dim <= 0:
            raise ValueError("motor_action_dim must be positive.")
        if execution_reward_index < 0 or foothold_reward_index < 0:
            raise ValueError("Reward indices must be non-negative.")
        if execution_reward_index == foothold_reward_index:
            raise ValueError("Execution and foothold reward indices must differ.")
        if foothold_surrogate_coef < 0.0:
            raise ValueError("foothold_surrogate_coef must be non-negative.")
        if foothold_entropy_coef is None:
            foothold_entropy_coef = self.entropy_coef
        if foothold_entropy_coef < 0.0:
            raise ValueError("foothold_entropy_coef must be non-negative.")
        if foothold_learning_rate <= 0.0:
            raise ValueError("foothold_learning_rate must be positive.")
        if foothold_desired_kl <= 0.0:
            raise ValueError("foothold_desired_kl must be positive.")
        if foothold_kl_stop_multiplier <= 1.0:
            raise ValueError(
                "foothold_kl_stop_multiplier must be greater than one."
            )
        if full_finite_check_interval <= 0:
            raise ValueError(
                "full_finite_check_interval must be positive."
            )

        required_policy_api = (
            "motor_parameters",
            "foothold_parameters",
            "motor_std",
            "foothold_std",
            "clip_motor_std",
            "clip_foothold_std",
        )
        missing_policy_api = tuple(
            name
            for name in required_policy_api
            if not hasattr(self.actor_critic, name)
        )
        if missing_policy_api:
            raise TypeError(
                "Event-gated foothold PPO requires the independent foothold "
                "actor-critic interface; missing "
                + ", ".join(missing_policy_api)
            )
        num_actions = int(
            self.actor_critic.motor_std.numel()
            + self.actor_critic.foothold_std.numel()
        )
        if num_actions != motor_action_dim + 2:
            raise ValueError(
                "Event-gated foothold PPO requires exactly two foothold "
                f"actions after {motor_action_dim} motor actions; got "
                f"{num_actions} total actions."
            )

        self.motor_action_dim = motor_action_dim
        self.execution_reward_index = execution_reward_index
        self.foothold_reward_index = foothold_reward_index
        self.foothold_initial_std_m = tuple(foothold_initial_std_m)
        self.foothold_min_std_m = tuple(foothold_min_std_m)
        self.foothold_max_std_m = tuple(foothold_max_std_m)
        self.foothold_reachability_radii_m = tuple(
            foothold_reachability_radii_m
        )
        self.foothold_learning_rate = float(foothold_learning_rate)
        self.foothold_desired_kl = float(foothold_desired_kl)
        self.foothold_kl_stop_multiplier = float(
            foothold_kl_stop_multiplier
        )
        self.foothold_surrogate_coef = foothold_surrogate_coef
        self.foothold_entropy_coef = foothold_entropy_coef
        self.full_finite_check_interval = int(
            full_finite_check_interval
        )
        # Direct diagnostic calls outside ``update`` remain fail-fast.
        self._run_full_finite_check = True

        # PPO.update resolves coefficients from the returned loss names.
        self.motor_surrogate_loss_coef = 1.0
        self.foothold_surrogate_loss_coef = foothold_surrogate_coef
        self.motor_entropy_coef = self.entropy_coef
        self.motor_value_loss_coef = self.value_loss_coef
        self.foothold_value_loss_coef = self.value_loss_coef
        # SAC subclasses can disable the legacy planner PPO branch while
        # retaining the motor/AMP implementation and diagnostics.
        self.use_foothold_ppo = True

        motor_parameters = tuple(self.actor_critic.motor_parameters())
        foothold_parameters = tuple(self.actor_critic.foothold_parameters())
        motor_parameter_ids = {id(parameter) for parameter in motor_parameters}
        foothold_parameter_ids = {
            id(parameter) for parameter in foothold_parameters
        }
        all_parameter_ids = {
            id(parameter) for parameter in self.actor_critic.parameters()
        }
        if (
            motor_parameter_ids & foothold_parameter_ids
            or motor_parameter_ids | foothold_parameter_ids
            != all_parameter_ids
        ):
            raise RuntimeError(
                "Motor and foothold parameter groups must be disjoint and "
                "exhaustive."
            )
        optimizer_class = getattr(optim, optimizer_class_name)
        # Replace the upstream all-parameter optimizer before any update.
        self.optimizer = optimizer_class(
            motor_parameters,
            lr=self.learning_rate,
        )
        self.foothold_optimizer = optimizer_class(
            foothold_parameters,
            lr=self.foothold_learning_rate,
        )

        normalized_std = normalized_foothold_std(
            self.foothold_initial_std_m,
            self.foothold_reachability_radii_m,
        ).to(
            device=self.actor_critic.foothold_std.device,
            dtype=self.actor_critic.foothold_std.dtype,
        )
        self.foothold_min_std = normalized_foothold_std(
            self.foothold_min_std_m,
            self.foothold_reachability_radii_m,
        ).to(device=normalized_std.device, dtype=normalized_std.dtype)
        self.foothold_max_std = normalized_foothold_std(
            self.foothold_max_std_m,
            self.foothold_reachability_radii_m,
        ).to(device=normalized_std.device, dtype=normalized_std.dtype)
        if torch.any(self.foothold_min_std > self.foothold_max_std):
            raise ValueError(
                "foothold_min_std_m must not exceed foothold_max_std_m."
            )
        if torch.any(normalized_std < self.foothold_min_std) or torch.any(
            normalized_std > self.foothold_max_std
        ):
            raise ValueError(
                "foothold_initial_std_m must lie within the configured bounds."
            )
        with torch.no_grad():
            self.actor_critic.foothold_std.copy_(normalized_std)

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        obs_format,
        num_actions,
        num_rewards=1,
    ):
        if num_actions != self.motor_action_dim + 2:
            raise ValueError(
                f"Expected {self.motor_action_dim + 2} actions, got "
                f"{num_actions}."
            )
        max_reward_index = max(
            self.execution_reward_index,
            self.foothold_reward_index,
        )
        if num_rewards <= max_reward_index:
            raise ValueError(
                "Reward tensor does not contain the configured execution and "
                "foothold reward groups."
            )

        obs_size = get_subobs_size(obs_format["policy"])
        critic_obs_size = (
            get_subobs_size(obs_format.get("critic"))
            if "critic" in obs_format
            else None
        )
        self.transition = FootholdTransition()
        self.storage = FootholdRolloutStorage(
            num_envs,
            num_transitions_per_env,
            [obs_size],
            [critic_obs_size],
            [num_actions],
            num_rewards=num_rewards,
            device=self.device,
        )
        self._init_wasabi_storage(
            num_envs,
            num_transitions_per_env,
            obs_format,
        )

    def _init_wasabi_storage(
        self,
        num_envs,
        num_transitions_per_env,
        obs_format,
    ) -> None:
        if ":" in self.discriminator_class_name:
            module_name, class_name = self.discriminator_class_name.split(
                ":",
                maxsplit=1,
            )
            discriminator_class = getattr(
                importlib.import_module(module_name),
                class_name,
            )
        else:
            discriminator_class = getattr(
                instinct_modules,
                self.discriminator_class_name,
            )
        self.discriminator = discriminator_class(
            input_segment=obs_format[self.actor_state_key],
            **self.discriminator_kwargs,
        ).to(self.device)
        optimizer_kwargs = dict(self.discriminator_optimizer_kwargs)
        optimizer_kwargs.setdefault("lr", self.learning_rate)
        self.discriminator_optimizer = getattr(
            optim,
            self.discriminator_optimizer_class_name,
        )(
            self.discriminator.parameters(),
            **optimizer_kwargs,
        )

        self.amp_transition = AmpStorage.Transition()
        actor_state_size = get_subobs_size(
            obs_format[self.actor_state_key]
        )
        reference_state_size = get_subobs_size(
            obs_format[self.reference_state_key]
        )
        if actor_state_size != reference_state_size:
            raise ValueError(
                "WASABI actor and reference state shapes must match."
            )
        self.amp_storage = AmpStorage(
            num_envs,
            num_transitions_per_env,
            [actor_state_size],
            [reference_state_size],
            device=self.device,
        )

    def process_env_step(
        self,
        rewards,
        dones,
        infos,
        next_obs,
        next_critic_obs,
    ):
        event_key = "learned_foothold_action_event"
        safe_event_key = "learned_foothold_nominal_safe_event"
        unsafe_event_key = "learned_foothold_nominal_unsafe_event"
        if event_key not in infos:
            raise KeyError(
                f"Missing causal PPO transition key: {event_key}"
            )
        if safe_event_key not in infos:
            raise KeyError(
                f"Missing causal PPO transition key: {safe_event_key}"
            )
        if unsafe_event_key not in infos:
            raise KeyError(
                f"Missing causal PPO transition key: {unsafe_event_key}"
            )
        event = infos[event_key]
        if event.dtype != torch.bool:
            raise TypeError("Learned foothold action event must use bool.")
        safe_event = infos[safe_event_key]
        unsafe_event = infos[unsafe_event_key]
        if safe_event.dtype != torch.bool or unsafe_event.dtype != torch.bool:
            raise TypeError("Learned foothold nominal event masks must use bool.")
        if safe_event.shape != event.shape or unsafe_event.shape != event.shape:
            raise ValueError(
                "Learned foothold nominal event masks must match event shape."
            )
        if torch.any(safe_event & unsafe_event).item():
            raise ValueError(
                "Learned foothold nominal event masks overlap."
            )
        if not torch.equal(safe_event | unsafe_event, event):
            raise ValueError(
                "Learned foothold nominal event masks must union to event."
            )
        self.transition.foothold_action_event = event.detach().clone()
        self.transition.foothold_nominal_safe_event = (
            safe_event.detach().clone()
        )
        self.transition.foothold_nominal_unsafe_event = (
            unsafe_event.detach().clone()
        )
        super().process_env_step(
            rewards,
            dones,
            infos,
            next_obs,
            next_critic_obs,
        )

    def compute_losses(self, minibatch: FootholdMiniBatch):
        if self._run_full_finite_check:
            _require_finite(
                "PPO minibatch",
                {
                    "obs": minibatch.obs,
                    "critic_obs": minibatch.critic_obs,
                    "actions": minibatch.actions,
                    "values": minibatch.values,
                    "advantages": minibatch.advantages,
                    "returns": minibatch.returns,
                    "old_mu": minibatch.old_mu,
                    "old_sigma": minibatch.old_sigma,
                },
            )
            if torch.any(minibatch.old_sigma <= 0.0):
                raise FloatingPointError(
                    "PPO minibatch old_sigma must be finite and positive."
                )
        if minibatch.advantages.shape[-1] <= max(
            self.execution_reward_index,
            self.foothold_reward_index,
        ):
            raise ValueError("PPO minibatch is missing a configured advantage.")

        actor_hidden_states = (
            minibatch.hidden_states.actor
            if self.actor_critic.is_recurrent
            else None
        )
        self.actor_critic.act(
            minibatch.obs,
            masks=minibatch.masks,
            hidden_states=actor_hidden_states,
        )
        distribution = self.actor_critic.distribution
        new_motor_log_prob, new_foothold_log_prob = grouped_log_prob(
            distribution,
            minibatch.actions,
            self.motor_action_dim,
        )
        old_distribution = torch.distributions.Normal(
            minibatch.old_mu,
            minibatch.old_sigma,
        )
        old_motor_log_prob, old_foothold_log_prob = grouped_log_prob(
            old_distribution,
            minibatch.actions,
            self.motor_action_dim,
        )
        motor_surrogate_loss, _ = (
            grouped_clipped_surrogates(
                new_motor_log_prob=new_motor_log_prob,
                old_motor_log_prob=old_motor_log_prob,
                new_foothold_log_prob=new_foothold_log_prob,
                old_foothold_log_prob=old_foothold_log_prob,
                execution_advantage=minibatch.advantages[
                    ..., self.execution_reward_index
                ],
                foothold_advantage=minibatch.advantages[
                    ..., self.foothold_reward_index
                ],
                event_mask=minibatch.foothold_action_event,
                clip_param=self.clip_param,
            )
        )
        foothold_surrogate_terms = _clipped_surrogate_terms(
            new_foothold_log_prob,
            old_foothold_log_prob,
            minibatch.advantages[..., self.foothold_reward_index],
            self.clip_param,
        )
        (
            foothold_surrogate_loss,
            nominal_safe_surrogate_loss,
            nominal_unsafe_surrogate_loss,
        ) = balanced_event_masked_mean(
            foothold_surrogate_terms,
            minibatch.foothold_action_event,
            minibatch.foothold_nominal_safe_event,
            minibatch.foothold_nominal_unsafe_event,
        )

        critic_hidden_states = (
            minibatch.hidden_states.critic
            if self.actor_critic.is_recurrent
            else None
        )
        value_batch = self.actor_critic.evaluate(
            minibatch.critic_obs,
            masks=minibatch.masks,
            hidden_states=critic_hidden_states,
        )
        if self.use_clipped_value_loss:
            value_clipped = minibatch.values + (
                value_batch - minibatch.values
            ).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - minibatch.returns).pow(2)
            value_losses_clipped = (
                value_clipped - minibatch.returns
            ).pow(2)
            value_loss = torch.maximum(
                value_losses,
                value_losses_clipped,
            )
        else:
            value_loss = (minibatch.returns - value_batch).pow(2)
        value_loss = value_loss.reshape(
            -1,
            value_loss.shape[-1],
        ).mean(dim=0)

        sigma_batch = self.actor_critic.action_std
        mu_batch = self.actor_critic.action_mean
        kl_each = (
            torch.log(sigma_batch / minibatch.old_sigma + 1.0e-5)
            + (
                minibatch.old_sigma.square()
                + (minibatch.old_mu - mu_batch).square()
            )
            / (2.0 * sigma_batch.square())
            - 0.5
        )
        motor_kl = kl_each[..., : self.motor_action_dim].sum(
            dim=-1
        ).mean()
        foothold_kl = event_masked_mean(
            kl_each[..., self.motor_action_dim :].sum(dim=-1),
            minibatch.foothold_action_event,
        )

        entropy_each = distribution.entropy()
        motor_entropy = entropy_each[
            ..., : self.motor_action_dim
        ].sum(dim=-1).mean()
        foothold_entropy = event_masked_mean(
            entropy_each[..., self.motor_action_dim :].sum(dim=-1),
            minibatch.foothold_action_event,
        )
        event_count = minibatch.foothold_action_event.sum().to(
            dtype=torch.float32
        )
        raw_foothold_action = minibatch.actions[
            ..., self.motor_action_dim :
        ]
        raw_out_of_range = torch.any(
            torch.abs(raw_foothold_action) > 1.0,
            dim=-1,
        ).to(dtype=torch.float32)
        square_clamped_action = raw_foothold_action.clamp(-1.0, 1.0)
        requires_ellipse_projection = (
            torch.linalg.vector_norm(
                square_clamped_action,
                dim=-1,
            )
            > 1.0
        ).to(dtype=torch.float32)

        losses = {
            "motor_surrogate_loss": motor_surrogate_loss,
            "foothold_surrogate_loss": foothold_surrogate_loss,
            "motor_value_loss": value_loss[self.execution_reward_index],
            "foothold_value_loss": value_loss[self.foothold_reward_index],
            "motor_entropy": -motor_entropy,
            "foothold_entropy": -foothold_entropy,
        }
        stats = {
            "motor_kl": motor_kl,
            "foothold_kl": foothold_kl,
            "foothold_event_count": event_count,
            "foothold_nominal_safe_event_count": minibatch.foothold_nominal_safe_event.sum().to(
                dtype=torch.float32
            ),
            "foothold_nominal_unsafe_event_count": minibatch.foothold_nominal_unsafe_event.sum().to(
                dtype=torch.float32
            ),
            "foothold_nominal_safe_advantage_mean": event_masked_mean(
                minibatch.advantages[..., self.foothold_reward_index],
                minibatch.foothold_nominal_safe_event,
            ),
            "foothold_nominal_unsafe_advantage_mean": event_masked_mean(
                minibatch.advantages[..., self.foothold_reward_index],
                minibatch.foothold_nominal_unsafe_event,
            ),
            "foothold_nominal_safe_surrogate_loss": nominal_safe_surrogate_loss,
            "foothold_nominal_unsafe_surrogate_loss": nominal_unsafe_surrogate_loss,
            "foothold_balanced_surrogate_loss": foothold_surrogate_loss,
            "foothold_raw_out_of_range_fraction": event_masked_mean(
                raw_out_of_range,
                minibatch.foothold_action_event,
            ),
            "foothold_ellipse_projection_fraction": event_masked_mean(
                requires_ellipse_projection,
                minibatch.foothold_action_event,
            ),
        }
        for index in range(minibatch.advantages.shape[-1]):
            stats[f"advantage_{index}"] = minibatch.advantages[
                ..., index
            ].detach().mean()
        for index in range(value_loss.numel()):
            stats[f"value_loss_{index}"] = value_loss.detach()[index]

        if self._run_full_finite_check:
            _require_finite("PPO losses", losses.values())
            _require_finite("PPO statistics", stats.values())
        inter_vars = {
            "new_motor_log_prob": new_motor_log_prob,
            "new_foothold_log_prob": new_foothold_log_prob,
            "motor_kl": motor_kl,
            "foothold_kl": foothold_kl,
        }
        return losses, inter_vars, stats

    def _sync_gradients(self, parameters) -> None:
        if dist.is_initialized():
            world_size = dist.get_world_size()
            for parameter in parameters:
                if parameter.grad is not None:
                    dist.all_reduce(
                        parameter.grad.data,
                        op=dist.ReduceOp.SUM,
                    )
                    parameter.grad.data /= world_size

    def _clip_gradients(self, parameters, group_name: str) -> torch.Tensor:
        try:
            grad_norm = nn.utils.clip_grad_norm_(
                parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError as exc:
            raise FloatingPointError(
                f"{group_name} gradient norm must be finite."
            ) from exc
        return grad_norm.detach()

    def _policy_gradient_step(
        self,
        *,
        motor_loss: torch.Tensor,
        foothold_loss: torch.Tensor,
        run_foothold: bool,
        average_stats: dict,
        run_motor: bool = True,
    ) -> None:
        """Step disjoint parameter groups from one shared forward graph."""

        if not run_motor and not run_foothold:
            return
        motor_parameters = tuple(self.actor_critic.motor_parameters())
        foothold_parameters = tuple(self.actor_critic.foothold_parameters())
        self.optimizer.zero_grad(set_to_none=True)
        self.foothold_optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        if run_motor:
            total_loss = total_loss + motor_loss
        if run_foothold:
            total_loss = total_loss + foothold_loss
        total_loss.backward()

        if run_motor:
            self._sync_gradients(motor_parameters)
            motor_grad_norm = self._clip_gradients(
                motor_parameters,
                "Motor actor-critic",
            )
            average_stats["motor_grad_norm"] = (
                average_stats.get("motor_grad_norm", 0.0)
                + motor_grad_norm
            )
            # Preserve the upstream diagnostic name for existing dashboards.
            average_stats["grad_norm"] = (
                average_stats.get("grad_norm", 0.0) + motor_grad_norm
            )
            self.optimizer.step()
        if run_foothold:
            self._sync_gradients(foothold_parameters)
            foothold_grad_norm = self._clip_gradients(
                foothold_parameters,
                "Foothold actor-critic",
            )
            average_stats["foothold_grad_norm"] = (
                average_stats.get("foothold_grad_norm", 0.0)
                + foothold_grad_norm
            )
            self.foothold_optimizer.step()

    def gradient_step(self, loss: torch.Tensor, average_stats: dict):
        """Upstream-compatible motor-only gradient boundary."""

        zero_foothold_loss = self.actor_critic.foothold_std.sum() * 0.0
        self._policy_gradient_step(
            motor_loss=loss,
            foothold_loss=zero_foothold_loss,
            run_foothold=False,
            average_stats=average_stats,
        )

    def _foothold_update_allowed(
        self,
        foothold_kl: torch.Tensor,
        *,
        event_count: torch.Tensor,
    ) -> bool:
        if event_count.detach().item() <= 0:
            return False
        return bool(
            foothold_kl.detach().item()
            <= self.foothold_desired_kl
            * self.foothold_kl_stop_multiplier
        )

    @torch.no_grad()
    def _clip_foothold_std_to_physical_bounds(self) -> None:
        self.actor_critic.foothold_std.copy_(
            torch.maximum(
                torch.minimum(
                    self.actor_critic.foothold_std,
                    self.foothold_max_std,
                ),
                self.foothold_min_std,
            )
        )

    def _distributed_scalar_mean(self, value: torch.Tensor) -> torch.Tensor:
        result = value.detach().clone()
        if dist.is_initialized():
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
            result /= dist.get_world_size()
        return result

    def _adaptive_rate(
        self,
        kl: torch.Tensor,
        *,
        desired_kl: float,
        learning_rate: float,
    ) -> float:
        with torch.inference_mode():
            kl_mean = self._distributed_scalar_mean(kl)
            if kl_mean > desired_kl * 2.0:
                learning_rate = max(1.0e-5, learning_rate / 1.5)
            elif 0.0 < kl_mean < desired_kl / 2.0:
                learning_rate = min(1.0e-2, learning_rate * 1.5)
            if dist.is_initialized():
                rate_tensor = torch.tensor(learning_rate, device=self.device)
                dist.broadcast(rate_tensor, src=0)
                learning_rate = rate_tensor.item()
        return learning_rate

    def _adjust_learning_rate_once(self, motor_kl: torch.Tensor) -> None:
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        self.learning_rate = self._adaptive_rate(
            motor_kl,
            desired_kl=self.desired_kl,
            learning_rate=self.learning_rate,
        )
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = self.learning_rate

    def _adjust_foothold_learning_rate_once(
        self,
        foothold_kl: torch.Tensor,
    ) -> None:
        if self.schedule != "adaptive":
            return
        self.foothold_learning_rate = self._adaptive_rate(
            foothold_kl,
            desired_kl=self.foothold_desired_kl,
            learning_rate=self.foothold_learning_rate,
        )
        for parameter_group in self.foothold_optimizer.param_groups:
            parameter_group["lr"] = self.foothold_learning_rate

    def wasabi_gradient_step(
        self,
        loss: torch.Tensor,
        average_stats: dict,
    ):
        """Apply the same non-finite boundary to the discriminator optimizer."""

        parameters = tuple(self.discriminator.parameters())
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            for parameter in parameters:
                if parameter.grad is not None:
                    dist.all_reduce(
                        parameter.grad.data,
                        op=dist.ReduceOp.SUM,
                    )
                    parameter.grad.data /= world_size
        try:
            nn.utils.clip_grad_norm_(
                parameters,
                float("inf"),
                error_if_nonfinite=True,
            )
        except RuntimeError as exc:
            raise FloatingPointError(
                "WASABI discriminator gradient norm must be finite."
            ) from exc
        self.discriminator_optimizer.step()

    def _update_policy(self, current_learning_iteration):
        """Run motor and foothold PPO updates with independent safeguards."""

        self.current_learning_iteration = current_learning_iteration
        mean_losses = defaultdict(float)
        average_stats = defaultdict(float)
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )

        foothold_updates_stopped = False
        update_count = 0
        foothold_skip_count = 0
        for minibatch in generator:
            losses, _, stats = self.compute_losses(minibatch)
            update_count += 1
            self._adjust_learning_rate_once(stats["motor_kl"])

            motor_loss = (
                self.motor_surrogate_loss_coef
                * losses["motor_surrogate_loss"]
                + self.motor_value_loss_coef
                * losses["motor_value_loss"]
                + self.motor_entropy_coef * losses["motor_entropy"]
            )
            foothold_loss = (
                self.foothold_surrogate_loss_coef
                * losses["foothold_surrogate_loss"]
                + self.foothold_value_loss_coef
                * losses["foothold_value_loss"]
                + self.foothold_entropy_coef * losses["foothold_entropy"]
            )

            has_foothold_event = bool(
                stats["foothold_event_count"].detach().item() > 0
            )
            run_foothold = False
            if self.use_foothold_ppo and has_foothold_event:
                if foothold_updates_stopped or not self._foothold_update_allowed(
                    stats["foothold_kl"],
                    event_count=stats["foothold_event_count"],
                ):
                    foothold_updates_stopped = True
                    foothold_skip_count += 1
                else:
                    run_foothold = True

            self._policy_gradient_step(
                motor_loss=motor_loss,
                foothold_loss=foothold_loss,
                run_foothold=run_foothold,
                average_stats=average_stats,
            )

            for key, value in losses.items():
                mean_losses[key] += value.detach()
            mean_losses["motor_total_loss"] += motor_loss.detach()
            mean_losses["foothold_total_loss"] += foothold_loss.detach()
            mean_losses["total_loss"] += (
                motor_loss.detach()
                + (foothold_loss.detach() if run_foothold else 0.0)
            )
            for key, value in stats.items():
                average_stats[key] += value.detach()

        if update_count == 0:
            raise RuntimeError("PPO storage produced no minibatches.")
        for key in tuple(mean_losses):
            mean_losses[key] /= update_count
        for key in tuple(average_stats):
            average_stats[key] /= update_count
        average_stats["foothold_kl_skip_count"] = torch.tensor(
            float(foothold_skip_count),
            device=self.device,
        )

        self.storage.clear()
        self.actor_critic.clip_motor_std(minimum=self.clip_min_std)
        self._clip_foothold_std_to_physical_bounds()
        self._adjust_foothold_learning_rate_once(
            average_stats["foothold_kl"]
        )

        foothold_std = self.actor_critic.foothold_std.detach()
        radii = torch.as_tensor(
            self.foothold_reachability_radii_m,
            device=foothold_std.device,
            dtype=foothold_std.dtype,
        )
        average_stats["motor_learning_rate"] = torch.tensor(
            self.learning_rate,
            device=self.device,
        )
        average_stats["foothold_learning_rate"] = torch.tensor(
            self.foothold_learning_rate,
            device=self.device,
        )
        average_stats["foothold_std_normalized_x"] = foothold_std[0]
        average_stats["foothold_std_normalized_y"] = foothold_std[1]
        average_stats["foothold_std_m_x"] = foothold_std[0] * radii[0]
        average_stats["foothold_std_m_y"] = foothold_std[1] * radii[1]
        return mean_losses, average_stats

    def update(self, current_learning_iteration):
        self._run_full_finite_check = should_run_full_finite_check(
            current_learning_iteration,
            self.full_finite_check_interval,
        )
        # Deliberately bypass WasabiAlgoMixin.update so the policy runs exactly
        # once before the unchanged discriminator phase below.
        mean_losses, average_stats = self._update_policy(
            current_learning_iteration,
        )
        if self._run_full_finite_check:
            _require_finite(
                "Updated actor-critic parameters",
                self.actor_critic.parameters(),
            )
            _require_finite(
                "Updated actor-critic optimizer state",
                _optimizer_state_tensors(self.optimizer),
            )
            _require_finite(
                "Updated foothold optimizer state",
                _optimizer_state_tensors(self.foothold_optimizer),
            )

        if self.discriminator.is_recurrent:
            amp_generator = (
                self.amp_storage.recurrent_mini_batch_generator(
                    self.num_mini_batches,
                    self.num_learning_epochs,
                )
            )
        else:
            amp_generator = self.amp_storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for amp_minibatch in amp_generator:
            losses, _, stats = self.compute_amp_losses(amp_minibatch)
            loss = 0.0
            for key, value in losses.items():
                loss += getattr(self, key + "_coef", 1.0) * value
                mean_losses[key] += value.detach() / num_updates
            mean_losses["amp_total_loss"] += loss.detach() / num_updates
            for key, value in stats.items():
                average_stats[key] += value.detach() / num_updates
            self.wasabi_gradient_step(loss, average_stats)

        if self._run_full_finite_check:
            _require_finite(
                "Updated WASABI parameters",
                self.discriminator.parameters(),
            )
            _require_finite(
                "Updated WASABI optimizer state",
                _optimizer_state_tensors(self.discriminator_optimizer),
            )
        if self.discriminator.normalizer is not None:
            self.discriminator.normalizer.sync_across_processes()
        self.amp_storage.clear()
        return mean_losses, average_stats

    def state_dict(self):
        state = super().state_dict()
        state["foothold_optimizer_state_dict"] = (
            self.foothold_optimizer.state_dict()
        )
        state["motor_learning_rate"] = float(self.learning_rate)
        state["foothold_learning_rate"] = float(
            self.foothold_learning_rate
        )
        return state

    def load_state_dict(self, state_dict):
        if "foothold_optimizer_state_dict" not in state_dict:
            raise KeyError(
                "Independent foothold checkpoint is missing "
                "foothold_optimizer_state_dict. Shared-head checkpoints "
                "cannot be resumed into this architecture."
            )
        super().load_state_dict(state_dict)
        self.foothold_optimizer.load_state_dict(
            state_dict["foothold_optimizer_state_dict"]
        )
        self.learning_rate = float(
            state_dict.get(
                "motor_learning_rate",
                self.optimizer.param_groups[0]["lr"],
            )
        )
        self.foothold_learning_rate = float(
            state_dict.get(
                "foothold_learning_rate",
                self.foothold_optimizer.param_groups[0]["lr"],
            )
        )
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        for group in self.foothold_optimizer.param_groups:
            group["lr"] = self.foothold_learning_rate


class EventGatedWasabiSAC(EventGatedWasabiPPO):
    """Hybrid learner: unchanged motor/AMP PPO plus event-only planner SAC.

    The existing actor-critic still owns the 29-D motor head and the 2-D
    planner head used at rollout time.  SAC supplies the planner optimizer and
    twin Q critics; the legacy planner PPO objective is disabled explicitly.
    """

    _SAC_STATE_VERSION = 2

    def __init__(
        self,
        *args,
        sac_hidden_dims: Sequence[int] = (128, 128),
        sac_replay_capacity: int = 100_000,
        sac_batch_size: int = 256,
        sac_warmup_events: int = 1_024,
        sac_target_sample_ratio: float = 0.125,
        sac_max_updates_per_rollout: int = 4,
        sac_actor_learning_rate: float = 1.0e-4,
        sac_critic_learning_rate: float = 1.0e-4,
        sac_alpha_learning_rate: float = 1.0e-4,
        sac_gamma: float = 0.99,
        sac_tau: float = 0.005,
        sac_target_entropy: float = -2.0,
        sac_max_grad_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_foothold_ppo = False
        self._sac_constructor_config = {
            "hidden_dims": tuple(int(width) for width in sac_hidden_dims),
            "replay_capacity": int(sac_replay_capacity),
            "batch_size": int(sac_batch_size),
            "warmup_events": int(sac_warmup_events),
            "target_sample_ratio": float(sac_target_sample_ratio),
            "max_updates_per_rollout": int(sac_max_updates_per_rollout),
            "actor_lr": float(sac_actor_learning_rate),
            "critic_lr": float(sac_critic_learning_rate),
            "alpha_lr": float(sac_alpha_learning_rate),
            "gamma": float(sac_gamma),
            "tau": float(sac_tau),
            "target_entropy": float(sac_target_entropy),
            "max_grad_norm": float(sac_max_grad_norm),
        }
        self.sac: FootholdSAC | None = None
        self._planner_events: PlannerEventAccumulator | None = None
        self._planner_events_since_update = 0

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        obs_format,
        num_actions,
        num_rewards=1,
    ):
        super().init_storage(
            num_envs,
            num_transitions_per_env,
            obs_format,
            num_actions,
            num_rewards,
        )
        feature_dim = int(self.actor_critic.foothold_actor[0].in_features)
        replay_obs_dim = int(self.storage.observations.shape[-1])
        sac_config = FootholdSACConfig(
            obs_dim=feature_dim,
            replay_obs_dim=replay_obs_dim,
            action_dim=2,
            **self._sac_constructor_config,
        )
        planner_parameters = (
            self.actor_critic.planner_actor_parameters()
            if hasattr(self.actor_critic, "planner_actor_parameters")
            else self.actor_critic.planner_policy_parameters()
        )
        planner_encoder_parameters = (
            self.actor_critic.planner_encoder_parameters()
            if hasattr(self.actor_critic, "planner_encoder_parameters")
            else ()
        )
        self.sac = FootholdSAC(
            sac_config,
            device=self.device,
            actor_distribution_fn=self.actor_critic.planner_distribution_from_features,
            actor_parameters=planner_parameters,
            feature_parameters=planner_encoder_parameters,
            feature_fn=lambda observations: self.actor_critic.planner_features(
                observations,
                detach_shared=True,
            ),
        )
        self._planner_events = PlannerEventAccumulator(
            num_envs=int(num_envs),
            obs_dim=replay_obs_dim,
            action_dim=2,
            device=self.device,
        )
        self._planner_events_since_update = 0

    def _require_sac(self) -> FootholdSAC:
        if self.sac is None:
            raise RuntimeError("Planner SAC must be initialized with init_storage.")
        return self.sac

    def process_env_step(
        self,
        rewards,
        dones,
        infos,
        next_obs,
        next_critic_obs,
    ):
        """Record planner transitions only at high-level event boundaries."""

        accumulator = self._planner_events
        if accumulator is None:
            raise RuntimeError("Planner event accumulator is not initialized.")
        event = infos.get("learned_foothold_action_event")
        if event is None:
            raise KeyError("Missing causal PPO transition key: learned_foothold_action_event")
        raw_event_reward = infos.get("learned_foothold_event_reward")
        if raw_event_reward is None:
            raise KeyError(
                "Missing unscaled planner event reward: "
                "learned_foothold_event_reward"
            )
        planner_rewards = torch.as_tensor(
            raw_event_reward,
            device=self.device,
            dtype=torch.float32,
        )
        expected_shape = (accumulator.num_envs,)
        if tuple(planner_rewards.shape) != expected_shape:
            raise ValueError(
                "learned_foothold_event_reward must have shape "
                f"{expected_shape}, got {tuple(planner_rewards.shape)}."
            )
        if not torch.isfinite(planner_rewards).all().item():
            raise FloatingPointError(
                "learned_foothold_event_reward must be finite."
            )

        def record_event(obs, action, reward, next_state, terminal):
            self._require_sac().observe(obs, action, reward, next_state, terminal)

        completed = accumulator.process_step(
            observations=self.transition.observations.detach(),
            # The environment action term performs this normalization before
            # publishing the planner action to the sensor.  Replay must store
            # the same bounded action consumed by the SAC critics, while the
            # rollout transition itself remains the raw action sent to env.
            actions=radial_squash(
                self.transition.actions[..., self.motor_action_dim :].detach()
            ),
            rewards=planner_rewards.detach(),
            next_observations=next_obs.detach(),
            dones=torch.as_tensor(dones, device=self.device, dtype=torch.bool),
            event_mask=torch.as_tensor(event, device=self.device, dtype=torch.bool),
            record=record_event,
        )
        self._planner_events_since_update += completed

        # The parent still records the ordinary motor/AMP PPO transition and
        # clears the short-horizon transition object after doing so.
        super().process_env_step(
            rewards,
            dones,
            infos,
            next_obs,
            next_critic_obs,
        )

    def act(self, obs, critic_obs):
        """Keep the motor PPO sample and replace only the planner action."""

        actions = super().act(obs, critic_obs)
        sac = self._require_sac()
        with torch.no_grad():
            features = self.actor_critic.planner_features(
                obs,
                detach_shared=True,
            )
            planner_action, planner_log_prob = sac.act_raw_with_log_prob(features)
            planner_distribution = (
                self.actor_critic.planner_distribution_from_features(features)
            )
        actions = actions.clone()
        actions[..., self.motor_action_dim :] = planner_action
        self.transition.actions = actions.detach()
        # Keep rollout diagnostics and legacy storage fields consistent with
        # the planner distribution and transformed action sent to the
        # environment.
        self.transition.action_mean = self.transition.action_mean.clone()
        self.transition.action_sigma = self.transition.action_sigma.clone()
        self.transition.action_mean[..., self.motor_action_dim :] = (
            planner_distribution.mean
        )
        self.transition.action_sigma[..., self.motor_action_dim :] = (
            planner_distribution.scale
        )
        motor_distribution = Normal(
            self.transition.action_mean[..., : self.motor_action_dim],
            self.transition.action_sigma[..., : self.motor_action_dim],
        )
        self.transition.actions_log_prob = (
            motor_distribution.log_prob(actions[..., : self.motor_action_dim]).sum(
                dim=-1,
                keepdim=True,
            )
            + planner_log_prob.unsqueeze(-1)
        ).detach()
        return self.transition.actions

    def update(self, current_learning_iteration):
        # Event transitions are appended when their next high-level boundary
        # is observed, so a rollout window ending in the middle of a swing
        # does not create a fake 20ms next state.
        event_count = self._planner_events_since_update
        mean_losses, average_stats = super().update(current_learning_iteration)
        sac = self._require_sac()
        sac_stats = sac.update(new_event_count=event_count)
        self._planner_events_since_update = 0
        for key, value in sac_stats.items():
            average_stats[key] = torch.tensor(float(value), device=self.device)
        average_stats["sac_event_count"] = torch.tensor(
            float(event_count),
            device=self.device,
        )
        # Keep normalized planner exploration within the physical bounds that
        # the existing configuration already validated.
        self._clip_foothold_std_to_physical_bounds()
        return mean_losses, average_stats

    def state_dict(self):
        state = super().state_dict()
        state["foothold_sac_version"] = self._SAC_STATE_VERSION
        if self.sac is not None:
            state["foothold_sac"] = self.sac.state_dict()
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        sac = self._require_sac()
        if "foothold_sac" in state_dict:
            version = int(state_dict.get("foothold_sac_version", 0))
            if version < self._SAC_STATE_VERSION:
                # Version 1 replay stored the unsquashed action and paired it
                # with the immediate post-step observation.  Neither field
                # can be repaired after saving, so silently loading it would
                # train the new critic on a different MDP.  The actor itself
                # is already restored by the parent checkpoint; start only
                # the SAC critics, temperature, optimizer and replay fresh.
                print(
                    "[Warning] Legacy foothold SAC state detected; discarding "
                    "its replay/critics because event transition semantics "
                    "changed. Planner actor weights remain loaded."
                )
            else:
                sac.load_state_dict(state_dict["foothold_sac"])
        elif "foothold_sac_version" in state_dict:
            raise KeyError(
                "SAC checkpoint is missing foothold_sac state."
            )
        else:
            print(
                "[Warning] PPO planner checkpoint loaded; SAC critics, "
                "temperature, and replay start fresh."
            )

    def distributed_data_parallel(self):
        super().distributed_data_parallel()
        if not dist.is_initialized() or self.sac is None:
            return
        for module in (
            self.sac.critic_1,
            self.sac.critic_2,
            self.sac.target_critic_1,
            self.sac.target_critic_2,
        ):
            for parameter in module.parameters():
                dist.broadcast(parameter.data, src=0)
        dist.broadcast(self.sac.log_alpha.data, src=0)


def register_event_gated_foothold_algorithm() -> None:
    """Register explicitly without changing legacy import behavior."""

    import instinct_rl.algorithms as algorithms

    algorithms.EventGatedWasabiPPO = EventGatedWasabiPPO
    algorithms.EventGatedWasabiSAC = EventGatedWasabiSAC
