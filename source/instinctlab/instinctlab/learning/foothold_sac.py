"""Small SAC learner used by the learned foothold planner."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable
from math import floor, isfinite
from time import perf_counter

import torch
from torch import nn, optim
from torch.distributions import Normal

from .foothold_sac_replay import FootholdReplayBatch, FootholdReplayBuffer


def sac_backup_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    target_q1: torch.Tensor,
    target_q2: torch.Tensor,
    next_log_prob: torch.Tensor,
    alpha: torch.Tensor | float,
    gamma: float,
) -> torch.Tensor:
    """Build a clipped-double-Q SAC Bellman target."""

    values = (rewards, dones, target_q1, target_q2, next_log_prob)
    if not all(torch.isfinite(value).all().item() for value in values):
        raise FloatingPointError("SAC target inputs must be finite.")
    if rewards.shape != dones.shape:
        raise ValueError("rewards and dones must share one shape.")
    if target_q1.shape != rewards.shape or target_q2.shape != rewards.shape:
        raise ValueError("target Q values must match rewards.")
    if next_log_prob.shape != rewards.shape:
        raise ValueError("next_log_prob must match rewards.")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    alpha_tensor = torch.as_tensor(alpha, device=rewards.device, dtype=rewards.dtype)
    if not torch.isfinite(alpha_tensor).all().item() or torch.any(alpha_tensor < 0.0):
        raise ValueError("alpha must be finite and non-negative.")
    bootstrap = torch.minimum(target_q1, target_q2) - alpha_tensor * next_log_prob
    return rewards + float(gamma) * (~dones.bool()).to(rewards.dtype) * bootstrap


@torch.no_grad()
def polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Move target parameters toward source parameters by ``tau``."""

    if not 0.0 < float(tau) <= 1.0:
        raise ValueError("tau must lie in (0, 1].")
    target_parameters = tuple(target.parameters())
    source_parameters = tuple(source.parameters())
    if len(target_parameters) != len(source_parameters):
        raise ValueError("source and target modules must have equal parameters.")
    for target_parameter, source_parameter in zip(
        target_parameters,
        source_parameters,
    ):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


def _mlp(input_dim: int, output_dim: int, hidden_dims: Iterable[int]) -> nn.Sequential:
    dimensions = (int(input_dim), *[int(width) for width in hidden_dims], int(output_dim))
    if any(width <= 0 for width in dimensions):
        raise ValueError("MLP dimensions must be positive.")
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(dimensions) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def radial_squash(raw_action: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained vector bijectively into the open unit ball."""

    if raw_action.ndim < 1 or raw_action.shape[-1] <= 0:
        raise ValueError("raw_action must have a non-empty final dimension.")
    squared_norm = raw_action.square().sum(dim=-1, keepdim=True)
    return raw_action / torch.sqrt(1.0 + squared_norm)


def radial_squash_log_abs_det_jacobian(raw_action: torch.Tensor) -> torch.Tensor:
    """Return ``log|det(da/du)|`` for :func:`radial_squash`."""

    if raw_action.ndim < 1 or raw_action.shape[-1] <= 0:
        raise ValueError("raw_action must have a non-empty final dimension.")
    dimension = raw_action.shape[-1]
    squared_norm = raw_action.square().sum(dim=-1)
    # Tangential eigenvalues have exponent -1/2 and the radial eigenvalue
    # exponent -3/2, giving -(d + 2) / 2 in d dimensions.
    return -0.5 * float(dimension + 2) * torch.log1p(squared_norm)


class PlannerEventAccumulator:
    """Close one SAC transition at the next planner decision event.

    The low-level simulator steps much faster than the foothold planner.  A
    planner action therefore remains pending while swing and touchdown are
    executed; a normal control frame must never be used as its ``next_obs``.
    ``record`` receives only completed event transitions.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str,
    ) -> None:
        if int(num_envs) <= 0 or int(obs_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError("event accumulator dimensions must be positive.")
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pending_obs = torch.zeros(
            self.num_envs, self.obs_dim, device=self.device
        )
        self.pending_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self.pending_rewards = torch.zeros(self.num_envs, device=self.device)

    def _batch(
        self,
        value: torch.Tensor,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=dtype)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}.")
        return tensor

    def process_step(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        event_mask: torch.Tensor,
        record: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            None,
        ],
    ) -> int:
        """Consume one low-level step and return completed event count."""

        obs = self._batch(
            observations,
            name="observations",
            shape=(self.num_envs, self.obs_dim),
            dtype=torch.float32,
        )
        action = self._batch(
            actions,
            name="actions",
            shape=(self.num_envs, self.action_dim),
            dtype=torch.float32,
        )
        reward = self._batch(
            rewards,
            name="rewards",
            shape=(self.num_envs,),
            dtype=torch.float32,
        )
        next_obs = self._batch(
            next_observations,
            name="next_observations",
            shape=(self.num_envs, self.obs_dim),
            dtype=torch.float32,
        )
        done = self._batch(
            dones,
            name="dones",
            shape=(self.num_envs,),
            dtype=torch.bool,
        )
        event = self._batch(
            event_mask,
            name="event_mask",
            shape=(self.num_envs,),
            dtype=torch.bool,
        )
        completed = 0

        # The reward from a non-event step belongs to the currently pending
        # high-level action.  A step that starts a new event belongs to the
        # new action instead, so it is deliberately not added here.
        continuing = self.pending & ~event
        self.pending_rewards[continuing] += reward[continuing]

        # The current observation is the state at which the next planner
        # action is about to be selected.  It is the correct next state for
        # the preceding event, not the post-step observation of this action.
        boundary = self.pending & event
        if torch.any(boundary):
            indices = boundary.nonzero(as_tuple=False).flatten()
            boundary_dones = done[indices]
            boundary_next_obs = torch.where(
                boundary_dones.unsqueeze(-1),
                next_obs[indices],
                obs[indices],
            )
            record(
                self.pending_obs[indices],
                self.pending_actions[indices],
                self.pending_rewards[indices],
                boundary_next_obs,
                boundary_dones,
            )
            completed += int(indices.numel())
            self.pending[indices] = False

        # A reset/fall before another planner event closes the pending event
        # with a real terminal next observation.
        terminal = self.pending & ~event & done
        if torch.any(terminal):
            indices = terminal.nonzero(as_tuple=False).flatten()
            record(
                self.pending_obs[indices],
                self.pending_actions[indices],
                self.pending_rewards[indices],
                next_obs[indices],
                torch.ones(indices.numel(), dtype=torch.bool, device=self.device),
            )
            completed += int(indices.numel())
            self.pending[indices] = False

        # Start the new event after closing the previous one.  Its reward is
        # initialized from the same simulator step as its action.
        if torch.any(event):
            indices = event.nonzero(as_tuple=False).flatten()
            self.pending_obs[indices] = obs[indices]
            self.pending_actions[indices] = action[indices]
            self.pending_rewards[indices] = reward[indices]
            self.pending[indices] = ~done[indices]

            # If a planner event and reset happen in one simulator step, the
            # event has a genuine terminal transition immediately.
            event_terminal = done[indices]
            if torch.any(event_terminal):
                terminal_indices = indices[event_terminal]
                record(
                    self.pending_obs[terminal_indices],
                    self.pending_actions[terminal_indices],
                    self.pending_rewards[terminal_indices],
                    next_obs[terminal_indices],
                    torch.ones(
                        terminal_indices.numel(),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                )
                completed += int(terminal_indices.numel())

        return completed

    def clear(self) -> None:
        """Discard pending events, e.g. when rebuilding the environment."""

        self.pending.zero_()
        self.pending_rewards.zero_()


@dataclass(frozen=True)
class FootholdSACConfig:
    obs_dim: int
    action_dim: int = 2
    hidden_dims: tuple[int, ...] = (128, 128)
    replay_capacity: int = 100_000
    batch_size: int = 256
    warmup_events: int = 1_024
    target_sample_ratio: float = 0.125
    max_updates_per_rollout: int = 4
    actor_lr: float = 1.0e-4
    critic_lr: float = 1.0e-4
    alpha_lr: float = 1.0e-4
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: float = -2.0
    max_grad_norm: float = 1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    # Replay can retain the flattened environment observation while the SAC
    # networks consume a compact planner feature vector produced by a
    # caller-supplied encoder.
    replay_obs_dim: int | None = None

    def __post_init__(self) -> None:
        if self.obs_dim <= 0 or self.action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive.")
        if self.replay_obs_dim is not None and self.replay_obs_dim <= 0:
            raise ValueError("replay_obs_dim must be positive when provided.")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if self.replay_capacity <= 0 or self.batch_size <= 0:
            raise ValueError("replay_capacity and batch_size must be positive.")
        if self.warmup_events < 0 or self.max_updates_per_rollout < 0:
            raise ValueError(
                "warmup_events and max_updates_per_rollout cannot be negative."
            )
        if not isfinite(self.target_sample_ratio) or self.target_sample_ratio < 0.0:
            raise ValueError("target_sample_ratio must be finite and non-negative.")
        for name in ("actor_lr", "critic_lr", "alpha_lr", "max_grad_norm"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must lie in (0, 1].")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max.")


class _GaussianActor(nn.Module):
    def __init__(self, config: FootholdSACConfig) -> None:
        super().__init__()
        self.backbone = _mlp(config.obs_dim, 2 * config.action_dim, config.hidden_dims)
        self.action_dim = config.action_dim
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def distribution(self, features: torch.Tensor) -> Normal:
        output = self.backbone(features)
        mean, log_std = output.split(self.action_dim, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp())


class _QNetwork(nn.Module):
    def __init__(self, config: FootholdSACConfig) -> None:
        super().__init__()
        self.network = _mlp(
            config.obs_dim + config.action_dim,
            1,
            config.hidden_dims,
        )

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((features, actions), dim=-1)).squeeze(-1)


class FootholdSAC(nn.Module):
    """SAC learner for sparse, event-triggered foothold actions."""

    def __init__(
        self,
        config: FootholdSACConfig,
        *,
        device: torch.device | str,
        actor_distribution_fn: Callable[[torch.Tensor], Normal] | None = None,
        actor_parameters: Iterable[nn.Parameter] | None = None,
        feature_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self._feature_fn = feature_fn
        if actor_distribution_fn is None:
            self.actor: nn.Module | None = _GaussianActor(config).to(self.device)
            self._actor_distribution_fn = self.actor.distribution
            actor_parameters = self.actor.parameters()
        else:
            self.actor = None
            self._actor_distribution_fn = actor_distribution_fn
            if actor_parameters is None:
                raise ValueError(
                    "actor_parameters are required with an external actor."
                )
        actor_parameters = tuple(actor_parameters)
        if not actor_parameters:
            raise ValueError("planner actor must expose trainable parameters.")
        self.critic_1 = _QNetwork(config).to(self.device)
        self.critic_2 = _QNetwork(config).to(self.device)
        self.target_critic_1 = _QNetwork(config).to(self.device)
        self.target_critic_2 = _QNetwork(config).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        for parameter in self.target_critic_1.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_critic_2.parameters():
            parameter.requires_grad_(False)

        self.log_alpha = nn.Parameter(torch.zeros((), device=self.device))
        self.actor_optimizer = optim.Adam(actor_parameters, lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(
            tuple(self.critic_1.parameters()) + tuple(self.critic_2.parameters()),
            lr=config.critic_lr,
        )
        self.alpha_optimizer = optim.Adam((self.log_alpha,), lr=config.alpha_lr)
        self.replay = FootholdReplayBuffer(
            config.replay_capacity,
            config.replay_obs_dim or config.obs_dim,
            config.action_dim,
            self.device,
        )
        self.total_updates = 0
        self.skipped_updates = 0
        # Fractional update credit keeps the long-run sample ratio stable when
        # event counts vary between rollouts (e.g. 64 vs. 4096 environments).
        # It is always reduced to the fractional remainder after a rollout;
        # integer work above the per-rollout cap is intentionally discarded.
        self.update_credit = 0.0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _check_features(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(features, device=self.device, dtype=torch.float32)
        if features.ndim != 2 or features.shape[-1] != self.config.obs_dim:
            raise ValueError(
                "planner features must have shape "
                f"[batch, {self.config.obs_dim}]."
            )
        if not torch.isfinite(features).all().item():
            raise FloatingPointError("planner features must be finite.")
        return features

    def _features_from_replay(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode raw replay observations for the SAC networks."""

        replay_obs_dim = self.config.replay_obs_dim or self.config.obs_dim
        observations = torch.as_tensor(
            observations,
            device=self.device,
            dtype=torch.float32,
        )
        if (
            observations.ndim != 2
            or observations.shape[-1] != replay_obs_dim
        ):
            raise ValueError(
                "replay observations must have shape "
                f"[batch, {replay_obs_dim}]."
            )
        if not torch.isfinite(observations).all().item():
            raise FloatingPointError("replay observations must be finite.")
        features = observations if self._feature_fn is None else self._feature_fn(observations)
        return self._check_features(features)

    def _sample_action_and_log_prob(
        self,
        features: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_action, log_prob = self._sample_raw_action_and_log_prob(
            features,
            deterministic=deterministic,
        )
        return radial_squash(raw_action), log_prob

    def _sample_raw_action_and_log_prob(
        self,
        features: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self._actor_distribution_fn(features)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        log_prob = (
            distribution.log_prob(raw_action).sum(dim=-1)
            - radial_squash_log_abs_det_jacobian(raw_action)
        )
        return raw_action, log_prob

    @torch.no_grad()
    def act(self, features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action, _ = self.act_with_log_prob(features, deterministic=deterministic)
        return action

    @torch.no_grad()
    def act_with_log_prob(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a normalized planner action and its transformed log-prob."""

        features = self._check_features(features)
        return self._sample_action_and_log_prob(
            features,
            deterministic=deterministic,
        )

    @torch.no_grad()
    def act_raw_with_log_prob(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the raw action expected by the environment action term.

        The action term applies the same radial squash before the planner
        consumes it.  SAC trains on the squashed action, but rollout must
        still pass the unsquashed sample so that the transform is applied
        exactly once.
        """

        features = self._check_features(features)
        return self._sample_raw_action_and_log_prob(
            features,
            deterministic=deterministic,
        )

    def observe(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.replay.add(obs, actions, rewards, next_obs, dones)

    def _finite_batch(self, batch: FootholdReplayBatch) -> None:
        if not all(torch.isfinite(value).all().item() for value in batch if value.dtype != torch.bool):
            raise FloatingPointError("SAC replay batch must be finite.")

    def _skip_nonfinite_update(self) -> None:
        """Discard pending gradients after a rejected SAC batch."""

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        self.skipped_updates += 1

    def update(self, new_event_count: int = 0) -> dict[str, float]:
        """Run event-scaled SAC updates for newly completed planner events.

        ``new_event_count`` counts complete high-level planner transitions
        appended since the previous call.  The update budget is accumulated as
        fractional credit so the scheduler remains well behaved for both small
        and large vectorized environments.
        """

        new_event_count = int(new_event_count)
        if new_event_count < 0:
            raise ValueError("new_event_count cannot be negative.")

        zero_stats = {
            "sac_update_count": 0.0,
            "sac_requested_update_count": 0.0,
            "sac_dropped_update_count": 0.0,
            "sac_new_event_count": float(new_event_count),
            "sac_sample_ratio": 0.0,
            "sac_update_credit": float(self.update_credit),
            "sac_update_time": 0.0,
            "replay_size": float(len(self.replay)),
            "sac_skipped_update_count": float(self.skipped_updates),
            "sac_actor_loss": 0.0,
            "sac_critic_loss": 0.0,
            "sac_alpha_loss": 0.0,
            "sac_alpha": float(self.alpha.detach().item()),
            "sac_q_mean": 0.0,
        }
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_events):
            self.skipped_updates += 1
            zero_stats["sac_skipped_update_count"] = float(self.skipped_updates)
            return zero_stats

        self.update_credit += (
            float(new_event_count) * self.config.target_sample_ratio
            / float(self.config.batch_size)
        )
        requested_updates = floor(self.update_credit)
        if requested_updates <= 0:
            return zero_stats | {
                "sac_update_credit": float(self.update_credit),
            }

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        alpha_loss_value = 0.0
        q_value = 0.0
        available_updates = len(self.replay) // self.config.batch_size
        updates = min(
            requested_updates,
            self.config.max_updates_per_rollout,
            available_updates,
        )
        # Consume the requested integer credit even when the per-rollout cap
        # limits actual work.  This prevents an event burst from creating an
        # unbounded backlog that would slow every later rollout.
        self.update_credit -= float(requested_updates)
        dropped_updates = requested_updates - updates
        if updates <= 0:
            self.skipped_updates += 1
            return zero_stats | {
                "sac_requested_update_count": float(requested_updates),
                "sac_dropped_update_count": float(dropped_updates),
                "sac_update_credit": float(self.update_credit),
                "sac_skipped_update_count": float(self.skipped_updates),
            }

        completed_updates = 0
        update_start = perf_counter()
        for _ in range(updates):
            try:
                batch = self.replay.sample(self.config.batch_size)
                self._finite_batch(batch)
                # The critic update must not backpropagate through the
                # planner actor or its depth encoder.  Recompute those
                # features below for the actor update, where their gradients
                # are intentional.
                features = self._features_from_replay(batch.obs).detach()
                with torch.no_grad():
                    next_features = self._features_from_replay(batch.next_obs)
                    next_action, next_log_prob = self._sample_action_and_log_prob(
                        next_features,
                        deterministic=False,
                    )
                    target_q1 = self.target_critic_1(next_features, next_action)
                    target_q2 = self.target_critic_2(next_features, next_action)
                    target = sac_backup_target(
                        batch.rewards,
                        batch.dones,
                        target_q1,
                        target_q2,
                        next_log_prob,
                        self.alpha.detach(),
                        self.config.gamma,
                    )

                current_q1 = self.critic_1(features, batch.actions)
                current_q2 = self.critic_2(features, batch.actions)
                critic_loss = 0.5 * (
                    (current_q1 - target).pow(2).mean()
                    + (current_q2 - target).pow(2).mean()
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                try:
                    nn.utils.clip_grad_norm_(
                        tuple(self.critic_1.parameters())
                        + tuple(self.critic_2.parameters()),
                        self.config.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    raise FloatingPointError("SAC critic gradients must be finite.") from exc
                self.critic_optimizer.step()

                # Recompute features so the actor update has a fresh graph and
                # can train a dedicated planner encoder without retaining the
                # critic graph. Shared motor features remain detached by fn.
                actor_features = self._features_from_replay(batch.obs)
                action, log_prob = self._sample_action_and_log_prob(
                    actor_features,
                    deterministic=False,
                )
                q1 = self.critic_1(actor_features, action)
                q2 = self.critic_2(actor_features, action)
                q_min = torch.minimum(q1, q2)
                actor_loss = (self.alpha.detach() * log_prob - q_min).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                try:
                    nn.utils.clip_grad_norm_(
                        tuple(self.actor_optimizer.param_groups[0]["params"]),
                        self.config.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    raise FloatingPointError("SAC actor gradients must be finite.") from exc
                self.actor_optimizer.step()

                alpha_loss = -(
                    self.log_alpha * (log_prob.detach() + self.config.target_entropy)
                ).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    self.log_alpha.clamp_(-20.0, 2.0)

                polyak_update(self.target_critic_1, self.critic_1, self.config.tau)
                polyak_update(self.target_critic_2, self.critic_2, self.config.tau)
                self.total_updates += 1
                completed_updates += 1
                actor_loss_value += float(actor_loss.detach().item())
                critic_loss_value += float(critic_loss.detach().item())
                alpha_loss_value += float(alpha_loss.detach().item())
                q_value += float(q_min.detach().mean().item())
            except FloatingPointError:
                self._skip_nonfinite_update()

        if completed_updates == 0:
            return zero_stats | {
                "sac_requested_update_count": float(requested_updates),
                "sac_dropped_update_count": float(dropped_updates),
                "sac_update_credit": float(self.update_credit),
                "sac_update_time": perf_counter() - update_start,
                "replay_size": float(len(self.replay)),
                "sac_skipped_update_count": float(self.skipped_updates),
            }

        divisor = float(completed_updates)
        return {
            "sac_update_count": float(completed_updates),
            "sac_requested_update_count": float(requested_updates),
            "sac_dropped_update_count": float(dropped_updates),
            "sac_new_event_count": float(new_event_count),
            "sac_sample_ratio": (
                float(completed_updates * self.config.batch_size)
                / float(new_event_count)
                if new_event_count > 0
                else 0.0
            ),
            "sac_update_credit": float(self.update_credit),
            "sac_update_time": perf_counter() - update_start,
            "replay_size": float(len(self.replay)),
            "sac_skipped_update_count": float(self.skipped_updates),
            "sac_actor_loss": actor_loss_value / divisor,
            "sac_critic_loss": critic_loss_value / divisor,
            "sac_alpha_loss": alpha_loss_value / divisor,
            "sac_alpha": float(self.alpha.detach().item()),
            "sac_q_mean": q_value / divisor,
        }

    def state_dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        state = super().state_dict(*args, **kwargs)
        state["actor_optimizer"] = self.actor_optimizer.state_dict()
        state["critic_optimizer"] = self.critic_optimizer.state_dict()
        state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        state["replay"] = self.replay.state_dict()
        state["total_updates"] = self.total_updates
        state["skipped_updates"] = self.skipped_updates
        state["update_credit"] = self.update_credit
        return state

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[no-untyped-def]
        optimizer_keys = {
            "actor_optimizer",
            "critic_optimizer",
            "alpha_optimizer",
            "replay",
            "total_updates",
            "skipped_updates",
            "update_credit",
        }
        module_state = {
            key: value for key, value in state_dict.items() if key not in optimizer_keys
        }
        result = super().load_state_dict(module_state, strict=strict)
        if "actor_optimizer" in state_dict:
            self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        if "critic_optimizer" in state_dict:
            self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        if "alpha_optimizer" in state_dict:
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        if "replay" in state_dict:
            self.replay.load_state_dict(state_dict["replay"])
        self.total_updates = int(state_dict.get("total_updates", 0))
        self.skipped_updates = int(state_dict.get("skipped_updates", 0))
        self.update_credit = float(state_dict.get("update_credit", 0.0))
        if not 0.0 <= self.update_credit < 1.0:
            raise ValueError("SAC update_credit must lie in [0, 1).")
        return result
