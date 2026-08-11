"""Rollout storage carrying causal learned-foothold action events."""

from collections import namedtuple

import torch

from instinct_rl.storage.rollout_storage import RolloutStorage


class FootholdTransition(RolloutStorage.Transition):
    """PPO transition extended with the learned-foothold consumption event."""

    def __init__(self):
        super().__init__()
        self.foothold_action_event: torch.Tensor | None = None


FootholdMiniBatch = namedtuple(
    "FootholdMiniBatch",
    [*RolloutStorage.MiniBatch._fields, "foothold_action_event"],
)


class FootholdRolloutStorage(RolloutStorage):
    """Keep sparse planner events aligned with their rollout transitions."""

    Transition = FootholdTransition
    MiniBatch = FootholdMiniBatch

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        critic_obs_shape,
        actions_shape,
        num_rewards=1,
        device="cpu",
    ):
        super().__init__(
            num_envs,
            num_transitions_per_env,
            obs_shape,
            critic_obs_shape,
            actions_shape,
            num_rewards=num_rewards,
            device=device,
        )
        self.foothold_action_event = torch.zeros(
            num_transitions_per_env,
            num_envs,
            dtype=torch.bool,
            device=self.device,
        )

    def add_transitions(self, transition: FootholdTransition):
        event = transition.foothold_action_event
        if event is None:
            raise ValueError("Foothold action event is required.")
        if event.dtype != torch.bool:
            raise TypeError("Foothold action event must use torch.bool.")
        if event.shape != (self.num_envs,):
            raise ValueError(
                "Foothold action event must have shape "
                f"({self.num_envs},), got {tuple(event.shape)}."
            )

        step = self.step
        super().add_transitions(transition)
        self.foothold_action_event[step].copy_(event)

    def compute_returns(self, last_values, gamma, lam):
        """Compute GAE and normalize each reward advantage independently."""

        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = (
                delta
                + next_is_not_terminal * gamma * lam * advantage
            )
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        mean = self.advantages.mean(dim=(0, 1), keepdim=True)
        std = self.advantages.std(
            dim=(0, 1),
            unbiased=False,
            keepdim=True,
        )
        self.advantages = (self.advantages - mean) / (std + 1.0e-8)

    def get_minibatch_from_selection(
        self,
        T_select,
        B_select,
        padded_B_slice=None,
        prev_done_mask=None,
    ):
        minibatch = super().get_minibatch_from_selection(
            T_select,
            B_select,
            padded_B_slice,
            prev_done_mask,
        )
        event_batch = self.foothold_action_event[T_select, B_select]
        return self.MiniBatch(*minibatch, event_batch)
