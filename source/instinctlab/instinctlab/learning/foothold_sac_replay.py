"""Small device-resident replay buffer for sparse foothold events."""

from __future__ import annotations

from collections import namedtuple

import torch


FootholdReplayBatch = namedtuple(
    "FootholdReplayBatch",
    ("obs", "actions", "rewards", "next_obs", "dones"),
)


class FootholdReplayBuffer:
    """Circular replay storage with fixed-size flat observations."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive.")
        if int(obs_dim) <= 0:
            raise ValueError("obs_dim must be positive.")
        if int(action_dim) <= 0:
            raise ValueError("action_dim must be positive.")

        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.obs = torch.zeros(
            self.capacity,
            self.obs_dim,
            device=self.device,
        )
        self.actions = torch.zeros(
            self.capacity,
            self.action_dim,
            device=self.device,
        )
        self.rewards = torch.zeros(self.capacity, device=self.device)
        self.next_obs = torch.zeros(
            self.capacity,
            self.obs_dim,
            device=self.device,
        )
        self.dones = torch.zeros(
            self.capacity,
            dtype=torch.bool,
            device=self.device,
        )
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def _batch_tensor(
        self,
        value: torch.Tensor,
        *,
        name: str,
        trailing_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=dtype)
        expected_ndim = 1 + len(trailing_shape)
        if tensor.ndim != expected_ndim or tuple(tensor.shape[1:]) != trailing_shape:
            raise ValueError(
                f"{name} must have shape [batch, {', '.join(map(str, trailing_shape))}]."
            )
        return tensor

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        obs = self._batch_tensor(
            obs,
            name="obs",
            trailing_shape=(self.obs_dim,),
            dtype=torch.float32,
        )
        actions = self._batch_tensor(
            actions,
            name="actions",
            trailing_shape=(self.action_dim,),
            dtype=torch.float32,
        )
        rewards = self._batch_tensor(
            rewards,
            name="rewards",
            trailing_shape=(),
            dtype=torch.float32,
        )
        next_obs = self._batch_tensor(
            next_obs,
            name="next_obs",
            trailing_shape=(self.obs_dim,),
            dtype=torch.float32,
        )
        dones = self._batch_tensor(
            dones,
            name="dones",
            trailing_shape=(),
            dtype=torch.bool,
        )
        batch_size = obs.shape[0]
        if not (
            actions.shape[0]
            == rewards.shape[0]
            == next_obs.shape[0]
            == dones.shape[0]
            == batch_size
        ):
            raise ValueError("all replay fields must have the same batch size.")
        if batch_size == 0:
            return

        if batch_size > self.capacity:
            obs = obs[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            dones = dones[-self.capacity :]
            batch_size = self.capacity

        indices = (
            torch.arange(batch_size, device=self.device, dtype=torch.long)
            + self.position
        ) % self.capacity
        self.obs.index_copy_(0, indices, obs)
        self.actions.index_copy_(0, indices, actions)
        self.rewards.index_copy_(0, indices, rewards)
        self.next_obs.index_copy_(0, indices, next_obs)
        self.dones.index_copy_(0, indices, dones)
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, batch_size: int) -> FootholdReplayBatch:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if self.size < int(batch_size):
            raise ValueError(
                "batch_size cannot exceed the number of replay transitions."
            )
        indices = torch.randint(
            0,
            self.size,
            (int(batch_size),),
            device=self.device,
        )
        return FootholdReplayBatch(
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "position": self.position,
            "size": self.size,
            "obs": self.obs.clone(),
            "actions": self.actions.clone(),
            "rewards": self.rewards.clone(),
            "next_obs": self.next_obs.clone(),
            "dones": self.dones.clone(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        for key, expected in (
            ("capacity", self.capacity),
            ("obs_dim", self.obs_dim),
            ("action_dim", self.action_dim),
        ):
            if int(state[key]) != expected:
                raise ValueError(f"Replay {key} does not match this buffer.")
        self.position = int(state["position"])
        self.size = int(state["size"])
        if not 0 <= self.position < self.capacity:
            raise ValueError("Replay position is out of range.")
        if not 0 <= self.size <= self.capacity:
            raise ValueError("Replay size is out of range.")
        for key in ("obs", "actions", "rewards", "next_obs", "dones"):
            value = torch.as_tensor(state[key], device=self.device)
            target = getattr(self, key)
            if value.shape != target.shape:
                raise ValueError(f"Replay tensor {key} shape does not match.")
            target.copy_(value.to(dtype=target.dtype))
