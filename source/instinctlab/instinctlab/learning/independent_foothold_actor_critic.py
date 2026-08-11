"""Actor-critic with isolated motor and learned-foothold policy heads."""

from collections.abc import Iterable
import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from instinct_rl.modules.actor_critic import get_activation
from instinct_rl.modules.encoder_actor_critic import EncoderActorCriticMixin
from instinct_rl.modules.moe_actor_critic import MoEActorCritic
from instinct_rl.utils.utils import get_subobs_by_components

if __package__:
    from .foothold_depth_encoder import FootholdDepthEncoder
else:  # Keep the focused source-file tests independent of IsaacLab imports.
    _depth_encoder_path = Path(__file__).with_name("foothold_depth_encoder.py")
    _depth_encoder_spec = importlib.util.spec_from_file_location(
        "foothold_depth_encoder", _depth_encoder_path
    )
    if _depth_encoder_spec is None or _depth_encoder_spec.loader is None:
        raise ImportError("Unable to load the foothold depth encoder module.")
    _depth_encoder_module = importlib.util.module_from_spec(_depth_encoder_spec)
    _depth_encoder_spec.loader.exec_module(_depth_encoder_module)
    FootholdDepthEncoder = _depth_encoder_module.FootholdDepthEncoder


class IndependentFootholdMoEActorCritic(MoEActorCritic):
    """Produce one environment action from two independent policy modules.

    The original MoE remains the 29-dimensional motor policy.  The final two
    action dimensions come from a small MLP whose input is detached from the
    shared observation feature.  The two value heads follow the same parameter
    boundary: critic zero belongs to motor execution and critic one belongs to
    foothold planning.
    """

    def __init__(
        self,
        obs_format,
        num_actions,
        *,
        motor_action_dim: int,
        foothold_hidden_dims=(128, 64),
        foothold_depth_output_size: int = 0,
        **kwargs,
    ):
        if motor_action_dim <= 0:
            raise ValueError("motor_action_dim must be positive.")
        if num_actions != motor_action_dim + 2:
            raise ValueError(
                "Independent foothold policy requires exactly two planner "
                f"actions after {motor_action_dim} motor actions; got "
                f"{num_actions} total actions."
            )
        if int(kwargs.get("num_rewards", 1)) != 2:
            raise ValueError(
                "Independent foothold policy requires two reward critics."
            )
        if not foothold_hidden_dims or any(
            int(width) <= 0 for width in foothold_hidden_dims
        ):
            raise ValueError("foothold_hidden_dims must contain positive sizes.")

        self.motor_action_dim = int(motor_action_dim)
        self.foothold_action_dim = 2
        self.foothold_hidden_dims = tuple(
            int(width) for width in foothold_hidden_dims
        )
        if foothold_depth_output_size < 0:
            raise ValueError("foothold_depth_output_size must be non-negative.")
        self.foothold_depth_output_size = int(foothold_depth_output_size)
        super().__init__(obs_format, num_actions, **kwargs)

        joint_std = self.std.detach().clone()
        del self.std
        self.motor_std = nn.Parameter(joint_std[: self.motor_action_dim])
        self.foothold_std = nn.Parameter(joint_std[self.motor_action_dim :])
        self.foothold_actor = self._build_foothold_actor()

    def _build_actor(self, _num_actions):
        # Keep the historical ``actor.*`` key space and only change its output
        # width from the environment action count to the motor action count.
        return super()._build_actor(self.motor_action_dim)

    def _build_foothold_actor(self) -> nn.Sequential:
        dimensions = (
            self.mlp_input_dim_a + self.foothold_depth_output_size,
            *self.foothold_hidden_dims,
            self.foothold_action_dim,
        )
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                layers.append(get_activation(self.activation))
        return nn.Sequential(*layers)

    def _action_mean(
        self,
        observations: torch.Tensor,
        foothold_depth_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        motor_mean = self.actor(observations)
        foothold_input = observations.detach()
        if self.foothold_depth_output_size:
            if foothold_depth_features is None:
                raise RuntimeError(
                    "Planner depth features are required by the learned "
                    "foothold actor."
                )
            if foothold_depth_features.shape[:-1] != observations.shape[:-1]:
                raise ValueError(
                    "Planner depth features must share the observation batch "
                    "dimensions."
                )
            if foothold_depth_features.shape[-1] != self.foothold_depth_output_size:
                raise ValueError("Planner depth feature width does not match policy configuration.")
            foothold_input = torch.cat(
                (foothold_input, foothold_depth_features),
                dim=-1,
            )
        foothold_mean = self.foothold_actor(foothold_input)
        return torch.cat((motor_mean, foothold_mean), dim=-1)

    def update_distribution(self, observations):
        mean = self._action_mean(observations)
        std = torch.cat((self.motor_std, self.foothold_std), dim=0)
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act_inference(self, observations):
        return self._action_mean(observations)

    def forward(self, observations):
        return self.act_inference(observations)

    def evaluate(self, critic_observations, **kwargs):
        if isinstance(critic_observations, list):
            if len(critic_observations) != 2:
                raise ValueError("Independent policy expects two critic inputs.")
            execution_input, foothold_input = critic_observations
        else:
            execution_input = critic_observations
            foothold_input = critic_observations
        return torch.cat(
            (
                self.critics[0](execution_input),
                self.critics[1](foothold_input.detach()),
            ),
            dim=-1,
        )

    def _parameters_with_prefixes(
        self,
        prefixes: Iterable[str],
        *,
        include: bool,
    ) -> tuple[nn.Parameter, ...]:
        prefixes = tuple(prefixes)
        selected: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            matches = any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in prefixes
            )
            if matches == include:
                selected.append(parameter)
        return tuple(selected)

    def foothold_parameters(self) -> tuple[nn.Parameter, ...]:
        return self._parameters_with_prefixes(
            (
                "foothold_actor",
                "foothold_depth_encoder",
                "foothold_std",
                "critics.1",
            ),
            include=True,
        )

    def motor_parameters(self) -> tuple[nn.Parameter, ...]:
        return self._parameters_with_prefixes(
            (
                "foothold_actor",
                "foothold_depth_encoder",
                "foothold_std",
                "critics.1",
            ),
            include=False,
        )

    @torch.no_grad()
    def clip_motor_std(self, minimum=None, maximum=None) -> None:
        self.motor_std.copy_(self.motor_std.clamp(min=minimum, max=maximum))

    @torch.no_grad()
    def clip_foothold_std(self, minimum=None, maximum=None) -> None:
        self.foothold_std.copy_(
            self.foothold_std.clamp(min=minimum, max=maximum)
        )

    @torch.no_grad()
    def clip_std(self, min=None, max=None) -> None:
        """Retain the upstream interface without merging the parameters."""

        self.clip_motor_std(minimum=min, maximum=max)
        self.clip_foothold_std(minimum=min, maximum=max)

    def export_as_onnx(self, observations, filedir):
        """Export the combined 29+2 policy instead of the motor head alone."""

        self.eval()
        output_path = os.path.join(filedir, "actor.onnx")
        with torch.no_grad():
            torch.onnx.export(
                self,
                observations,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
            )
        print(f"Exported independent foothold policy to {output_path}")


class IndependentFootholdEncoderMoEActorCritic(
    EncoderActorCriticMixin,
    IndependentFootholdMoEActorCritic,
):
    """Encoded parkour variant; the encoder is evaluated only once per call."""

    def __init__(
        self,
        obs_format,
        num_actions,
        *,
        foothold_depth_output_size=0,
        foothold_depth_hidden_channels=8,
        **kwargs,
    ):
        depth_output_size = int(foothold_depth_output_size)
        policy_segments = obs_format.get("policy", {})
        depth_shape = policy_segments.get("depth_image")
        if depth_output_size > 0 and depth_shape is None:
            raise ValueError(
                "Planner depth encoding requires a policy depth_image "
                "observation."
            )
        self._foothold_depth_input_shape = (
            tuple(int(value) for value in depth_shape)
            if depth_shape is not None
            else None
        )
        self._foothold_depth_hidden_channels = int(foothold_depth_hidden_channels)
        super().__init__(
            obs_format,
            num_actions,
            foothold_depth_output_size=depth_output_size,
            **kwargs,
        )
        if self.foothold_depth_output_size:
            self.foothold_depth_encoder = FootholdDepthEncoder(
                self._foothold_depth_input_shape,
                output_size=self.foothold_depth_output_size,
                hidden_channels=self._foothold_depth_hidden_channels,
            )

    def _encoded_action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        encoded_observations = self.encoders(observations)
        if not self.foothold_depth_output_size:
            # Saved configurations produced before the dedicated planner
            # encoder keep their original network shape and remain playable.
            return self._action_mean(encoded_observations)
        depth_image = get_subobs_by_components(
            observations,
            ["depth_image"],
            self.obs_segments,
        ).reshape(-1, *self._foothold_depth_input_shape)
        depth_features = self.foothold_depth_encoder(depth_image)
        return self._action_mean(encoded_observations, depth_features)

    def update_distribution(self, observations):
        mean = self._encoded_action_mean(observations)
        std = torch.cat((self.motor_std, self.foothold_std), dim=0)
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self._encoded_action_mean(observations)

    def forward(self, observations):
        return self.act_inference(observations)

    def export_as_onnx(
        self,
        observations,
        filedir,
        encoder_as_seperate_file=False,
    ):
        # A split export would send only the 29D motor actor through the
        # upstream path.  Always export the complete encoded 31D policy.
        return EncoderActorCriticMixin.export_as_onnx(
            self,
            observations,
            filedir,
            encoder_as_seperate_file=False,
        )
