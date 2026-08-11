"""Small depth encoder used only by the learned foothold policy head."""

from __future__ import annotations

import torch
from torch import nn


class FootholdDepthEncoder(nn.Module):
    """Encode the existing camera depth tensor for foothold decisions.

    The encoder intentionally consumes the same preprocessed depth image that
    is available to the motor policy.  It does not use simulation-only height
    scanners or terrain labels.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        *,
        output_size: int = 64,
        hidden_channels: int = 8,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or any(int(value) <= 0 for value in input_shape):
            raise ValueError("input_shape must be a positive (C, H, W) tuple.")
        if output_size <= 0 or hidden_channels <= 0:
            raise ValueError("output_size and hidden_channels must be positive.")

        self.input_shape = tuple(int(value) for value in input_shape)
        input_channels, input_height, input_width = self.input_shape
        pooled_shape = (min(4, input_height), min(8, input_width))
        self.output_size = int(output_size)
        self.features = nn.Sequential(
            # A depthwise spatial filter followed by a pointwise temporal/
            # channel mixer preserves image layout at a fraction of the cost
            # of two dense convolutions over all 4096 environments.
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                padding=1,
                groups=input_channels,
            ),
            nn.ELU(),
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(pooled_shape),
        )
        self.projection = nn.Sequential(
            nn.Linear(
                hidden_channels * pooled_shape[0] * pooled_shape[1],
                self.output_size,
            ),
            nn.ELU(),
        )

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        if depth_image.ndim != 4:
            raise ValueError("depth_image must have shape (batch, C, H, W).")
        if tuple(depth_image.shape[1:]) != self.input_shape:
            raise ValueError(
                "depth_image shape does not match the configured planner "
                f"input: {tuple(depth_image.shape[1:])} != {self.input_shape}."
            )
        depth_image = torch.nan_to_num(
            depth_image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return self.projection(self.features(depth_image).flatten(1))
