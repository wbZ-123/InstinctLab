"""Audited initialization from a legacy 29-action parkour checkpoint."""

from dataclasses import dataclass, field
import re
from collections.abc import Mapping, Sequence
from math import isfinite, prod
from pathlib import Path

import torch


@dataclass
class FootholdCheckpointMigrationReport:
    """Exact record of copied and deliberately initialized parameters."""

    copied: list[str] = field(default_factory=list)
    expanded: list[str] = field(default_factory=list)
    initialized: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    source_learning_rate: float | None = None


_FIRST_INPUT_WEIGHT = re.compile(
    r"^(actor|critics\.0)\.(gate\.0|experts\.\d+\.0)\.weight$"
)


def learned_foothold_policy_input_expansion(
    *,
    nominal_foothold_dim: int,
) -> int:
    """Return the exact flattened policy-input growth in learned mode."""

    if (
        not isinstance(nominal_foothold_dim, int)
        or isinstance(nominal_foothold_dim, bool)
        or nominal_foothold_dim <= 0
    ):
        raise ValueError("nominal_foothold_dim must be a positive integer.")
    return nominal_foothold_dim


def build_legacy_input_column_map(
    destination_segments: Mapping[str, Sequence[int]],
    *,
    temporal_appends: Mapping[str, tuple[int, int]],
    new_components: set[str] | frozenset[str] = frozenset(),
) -> list[int]:
    """Map legacy flattened columns into an expanded temporal observation.

    Each ``temporal_appends`` entry is ``(new_values_per_frame, history)``.
    New values are appended within every frame, so a single prefix copy would
    shift all later frames and observation components.
    """

    unknown = sorted(set(temporal_appends) - set(destination_segments))
    if unknown:
        raise ValueError(
            "Temporal append components are missing from destination: "
            + ", ".join(unknown)
        )
    unknown_new = sorted(set(new_components) - set(destination_segments))
    if unknown_new:
        raise ValueError(
            "New components are missing from destination: "
            + ", ".join(unknown_new)
        )
    overlap = sorted(set(temporal_appends) & set(new_components))
    if overlap:
        raise ValueError(
            "Components cannot be both temporal appends and wholly new: "
            + ", ".join(overlap)
        )

    column_map: list[int] = []
    destination_offset = 0
    for component_name, shape in destination_segments.items():
        component_size = prod(shape)
        if component_size <= 0:
            raise ValueError(
                f"Destination component {component_name!r} must be non-empty."
            )
        expansion = temporal_appends.get(component_name)
        if component_name in new_components:
            pass
        elif expansion is None:
            column_map.extend(
                range(
                    destination_offset,
                    destination_offset + component_size,
                )
            )
        else:
            appended_per_frame, history_length = expansion
            if appended_per_frame <= 0 or history_length <= 0:
                raise ValueError(
                    "Temporal append size and history must be positive."
                )
            if component_size % history_length != 0:
                raise ValueError(
                    f"Component {component_name!r} size {component_size} "
                    f"is not divisible by history {history_length}."
                )
            destination_frame_size = component_size // history_length
            source_frame_size = (
                destination_frame_size - appended_per_frame
            )
            if source_frame_size <= 0:
                raise ValueError(
                    f"Component {component_name!r} has no legacy values."
                )
            for frame_index in range(history_length):
                frame_offset = (
                    destination_offset
                    + frame_index * destination_frame_size
                )
                column_map.extend(
                    range(
                        frame_offset,
                        frame_offset + source_frame_size,
                    )
                )
        destination_offset += component_size
    return column_map


def _source_key_for_destination(destination_key: str) -> str | None:
    if (
        destination_key.startswith("critics.1.")
        or destination_key.startswith("foothold_actor.")
        or destination_key.startswith("foothold_depth_encoder.")
        or destination_key == "foothold_std"
    ):
        return None
    if destination_key.startswith("critics.0."):
        return "critic." + destination_key.removeprefix("critics.0.")
    if destination_key == "motor_std":
        return "std"
    return destination_key


def _copy_like(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    return source.to(
        device=destination.device,
        dtype=destination.dtype,
    )


def migrate_foothold_model_state(
    source_state: Mapping[str, torch.Tensor],
    destination_state: Mapping[str, torch.Tensor],
    *,
    motor_action_dim: int,
    input_column_maps: Mapping[str, Sequence[int]],
    foothold_normalized_std: Sequence[float],
) -> tuple[dict[str, torch.Tensor], FootholdCheckpointMigrationReport]:
    """Migrate only the explicitly approved legacy-to-learned expansions."""

    if motor_action_dim <= 0:
        raise ValueError("motor_action_dim must be positive.")
    missing_maps = {"actor", "critics.0"} - set(input_column_maps)
    if missing_maps:
        raise ValueError(
            "Missing legacy input column maps: "
            + ", ".join(sorted(missing_maps))
        )
    normalized_std = torch.as_tensor(
        foothold_normalized_std,
        dtype=torch.float32,
    )
    if normalized_std.shape != (2,) or not torch.isfinite(
        normalized_std
    ).all():
        raise ValueError("foothold_normalized_std must be a finite XY pair.")
    source_std = source_state.get("std")
    if source_std is None or source_std.shape != (motor_action_dim,):
        raise ValueError(
            "Legacy initialization requires a 29-action motor checkpoint; "
            "shared-head learned-foothold checkpoints are not compatible."
        )

    migrated = {
        key: value.detach().clone()
        for key, value in destination_state.items()
    }
    report = FootholdCheckpointMigrationReport()
    consumed_source: set[str] = set()

    for destination_key, destination in destination_state.items():
        source_key = _source_key_for_destination(destination_key)
        if source_key is None:
            if destination_key == "foothold_std":
                migrated[destination_key] = normalized_std.to(
                    device=destination.device,
                    dtype=destination.dtype,
                )
            report.initialized.append(destination_key)
            continue
        if source_key not in source_state:
            report.unexpected.append(
                f"{destination_key}: missing source {source_key}"
            )
            continue

        source = source_state[source_key]
        consumed_source.add(source_key)
        if source.shape == destination.shape:
            migrated[destination_key] = _copy_like(source, destination)
            report.copied.append(destination_key)
            continue

        if (
            _FIRST_INPUT_WEIGHT.fullmatch(destination_key)
            and source.ndim == 2
            and destination.ndim == 2
            and destination.shape[0] == source.shape[0]
        ):
            map_name = (
                "actor"
                if destination_key.startswith("actor.")
                else "critics.0"
            )
            destination_columns = tuple(input_column_maps[map_name])
            if (
                len(destination_columns) != source.shape[1]
                or len(set(destination_columns)) != len(destination_columns)
                or any(
                    column < 0 or column >= destination.shape[1]
                    for column in destination_columns
                )
            ):
                report.unexpected.append(
                    f"{destination_key}: invalid {map_name} input column map "
                    f"for {tuple(source.shape)} -> "
                    f"{tuple(destination.shape)}"
                )
                continue
            migrated[destination_key].zero_()
            migrated[destination_key][
                :,
                list(destination_columns),
            ] = _copy_like(
                source,
                destination[:, list(destination_columns)],
            )
            report.expanded.append(
                f"{destination_key}[:, legacy_columns]"
            )
            report.initialized.append(
                f"{destination_key}[:, new_columns]=0"
            )
            continue

        report.unexpected.append(
            f"{destination_key}: {tuple(source.shape)} -> "
            f"{tuple(destination.shape)}"
        )

    unconsumed = sorted(set(source_state) - consumed_source)
    report.unexpected.extend(unconsumed)
    if report.unexpected:
        raise ValueError(
            "Unexpected checkpoint migration entries: "
            + ", ".join(report.unexpected)
        )
    return migrated, report


def _checkpoint_learning_rate(checkpoint: Mapping) -> float:
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise KeyError(
            "Legacy checkpoint is missing optimizer_state_dict needed to "
            "recover its final learning rate."
        )
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, Sequence) or not param_groups:
        raise KeyError(
            "Legacy checkpoint optimizer has no parameter groups."
        )
    learning_rates = {float(group["lr"]) for group in param_groups}
    if (
        len(learning_rates) != 1
        or not all(isfinite(value) and value > 0.0 for value in learning_rates)
    ):
        raise ValueError(
            "Legacy checkpoint must contain one finite positive learning "
            "rate shared by all optimizer parameter groups."
        )
    return learning_rates.pop()


def initialize_runner_from_legacy_checkpoint(
    runner,
    checkpoint_path: str | Path,
    *,
    motor_action_dim: int,
    input_column_maps: Mapping[str, Sequence[int]],
    foothold_normalized_std: Sequence[float],
) -> FootholdCheckpointMigrationReport:
    """Initialize networks only; intentionally do not restore optimizer/iter."""

    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=runner.alg.actor_critic.motor_std.device
        if hasattr(runner.alg.actor_critic, "motor_std")
        else "cpu",
        weights_only=True,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError("Legacy checkpoint is missing model_state_dict.")
    destination_state = runner.alg.actor_critic.state_dict()
    migrated, report = migrate_foothold_model_state(
        checkpoint["model_state_dict"],
        destination_state,
        motor_action_dim=motor_action_dim,
        input_column_maps=input_column_maps,
        foothold_normalized_std=foothold_normalized_std,
    )
    runner.alg.actor_critic.load_state_dict(migrated, strict=True)
    source_learning_rate = _checkpoint_learning_rate(checkpoint)
    if not hasattr(runner.alg, "optimizer") or not hasattr(
        runner.alg, "learning_rate"
    ):
        raise AttributeError(
            "Runner algorithm must expose optimizer and learning_rate."
        )
    for parameter_group in runner.alg.optimizer.param_groups:
        parameter_group["lr"] = source_learning_rate
    runner.alg.learning_rate = source_learning_rate
    report.source_learning_rate = source_learning_rate

    discriminator = getattr(runner.alg, "discriminator", None)
    if discriminator is not None:
        if "discriminator" not in checkpoint:
            raise KeyError(
                "Legacy checkpoint is missing discriminator state."
            )
        discriminator.load_state_dict(
            checkpoint["discriminator"],
            strict=True,
        )

    # This is initialization of a new optimization problem, not resume.
    runner.current_learning_iteration = 0
    return report
