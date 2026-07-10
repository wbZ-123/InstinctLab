#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import NamedTuple


class SeriesSummary(NamedTuple):
    last: float
    mean: float
    min: float
    max: float
    first_window_mean: float
    last_window_mean: float
    delta: float
    delta_percent: float
    trend: str
    count: int


class ReportRow(NamedTuple):
    name: str
    tag: str
    summary: SeriesSummary
    status: str


CORE_TAGS = (
    "Step_Monitor/foothold_planner_nonfinite_fraction",
    "Step_Monitor/foothold_planner_plan_invalid_fraction",
    "Step_Monitor/foothold_planner_safe_target_search_rate",
    "Step_Monitor/foothold_planner_safe_target_final_valid_fraction",
    "Step_Monitor/foothold_planner_safe_target_fallback_fraction",
    "Step_Monitor/foothold_planner_safe_target_score_mean",
    "Step_Monitor/foothold_planner_safe_target_score_max",
    "Step_Monitor/foothold_planner_safe_target_nominal_inside_ellipse_fraction",
    "Step_Monitor/foothold_planner_safe_target_nominal_obstacle_safe_fraction",
    "Step_Monitor/foothold_planner_safe_target_nominal_valid_fraction",
    "Step_Monitor/foothold_planner_safe_target_candidate_count_mean",
    "Step_Monitor/foothold_planner_safe_target_candidate_inside_ellipse_count_mean",
    "Step_Monitor/foothold_planner_safe_target_candidate_obstacle_safe_count_mean",
    "Step_Monitor/foothold_planner_safe_target_candidate_valid_count_mean",
    "Step_Monitor/foothold_planner_swing_fraction",
    "Step_Monitor/foothold_planner_overdue_fraction",
    "Step_Monitor/foothold_planner_early_contact_fraction",
    "Step_Monitor/foothold_planner_stance_lost_fraction",
    "Step_Monitor/foothold_planner_touchdown_confirm_step_rate",
    "Train/time/mean_reward_0",
    "Train/time/mean_episode_length",
)

LEGACY_TAGS = (
    "Step_Monitor/foothold_planner_safe_target_valid_fraction",
)


def summarize_series(values: list[float]) -> SeriesSummary:
    if not values:
        raise ValueError("Cannot summarize an empty scalar series.")

    first_window = values[: max(1, len(values) // 4)]
    last_window = values[-max(1, len(values) // 4) :]
    first_mean = mean(first_window)
    last_mean = mean(last_window)
    delta = last_mean - first_mean
    if abs(first_mean) > 1e-12:
        delta_percent = 100.0 * delta / abs(first_mean)
    else:
        delta_percent = 0.0
    tolerance = max(1e-8, abs(first_mean) * 0.05)
    if last_mean > first_mean + tolerance:
        trend = "up"
    elif last_mean < first_mean - tolerance:
        trend = "down"
    else:
        trend = "flat"

    return SeriesSummary(
        last=values[-1],
        mean=mean(values),
        min=min(values),
        max=max(values),
        first_window_mean=first_mean,
        last_window_mean=last_mean,
        delta=delta,
        delta_percent=delta_percent,
        trend=trend,
        count=len(values),
    )


def _short_name(tag: str) -> str:
    return tag.rsplit("/", maxsplit=1)[-1]


def _status_for(
    tag: str,
    summary: SeriesSummary,
    summaries_by_name: dict[str, SeriesSummary] | None = None,
) -> str:
    name = _short_name(tag)
    if name in {
        "foothold_planner_nonfinite_fraction",
        "foothold_planner_plan_invalid_fraction",
    }:
        return "OK" if summary.max <= 0.0 else "BAD"
    if name in {
        "foothold_planner_safe_target_final_valid_fraction",
        "foothold_planner_safe_target_valid_fraction",
        "foothold_planner_safe_target_nominal_inside_ellipse_fraction",
        "foothold_planner_safe_target_nominal_obstacle_safe_fraction",
        "foothold_planner_safe_target_nominal_valid_fraction",
    }:
        if summary.last >= 0.8:
            return "OK"
        if summary.last >= 0.5:
            return "WATCH"
        return "BAD"
    if name == "foothold_planner_safe_target_candidate_valid_count_mean":
        candidate_count = None
        fallback_fraction = None
        final_valid_fraction = None
        if summaries_by_name is not None:
            candidate_count = summaries_by_name.get(
                "foothold_planner_safe_target_candidate_count_mean"
            )
            fallback_fraction = summaries_by_name.get(
                "foothold_planner_safe_target_fallback_fraction"
            )
            final_valid_fraction = summaries_by_name.get(
                "foothold_planner_safe_target_final_valid_fraction"
            )

        if (
            candidate_count is not None
            and candidate_count.last <= 0.0
            and (
                fallback_fraction is None
                or fallback_fraction.last <= 0.0
            )
            and (
                final_valid_fraction is None
                or final_valid_fraction.last >= 0.8
            )
        ):
            return "INFO"
        if summary.last >= 4.0:
            return "OK"
        if summary.last > 0.0:
            return "WATCH"
        return "BAD"
    if name == "foothold_planner_safe_target_fallback_fraction":
        if summary.last < 0.2:
            return "OK"
        if summary.last < 0.5:
            return "WATCH"
        return "BAD"
    if name in {
        "foothold_planner_overdue_fraction",
        "foothold_planner_early_contact_fraction",
        "foothold_planner_stance_lost_fraction",
    }:
        return "OK" if summary.last < 0.05 else "WATCH"
    if name in {
        "mean_reward_0",
        "mean_episode_length",
    }:
        return "OK" if summary.trend == "up" else "WATCH"
    return "INFO"


def build_report_rows(
    scalars: dict[str, list[float]],
    tags: tuple[str, ...] = CORE_TAGS,
) -> list[ReportRow]:
    rows: list[ReportRow] = []
    summaries_by_name: dict[str, SeriesSummary] = {}
    summaries_by_tag: dict[str, SeriesSummary] = {}

    for tag in tags:
        values = scalars.get(tag)
        if not values:
            continue
        summary = summarize_series(values)
        summaries_by_name[_short_name(tag)] = summary
        summaries_by_tag[tag] = summary

    for tag in tags:
        summary = summaries_by_tag.get(tag)
        if summary is None:
            continue
        rows.append(
            ReportRow(
                name=_short_name(tag),
                tag=tag,
                summary=summary,
                status=_status_for(tag, summary, summaries_by_name),
            )
        )
    return rows


def read_scalar_events(run_dir: Path) -> dict[str, list[float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:
        raise RuntimeError(
            "需要 tensorboard 包才能读取 event 文件；当前环境没有导入成功。"
        ) from exc

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    scalars: dict[str, list[float]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        scalars[tag] = [event.value for event in accumulator.Scalars(tag)]
    return scalars


def find_latest_run(logdir: Path, pattern: str) -> Path:
    candidates = sorted(
        path for path in logdir.iterdir() if path.is_dir() and pattern in path.name
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run directory matching '*{pattern}*' under {logdir}"
        )
    return candidates[-1]


def format_rows(rows: list[ReportRow]) -> str:
    if not rows:
        return "No matching scalar tags found."

    lines = [
        (
            f"{'metric':52} {'last':>10} {'mean':>10} {'min':>10} "
            f"{'max':>10} {'first_q':>10} {'last_q':>10} "
            f"{'delta':>10} {'delta_%':>9} {'trend':>7} "
            f"{'status':>7} {'n':>5}"
        ),
        "-" * 163,
    ]
    for row in rows:
        summary = row.summary
        lines.append(
            f"{row.name:52} "
            f"{summary.last:10.4g} "
            f"{summary.mean:10.4g} "
            f"{summary.min:10.4g} "
            f"{summary.max:10.4g} "
            f"{summary.first_window_mean:10.4g} "
            f"{summary.last_window_mean:10.4g} "
            f"{summary.delta:10.4g} "
            f"{summary.delta_percent:9.3g} "
            f"{summary.trend:>7} "
            f"{row.status:>7} "
            f"{summary.count:5d}"
        )
    return "\n".join(lines)


def _row_to_payload(row: ReportRow) -> dict[str, object]:
    summary = row.summary
    return {
        "metric": row.name,
        "tag": row.tag,
        "status": row.status,
        "summary": {
            "last": summary.last,
            "mean": summary.mean,
            "min": summary.min,
            "max": summary.max,
            "first_q": summary.first_window_mean,
            "last_q": summary.last_window_mean,
            "delta": summary.delta,
            "delta_percent": summary.delta_percent,
            "trend": summary.trend,
            "n": summary.count,
        },
    }


def save_report(
    *,
    run_dir: Path,
    rows: list[ReportRow],
    save_dir: Path = Path("logs/foothold_reports"),
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / f"{run_dir.name}.json"
    payload = {
        "run": str(run_dir),
        "rows": [_row_to_payload(row) for row in rows],
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize foothold planner TensorBoard scalar curves."
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="A single TensorBoard run directory.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("logs/instinct_rl/g1_parkour"),
        help="Root log directory used with --latest-pattern.",
    )
    parser.add_argument(
        "--latest-pattern",
        default="foothold",
        help="Pick the latest run whose directory name contains this string.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Extra scalar tag to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("logs/foothold_reports"),
        help="Directory for machine-readable JSON summaries.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only print the table; do not write a JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run
    if run_dir is None:
        run_dir = find_latest_run(args.logdir, args.latest_pattern)

    scalars = read_scalar_events(run_dir)
    tags = CORE_TAGS + LEGACY_TAGS + tuple(args.tag)
    rows = build_report_rows(scalars, tags=tags)

    print(f"run: {run_dir}")
    print(format_rows(rows))
    if not args.no_save:
        report_path = save_report(
            run_dir=run_dir,
            rows=rows,
            save_dir=args.save_dir,
        )
        print(f"saved_json: {report_path}")


if __name__ == "__main__":
    main()
