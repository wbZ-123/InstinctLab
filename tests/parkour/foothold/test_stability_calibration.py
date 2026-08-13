import torch
import pytest

from instinctlab_foothold.stability_calibration import (
    calibrate_stability_bounds,
    load_stability_bounds,
)


def test_calibration_uses_successful_hold_samples_only():
    values = torch.tensor(
        [
            [0.10, 0.20, 0.10, 0.01],
            [0.15, 0.30, 0.20, 0.02],
            [0.20, 0.40, 0.30, 0.03],
            [9.00, 9.00, 9.00, 9.00],
        ]
    )
    valid = torch.tensor([True, True, True, False])

    result = calibrate_stability_bounds(
        values,
        valid,
        quantile=1.0,
        dwell_s=0.10,
    )

    assert result["max_tilt_rad"] == pytest.approx(0.20)
    assert result["max_angular_speed_rad_s"] == pytest.approx(0.40)
    assert result["max_horizontal_speed_m_s"] == pytest.approx(0.30)
    assert result["max_support_slip_m_s"] == pytest.approx(0.03)
    assert result["dwell_s"] == pytest.approx(0.10)


def test_calibration_default_dwell_matches_recovery_seed():
    values = torch.tensor([[0.10, 0.20, 0.10, 0.01]])
    result = calibrate_stability_bounds(values, torch.tensor([True]))

    assert result["dwell_s"] == pytest.approx(0.04)


def test_calibration_rejects_empty_or_nonfinite_samples():
    values = torch.tensor([[float("nan"), 0.2, 0.1, 0.01]])
    valid = torch.tensor([True])

    try:
        calibrate_stability_bounds(values, valid, quantile=0.99, dwell_s=0.10)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("expected non-finite calibration input to fail")


def test_load_stability_bounds_requires_all_positive_named_values(tmp_path):
    path = tmp_path / "bounds.json"
    path.write_text(
        '{"max_tilt_rad": 0.2, "max_angular_speed_rad_s": 0.4, '
        '"max_horizontal_speed_m_s": 0.3, "max_support_slip_m_s": 0.03, '
        '"dwell_s": 0.1}'
    )

    bounds = load_stability_bounds(path)

    assert bounds.max_tilt_rad == pytest.approx(0.2)
    assert bounds.dwell_s == pytest.approx(0.1)
