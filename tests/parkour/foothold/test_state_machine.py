import torch

import pytest

from instinctlab_foothold.state_machine import (
    GaitMachineConfig,
    advance_gait,
    gait_phase,
    initial_gait_state,
)

from instinctlab_foothold.types import GaitState


def test_reset_holds_for_exactly_point_four_seconds():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    contact = torch.tensor([[True, True]])
    touchdown_accepted = torch.tensor([False])
    planner_valid = torch.tensor([True])

    for _ in range(19):
        state = advance_gait(
            state=state,
            contact=contact,
            touchdown_accepted=touchdown_accepted,
            planner_valid=planner_valid,
            dt=0.02,
            cfg=cfg,
        )
        assert state.mode.item() == GaitState.HOLD

    state = advance_gait(
        state=state,
        contact=contact,
        touchdown_accepted=touchdown_accepted,
        planner_valid=planner_valid,
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.LEFT_SWING


def test_reset_hold_keeps_time_but_requires_confirmed_contact_to_start():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    touchdown_accepted = torch.tensor([False])
    planner_valid = torch.tensor([True])

    for _ in range(20):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[False, False]]),
            touchdown_accepted=touchdown_accepted,
            planner_valid=planner_valid,
            dt=0.02,
            cfg=cfg,
        )

    torch.testing.assert_close(
        state.hold_elapsed_s,
        torch.tensor([0.40]),
    )
    assert state.mode.item() == GaitState.HOLD

    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=touchdown_accepted,
        planner_valid=planner_valid,
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.HOLD

    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=touchdown_accepted,
        planner_valid=planner_valid,
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.LEFT_SWING


def test_reset_hold_does_not_start_without_confirmed_contact():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    for _ in range(25):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[False, False]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.HOLD


def test_invalid_plan_is_visible_then_enters_recovery():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([False]),
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.PLAN_INVALID

    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.RECOVERY


def test_swing_elapsed_time_advances_after_hold():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    for _ in range(20):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.LEFT_SWING

    state = advance_gait(
        state=state,
        contact=torch.tensor([[False, True]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )

    torch.testing.assert_close(
        state.elapsed_s,
        torch.tensor([0.02]),
    )


def test_liftoff_requires_point_zero_four_seconds_without_contact():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    for _ in range(20):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.LEFT_SWING

    state = advance_gait(
        state=state,
        contact=torch.tensor([[False, True]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )

    assert not state.swing_has_lifted.item()

    state = advance_gait(
        state=state,
        contact=torch.tensor([[False, True]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )

    assert state.swing_has_lifted.item()
    torch.testing.assert_close(
        state.no_contact_elapsed_s,
        torch.tensor([[0.04, 0.0]]),
    )


def test_confirmed_contact_before_phase_threshold_is_early_contact():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    # 完成 0.4 秒启动保持，进入左脚摆动。
    for _ in range(20):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    # 左脚连续无接触 0.04 秒，确认已经离地。
    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[False, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.swing_has_lifted.item()

    # 左脚过早恢复接触，并持续 0.04 秒。
    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    phase = state.elapsed_s / cfg.swing_s

    assert phase.item() < cfg.early_contact_phase
    assert state.mode.item() == GaitState.EARLY_CONTACT


def test_confirmed_stance_loss_is_reported():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    # 进入左脚摆动。
    for _ in range(20):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.LEFT_SWING
    assert state.swing_side.item() == 0

    # 左脚摆动期间，右支撑脚连续失去接触 0.04 秒。
    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[False, False]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    torch.testing.assert_close(
        state.no_contact_elapsed_s,
        torch.tensor([[0.04, 0.04]]),
    )
    assert state.mode.item() == GaitState.STANCE_LOST


def test_accepted_late_touchdown_swaps_to_right_swing():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    # 进入左脚摆动。
    for _ in range(20):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    # 左脚离地确认。
    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[False, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    # 保持左脚悬空，推进到晚摆动窗口附近。
    for _ in range(7):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[False, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    # 第一帧接触尚未满足 0.04 秒消抖。
    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([True]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.LEFT_SWING

    # 第二帧接触确认，进入一次 TOUCHDOWN_CONFIRM。
    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([True]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.TOUCHDOWN_CONFIRM

    # 下一拍交换左右脚。
    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([True]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.RIGHT_SWING
    assert state.swing_side.item() == 1
    torch.testing.assert_close(
        state.elapsed_s,
        torch.zeros(1),
    )


def test_rejected_late_touchdown_eventually_becomes_overdue():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    # 进入左脚摆动。
    for _ in range(20):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    # 确认左脚离地。
    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[False, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    # 推进到晚摆动窗口。
    for _ in range(7):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[False, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    # 左脚接触得到确认，但落点位置不合格。
    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.LEFT_SWING

    # 继续等待直到总摆动时间达到 0.44 秒。
    for _ in range(11):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    torch.testing.assert_close(
        state.elapsed_s,
        torch.tensor([0.44]),
    )
    assert state.mode.item() == GaitState.OVERDUE


@pytest.mark.parametrize(
    "reason",
    [
        GaitState.EARLY_CONTACT,
        GaitState.OVERDUE,
        GaitState.STANCE_LOST,
    ],
)
def test_failure_reason_is_visible_then_recovery_is_latched(reason):
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    # 直接构造“上一拍已经报告失败原因”的状态。
    state.mode.fill_(reason)

    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.RECOVERY

    # RECOVERY 不会自行恢复步态，等待 Sensor 或环境重置。
    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.RECOVERY


def test_invalid_plan_during_swing_has_highest_priority():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    # 进入左脚摆动。
    for _ in range(20):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.LEFT_SWING

    # 摆动期间规划突然失效。
    state = advance_gait(
        state=state,
        contact=torch.tensor([[False, False]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([False]),
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.PLAN_INVALID

    # 下一拍进入恢复。
    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.RECOVERY


def test_invalid_plan_blocks_swap_after_touchdown_confirm():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    # 构造：左脚刚刚成功触地，下一拍原本应切换为右脚摆动。
    state.mode.fill_(GaitState.TOUCHDOWN_CONFIRM)
    state.swing_side.fill_(0)

    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([True]),
        planner_valid=torch.tensor([False]),
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.PLAN_INVALID

    # 规划无效时不能偷偷把下一摆动脚改成右脚。
    assert state.swing_side.item() == 0


def test_gait_phase_is_normalized_and_clamped():
    cfg = GaitMachineConfig(swing_s=0.32)
    state = initial_gait_state(3, device="cpu")

    state.elapsed_s.copy_(
        torch.tensor([-0.10, 0.16, 0.64])
    )

    torch.testing.assert_close(
        gait_phase(state, cfg),
        torch.tensor([0.0, 0.5, 1.0]),
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "reset_hold_s",
        "swing_s",
        "contact_confirm_s",
        "overdue_s",
    ],
)
def test_non_positive_timing_configuration_is_rejected(
    field_name,
):
    with pytest.raises(ValueError, match="positive"):
        GaitMachineConfig(
            **{field_name: 0.0}
        )


@pytest.mark.parametrize(
    "dt",
    [0.0, -0.02],
)
def test_non_positive_control_period_is_rejected(dt):
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    with pytest.raises(ValueError, match="positive"):
        advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=dt,
            cfg=cfg,
        )


@pytest.mark.parametrize(
    "phase",
    [0.0, 1.0, -0.1, 1.1],
)
def test_early_contact_phase_must_be_inside_unit_interval(
    phase,
):
    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        GaitMachineConfig(
            early_contact_phase=phase
        )


def test_4096_environments_advance_independently():
    num_envs = 4096
    cfg = GaitMachineConfig()
    state = initial_gait_state(
        num_envs,
        device="cpu",
    )

    planner_valid = (
        torch.arange(num_envs).remainder(2) == 0
    )
    contact = torch.ones(
        (num_envs, 2),
        dtype=torch.bool,
    )

    state = advance_gait(
        state=state,
        contact=contact,
        touchdown_accepted=torch.zeros(
            num_envs,
            dtype=torch.bool,
        ),
        planner_valid=planner_valid,
        dt=0.02,
        cfg=cfg,
    )

    valid_envs = planner_valid
    invalid_envs = ~planner_valid

    assert torch.all(
        state.mode[valid_envs] == GaitState.HOLD
    )
    assert torch.all(
        state.mode[invalid_envs]
        == GaitState.PLAN_INVALID
    )

    torch.testing.assert_close(
        state.hold_elapsed_s[valid_envs],
        torch.full((num_envs // 2,), 0.02),
    )
    torch.testing.assert_close(
        state.hold_elapsed_s[invalid_envs],
        torch.zeros(num_envs // 2),
    )



def test_task3_types_are_public_package_api():
    import instinctlab_foothold

    expected_names = (
        "GaitMachineConfig",
        "GaitMachineState",
        "SwingReference",
        "advance_gait",
        "gait_phase",
        "initial_gait_state",
        "quintic_swing_reference",
    )

    for name in expected_names:
        assert hasattr(instinctlab_foothold, name)
