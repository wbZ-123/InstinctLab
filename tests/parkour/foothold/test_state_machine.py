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


def test_initial_gait_state_alternates_swing_side_across_environments():
    state = initial_gait_state(6, device="cpu")

    torch.testing.assert_close(
        state.swing_side,
        torch.tensor([0, 1, 0, 1, 0, 1]),
    )


def test_hold_state_starts_configured_swing_side_after_confirmed_contact():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")
    state.swing_side.fill_(1)

    for _ in range(20):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.RIGHT_SWING
    assert state.swing_side.item() == 1


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

    for _ in range(2):
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


def test_accepted_late_touchdown_holds_briefly_before_next_swing():
    cfg = GaitMachineConfig(step_hold_s=0.04)
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

    # 下一拍先交换下一摆动脚，但进入短双足支撑 HOLD。
    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([True]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.HOLD
    assert state.swing_side.item() == 1

    # HOLD 中确认双脚稳定接触满 step_hold_s 后，再启动右脚摆动。
    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
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


def test_touchdown_hold_duration_can_be_overridden_per_environment():
    cfg = GaitMachineConfig(step_hold_s=0.08)
    state = initial_gait_state(2, device="cpu")
    state.mode[:] = GaitState.TOUCHDOWN_CONFIRM
    state.swing_side[:] = 0

    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True], [True, True]]),
        touchdown_accepted=torch.tensor([True, True]),
        planner_valid=torch.tensor([True, True]),
        dt=0.02,
        cfg=cfg,
        step_hold_s=torch.tensor([0.0, 0.08]),
    )

    assert torch.all(state.mode == GaitState.HOLD)
    torch.testing.assert_close(
        state.hold_required_s,
        torch.tensor([0.0, 0.08]),
    )

    # First hold step: neither env has confirmed stable double contact yet.
    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True], [True, True]]),
        touchdown_accepted=torch.tensor([False, False]),
        planner_valid=torch.tensor([True, True]),
        dt=0.02,
        cfg=cfg,
        step_hold_s=torch.tensor([0.0, 0.08]),
    )

    assert torch.all(state.mode == GaitState.HOLD)

    # Second hold step: env 0 can start because contact is confirmed and
    # extra hold is zero; env 1 must keep waiting for its longer hold.
    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True], [True, True]]),
        touchdown_accepted=torch.tensor([False, False]),
        planner_valid=torch.tensor([True, True]),
        dt=0.02,
        cfg=cfg,
        step_hold_s=torch.tensor([0.0, 0.08]),
    )

    assert state.mode[0].item() == GaitState.RIGHT_SWING
    assert state.mode[1].item() == GaitState.HOLD


def test_accepted_late_touchdown_takes_priority_over_old_stance_loss():
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

    # 进入晚摆动窗口。
    for _ in range(7):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[False, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    # 新左脚已经连续确认落地；同一瞬间旧右支撑脚开始/保持离地。
    # 这应该被视为正常换支撑，而不是旧支撑脚丢失失败。
    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, False]]),
        touchdown_accepted=torch.tensor([True]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )
    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, False]]),
        touchdown_accepted=torch.tensor([True]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.TOUCHDOWN_CONFIRM


def test_late_physical_touchdown_is_accepted_even_when_target_error_is_bad():
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

    # 左脚接触得到确认；即使落点质量不合格，状态机也应该承认
    # “物理落地”并完成支撑切换，落点误差交给 reward/monitor 处理。
    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.TOUCHDOWN_CONFIRM


@pytest.mark.parametrize(
    "reason",
    [
        GaitState.EARLY_CONTACT,
        GaitState.OVERDUE,
        GaitState.STANCE_LOST,
    ],
)
def test_failure_reason_is_visible_then_recovery_returns_to_hold(reason):
    cfg = GaitMachineConfig(recovery_hold_s=0.04)
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

    # 双脚重新稳定接触且规划有效后，RECOVERY 等待一小段时间回 HOLD。
    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.HOLD
    torch.testing.assert_close(state.hold_elapsed_s, torch.zeros(1))


def test_recovery_waits_for_valid_plan_then_hold_waits_for_contact():
    cfg = GaitMachineConfig(recovery_hold_s=0.04)
    state = initial_gait_state(1, device="cpu")
    state.mode.fill_(GaitState.RECOVERY)

    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([False]),
        dt=0.02,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.RECOVERY

    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, False]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.HOLD

    # HOLD owns the "wait for stable double contact" gate, so it should not
    # start a new swing while contact is still unstable.
    for _ in range(25):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, False]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.HOLD


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


def test_recovery_marks_next_swing_as_recovery_step_until_touchdown():
    cfg = GaitMachineConfig(recovery_hold_s=0.04, reset_hold_s=0.04)
    state = initial_gait_state(1, device="cpu")
    state.mode[:] = GaitState.RECOVERY

    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )

    assert state.mode.item() == GaitState.HOLD
    assert state.recovery_step_pending.item()

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
    assert state.recovery_step_pending.item()
    assert state.recovery_step_active.item()

    # A successful touchdown-confirm clears recovery-step bookkeeping.
    state.mode[:] = GaitState.TOUCHDOWN_CONFIRM
    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([True]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )

    assert state.mode.item() == GaitState.HOLD
    assert not state.recovery_step_pending.item()
    assert not state.recovery_step_active.item()

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
        "recovery_hold_s",
        "step_hold_s",
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
