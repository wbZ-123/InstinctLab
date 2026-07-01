from instinctlab_foothold.types import (
    FOOTHOLD_OBSERVATION_DIM,
    GaitState,
    ObservationSlice,
)


def test_observation_layout_is_contiguous_and_44d():
    slices = [member.value for member in ObservationSlice]

    assert slices[0].start == 0
    assert all(left.stop == right.start for left, right in zip(slices, slices[1:]))
    assert slices[-1].stop == FOOTHOLD_OBSERVATION_DIM == 44


def test_gait_state_values_are_stable_for_checkpoints():
    assert GaitState.HOLD == 0
    assert GaitState.LEFT_SWING == 1
    assert GaitState.RIGHT_SWING == 2
    assert GaitState.RECOVERY == 8


def test_public_package_exports_core_types():
    import instinctlab_foothold

    assert instinctlab_foothold.FOOTHOLD_OBSERVATION_DIM == 44
    assert instinctlab_foothold.GaitState is GaitState
    assert instinctlab_foothold.ObservationSlice is ObservationSlice
