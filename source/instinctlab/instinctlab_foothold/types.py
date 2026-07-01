from enum import Enum, IntEnum


class ObservationSlice(Enum):
    SWING_ONE_HOT = slice(0, 2)
    PHASE_SIN_COS = slice(2, 4)
    NORMALIZED_TIME = slice(4, 5)
    REFERENCE_POSITION = slice(5, 8)
    REFERENCE_VELOCITY = slice(8, 11)
    CURRENT_POSITION = slice(11, 14)
    CURRENT_YAW_SIN_COS = slice(14, 16)
    CURRENT_NORMAL = slice(16, 19)
    NEXT_POSITION = slice(19, 22)
    NEXT_YAW_SIN_COS = slice(22, 24)
    NEXT_NORMAL = slice(24, 27)
    FEASIBLE_VELOCITY = slice(27, 30)
    POSITION_ERROR = slice(30, 33)
    VELOCITY_ERROR = slice(33, 36)
    APEX_HEIGHT = slice(36, 37)
    PLANNER_VALID = slice(37, 38)
    NEXT_VALID = slice(38, 39)
    TERRAIN_CONFIDENCE = slice(39, 40)
    SUPPORT_MARGIN = slice(40, 41)
    EDGE_RISK = slice(41, 42)
    UNKNOWN_FRACTION = slice(42, 43)
    RECOVERY_STATE = slice(43, 44)


FOOTHOLD_OBSERVATION_DIM = 44


class GaitState(IntEnum):
    HOLD = 0
    LEFT_SWING = 1
    RIGHT_SWING = 2
    TOUCHDOWN_CONFIRM = 3
    EARLY_CONTACT = 4
    OVERDUE = 5
    STANCE_LOST = 6
    PLAN_INVALID = 7
    RECOVERY = 8
