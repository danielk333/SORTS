"""
NOTE: this is tmp kept as a reference alternative implementation of 'schedule_metadata.py'

TODO: remove?
"""

import typing as t, enum
import numpy as np
from numpy.typing import DTypeLike


class CoordinateSystem(enum.IntEnum):
    AZELR = enum.auto()
    NED = enum.auto()
    ENU = enum.auto()


# ref: https://docs.python.org/3.10/library/enum.html#others
class ScheduleColumnName(str, enum.Enum):
    STT_TSTMP = "stt_tstmp"
    "timestamp, the start time of the schedule slice"

    COORDINATE_SYSTEM = "coordinate_system"
    COH_INT_BANDWIDTH = "coh_int_bandwidth"
    POINTING = "pointing"
    IPP = "ipp"
    PULSE_LENGTH = "pulse_length"


schedule_column_dtypes: dict[ScheduleColumnName, DTypeLike] = {
    ScheduleColumnName.STT_TSTMP: np.int8,
    ScheduleColumnName.COORDINATE_SYSTEM: np.float64,
    ScheduleColumnName.COH_INT_BANDWIDTH: np.float64,
    ScheduleColumnName.POINTING: np.float64,
    ScheduleColumnName.IPP: np.float64,
    ScheduleColumnName.PULSE_LENGTH: np.float64,
}

# schedule_column_dtypes: dict[
#     t.Literal[ScheduleColumnName.STT_TSTMP]
#     | t.Literal[ScheduleColumnName.COORDINATE_SYSTEM]
#     | t.Literal[ScheduleColumnName.COH_INT_BANDWIDTH]
#     | t.Literal[ScheduleColumnName.POINTING]
#     | t.Literal[ScheduleColumnName.IPP]
#     | t.Literal[ScheduleColumnName.PULSE_LENGTH],
#     DTypeLike,
# ] = {
#     ScheduleColumnName.STT_TSTMP: np.int8,
#     ScheduleColumnName.COORDINATE_SYSTEM: np.float64,
#     ScheduleColumnName.COH_INT_BANDWIDTH: np.float64,
#     ScheduleColumnName.POINTING: np.float64,
#     ScheduleColumnName.IPP: np.float64,
#     ScheduleColumnName.PULSE_LENGTH: np.float64,
# }


schedule_nprecord_dtype = [(k, v) for k, v in schedule_column_dtypes.items()]
# schedule_nprecord_dtype = {k: (k, v) for k, v in schedule_column_dtypes.items()}
