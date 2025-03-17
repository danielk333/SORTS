import typing as t, enum
import numpy as np
from numpy.typing import DTypeLike


class CoordinateSystem(enum.IntEnum):
    AZELR = enum.auto()
    NED = enum.auto()
    ENU = enum.auto()


ScheduleColumnName = t.Literal[
    "stt_tstmp",
    "coordinate_system",
    "coh_int_bandwidth",
    "pointing",
    "ipp",
    "pulse_length",
]
"""
accepted column name in schedule

`stt_tstmp` refers to the start time of the schedule slice
"""

schedule_column_names: dict[ScheduleColumnName, ScheduleColumnName] = {
    v: v for v in t.get_args(ScheduleColumnName)
}
"""
a dict for conveniently using `ScheduleColumnName` as literal values
"""

schedule_column_dtypes: dict[ScheduleColumnName, DTypeLike] = {
    "stt_tstmp": np.int8,
    "coordinate_system": np.float64,
    "coh_int_bandwidth": np.float64,
    "pointing": [("p1", np.float64), ("p2", np.float64), ("p3", np.float64)],
    "ipp": np.float64,
    "pulse_length": np.float64,
}


schedule_nprecord_dtype: list[tuple[ScheduleColumnName, DTypeLike]] = [
    (k, v) for k, v in schedule_column_dtypes.items()
]
