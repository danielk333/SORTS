import typing as t, enum
import numpy as np
from numpy.typing import DTypeLike

ScheduleColumnName = t.Literal[
    "stt_tstmp",
    "coh_int_bandwidth",
    "pointing_az",
    "pointing_el",
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

# TODO: remove?
#   seems pandas dataframe is a better choice for schedule,
#   so this seems not needed
schedule_column_dtypes: dict[ScheduleColumnName, DTypeLike] = {
    "stt_tstmp": "datetime64[ns]",
    "coh_int_bandwidth": np.float64,
    "pointing_az": np.float64,
    "pointing_el": np.float64,
    "ipp": np.float64,
    "pulse_length": np.float64,
}

# TODO: remove?
#   seems pandas dataframe is a better choice for schedule,
#   so this seems not needed
schedule_ndarray_dtype: list[tuple[ScheduleColumnName, DTypeLike]] = [
    (k, v) for k, v in schedule_column_dtypes.items()
]
"""
dtype for a schedule. generated from `schedule_column_dtypes`
"""

__all__ = [
    "ScheduleColumnName",
    "schedule_column_names",
    "schedule_column_dtypes",
    "schedule_ndarray_dtype",
]
