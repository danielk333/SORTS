from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts.types import Datetime64_us, Float64_as_deg
from sorts.schedule_v2.types import ExperimentDetail, ScheduleNdarrayDict, ScheduleNdarrayDict2
from sorts.schedule_v2.schedule_data import (
    schedule_data_keys,
    schedule_coord_keys,
    schedule_attr_keys,
    schedule_keys,
    ScheduleXrds,
    from_ndarrays,
    from_ndarrays_2,
    empty,
    to_dataframe as to_dataframe_,
)
from sorts.schedule_v2.priority_scheduling import priority_scheduling

logger = logging.getLogger(__name__)

# TODO: move it, superseded by `ScheduleDataKey`, `ScheduleCoordKey`, `ScheduleAttrKey`
ScheduleFieldKey = t.Literal[
    "exp_detail_map", "start_time", "pointing_az", "pointing_el", "exp_num"
]

# Define the column names used when exported as a `DataFrame` (pandas or alike)
NonDerivedDataFrameColumnName = t.Literal["start_time", "pointing_az", "pointing_el", "exp_num"]
DerivedDataFrameColumnName = t.Literal["end_time"]
DataFrameColumnName = t.Literal["start_time", "end_time", "pointing_az", "pointing_el", "exp_num"]

assert all((n in t.get_args(ScheduleFieldKey) for n in t.get_args(NonDerivedDataFrameColumnName)))
assert set(t.get_args(DataFrameColumnName)) == set(
    [*t.get_args(NonDerivedDataFrameColumnName), *t.get_args(DerivedDataFrameColumnName)]
)


data_frame_column_names: t.Final[dict[DataFrameColumnName, str]] = {
    n: n for n in t.get_args(DataFrameColumnName)
}
"""A dict of `DataFrameColumnName` as string key-value pair for convenience."""

cn = data_frame_column_names
"""An alias of `data_frame_column_names`"""


# TODO: can be removed? xarray dataset class is already dataframe like, and have pandas conversion methods
def from_dataframe(
    df: pd.DataFrame, exp_detail_map: dict[int, ExperimentDetail]
) -> ScheduleNdarrayDict2:
    sch = ScheduleNdarrayDict2(
        **{k: df[k].to_numpy() for k in t.get_args(NonDerivedDataFrameColumnName)},
        exp_detail_map=exp_detail_map,
    )

    return sch


# TODO: remove its usage, then remove this func
def create_mask_by_time_range(
    sch: ScheduleNdarrayDict2, time_range: tuple[Datetime64_us, Datetime64_us]
) -> npt.NDArray[np.bool]:
    """
    Return a mask that filters out schedule entries that are not inside `time_range`.
    (a right-open interval)

    NOTE: right now it only check against `start_time`
    """

    start_time, end_time = time_range

    # TODO: better include end_time in the schedule and check against that
    mask: npt.NDArray[np.bool] = np.logical_and(
        sch["start_time"] >= start_time,
        sch["start_time"] <= end_time,
    )

    return mask


# TODO: remove its usage, then remove this func
def filter_by_mask(sch: ScheduleNdarrayDict2, mask: npt.NDArray[np.bool]) -> ScheduleNdarrayDict2:
    """Return a slice of the origin schedule based on the `mask`"""

    filtered_sch = ScheduleNdarrayDict2(
        exp_detail_map=sch["exp_detail_map"],
        start_time=sch["start_time"][mask],
        exp_num=sch["exp_num"][mask],
        pointing_az=sch["pointing_az"][mask],
        pointing_el=sch["pointing_el"][mask],
    )

    return filtered_sch


# TODO: remove its usage, then remove this func
def filter_by_time_range(
    sch: ScheduleNdarrayDict2, time_range: tuple[Datetime64_us, Datetime64_us]
) -> ScheduleNdarrayDict2:
    """
    Return a slice of the origin schedule based on the `time_range`
    (a right-open interval)
    """

    return filter_by_mask(sch, create_mask_by_time_range(sch, time_range))


# TODO: add schedule validation?
class Schedule:
    """
    Provides methods for manipuating the schedule data and enforce that the require columns/data are set.

    Schedule data is stored in the `data` attribute, and some additional helper metadata are stored in other attributes.

    Methods of this class are mostly just redirection to equivalent module level functions.
    """

    data_keys = schedule_data_keys
    """shortcut to module attribute `schedule_data_keys`"""
    coord_keys = schedule_coord_keys
    """shortcut to module attribute `schedule_coord_keys`"""
    attr_keys = schedule_attr_keys
    """shortcut to module attribute `schedule_attr_keys`"""

    keys = schedule_keys
    """shortcut to module attribute `schedule_keys`"""

    def __init__(self, data: ScheduleXrds):
        self.data: ScheduleXrds = data

    @classmethod
    def from_ndarrays(cls, data: ScheduleNdarrayDict) -> Schedule:
        return Schedule(data=from_ndarrays(data))

    # TODO: remove its usage, then remove this method
    @classmethod
    def from_ndarrays_2(cls, data: ScheduleNdarrayDict2) -> Schedule:
        return Schedule(data=from_ndarrays_2(data))

    @classmethod
    def empty(cls) -> Schedule:
        return Schedule(data=empty())

    @classmethod
    def priority_scheduling(cls, schs: list[Schedule]):
        resultant_sch_data = priority_scheduling([sch.data for sch in schs])
        return Schedule(data=resultant_sch_data)

    def to_ndarrays(self) -> ScheduleNdarrayDict:
        arr_dict: ScheduleNdarrayDict = {
            "exp_detail_map": self.data.attrs[self.attr_keys["exp_detail_map"]],
            "start_time": self.data[self.coord_keys["start_time"]].to_numpy(),
            "end_time": self.data[self.coord_keys["end_time"]].to_numpy(),
            "exp_num": self.data[self.data_keys["exp_num"]].to_numpy(),
            "pointing": self.data[self.data_keys["pointing"]].to_numpy(),
        }

        return arr_dict

    # TODO: remove its usage, then remove this method
    def to_ndarrays_2(self) -> ScheduleNdarrayDict2:
        arr_dict: ScheduleNdarrayDict2 = {
            "exp_detail_map": self.data.attrs[self.attr_keys["exp_detail_map"]],
            "start_time": self.data[self.coord_keys["start_time"]].to_numpy(),
            "exp_num": self.data[self.data_keys["exp_num"]].to_numpy(),
            "pointing_az": self.data[self.data_keys["pointing"]].to_numpy()[0],
            "pointing_el": self.data[self.data_keys["pointing"]].to_numpy()[1],
        }

        return arr_dict

    def to_dataframe(self) -> pd.DataFrame:
        return to_dataframe_(self.data)
