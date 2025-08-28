from __future__ import annotations
import logging
import numpy as np
import xarray as xr
import xarray as xr
from sorts.schedule_v2.types import ScheduleNdarrayDict, ScheduleNdarrayDict2

logger = logging.getLogger(__name__)

# TODO: rename to just `ScheduleData` when xarray adoptation is done?
ScheduleXrds = xr.Dataset
"""An xarray `Dataset` that contains the schedule data"""


def from_ndarrays(data: ScheduleNdarrayDict) -> ScheduleXrds:
    sch_data = xr.Dataset(
        coords={
            "start_time": data["start_time"],
            "end_time": ("start_time", data["end_time"]),
            "azelr": ["az", "el", "r"],
        },
        data_vars={
            "pointing": (
                ("azelr", "start_time"),
                data["pointing"],
            ),
            "exp_num": ("start_time", data["exp_num"]),
        },
        attrs={"exp_detail_map": data["exp_detail_map"]},
    )

    return sch_data


# TODO: remove its usage, then remove this method
def from_ndarrays_2(data: ScheduleNdarrayDict2) -> ScheduleXrds:
    sch_data = xr.Dataset(
        coords={
            "start_time": data["start_time"],
            "end_time": ("start_time", data["start_time"]),
            "azelr": ["az", "el", "r"],
        },
        data_vars={
            "pointing": (
                ("azelr", "start_time"),
                np.array(
                    [
                        data["pointing_az"],
                        data["pointing_el"],
                        np.full(len(data["pointing_az"]), 1.0, dtype=np.float64),
                    ]
                ),
            ),
            "exp_num": ("start_time", data["exp_num"]),
        },
        attrs={"exp_detail_map": data["exp_detail_map"]},
    )

    return sch_data


def empty() -> ScheduleXrds:
    sch_data = from_ndarrays(
        {
            "exp_detail_map": {},
            "start_time": np.empty(0, dtype="datetime64[us]"),
            "end_time": np.empty(0, dtype="datetime64[us]"),
            "exp_num": np.empty(0, dtype=np.int64),
            "pointing": np.empty((3, 0), dtype=np.float64),
        }
    )

    return sch_data
