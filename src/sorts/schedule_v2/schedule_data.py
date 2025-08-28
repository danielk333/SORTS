from __future__ import annotations
import logging, typing as t
import numpy as np
import pandas as pd
import xarray as xr
from sorts.schedule_v2.types import ScheduleNdarrayDict, ScheduleNdarrayDict2

logger = logging.getLogger(__name__)

ScheduleDataKey = t.Literal["pointing", "exp_num"]
ScheduleCoordKey = t.Literal["start_time", "end_time"]
ScheduleAttrKey = t.Literal["exp_detail_map"]
ScheduleKey = t.Literal[ScheduleDataKey, ScheduleCoordKey, ScheduleAttrKey]

schedule_data_keys: dict[ScheduleDataKey, str] = {k: k for k in t.get_args(ScheduleDataKey)}
schedule_coord_keys: dict[ScheduleCoordKey, str] = {k: k for k in t.get_args(ScheduleCoordKey)}
schedule_attr_keys: dict[ScheduleAttrKey, str] = {k: k for k in t.get_args(ScheduleAttrKey)}
schedule_keys: dict[ScheduleKey, str] = {k: k for k in t.get_args(ScheduleKey)}

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


def to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    # define some column names/keys
    keys = schedule_keys

    df = pd.concat(
        t.cast(
            list[pd.DataFrame],
            [
                ds[keys["end_time"]].transpose().to_pandas(),
                ds[keys["pointing"]].transpose().to_pandas(),
            ],
        ),
        axis=1,
        copy=False,
    )

    return df
