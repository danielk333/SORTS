from __future__ import annotations
import logging
import numpy as np
import xarray as xr
from sorts.schedule_v2.types import ScheduleNdarrayDict2

logger = logging.getLogger(__name__)


# TODO: remove its usage, then remove this method
def from_ndarrays_2(data: ScheduleNdarrayDict2):
    from sorts.schedule_v2.schedule import ScheduleXrds

    sch_data: ScheduleXrds = xr.Dataset(
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
        attrs={
            "stn_id": data.get("stn_id", "__NO_STN_ID__"),
            "exp_detail_map": data["exp_detail_map"],
        },
    )

    return sch_data
