from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import xarray as xr
from sorts.types import Datetime64_us, AzelrCoordinates_DegM, TimeRange_us
from sorts.radar.tx_rx import StationId
from sorts.schedule_v2.types import ExperimentDetail, ScheduleNdarrayDict2

logger = logging.getLogger(__name__)


class ScheduleNdarrayDict(t.TypedDict):
    """
    A TypedDict, stores a collection of "control slices" (or "slices" in short).

    - Slice data are stored as columns of fields, each of which is a `ndarray`.
    - Metadata (`ExperimentDetail`s) are stored as a dict inside the `exp_detail_map` field.
    """

    stn_id: StationId

    exp_detail_map: dict[int, ExperimentDetail]

    start_time: npt.NDArray[Datetime64_us]
    end_time: npt.NDArray[Datetime64_us]

    # TODO: re-eval the size of `exp_num`
    exp_num: npt.NDArray[np.int64]

    pointing: AzelrCoordinates_DegM


# TODO: rename to just `ScheduleData` when xarray adoptation is done?
ScheduleXrds = xr.Dataset
"""An xarray `Dataset` that contains the schedule data"""


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
        attrs={
            "stn_id": data.get("stn_id", "__NO_STN_ID__"),
            "exp_detail_map": data["exp_detail_map"],
        },
    )

    return sch_data
