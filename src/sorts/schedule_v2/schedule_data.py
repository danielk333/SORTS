from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import xarray as xr
from sorts.types import Timedelta64_us, Datetime64_us, Float64_as_deg
import xarray as xr

logger = logging.getLogger(__name__)

# Schedule

ScheduleDataKey = t.Literal["pointing", "exp_num"]
ScheduleCoordKey = t.Literal["start_time", "end_time"]
ScheduleAttrKey = t.Literal["exp_detail_map"]
ScheduleKey = t.Literal[ScheduleDataKey, ScheduleCoordKey, ScheduleAttrKey]

schedule_data_keys: dict[ScheduleDataKey, str] = {k: k for k in t.get_args(ScheduleDataKey)}
schedule_coord_keys: dict[ScheduleCoordKey, str] = {k: k for k in t.get_args(ScheduleCoordKey)}
schedule_attr_keys: dict[ScheduleAttrKey, str] = {k: k for k in t.get_args(ScheduleAttrKey)}
schedule_keys: dict[ScheduleKey, str] = {k: k for k in t.get_args(ScheduleKey)}


class ExperimentDetail(t.TypedDict):
    """A TypedDict of params"""

    id: int

    coh_int_bandwidth: float  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    ipp: float  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    pulse_length: float  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    power: float
    bandwidth: float
    duty_cycle: float  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    noise_temp: float

    slice_duration: Timedelta64_us
    "Duration of a control slice, in micro-second"

    # TODO: this is a temp workaround to get multiple simutaneous rx pointings working
    num_simutaneous_pointings: t.NotRequired[int]


# TODO: try to remove/dissolve this class?
class ScheduleNdarrayDict2(t.TypedDict):
    """
    A TypedDict, stores a collection of "control slices" (or "slices" in short).

    - Slice data are stored as columns of fields, each of which is a `ndarray`.
    - Metadata (`ExperimentDetail`s) are stored as a dict inside the `exp_detail_map` field.
    """

    # TODO: add `Station` into this class?
    exp_detail_map: dict[int, ExperimentDetail]

    start_time: npt.NDArray[Datetime64_us]

    # TODO: re-eval the size of `exp_num`
    exp_num: npt.NDArray[np.int64]

    pointing_az: npt.NDArray[Float64_as_deg]
    pointing_el: npt.NDArray[Float64_as_deg]


# TODO: rename to just `ScheduleData` when xarray adoptation is done?
ScheduleXrds = xr.Dataset
"""An xarray `Dataset` that contains the schedule data"""
