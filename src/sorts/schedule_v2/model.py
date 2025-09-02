from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from sorts.types import Datetime64_us, TimeRange_us, Timedelta64_us, AzelrCoordinates_DegM
from sorts.utils import assert_class_attributes_equal_to
from sorts.radar.tx_rx import StationId
from . import funcs


logger = logging.getLogger(__name__)

DataKey = t.Literal["pointing", "exp_num"]
CoordKey = t.Literal["start_time", "end_time"]
AttrKey = t.Literal["stn_id", "exp_detail_map"]
Key = t.Literal[DataKey, CoordKey, AttrKey]


class _K:
    """Internal helper class for accessing string keys consistently"""

    pointing: t.Final = "pointing"
    exp_num: t.Final = "exp_num"
    start_time: t.Final = "start_time"
    end_time: t.Final = "end_time"
    stn_id: t.Final = "stn_id"
    exp_detail_map: t.Final = "exp_detail_map"


assert_class_attributes_equal_to(_K, t.get_args(Key))


ScheduleData = xr.Dataset
"""
A xarray `Dataset` with:
  ```
  Dimensions:     (azelr: 3, start_time: n)
  Coordinates:
  * start_time  (start_time) datetime64[us]
      end_time    (start_time) datetime64[us]
  * azelr       (azelr) <U2 24B 'az' 'el' 'r'
  Data variables:
      pointing    (azelr, start_time) float64
      exp_num     (start_time) int64
  Attributes:
      stn_id:          str
      exp_detail_map:  dict[int, ExperimentDetail]
  ```
"""


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


TimeRangeIndexer = TimeRange_us
"""Contains info to get a subset of entries from a `Schedule`"""

XrDataArrayIndexer = xr.DataArray
"""Contains info to get a subset of entries from a `Schedule`"""


# TODO: add schedule validation?
class Schedule:
    """
    Provides methods for manipuating the schedule data and enforce that the require columns/data are set.
    Also contains some related metadata.

    Schedule data is stored in the private attribute `_data`,
    and is not intended for consumption from outside of this library.

    NOTE:
        We are still evaluating which backing data structure to use and is subject to change
    """

    _K = funcs._K
    """shortcut to module attribute"""

    def __init__(self, data: ScheduleData):
        self._data: ScheduleData = data

    @classmethod
    def from_ndarrays(cls, data: ScheduleNdarrayDict) -> t.Self:
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
            attrs={
                "stn_id": data["stn_id"],
                "exp_detail_map": data["exp_detail_map"],
            },
        )

        return cls(data=sch_data)

    @classmethod
    def empty(cls) -> t.Self:
        return cls.from_ndarrays(
            {
                "stn_id": "__EMPTY_ID__",
                "exp_detail_map": {},
                "start_time": np.empty(0, dtype="datetime64[us]"),
                "end_time": np.empty(0, dtype="datetime64[us]"),
                "exp_num": np.empty(0, dtype=np.int64),
                "pointing": np.empty((3, 0), dtype=np.float64),
            }
        )

    @classmethod
    def priority_scheduling(cls, schs: t.Sequence[Schedule]):
        """
        Merge a sequence of schedules for a single station into one,
        schedule with smaller index in the sequence is given priority over those with larger index.

        Note: It is assumed (and not checked) that each of the schedule itself does not contain overlapping entries.
        """

        # NOTE: used lazy import here to avoid circular import
        from sorts.schedule_v2.priority_scheduling import priority_scheduling

        resultant_sch_data = priority_scheduling([sch._data for sch in schs])
        return cls(data=resultant_sch_data)

    def __repr__(self):
        return f"<sorts.Schedule> with data:\n{self._data.__repr__()}"

    def to_ndarrays(self) -> ScheduleNdarrayDict:
        arr_dict: ScheduleNdarrayDict = {
            "stn_id": self._data.attrs[_K.stn_id],
            "exp_detail_map": self._data.attrs[_K.exp_detail_map],
            "start_time": self._data[_K.start_time].to_numpy(),
            "end_time": self._data[_K.end_time].to_numpy(),
            "exp_num": self._data[_K.exp_num].to_numpy(),
            "pointing": self._data[_K.pointing].to_numpy(),
        }

        return arr_dict

    def to_dataframe(self) -> pd.DataFrame:
        return funcs.to_dataframe(self._data)

    def filter_by_time_range(self, time_range: TimeRange_us) -> t.Self:
        cls = type(self)
        filtered_data = funcs.filter_by_time_range(self._data, time_range)
        return cls(data=filtered_data)

    # TODO: we can probably inject the schedule is tx or rx into `Schedule` class and remove param `is_split_simu`
    def get_indexer_per_measurement(self, is_split_simu: bool) -> list[XrDataArrayIndexer]:
        return funcs.get_indexer_per_measurement(self._data, is_split_simu)
