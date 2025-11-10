"""
Defines the NewType `Schedule` and functions for its functionalities
"""

from __future__ import annotations
import logging, typing as t
from dataclasses import dataclass
from functools import reduce
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from sorts import types, utils, radar


logger = logging.getLogger(__name__)

CoordKey = t.Literal[
    "multi_index", "start_time", "exp_num", "stn_num", "simult_num", "enu", "e", "n", "u"
]
DataKey = t.Literal["end_time", "pointing"]
Key = t.Literal[DataKey, CoordKey]


class _K:
    """Internal helper class for accessing string keys consistently"""

    multi_index: t.Final = "multi_index"
    start_time: t.Final = "start_time"
    exp_num: t.Final = "exp_num"
    stn_num: t.Final = "stn_num"
    simult_num: t.Final = "simult_num"
    enu: t.Final = "enu"
    e: t.Final = "e"
    n: t.Final = "n"
    u: t.Final = "u"
    end_time: t.Final = "end_time"
    pointing: t.Final = "pointing"


utils.assert_class_attributes_equal_to(_K, t.get_args(Key))

Schedule = t.NewType("Schedule", xr.Dataset)
"""
A xarray `Dataset` of:
  ```
  Dimensions:      (multi_index: n, enu: 3)
  Coordinates:
    * multi_index  (multi_index) object MultiIndex ('exp_num', 'stn_num', 'simult_num', 'start_time')
    * start_time   (multi_index) datetime64[us]
    * exp_num      (multi_index) int16
    * stn_num      (multi_index) int16
    * simult_num   (multi_index) int16
    * enu          (enu) 'e' 'n' 'u'
  Data variables:
      end_time     (multi_index) datetime64[us]
      pointing     (enu, multi_index) float64
  ```
"""

SimultaneousNum = int
"""An int16 that corresponds to the order in simultaneous pointings"""

ExperimentId = int
"""A unique int16 that identifies an experiment"""


@dataclass(kw_only=True)
class ExperimentDetail:
    id: ExperimentId

    coh_int_bandwidth: float
    ipp: float
    pulse_length: float
    power: float
    bandwidth: float
    duty_cycle: float
    noise_temp: float

    slice_duration: types.Timedelta64_us
    "Duration of a control slice, in micro-second"


ExperimentDetailMap = dict[ExperimentId, ExperimentDetail]


ExperimentIdStationIdPairsMap = dict[ExperimentId, list[tuple[radar.StationId, radar.StationId]]]


# TODO: this helper should ideally be part of radar module/subpackage,
#   but we can move it after that module is refactored
def default_station():
    return radar.Station(
        lat=0.0,
        lon=0.0,
        alt=0.0,
        min_elevation=0.0,
        beam=None,  # might not work when the `Station` typing is tightened
        uid=0,
    )


def empty() -> Schedule:
    multi_index = pd.MultiIndex.from_arrays(
        [
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype="datetime64[us]"),
        ],
        names=(_K.exp_num, _K.stn_num, _K.simult_num, _K.start_time),
    )

    sch = xr.Dataset(
        coords={
            **xr.Coordinates.from_pandas_multiindex(multi_index, _K.multi_index),
            _K.enu: [_K.e, _K.n, _K.u],
        },
        data_vars={
            _K.end_time: (_K.multi_index, np.empty(0, dtype="datetime64[us]")),
            _K.pointing: ((_K.enu, _K.multi_index), np.empty((3, 0), dtype=np.float64)),
        },
    )

    return Schedule(sch)


def from_ndarrays(
    start_time: npt.NDArray[types.Datetime64_us],
    end_time: npt.NDArray[types.Datetime64_us],
    exp_num: npt.NDArray[np.int16],
    stn_num: npt.NDArray[np.int16],
    simult_num: npt.NDArray[np.int16],
    pointing: types.EnuCoordinates,
) -> Schedule:
    multi_index = pd.MultiIndex.from_arrays(
        [exp_num, stn_num, simult_num, start_time],
        names=(_K.exp_num, _K.stn_num, _K.simult_num, _K.start_time),
    )

    sch = xr.Dataset(
        coords={
            **xr.Coordinates.from_pandas_multiindex(multi_index, _K.multi_index),
            _K.enu: [_K.e, _K.n, _K.u],
        },
        data_vars={
            _K.end_time: (_K.multi_index, end_time),
            _K.pointing: ((_K.enu, _K.multi_index), pointing),
        },
    )

    return Schedule(sch)


# TODO: remove?
def to_dataframe(sch: Schedule) -> pd.DataFrame:
    df = pd.concat(
        t.cast(
            list[pd.DataFrame],
            [
                sch[_K.end_time].transpose().to_pandas(),
                sch[_K.pointing].transpose().to_pandas(),
            ],
        ),
        axis=1,
        copy=False,
    ).reset_index()

    return df


def filter_by_time_range(sch: Schedule, time_range: types.TimeRange_us) -> Schedule:
    mask = (sch[_K.start_time] >= time_range[0]) & (sch[_K.end_time] <= time_range[1])

    ds_masked = sch[{_K.multi_index: mask}]

    return ds_masked


def filter_by_time_ranges(sch: Schedule, time_ranges: t.Sequence[types.TimeRange_us]) -> Schedule:
    resultant_mask: xr.DataArray = reduce(
        xr.ufuncs.logical_or,
        [
            (sch[_K.start_time] >= time_range[0]) & (sch[_K.end_time] <= time_range[1])
            for time_range in time_ranges
        ],
    )

    ds_masked = sch[{_K.multi_index: resultant_mask}]

    return ds_masked
