from __future__ import annotations
import logging, typing as t
from functools import reduce
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from sorts.types import Datetime64_us, TimeRange_us, Timedelta64_us, EnuCoordinates
from sorts.utils import assert_class_attributes_equal_to
from sorts.radar import Station, StationId


logger = logging.getLogger(__name__)


CoordKey = t.Literal[
    "multi_index", "start_time", "exp_num", "stn_num", "simult_num", "enu", "e", "n", "u"
]
DataKey = t.Literal["end_time", "pointing"]
AttrKey = t.Literal["exp_detail_map"]
Key = t.Literal[DataKey, CoordKey, AttrKey]


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
    exp_detail_map: t.Final = "exp_detail_map"


assert_class_attributes_equal_to(_K, t.get_args(Key))

ScheduleData = xr.Dataset
"""
A xarray `Dataset` of:
  ```
  Dimensions:      (multi_index: n, enu: 3)
  Coordinates:
    * multi_index  (multi_index) object MultiIndex ('start_time', 'exp_num', 'stn_num', 'simult_num')
    * start_time   (multi_index) datetime64[us]
    * exp_num      (multi_index) int16
    * stn_num      (multi_index) int16
    * simult_num   (multi_index) int16
    * enu          (enu) 'e' 'n' 'u'
  Data variables:
      end_time     (multi_index) datetime64[us]
      pointing     (enu, multi_index) float64
  Attributes:
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

    # TODO: re-eval if we should we `t.NotRequired` here
    stn_pairs: t.NotRequired[list[tuple[int, int]]]
    """TX-RX station number pairs"""


# TODO: replace existing usage of `dict[int, ExperimentDetail]` by this type
ExperimentDetailMap = dict[int, ExperimentDetail]


class ScheduleNdarrayDict(t.TypedDict):
    """
    A TypedDict, stores a collection of "control slices" (or "slices" in short).

    - Slice data are stored as columns of fields, each of which is a `ndarray`.
    - Metadata (`ExperimentDetail`s) are stored as a dict inside the `exp_detail_map` field.
    """

    exp_detail_map: ExperimentDetailMap

    start_time: npt.NDArray[Datetime64_us]
    end_time: npt.NDArray[Datetime64_us]

    # TODO: re-eval the size of `exp_num`, `stn_num`, `simult_num`
    exp_num: npt.NDArray[np.int16]
    stn_num: npt.NDArray[np.int16]
    simult_num: npt.NDArray[np.int16]

    pointing: EnuCoordinates


# TODO: this helper should ideally be part of radar module/subpackage,
#   but we can move it after that module is refactored
def default_station():
    return Station(
        lat=0.0,
        lon=0.0,
        alt=0.0,
        min_elevation=0.0,
        beam=None,  # might not work when the `Station` typing is tightened
        uid=0,
    )


def empty_data() -> ScheduleData:
    multi_index = pd.MultiIndex.from_arrays(
        [
            np.empty(0, dtype="datetime64[us]"),
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=np.int16),
        ],
        names=(_K.start_time, _K.exp_num, _K.stn_num, _K.simult_num),
    )

    sch_data = xr.Dataset(
        coords={
            **xr.Coordinates.from_pandas_multiindex(multi_index, _K.multi_index),
            _K.enu: [_K.e, _K.n, _K.u],
        },
        data_vars={
            _K.end_time: (_K.multi_index, np.empty(0, dtype="datetime64[us]")),
            _K.pointing: ((_K.enu, _K.multi_index), np.empty((3, 0), dtype=np.float64)),
        },
        attrs={_K.exp_detail_map: {}},
    )

    return sch_data


def from_ndarrays(data: ScheduleNdarrayDict) -> ScheduleData:
    multi_index = pd.MultiIndex.from_arrays(
        [data[_K.start_time], data[_K.exp_num], data[_K.stn_num], data[_K.simult_num]],
        names=(_K.start_time, _K.exp_num, _K.stn_num, _K.simult_num),
    )

    sch_data = xr.Dataset(
        coords={
            **xr.Coordinates.from_pandas_multiindex(multi_index, _K.multi_index),
            _K.enu: [_K.e, _K.n, _K.u],
        },
        data_vars={
            _K.end_time: (_K.multi_index, data[_K.end_time]),
            _K.pointing: ((_K.enu, _K.multi_index), data[_K.pointing]),
        },
        attrs={_K.exp_detail_map: data[_K.exp_detail_map]},
    )

    return sch_data


def to_ndarrays(data: ScheduleData) -> ScheduleNdarrayDict:
    arr_dict: ScheduleNdarrayDict = {
        _K.exp_detail_map: data.attrs[_K.exp_detail_map],
        _K.start_time: data[_K.start_time].to_numpy(),
        _K.end_time: data[_K.end_time].to_numpy(),
        _K.exp_num: data[_K.exp_num].to_numpy(),
        _K.stn_num: data[_K.stn_num].to_numpy(),
        _K.simult_num: data[_K.simult_num].to_numpy(),
        _K.pointing: data[_K.pointing].to_numpy(),
    }

    return arr_dict


# TODO: remove?
def to_dataframe(ds: ScheduleData) -> pd.DataFrame:
    df = pd.concat(
        t.cast(
            list[pd.DataFrame],
            [
                ds[_K.end_time].transpose().to_pandas(),
                ds[_K.pointing].transpose().to_pandas(),
            ],
        ),
        axis=1,
        copy=False,
    ).reset_index()

    return df


def merge_attrs(attrs_dicts: list[dict[AttrKey, t.Any]]) -> dict:
    """
    Merging attrs dict, latter attrs dict will override former attrs dict, just like `.update()` method of `dict`.

    Returns a shallow copy.
    """

    match len(attrs_dicts):
        case 0:
            return {}

        case 1:
            # returns a shallow copy, the dict 'exp_detail_map' will be a new shallow copy as well
            return {
                **attrs_dicts[0],
                _K.exp_detail_map: attrs_dicts[0][_K.exp_detail_map].copy(),
            }

        case _:
            # returns a shallow copy, the dict 'exp_detail_map' will be a new shallow copy as well
            result: dict[AttrKey, t.Any] = {
                **attrs_dicts[0],
                _K.exp_detail_map: attrs_dicts[0][_K.exp_detail_map].copy(),
            }

            for attrs_dict in attrs_dicts[0:]:
                result[_K.exp_detail_map].update(attrs_dict[_K.exp_detail_map])

    return result


def filter_by_time_range(ds: ScheduleData, time_range: TimeRange_us) -> ScheduleData:
    mask = (ds[_K.start_time] >= time_range[0]) & (ds[_K.end_time] <= time_range[1])

    ds_masked = ds[{_K.multi_index: mask}]

    return ds_masked


# TODO: this is very similar to `rx_time_mask: xr.DataArray = reduce(...)` in `simulation_unit.py`,
#   maybe one of them can be dissolved?
def filter_by_time_ranges(ds: ScheduleData, time_ranges: t.Sequence[TimeRange_us]) -> ScheduleData:
    resultant_mask: xr.DataArray = reduce(
        xr.ufuncs.logical_or,
        [
            (ds[_K.start_time] >= time_range[0]) & (ds[_K.end_time] <= time_range[1])
            for time_range in time_ranges
        ],
    )

    ds_masked = ds[{_K.multi_index: resultant_mask}]

    return ds_masked


# TODO: can probably be simplified, or even dissovled, now that we have `simult_num` in `ScheduleData`
def get_indexer_per_measurement(
    ds: ScheduleData, is_split_simult: bool, is_copy=False
) -> list[xr.DataArray]:
    """
    Split a schedule data by measurements.

    i.e. By `exp_num` and optionally per each of the simutaneous pointings (controlled by `is_split_simult`)
    """

    idxers: list[xr.DataArray] = []

    # early return special case
    if len(ds[_K.multi_index]) == 0:
        return idxers

    # identify where `exp_num` changes
    exp_num_chg_pts = ds[_K.exp_num] != ds[_K.exp_num].shift({_K.multi_index: 1})
    exp_num_split_ids = exp_num_chg_pts.cumsum()

    for _, ds_split in ds.groupby(exp_num_split_ids):
        if is_split_simult:
            # further spliting according to number of simutaneous rx pointings
            for simult_num in np.unique(ds_split[_K.simult_num].to_numpy()):
                idxer = ds_split[_K.multi_index].loc[{_K.simult_num: simult_num}]
                idxers.append(idxer if not is_copy else idxer.copy())

        else:
            idxer = ds_split[_K.multi_index]
            idxers.append(idxer if not is_copy else idxer.copy())

    return idxers
