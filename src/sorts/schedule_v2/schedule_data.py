"""
Types and functions for `ScheduleData` manipuations.

- Not intended for consumption from outside of this library
- Intended to be imported as a module when consuming (e.g. `from sorts.schedule_v2 import schedule_data)
"""

from __future__ import annotations
import logging, typing as t
import numpy as np
import pandas as pd
import xarray as xr
from sorts.types import TimeRange_us
from sorts.utils import assert_class_attributes_equal_to

if t.TYPE_CHECKING:
    from sorts.schedule_v2.schedule import ExperimentDetail


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


def to_dataframe(ds: ScheduleData) -> pd.DataFrame:
    df = pd.concat(
        t.cast(
            list[pd.DataFrame],
            [
                ds[_K.end_time].transpose().to_pandas(),
                ds[_K.pointing].transpose().to_pandas(),
                ds[_K.exp_num].transpose().to_pandas(),
            ],
        ),
        axis=1,
        copy=False,
    ).reset_index()

    return df


def merge_attrs(attrs_dicts: list[dict[AttrKey, t.Any]]) -> dict:
    """Merging attrs dict, latter attrs dict will override former attrs dict, just like `.update()` method of `dict`"""

    match len(attrs_dicts):
        case 0:
            return {}
        case 1:
            return attrs_dicts[0]
        case _:
            result: dict[AttrKey, t.Any] = attrs_dicts[0]
            for attrs_dict in attrs_dicts[0:]:
                result[_K.stn_id] = attrs_dict[_K.stn_id]
                result[_K.exp_detail_map].update(attrs_dict[_K.exp_detail_map])

    return result


def filter_by_time_range(ds: ScheduleData, time_range: TimeRange_us) -> ScheduleData:
    mask = (ds[_K.start_time] >= time_range[0]) & (ds[_K.start_time] <= time_range[1])

    ds_masked = ds[{_K.start_time: mask}]

    return ds_masked


# TODO: maybe saving a `simu_grp` number in schedule is more memory efficient?
#   (or not, because measurement is very sparse over schedule)
def get_indexer_per_measurement(ds: ScheduleData, is_split_simu: bool) -> list[xr.DataArray]:
    """
    Split a schedule data by measurements.

    i.e. By `exp_num` and optionally per each of the simutaneous pointings (controlled by `is_split_simu`)
    """

    # identify where `exp_num` changes
    chg_pts = ds[_K.exp_num] != ds[_K.exp_num].shift({_K.start_time: 1})
    split_ids = chg_pts.cumsum()

    exp_detail_map: dict[int, ExperimentDetail] = ds.attrs[_K.exp_detail_map]
    idxers: list[xr.DataArray] = []
    for _, ds_split in ds.groupby(split_ids):
        if is_split_simu:
            # further spliting according to number of simutaneous rx pointings
            simu_num = exp_detail_map[ds_split[_K.exp_num][0].item()].get(
                "num_simutaneous_pointings", 1
            )
            for i in range(simu_num):
                idxers.append(
                    xr.DataArray(
                        (np.arange(len(ds_split[_K.start_time])) - i) % simu_num == 0,
                        dims=_K.start_time,
                    )
                )
        else:
            idxers.append(
                xr.DataArray(
                    np.full(len(ds_split[_K.start_time]), True, dtype=np.bool),
                    dims=_K.start_time,
                )
            )

    return idxers
