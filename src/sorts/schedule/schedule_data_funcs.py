from __future__ import annotations
import logging, typing as t
import numpy as np
import pandas as pd
import xarray as xr
from sorts.types import TimeRange_us
from sorts.radar import Station
from .types import _K, AttrKey

if t.TYPE_CHECKING:
    from .schedule import ScheduleData, ExperimentDetail, ScheduleNdarrayDict


logger = logging.getLogger(__name__)


# TODO: this helper should ideally be part of radar module/subpackage,
#   but we can move it after that module is refactored
def default_station():
    return Station(
        lat=0.0,
        lon=0.0,
        alt=0.0,
        min_elevation=0.0,
        beam=None,  # might not work when the `Station` typing is tightened
        uid=f"__generated_by_{default_station.__name__}",
    )


def empty_data() -> ScheduleData:
    sch_data = xr.Dataset(
        coords={
            _K.start_time: np.empty(0, dtype="datetime64[us]"),
            _K.end_time: (_K.start_time, np.empty(0, dtype="datetime64[us]")),
            _K.enu: [_K.e, _K.n, _K.u],
        },
        data_vars={
            _K.pointing: ((_K.enu, _K.start_time), np.empty((3, 0), dtype=np.float64)),
            _K.exp_num: (_K.start_time, np.empty(0, dtype=np.int64)),
        },
        attrs={
            _K.stn_id: f"__generated_by_{empty_data.__name__}",
            _K.exp_detail_map: {},
        },
    )

    return sch_data


def from_ndarrays(data: ScheduleNdarrayDict) -> ScheduleData:
    sch_data = xr.Dataset(
        coords={
            _K.start_time: data[_K.start_time],
            _K.end_time: (_K.start_time, data[_K.end_time]),
            _K.enu: [_K.e, _K.n, _K.u],
        },
        data_vars={
            _K.pointing: ((_K.enu, _K.start_time), data[_K.pointing]),
            _K.exp_num: (_K.start_time, data[_K.exp_num]),
            _K.simult_num: (_K.start_time, data[_K.simult_num]),
        },
        attrs={
            _K.stn_id: data[_K.stn_id],
            _K.exp_detail_map: data[_K.exp_detail_map],
        },
    )

    return sch_data


def to_ndarrays(data: ScheduleData) -> ScheduleNdarrayDict:
    arr_dict: ScheduleNdarrayDict = {
        _K.stn_id: data.attrs[_K.stn_id],
        _K.exp_detail_map: data.attrs[_K.exp_detail_map],
        _K.start_time: data[_K.start_time].to_numpy(),
        _K.end_time: data[_K.end_time].to_numpy(),
        _K.exp_num: data[_K.exp_num].to_numpy(),
        _K.simult_num: data[_K.simult_num].to_numpy(),
        _K.pointing: data[_K.pointing].to_numpy(),
    }

    return arr_dict


def to_dataframe(ds: ScheduleData) -> pd.DataFrame:
    df = pd.concat(
        t.cast(
            list[pd.DataFrame],
            [
                ds[_K.end_time].transpose().to_pandas(),
                ds[_K.pointing].transpose().to_pandas(),
                ds[_K.exp_num].transpose().to_pandas(),
                ds[_K.simult_num].transpose().to_pandas(),
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
    mask = (ds[_K.start_time] >= time_range[0]) & (ds[_K.end_time] <= time_range[1])

    ds_masked = ds[{_K.start_time: mask}]

    return ds_masked


# TODO: can probably be simplified, or even dissovled, now that we have `simult_num` in `ScheduleData`
def get_indexer_per_measurement(ds: ScheduleData, is_split_simu: bool) -> list[xr.DataArray]:
    """
    Split a schedule data by measurements.

    i.e. By `exp_num` and optionally per each of the simutaneous pointings (controlled by `is_split_simu`)
    """

    # identify where `exp_num` changes
    exp_num_chg_pts = ds[_K.exp_num] != ds[_K.exp_num].shift({_K.start_time: 1})
    exp_num_split_ids = exp_num_chg_pts.cumsum()

    exp_detail_map: dict[int, ExperimentDetail] = ds.attrs[_K.exp_detail_map]
    idxers: list[xr.DataArray] = []
    for _, ds_split in ds.groupby(exp_num_split_ids):
        if is_split_simu:
            # further spliting according to number of simutaneous rx pointings
            simu_num = exp_detail_map[ds_split[_K.exp_num][0].item()].get(
                "num_simutaneous_pointings", 1
            )
            for i in range(simu_num):
                idxers.append(
                    xr.DataArray(
                        (ds_split[_K.simult_num] == i).to_numpy(),
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
