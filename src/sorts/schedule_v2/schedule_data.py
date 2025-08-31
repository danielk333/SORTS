from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from sorts.utils import assert_class_attributes_equal_to
from sorts.types import Datetime64_us, AzelrCoordinates_DegM, TimeRange_us
from sorts.radar.tx_rx import StationId
from sorts.schedule_v2.types import ExperimentDetail, ScheduleNdarrayDict2

logger = logging.getLogger(__name__)

ScheduleDataKey = t.Literal["pointing", "exp_num"]
ScheduleCoordKey = t.Literal["start_time", "end_time"]
ScheduleAttrKey = t.Literal["stn_id", "exp_detail_map"]
ScheduleKey = t.Literal[ScheduleDataKey, ScheduleCoordKey, ScheduleAttrKey]

schedule_data_keys: dict[ScheduleDataKey, str] = {k: k for k in t.get_args(ScheduleDataKey)}
schedule_coord_keys: dict[ScheduleCoordKey, str] = {k: k for k in t.get_args(ScheduleCoordKey)}
schedule_attr_keys: dict[ScheduleAttrKey, str] = {k: k for k in t.get_args(ScheduleAttrKey)}
schedule_keys: dict[ScheduleKey, str] = {k: k for k in t.get_args(ScheduleKey)}


class _K:
    """Internal helper class for accessing string keys consistently"""

    pointing: t.Final = "pointing"
    exp_num: t.Final = "exp_num"
    start_time: t.Final = "start_time"
    end_time: t.Final = "end_time"
    stn_id: t.Final = "stn_id"
    exp_detail_map: t.Final = "exp_detail_map"


assert_class_attributes_equal_to(_K, t.get_args(ScheduleKey))


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

# TODO: add a validation function which assure the expected coord/data_key/label are there


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
        attrs={
            "stn_id": data["stn_id"],
            "exp_detail_map": data["exp_detail_map"],
        },
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
        attrs={
            "stn_id": data.get("stn_id", "__NO_STN_ID__"),
            "exp_detail_map": data["exp_detail_map"],
        },
    )

    return sch_data


def empty() -> ScheduleXrds:
    sch_data = from_ndarrays(
        {
            "stn_id": "__EMPTY_ID__",
            "exp_detail_map": {},
            "start_time": np.empty(0, dtype="datetime64[us]"),
            "end_time": np.empty(0, dtype="datetime64[us]"),
            "exp_num": np.empty(0, dtype=np.int64),
            "pointing": np.empty((3, 0), dtype=np.float64),
        }
    )

    return sch_data


def to_dataframe(ds: ScheduleXrds) -> pd.DataFrame:
    # define some column names/keys
    keys = schedule_keys

    df = pd.concat(
        t.cast(
            list[pd.DataFrame],
            [
                ds[keys["end_time"]].transpose().to_pandas(),
                ds[keys["pointing"]].transpose().to_pandas(),
                ds[keys["exp_num"]].transpose().to_pandas(),
            ],
        ),
        axis=1,
        copy=False,
    )

    return df


def merge_attrs(attrs_dicts: list[dict[ScheduleAttrKey, t.Any]]) -> dict:
    """Merging attrs dict, latter attrs dict will override former attrs dict, just like `.update()` method of `dict`"""

    match len(attrs_dicts):
        case 0:
            return {}
        case 1:
            return attrs_dicts[0]
        case _:
            result: dict[ScheduleAttrKey, t.Any] = attrs_dicts[0]
            for attrs_dict in attrs_dicts[0:]:
                result["stn_id"] = attrs_dict["stn_id"]
                result["exp_detail_map"].update(attrs_dict["exp_detail_map"])

    return result


def filter_by_time_range(ds: ScheduleXrds, time_range: TimeRange_us) -> ScheduleXrds:
    mask = (ds[schedule_keys["start_time"]] >= time_range[0]) & (
        ds[schedule_keys["start_time"]] <= time_range[1]
    )

    ds_masked = ds[{schedule_keys["start_time"]: mask}]

    return ds_masked


# TODO: maybe saving a `simu_grp` number in schedule is more memory efficient
def get_indexer_per_measurement(ds: ScheduleXrds, is_split_simu: bool) -> list[xr.DataArray]:
    """
    Split a schedule data by measurements.

    i.e. By `exp_num` and optionally per each of the simutaneous pointings (controlled by `is_split_simu`)
    """

    k = schedule_keys

    # identify where `exp_num` changes
    chg_pts = ds[k["exp_num"]] != ds[k["exp_num"]].shift({k["start_time"]: 1})
    split_ids = chg_pts.cumsum()

    exp_detail_map: dict[int, ExperimentDetail] = ds.attrs[k["exp_detail_map"]]
    idxers: list[xr.DataArray] = []
    for _, ds_split in ds.groupby(split_ids):
        if is_split_simu:
            # further spliting according to number of simutaneous rx pointings
            simu_num = exp_detail_map[ds_split[k["exp_num"]][0].item()].get(
                "num_simutaneous_pointings", 1
            )
            for i in range(simu_num):
                idxers.append(
                    xr.DataArray(
                        (np.arange(len(ds_split[k["start_time"]])) - i) % simu_num == 0,
                        dims=k["start_time"],
                    )
                )
        else:
            idxers.append(
                xr.DataArray(
                    np.full(len(ds_split[k["start_time"]]), True, dtype=np.bool),
                    dims=k["start_time"],
                )
            )

    return idxers
