from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from sorts.types import Datetime64_us, TimeRange_us, AzelrCoordinates_DegM
from sorts.utils import assert_class_attributes_equal_to
from sorts.radar.tx_rx import StationId
from sorts.schedule_v2.types import ExperimentDetail, ScheduleNdarrayDict2
from sorts.schedule_v2.schedule_data import from_ndarrays_2
from sorts.schedule_v2.priority_scheduling import priority_scheduling

logger = logging.getLogger(__name__)


ScheduleDataKey = t.Literal["pointing", "exp_num"]
ScheduleCoordKey = t.Literal["start_time", "end_time"]
ScheduleAttrKey = t.Literal["stn_id", "exp_detail_map"]
ScheduleKey = t.Literal[ScheduleDataKey, ScheduleCoordKey, ScheduleAttrKey]


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


ScheduleData = xr.Dataset


def schedule_data_to_dataframe(ds: ScheduleData) -> pd.DataFrame:
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


def filter_schedule_data_by_time_range(ds: ScheduleData, time_range: TimeRange_us) -> ScheduleData:
    mask = (ds[_K.start_time] >= time_range[0]) & (ds[_K.start_time] <= time_range[1])

    ds_masked = ds[{_K.start_time: mask}]

    return ds_masked


# TODO: remove its usage, then remove this func
def create_mask_by_time_range(
    sch: ScheduleNdarrayDict2, time_range: tuple[Datetime64_us, Datetime64_us]
) -> npt.NDArray[np.bool]:
    """
    Return a mask that filters out schedule entries that are not inside `time_range`.
    (a right-open interval)

    NOTE: right now it only check against `start_time`
    """

    start_time, end_time = time_range

    # TODO: better include end_time in the schedule and check against that
    mask: npt.NDArray[np.bool] = np.logical_and(
        sch["start_time"] >= start_time,
        sch["start_time"] <= end_time,
    )

    return mask


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


# TODO: remove its usage, then remove this func
def filter_by_mask(sch: ScheduleNdarrayDict2, mask: npt.NDArray[np.bool]) -> ScheduleNdarrayDict2:
    """Return a slice of the origin schedule based on the `mask`"""

    filtered_sch = ScheduleNdarrayDict2(
        exp_detail_map=sch["exp_detail_map"],
        start_time=sch["start_time"][mask],
        exp_num=sch["exp_num"][mask],
        pointing_az=sch["pointing_az"][mask],
        pointing_el=sch["pointing_el"][mask],
    )

    return filtered_sch


# TODO: remove its usage, then remove this func
def filter_by_time_range(
    sch: ScheduleNdarrayDict2, time_range: tuple[Datetime64_us, Datetime64_us]
) -> ScheduleNdarrayDict2:
    """
    Return a slice of the origin schedule based on the `time_range`
    (a right-open interval)
    """

    return filter_by_mask(sch, create_mask_by_time_range(sch, time_range))


TimeRangeIndexer = TimeRange_us
"""Contains info to get a subset of entries from a `Schedule`"""

XrDataArrayIndexer = xr.DataArray
"""Contains info to get a subset of entries from a `Schedule`"""


# TODO: add schedule validation?
class Schedule:
    """
    Provides methods for manipuating the schedule data and enforce that the require columns/data are set.

    Schedule data is stored in a private attribute `_data` attribute, and some additional helper metadata are stored in other attributes.

    NOTE:
        We are still evaluating which backing data structure to use and is subject to change
    """

    _K = _K
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

    # TODO: remove its usage, then remove this method
    @classmethod
    def from_ndarrays_2(cls, data: ScheduleNdarrayDict2) -> t.Self:
        return cls(data=from_ndarrays_2(data))

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
        return schedule_data_to_dataframe(self._data)

    def filter_by_time_range(self, time_range: TimeRange_us) -> t.Self:
        cls = type(self)
        filtered_data = filter_schedule_data_by_time_range(self._data, time_range)
        return cls(data=filtered_data)

    # TODO: we can probably inject the schedule is tx or rx into `Schedule` class and remove param `is_split_simu`
    def get_indexer_per_measurement(self, is_split_simu: bool) -> list[XrDataArrayIndexer]:
        return get_indexer_per_measurement(self._data, is_split_simu)
