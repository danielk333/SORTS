from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import xarray as xr
import pandas as pd
from sorts.types import Timedelta64_us, Datetime64_us, Float64_as_deg, Timedelta_Like
from sorts.utils import to_timedelta64_us

logger = logging.getLogger(__name__)


ScheduleFieldKey = t.Literal[
    "exp_detail_map", "start_time", "pointing_az", "pointing_el", "exp_num"
]

# Define the column names used when exported as a `DataFrame` (pandas or alike)
NonDerivedDataFrameColumnName = t.Literal["start_time", "pointing_az", "pointing_el", "exp_num"]
DerivedDataFrameColumnName = t.Literal["end_time"]
DataFrameColumnName = t.Literal["start_time", "end_time", "pointing_az", "pointing_el", "exp_num"]

assert all((n in t.get_args(ScheduleFieldKey) for n in t.get_args(NonDerivedDataFrameColumnName)))
assert set(t.get_args(DataFrameColumnName)) == set(
    [*t.get_args(NonDerivedDataFrameColumnName), *t.get_args(DerivedDataFrameColumnName)]
)


data_frame_column_names: t.Final[dict[DataFrameColumnName, str]] = {
    n: n for n in t.get_args(DataFrameColumnName)
}
"""A dict of `DataFrameColumnName` as string key-value pair for convenience."""

cn = data_frame_column_names
"""An alias of `data_frame_column_names`"""


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


# TODO: rename to just `Schedule` when xarray adoptation is done?
ScheduleXrds = xr.Dataset


# TODO: try to remove/dissolve this class?
class ScheduleNdarrayDict(t.TypedDict):
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


def to_schedule_ndarray_dict(sch: ScheduleXrds) -> ScheduleNdarrayDict:
    sch_dict = ScheduleNdarrayDict(
        exp_detail_map=sch.attrs["exp_detail_map"],
        start_time=sch["start_time"].to_numpy(),
        exp_num=sch["exp_num"].to_numpy(),
        pointing_az=sch["pointing"].to_numpy()[0],
        pointing_el=sch["pointing"].to_numpy()[1],
    )

    return sch_dict


# TODO: can be removed? will be automatically enforced by xarray dataset class
def validate_schedule_length(sch: ScheduleNdarrayDict) -> ScheduleNdarrayDict:
    """
    Throw exception if schedule fields are not consistent (same length).

    Returns the schedule itself.
    """

    k_0, *k_rests = [k for k in sch if k in t.get_args(NonDerivedDataFrameColumnName)]
    field_0, *field_rests = t.cast(
        tuple[npt.NDArray, ...],
        [sch[k] for k in t.get_args(NonDerivedDataFrameColumnName)],
    )  # actual value of the fields

    for k, f in zip(k_rests, field_rests):
        if f.shape != field_0.shape:
            raise RuntimeError(
                "fields of a `Schedule` must have equal lengths. "
                + f"but shape of {k} is {f.shape}, "
                f"while shape of {k_0} is {field_0.shape} "
            )

    return sch


def empty() -> ScheduleXrds:
    """A convenience method for generating an empty schedule"""

    sch = xr.Dataset(
        coords={
            "start_time": np.empty(0, dtype="datetime64[us]"),
            "azelr": ["az", "el", "r"],
        },
        data_vars={
            "pointing": (("azelr", "start_time"), []),
            "exp_num": (
                "start_time",
                np.empty(0, dtype=np.int64),
            ),
        },
        attrs={},
    )

    return sch


# TODO: remove its usage, then remove this func
def empty_npardict() -> ScheduleNdarrayDict:
    """A convenience method for generating an empty schedule"""

    sch = ScheduleNdarrayDict(
        exp_detail_map={},
        start_time=np.empty(0, "datetime64[us]"),
        exp_num=np.empty(0, np.int64),
        pointing_az=np.empty(0, Float64_as_deg),
        pointing_el=np.empty(0, Float64_as_deg),
    )
    validate_schedule_length(sch)

    return sch


# TODO: can be removed? xarray dataset class is already dataframe like, and have pandas conversion methods
def from_dataframe(
    df: pd.DataFrame, exp_detail_map: dict[int, ExperimentDetail]
) -> ScheduleNdarrayDict:
    sch = ScheduleNdarrayDict(
        **{k: df[k].to_numpy() for k in t.get_args(NonDerivedDataFrameColumnName)},
        exp_detail_map=exp_detail_map,
    )

    return sch


# TODO: can be removed? xarray dataset class is already dataframe like, and have pandas conversion methods
def to_dataframe(sch: ScheduleNdarrayDict) -> pd.DataFrame:
    """
    Convert `Schedule` into a pandas `DataFrame`.

    Some extra derived columns are generated in the resultant `DataFrame`, while metadata field(s) are not included.

    Handy for manipulation and plotting.
    """

    df = pd.DataFrame({k: sch[k] for k in t.get_args(NonDerivedDataFrameColumnName)})

    # add "end_time" column
    df[cn["end_time"]] = df[cn["start_time"]] + np.array(
        [sch["exp_detail_map"][n]["slice_duration"] for n in sch["exp_num"]],
        # NOTE: `dtype` have to be stated explicitly, otherwise numpy will assume `float64` which is incorrect here
        dtype="timedelta64[us]",
    )

    return df


# TODO: remove its usage, then remove this func
def create_mask_by_time_range(
    sch: ScheduleNdarrayDict, time_range: tuple[Datetime64_us, Datetime64_us]
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


# TODO: remove its usage, then remove this func
def filter_by_mask(sch: ScheduleNdarrayDict, mask: npt.NDArray[np.bool]) -> ScheduleNdarrayDict:
    """Return a slice of the origin schedule based on the `mask`"""

    filtered_sch = ScheduleNdarrayDict(
        exp_detail_map=sch["exp_detail_map"],
        start_time=sch["start_time"][mask],
        exp_num=sch["exp_num"][mask],
        pointing_az=sch["pointing_az"][mask],
        pointing_el=sch["pointing_el"][mask],
    )

    return filtered_sch


# TODO: remove its usage, then remove this func
def filter_by_time_range(
    sch: ScheduleNdarrayDict, time_range: tuple[Datetime64_us, Datetime64_us]
) -> ScheduleNdarrayDict:
    """
    Return a slice of the origin schedule based on the `time_range`
    (a right-open interval)
    """

    return filter_by_mask(sch, create_mask_by_time_range(sch, time_range))
