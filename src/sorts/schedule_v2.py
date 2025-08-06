from __future__ import annotations
import logging, typing as t
from dataclasses import dataclass, fields
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts.types import Timedelta64_us, Datetime64_us, Float64_as_deg

logger = logging.getLogger(__name__)


ScheduleFieldKey = t.Literal["meta", "start_time", "pointing_az", "pointing_el", "exp_num"]

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

Cn = DataFrameColumnName
"""An alias of `DataFrameColumnName`"""
cns = data_frame_column_names
"""An alias of `data_frame_column_names`"""


# TODO: rename to sth like `ControlSliceDetail`?
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


@dataclass(kw_only=True)
class Schedule:
    """
    A collection of "control slices" (or "slices" in short).

    Slice data are stored as columns of fields, each of which is a `ndarray`.
    Metadata (`ExperimentDetail`s) are stored as a dict inside the `meta` field.
    """

    # TODO: add `Station` into this class, maybe inside `meta`
    meta: dict[int, ExperimentDetail]

    start_time: npt.NDArray[Datetime64_us]

    # TODO: re-eval the size of `exp_num`
    exp_num: npt.NDArray[np.int64]

    pointing_az: npt.NDArray[Float64_as_deg]
    pointing_el: npt.NDArray[Float64_as_deg]

    Cn: t.ClassVar = data_frame_column_names
    """A shortcut to `data_frame_column_names`"""

    @classmethod
    def empty(cls) -> Schedule:
        """A convenience method for generating an empty schedule"""

        sch = Schedule(
            meta={},
            start_time=np.empty(0, "datetime64[us]"),
            exp_num=np.empty(0, np.int64),
            pointing_az=np.empty(0, Float64_as_deg),
            pointing_el=np.empty(0, Float64_as_deg),
        )

        return sch

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, meta: dict[int, ExperimentDetail]) -> Schedule:
        sch_dict = {c: df[c].to_numpy() for c in t.get_args(NonDerivedDataFrameColumnName)}
        sch = Schedule(**sch_dict, meta=meta)

        return sch

    def __post_init__(self):
        f_0, *f_rests = [
            f for f in fields(Schedule) if f.name in t.get_args(NonDerivedDataFrameColumnName)
        ]
        fv_0, *fv_rests = t.cast(
            tuple[npt.NDArray, ...],
            [getattr(self, c) for c in t.get_args(NonDerivedDataFrameColumnName)],
        )  # actual value of the fields

        for idx, f in enumerate(fv_rests):
            if f.shape != fv_0.shape:
                raise RuntimeError(
                    "fields of a `Schedule` must have equal lengths. "
                    + f"but shape of {f_rests[idx].name} is {fv_rests[idx].shape}, "
                    f"while shape of {f_0.name} is {fv_0.shape} "
                )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert `Schedule` into a pandas `DataFrame`.

        Some extra derived columns are generated in the resultant `DataFrame`, while metadata field(s) are not included.

        Handy for manipulation and plotting.
        """

        df = pd.DataFrame({c: getattr(self, c) for c in t.get_args(NonDerivedDataFrameColumnName)})

        # add "end_time" column
        df[self.Cn["end_time"]] = df[self.Cn["start_time"]] + np.array(
            [self.meta[n]["slice_duration"] for n in self.exp_num],
            # NOTE: `dtype` have to be stated explicitly, otherwise numpy will assume `float64` which is incorrect here
            dtype="timedelta64[us]",
        )

        return df

    def create_mask_by_time_range(self, time_range: tuple[Datetime64_us, Datetime64_us]):
        """
        Return a mask that filters out schedule entries that are not inside `time_range`.
        (start time and end time inclusive)
        """

        start_time, end_time = time_range

        sch_dt_s_arr_pass_mask: npt.NDArray[np.bool] = np.logical_and(
            self.start_time >= start_time,
            self.start_time <= end_time,
        )

        return sch_dt_s_arr_pass_mask

    def filter_by_mask(self, mask: npt.NDArray[np.bool]):
        """Return a slice of the origin schedule based on the `mask`"""

        filtered_sch = Schedule(
            meta=self.meta,
            start_time=self.start_time[mask],
            exp_num=self.exp_num[mask],
            pointing_az=self.pointing_az[mask],
            pointing_el=self.pointing_el[mask],
        )

        return filtered_sch

    def filter_by_time_range(self, time_range: tuple[Datetime64_us, Datetime64_us]):
        """
        Return a slice of the origin schedule based on the `time_range`
        (start time and end time inclusive)
        """

        return self.filter_by_mask(self.create_mask_by_time_range(time_range))
