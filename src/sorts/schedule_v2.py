from __future__ import annotations
import logging, typing as t
from dataclasses import dataclass, fields
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts.types import Timedelta64_us, Datetime64_us, Float64_as_deg

logger = logging.getLogger(__name__)


class DataFrameColumnNames:
    """
    Define the `pandas` `DataFrame` column names of a `Schedule` as class member.
    """

    # TODO: add test to ensure this file up-to-date with `Schedule class

    start_time: t.Final = "stt_tstmp_us"
    end_time: t.Final = "end_time"
    pointing_az: t.Final = "pointing_az"
    pointing_el: t.Final = "pointing_el"
    exp_num: t.Final = "exp_num"

    @classmethod
    def all(cls) -> list[str]:
        """Return a list of all column names."""

        return [
            t.cast(str, v)
            for k, v in vars(DataFrameColumnNames).items()
            if (not k.startswith("__")) and (not isinstance(v, classmethod))
        ]

    @classmethod
    def all_non_derived(cls) -> list[str]:
        """
        Return a list of all non-derived column names.
        (i.e. It is not generated and has a corresponding field in the `Schedule` dataclass)
        """

        return [c for c in cls.all() if c not in [cls.end_time]]


# TODO: rename to sth like `ControlSliceDetail`?
@dataclass(kw_only=True)
class ExperimentDetail:
    id: int

    coh_int_bandwidth: float
    ipp: float
    pulse_length: float
    power: float
    bandwidth: float
    duty_cycle: float
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

    meta: dict[int, ExperimentDetail]

    # TODO: if `end_tstmp_ms` is not needed, this can be renamed to just `tstmp_ms`?
    # TODO: `end_tstmp_ms` is v. likely not needed, rename it to just `time`? (and add docs that this is the start timestamp)
    stt_tstmp_us: npt.NDArray[Datetime64_us]

    # TODO: re-eval the size of `exp_num`
    exp_num: npt.NDArray[np.int64]

    pointing_az: npt.NDArray[Float64_as_deg]
    pointing_el: npt.NDArray[Float64_as_deg]

    @classmethod
    def empty(cls) -> Schedule:
        """A convenience method for generating an empty schedule"""

        sch = Schedule(
            meta={},
            stt_tstmp_us=np.empty(0, "datetime64[us]"),
            exp_num=np.empty(0, np.int64),
            pointing_az=np.empty(0, Float64_as_deg),
            pointing_el=np.empty(0, Float64_as_deg),
        )

        return sch

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, meta: dict[int, ExperimentDetail]) -> Schedule:
        sch_dict = {c: df[c].to_numpy() for c in DataFrameColumnNames.all_non_derived()}
        sch = Schedule(**sch_dict, meta=meta)

        return sch

    def __post_init__(self):
        f_0, *f_rests = [
            f for f in fields(Schedule) if f.name in DataFrameColumnNames.all_non_derived()
        ]
        fv_0, *fv_rests = t.cast(
            tuple[npt.NDArray, ...],
            [getattr(self, c) for c in DataFrameColumnNames.all_non_derived()],
        )  # actual value of the fields

        for idx, f in enumerate(fv_rests):
            if f.shape != fv_0.shape:
                raise RuntimeError(
                    "fields of a `Schedule` must have equal lengths. "
                    + f"but shape of {f_rests[idx].name} is {fv_rests[idx].shape}, "
                    f"while shape of {f_0.name} is {fv_0.shape} "
                )

    @property
    def cn(self):
        """A shortcut to return the DataFrameColumnNames class"""

        return DataFrameColumnNames

    def as_dataframe(self) -> pd.DataFrame:
        """
        Convert `Schedule` into a pandas `DataFrame`.

        Some extra derived columns are generated in the resultant `DataFrame`, while metadata field(s) are not included.

        Handy for manipulation and plotting.
        """

        df = pd.DataFrame({c: getattr(self, c) for c in DataFrameColumnNames.all_non_derived()})

        # add "end_time" column
        df[self.cn.end_time] = df[self.cn.start_time] + np.array(
            [self.meta[n].slice_duration for n in self.exp_num],
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
            self.stt_tstmp_us >= start_time,
            self.stt_tstmp_us <= end_time,
        )

        return sch_dt_s_arr_pass_mask

    def filter_by_mask(self, mask: npt.NDArray[np.bool]):
        """Return a slice of the origin schedule based on the `mask`"""

        filtered_sch = Schedule(
            meta=self.meta,
            stt_tstmp_us=self.stt_tstmp_us[mask],
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
