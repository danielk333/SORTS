from __future__ import annotations
import logging, typing as t
from dataclasses import dataclass, fields
import numpy as np
import numpy.typing as npt
from sorts.types import Datetime64_us

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Schedule:
    """
    a dataclass that holds fields of ndarrays which forms a schedule
    """

    # TODO: if `end_tstmp_ms` is not needed, this can be renamed to just `tstmp_ms`?
    stt_tstmp_us: npt.NDArray[Datetime64_us]

    # TODO: re-eval the size of `exp_num`
    exp_num: npt.NDArray[np.int64]

    pointing_az: npt.NDArray[np.float64]
    pointing_el: npt.NDArray[np.float64]

    def __post_init__(self):
        f_0, *f_rests = fields(self)  # Field objects
        fv_0, *fv_rests = t.cast(
            tuple[npt.NDArray, ...], [getattr(self, f.name) for f in fields(self)]
        )  # actual value of the fields

        for idx, f in enumerate(fv_rests):
            if f.shape != fv_0.shape:
                raise RuntimeError(
                    "fields of a `Schedule` must have equal lengths. "
                    + f"but shape of {f_rests[idx].name} is {fv_rests[idx].shape}, "
                    f"while shape of {f_0.name} is {fv_0.shape} "
                )

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
