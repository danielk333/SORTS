import logging, typing as t, math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from .. import scheduler_v2 as schr
from .. import controller_v2 as ctrlr

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RandomUniformScansController(ctrlr.ControllerProtocol):
    """
    a controller that generate random uniform scans
    """

    min_elevation_deg: float = 30.0
    dwell_us: float = 1000  # 1ms
    npoints: int = 10_000

    coh_int_bandwidth: float = 1.0
    ipp: float = 1.0
    pulse_length: float = 1.0

    # TODO: eval if we want to use `pydantic` (https://docs.pydantic.dev/)
    def __post_init__(self):
        if self.min_elevation_deg < 0 or self.min_elevation_deg > 90:
            raise RuntimeError(f"`min_elevation_deg` has to be in the range [0, 90]")

    def generate(
        self,
        stt_tstmp=datetime.now(timezone.utc),
        end_tstmp=datetime.now(timezone.utc) + timedelta(hours=24),
        res_us=1000,
    ) -> pd.DataFrame:
        """
        Parameters
        ---

        stt_tstmp
            start timestamp, irrelevant in this controller
        end_time
            end timestamp, irrelevant in this controller

        Returns
        ---
        a DataFrame with these columns:

        |index     |coh_int_bandwidth|pointing         |ipp    |pulse_length|
        |:-        |:-               |:-               |:-     |:-          |
        |datetime64|float64          |(float64,float64)|float64|float64     |
        """
        ...

        max_points_by_res_us = math.floor((end_tstmp - stt_tstmp) / timedelta(microseconds=res_us))
        if max_points_by_res_us < self.npoints:
            raise RuntimeError(
                f"npoints: {self.npoints} larger than res_us allows: {max_points_by_res_us}"
            )

        max_points_by_dwell_us = math.floor(
            (end_tstmp - stt_tstmp) / timedelta(microseconds=self.dwell_us)
        )
        if max_points_by_dwell_us < self.npoints:
            raise RuntimeError(
                f"npoints: {self.npoints} larger than dwell_us allows: {max_points_by_dwell_us}"
            )

        min_el = np.radians(self.min_elevation_deg)

        arr = np.recarray((self.npoints,), dtype=schr.schedule_ndarray_dtype)
        cn = schr.schedule_column_names

        arr[cn["stt_tstmp"]] = pd.date_range(
            start=stt_tstmp, end=end_tstmp, periods=self.npoints
        ).values

        arr[cn["coh_int_bandwidth"]].fill(self.coh_int_bandwidth)

        # TODO: chk the math and add a plot function in test?
        arr[cn["pointing_az"]] = np.random.uniform(low=0, high=2 * np.pi, size=self.npoints)
        arr[cn["pointing_el"]] = np.random.uniform(low=min_el, high=np.pi / 2, size=self.npoints)

        arr[cn["ipp"]].fill(self.ipp)
        arr[cn["ipp"]].fill(self.pulse_length)

        ret_df = pd.DataFrame(
            arr,
            columns=[*schr.schedule_column_names.values()],
        )

        # align the rows to res_us
        ret_df[cn["stt_tstmp"]] = t.cast(pd.Series, ret_df[cn["stt_tstmp"]]).dt.floor(f"{res_us}ns")

        ret_df.set_index(cn["stt_tstmp"], inplace=True)

        return ret_df
