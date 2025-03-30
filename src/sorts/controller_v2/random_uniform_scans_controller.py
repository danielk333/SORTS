import logging, typing as t, math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import numpy as np
from .. import scheduler_v2 as schr
from .. import controller_v2 as ctrlr

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RandomUniformScansController(ctrlr.ControllerProtocol):
    """
    a controller that generate random uniform scans
    """

    exp_num: int = 0
    min_elevation_deg: float = 30.0
    time_slice_us: float = 1000  # 1ms
    npoints: int = 10_000

    coh_int_bandwidth: float = 1.0
    ipp: float = 1.0
    pulse_length: float = 1.0

    # TODO: eval if we want to use `pydantic` (https://docs.pydantic.dev/)
    def __post_init__(self):
        if self.min_elevation_deg < 0 or self.min_elevation_deg > 90:
            raise RuntimeError(
                f"`min_elevation_deg` ({self.min_elevation_deg}) has to be in the range [0, 90]"
            )

        if self.time_slice_us < self.ipp * self.npoints:
            raise RuntimeError(
                f"`time_slice_us` ({self.time_slice_us}) cannot be small than `ipp * npoints` ({self.ipp * self.npoints})"
            )

    def generate(
        self,
        stt_tstmp,
        end_tstmp,
        res_us=1000,
    ) -> schr.Schedule:
        """
        Parameters
        ---

        stt_tstmp
            start timestamp, irrelevant in this controller
        end_tstmp
            end timestamp, irrelevant in this controller
        res_us
            resolution in microseconds
        """
        ...

        if self.time_slice_us < res_us:
            raise RuntimeError(
                f"`time_slice_us` ({self.time_slice_us}) cannot finer than `res_us` ({res_us})"
            )

        max_points_by_time_slice_us = math.floor(
            (end_tstmp - stt_tstmp) / timedelta(microseconds=self.time_slice_us)
        )
        if max_points_by_time_slice_us < self.npoints:
            raise RuntimeError(
                f"npoints: {self.npoints} larger than `time_slice_us` allows: {max_points_by_time_slice_us}"
            )

        min_el = np.radians(self.min_elevation_deg)

        # TODO: chk the math and add a plot function in test?
        ret_sch = schr.Schedule(
            stt_tstmp_ms=np.arange(
                stt_tstmp,
                end_tstmp,
                np.timedelta64((end_tstmp - stt_tstmp) / self.npoints),
                dtype="datetime64[us]",
            ),
            exp_num=np.full(self.npoints, self.exp_num),
            pointing_az=np.random.uniform(low=0, high=2 * np.pi, size=self.npoints),
            pointing_el=np.random.uniform(low=min_el, high=np.pi / 2, size=self.npoints),
            coh_int_bandwidth=np.full(self.npoints, self.coh_int_bandwidth),
            ipp=np.full(self.npoints, self.ipp),
            pulse_length=np.full(self.npoints, self.pulse_length),
        )

        return ret_sch
