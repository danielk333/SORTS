import logging
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import numpy.typing as npt
import pyant
from .. import scheduler_v2 as schr
from .. import controller_v2 as ctrlr
from .. import passes_v2
from ..radar.radars.composite_key import RadarStationCompositeKey

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TrackerController(ctrlr.ControllerProtocol):
    """
    a controller for tracking `Pass` object
    """

    # TODO: add `is_radian` as `Pass` member field? default to `False`?
    pass_obj: passes_v2.Pass

    exp_num: int = 0
    time_slice_us: float = 1000  # 1ms

    coh_int_bandwidth: float = 1.0
    ipp: float = 1.0
    pulse_length: float = 1.0

    def generate(
        self,
        stt_tstmp,
        end_tstmp,
        res_us=1000,
    ) -> dict[RadarStationCompositeKey, schr.Schedule]:
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

        # TODO: add conflict checks between schedule for passes?
        # TODO: add checks for lowest  time_slice (or schedule row) >= res_us
        stt_tstmp_arr = self.pass_obj.t[self.pass_obj.t < np.datetime64(end_tstmp, "us")]
        sch_total_rows = stt_tstmp_arr.shape[0]

        sch_dict: dict[RadarStationCompositeKey, schr.Schedule] = {}
        for idx, station_key in enumerate(self.pass_obj.radar_station_composite_keys):
            azelr_arr = pyant.coordinates.cart_to_sph(self.pass_obj.enu[idx])
            sch = schr.Schedule(
                stt_tstmp_us=stt_tstmp_arr,
                exp_num=np.full(sch_total_rows, self.exp_num),
                pointing_az=azelr_arr[0],
                pointing_el=azelr_arr[1],
                coh_int_bandwidth=np.full(sch_total_rows, self.coh_int_bandwidth),
                ipp=np.full(sch_total_rows, self.ipp),
                pulse_length=np.full(sch_total_rows, self.pulse_length),
            )

            sch_dict[station_key] = sch

        return sch_dict


__all__ = ["TrackerController"]
