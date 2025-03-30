import logging, typing as t, math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import numpy as np
import numpy.typing as npt
from .. import scheduler_v2 as schr
from .. import controller_v2 as ctrlr
from .. import passes

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TrackerController(ctrlr.ControllerProtocol):
    """
    a controller for tracking `Pass` objects
    """

    passes: list[passes.Pass]

    exp_num: int = 0
    min_elevation_deg: float = 30.0
    time_slice_us: float = 1000  # 1ms

    coh_int_bandwidth: float = 1.0
    ipp: float = 1.0
    pulse_length: float = 1.0

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

        _azelr_of_passes = [enu_to_azelr(pass_obj.enu) for pass_obj in self.passes]
        azelr_of_passes: EnuToAzelrRet = {
            "az": np.concat([azelr_dict["az"] for azelr_dict in _azelr_of_passes]),
            "el": np.concat([azelr_dict["el"] for azelr_dict in _azelr_of_passes]),
            "r": np.concat([azelr_dict["r"] for azelr_dict in _azelr_of_passes]),
        }

        # TODO: add conflict checks between schedule for passes?
        sch_total_rows = azelr_of_passes["r"].size

        ret_sch = schr.Schedule(
            stt_tstmp_us=np.arange(
                stt_tstmp,
                end_tstmp,
                np.timedelta64((end_tstmp - stt_tstmp) / sch_total_rows),
                dtype="datetime64[us]",
            ),
            exp_num=np.full(sch_total_rows, self.exp_num),
            pointing_az=azelr_of_passes["az"],
            pointing_el=azelr_of_passes["el"],
            coh_int_bandwidth=np.full(sch_total_rows, self.coh_int_bandwidth),
            ipp=np.full(sch_total_rows, self.ipp),
            pulse_length=np.full(sch_total_rows, self.pulse_length),
        )

        logger.error(f"generate() is wip, dummy values will be returned")
        return ret_sch


EnuToAzelrRet = dict[t.Literal["az", "el", "r"], npt.NDArray[np.float64]]


def enu_to_azelr(enu: npt.NDArray[np.float64]) -> EnuToAzelrRet:
    """
    wip, probably similar to `enu_to_ecef()`?
    """

    az = np.random.uniform(size=100)
    el = np.random.uniform(size=100)
    r = np.random.uniform(low=160e3, high=2000e3, size=100)  # NOTE: a rough LEO range as an example

    ret: EnuToAzelrRet = {
        "az": az,
        "el": el,
        "r": r,
    }

    return ret
