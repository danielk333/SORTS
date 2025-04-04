import logging, typing as t, math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import numpy as np
import numpy.typing as npt
import pyant
from .. import scheduler_v2 as schr
from .. import controller_v2 as ctrlr
from .. import passes

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TrackerController(ctrlr.ControllerProtocol):
    """
    a controller for tracking `Pass` object
    """

    # TODO: add `is_radian` as `Pass` member field? default to `False`?
    pass_obj: passes.Pass

    exp_num: int = 0
    time_slice_us: float = 1000  # 1ms

    coh_int_bandwidth: float = 1.0
    ipp: float = 1.0
    pulse_length: float = 1.0

    def __post_init__(self):
        # TODO: move this check to `Pass` object?
        # ensure the list len across different fields are consistent if a list of enu are in the Pass object
        if isinstance(self.pass_obj.enu, list):
            if not (
                isinstance(self.pass_obj.station_id, list)
                and len(self.pass_obj.enu) == len(self.pass_obj.station_id)
            ):
                raise RuntimeError(
                    f"`length of `enu` and `station_id` have to be equal in a `Pass` object"
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

        # wrap `enu` ndarray and `station_id` in a list, if they are not a list already
        enu_arr_list: list[npt.NDArray[np.float64]] = (
            self.pass_obj.enu if isinstance(self.pass_obj.enu, list) else [self.pass_obj.enu]
        )
        station_id_list: list[int] = (
            self.pass_obj.station_id
            if isinstance(self.pass_obj.station_id, list)
            else [self.pass_obj.station_id]
        )

        # TODO: maybe use `broadcast_to()` to avoid duplicating elements?
        # compute stt_tstmp_arr by
        #   - adjust `Pass.t` using `stt_tstmp`
        #   - filtering out datetime >= end_tstmp
        #   - duplicate and concat itself to match number of stations
        stt_tstmp_arr = map_pass_t_to_tstmp_arr(self.pass_obj.t, stt_tstmp)
        stt_tstmp_arr = stt_tstmp_arr[stt_tstmp_arr < np.datetime64(end_tstmp, "us")]
        stt_tstmp_arr = np.concat([stt_tstmp_arr for _ in range(len(station_id_list))])

        azelr_arr: npt.NDArray[np.float64] = np.concat(
            [pyant.coordinates.cart_to_sph(enu_arr) for enu_arr in enu_arr_list],
            axis=1,
        )
        station_id_arr = np.concat(
            [
                np.full(enu_arr.shape[1], station_id_list[idx])
                for idx, enu_arr in enumerate(enu_arr_list)
            ]
        )

        # TODO: add conflict checks between schedule for passes?
        # TODO: add checks for lowest  time_slice (or schedule row) >= res_us
        sch_total_rows = azelr_arr.shape[1]

        ret_sch = schr.Schedule(
            stt_tstmp_us=stt_tstmp_arr,
            exp_num=np.full(sch_total_rows, self.exp_num),
            station_id=station_id_arr,
            pointing_az=azelr_arr[0],
            pointing_el=azelr_arr[1],
            coh_int_bandwidth=np.full(sch_total_rows, self.coh_int_bandwidth),
            ipp=np.full(sch_total_rows, self.ipp),
            pulse_length=np.full(sch_total_rows, self.pulse_length),
        )

        logger.error(f"generate() is wip, dummy values will be returned")
        return ret_sch


def map_pass_t_to_tstmp_arr(t_arr: npt.NDArray[np.float64], stt_tstmp: datetime):
    """
    instance member `Pass.t` starts from 0 and is based in seconds,
    this method covert it to `timedelta64[us]` used in scheduler

    returns a new ndarray with modified values


    WIP; likely need to adjust `epoch` param of the orbit based on `stt_tstmp`
    """

    # NOTE: equidistant_sampling returns npt.NDArray[np.float64]
    tstmp_arr = (t_arr.copy() * 1e6).astype("timedelta64[us]")
    tstmp_arr = tstmp_arr + np.datetime64(stt_tstmp, "us")

    return tstmp_arr
