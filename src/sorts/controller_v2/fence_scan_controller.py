import logging, math
from dataclasses import dataclass, fields
from datetime import datetime
import numpy as np
import numpy.typing as npt
from ..radar.radars.composite_key import RadarStationCompositeKey
from sorts.radar.tx_rx import Station
from sorts import scheduler_v2 as scheduler
from sorts import controller_v2 as controller
from sorts.controller_v2 import pointing_patterns

logger = logging.getLogger(__name__)


# TODO: remove 'scanner_controller.py'
# TODO: update `controller.ControllerProtocol` and inherit from it
@dataclass(kw_only=True)
class FenceScanController:
    """
    TODO: make it support multi-rx, by taking a list of rx station
    """

    tx_station: Station
    rx_station: Station

    azimuth_deg: float
    min_elevation_deg: float
    dwell_s: float
    num: int
    start_time: datetime
    end_time: datetime

    exp_num: int = 0

    def __post_init__(self):
        interval_s = (self.end_time - self.start_time).total_seconds() / self.num
        self.start_time_us_arr = np.arange(
            self.start_time, self.end_time, np.timedelta64(math.floor(interval_s * 1e6), "us")
        )
        self.end_time_us_arr = self.start_time_us_arr + np.timedelta64(
            math.floor(self.dwell_s * 1e6), "us"
        )

        self.tx_pointings = pointing_patterns.fence_pointing(
            azimuth_deg=self.azimuth_deg,
            min_elevation_deg=self.min_elevation_deg,
            dwell_s=self.dwell_s,
            num=self.num,
            start_time=self.start_time,
            end_time=self.end_time,
        )

        # TODO: this is a shortcut for tx rx very close togther
        #   for generic cases, need to clarify the math in
        #   `src/sorts/controller/scanner.py`
        self.rx_pointings = self.tx_pointings.copy()

    def generate(
        self,
        stt_tstmp,
        end_tstmp,
        res_us=1000,
    ) -> tuple[scheduler.Schedule, scheduler.Schedule]:
        """Returns `(tx_schedule, rx_schedule)`"""

        time_range_mask = (self.start_time_us_arr >= np.datetime64(stt_tstmp)) & (
            self.end_time_us_arr <= np.datetime64(end_tstmp)
        )
        schedule_size = np.count_nonzero(time_range_mask)

        tx_schedule = scheduler.Schedule(
            stt_tstmp_us=self.start_time_us_arr[time_range_mask],
            exp_num=np.full(schedule_size, self.exp_num, dtype=np.int64),
            pointing_az=self.tx_pointings[0],
            pointing_el=self.tx_pointings[1],
            coh_int_bandwidth=np.full(schedule_size, 1.0, dtype=np.float64),
            ipp=np.full(schedule_size, 1.0, dtype=np.float64),
            pulse_length=np.full(schedule_size, 1.0, dtype=np.float64),
        )

        rx_schedule = scheduler.Schedule(
            stt_tstmp_us=self.start_time_us_arr[time_range_mask],
            exp_num=np.full(schedule_size, self.exp_num, dtype=np.int64),
            pointing_az=self.rx_pointings[0],
            pointing_el=self.rx_pointings[1],
            coh_int_bandwidth=np.full(schedule_size, 1.0, dtype=np.float64),
            ipp=np.full(schedule_size, 1.0, dtype=np.float64),
            pulse_length=np.full(schedule_size, 1.0, dtype=np.float64),
        )

        return (tx_schedule, rx_schedule)


__all__ = ["FenceScanController"]
