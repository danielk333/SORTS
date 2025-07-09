import logging, math
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sorts.radar.tx_rx import Station
from sorts.schedule_v2 import Schedule
from sorts.controller_v2.controller_protocol import ControllerProtocol
from sorts.controller_v2 import pointing_patterns

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class FenceScanController(ControllerProtocol):
    """
    NOTE: `num` refers to the number of pointings within a cycle.

    TODO: make it support multi-rx, by taking a list of rx station
    TODO: should take radar/station `azimuth_deg`, `elevation_deg` limitation into account?
    """

    tx_station: Station  # TODO: not used; remove?
    rx_station: Station  # TODO: not used; remove?

    azimuth_deg: float
    min_elevation_deg: float
    dwell_s: float
    num: int
    start_time: datetime
    end_time: datetime

    exp_num: int = 0

    def __post_init__(self):
        self._total_duration_s = (self.end_time - self.start_time).total_seconds()
        if self._total_duration_s < self.dwell_s:
            raise RuntimeError(
                f"The specified time range ({self.start_time.isoformat()} to {self.end_time.isoformat()}) "
                + f"cannot be smaller than the dwell ({self.dwell_s} sec)."
            )

        # TODO: we shouldn't need to pregenerate the time arr (and pointing arrays?),
        #   they can be computed when generate method is called
        self.start_time_us_arr = np.arange(
            self.start_time, self.end_time, np.timedelta64(math.floor(self.dwell_s * 1e6), "us")
        )
        self.end_time_us_arr = self.start_time_us_arr + np.timedelta64(
            math.floor(self.dwell_s * 1e6), "us"
        )

        self._tx_pts_within_a_cycle = pointing_patterns.fence_pointing(
            azimuth=self.azimuth_deg,
            min_elevation=self.min_elevation_deg,
            pointings_per_cycle=self.num,
        )

        # TODO: this is a shortcut for tx rx very close togther
        #   for generic cases, need to clarify the math in
        #   `src/sorts/controller/scanner.py`
        self._rx_pts_within_a_cycle = self._tx_pts_within_a_cycle.copy()

    def get_pointing_idx_within_a_cycle(self, t: datetime):
        idx = (
            np.mod((t - self.start_time).total_seconds() / (self.num * self.dwell_s), 1) * self.num
        ).astype(np.int64)
        return idx

    def generate(
        self,
        stt_tstmp: datetime,
        end_tstmp: datetime,
        res_us=1000,
    ) -> tuple[Schedule, Schedule]:
        """Returns `(tx_schedule, rx_schedule)`"""

        time_range_mask = (self.start_time_us_arr >= np.datetime64(stt_tstmp)) & (
            self.end_time_us_arr <= np.datetime64(end_tstmp)
        )
        schedule_size = np.count_nonzero(time_range_mask)

        tx_pointing = np.tile(
            self._tx_pts_within_a_cycle, (schedule_size + self.num - 1) // self.num
        )[:schedule_size]
        rx_pointing = np.tile(
            self._rx_pts_within_a_cycle, (schedule_size + self.num - 1) // self.num
        )[:schedule_size]

        tx_schedule = Schedule(
            meta={},  # TODO: replace this dummy with actual implementation
            stt_tstmp_us=self.start_time_us_arr[time_range_mask],
            exp_num=np.full(schedule_size, self.exp_num, dtype=np.int64),
            pointing_az=tx_pointing[0],
            pointing_el=tx_pointing[1],
        )

        rx_schedule = Schedule(
            meta={},  # TODO: replace this dummy with actual implementation
            stt_tstmp_us=self.start_time_us_arr[time_range_mask],
            exp_num=np.full(schedule_size, self.exp_num, dtype=np.int64),
            pointing_az=rx_pointing[0],
            pointing_el=rx_pointing[1],
        )

        return (tx_schedule, rx_schedule)
