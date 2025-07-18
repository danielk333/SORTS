import logging, math, typing as t
from dataclasses import dataclass
import numpy as np
from astropy.time import Time
from sorts.radar.tx_rx import Station
from sorts.types import Float_as_deg, AzelrCoordinates_DegM
from sorts.utils import astropy_time_to_datetime64_us
from sorts.schedule_v2 import Schedule, ExperimentDetail
from sorts.controller_v2 import pointing_patterns

logger = logging.getLogger(__name__)


class FenceScanControllerOutput(t.NamedTuple):
    tx_schedule: Schedule
    rx_schedules: t.Sequence[Schedule]


@dataclass(kw_only=True)
class FenceScanController:
    """
    TODO: should take radar/station `azimuth_deg`, `elevation_deg` limitation into account?
    TODO: this is WIP
    """

    tx_station: Station
    rx_station: t.Sequence[Station]
    exp_datail: ExperimentDetail

    azimuth: Float_as_deg
    min_elevation: Float_as_deg
    pointings_per_cycle: int

    def __post_init__(self):
        self._cached_output: FenceScanControllerOutput | None = None
        self._cached_tx_pointings_of_a_cycle: AzelrCoordinates_DegM | None = None
        self._cached_rx_pointings_of_a_cycle: AzelrCoordinates_DegM | None = None

        # TODO: update/adapt or remove?
        # self._total_duration_s = (self.end_time - self.start_time).total_seconds()
        # if self._total_duration_s < self.dwell_s:
        #     raise RuntimeError(
        #         f"The specified time range ({self.start_time.isoformat()} to {self.end_time.isoformat()}) "
        #         + f"cannot be smaller than the dwell ({self.dwell_s} sec)."
        #     )

    def generate(self, start_time: Time, end_time: Time) -> FenceScanControllerOutput:
        start_time_np = astropy_time_to_datetime64_us(start_time)
        end_time_np = astropy_time_to_datetime64_us(end_time)

        start_time_arr = np.arange(start_time_np, end_time_np, self.exp_datail.slice_duration)
        schedule_size = math.floor((end_time_np - start_time_np) / self.exp_datail.slice_duration)

        self._cached_tx_pointings_of_a_cycle = pointing_patterns.fence_pointing(
            azimuth=self.azimuth,
            min_elevation=self.min_elevation,
            pointings_per_cycle=self.pointings_per_cycle,
        )

        # TODO: this is a shortcut for tx rx very close togther
        #   for generic cases, need to clarify the math in
        #   `src/sorts/controller/scanner.py`
        self._cached_rx_pointings_of_a_cycle = self._cached_tx_pointings_of_a_cycle.copy()

        # repeat `self._cached_tx_pointings_of_a_cycle` until it is at least the size of `schedule_size`
        # then trim to exactly `schedule_size` long
        tx_pointing = np.tile(
            self._cached_tx_pointings_of_a_cycle,
            (schedule_size + self.pointings_per_cycle - 1) // self.pointings_per_cycle,
        )[:, :schedule_size]
        rx_pointing = np.tile(
            self._cached_rx_pointings_of_a_cycle,
            (schedule_size + self.pointings_per_cycle - 1) // self.pointings_per_cycle,
        )[:, :schedule_size]

        tx_schedule = Schedule(
            meta={self.exp_datail.id: self.exp_datail},
            start_time=start_time_arr,
            exp_num=np.full(schedule_size, self.exp_datail.id, dtype=np.int64),
            pointing_az=tx_pointing[0],
            pointing_el=tx_pointing[1],
        )

        rx_schedule = Schedule(
            meta={self.exp_datail.id: self.exp_datail},
            start_time=start_time_arr,
            exp_num=np.full(schedule_size, self.exp_datail.id, dtype=np.int64),
            pointing_az=rx_pointing[0],
            pointing_el=rx_pointing[1],
        )

        return FenceScanControllerOutput(tx_schedule, [rx_schedule])
