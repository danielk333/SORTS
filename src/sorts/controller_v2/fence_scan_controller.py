import logging, math, typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sorts.radar.tx_rx import Station
from sorts.frames import azel_to_ecef, ecef_to_enu, cart_to_sph
from sorts.types import (
    Float_as_deg,
    AzelrCoordinates_DegM,
    Float64_as_m,
    EcefCoordinates,
    EnuCoordinates,
    Datetime_like,
)
from sorts.utils import to_datetime64_us, wrap_azimuths_elevations
from sorts.schedule_v2 import Schedule, ExperimentDetail
from sorts.controller_v2 import pointing_patterns

logger = logging.getLogger(__name__)


class FenceScanControllerOutput(t.NamedTuple):
    tx_schedule: Schedule
    rx_schedules: t.Sequence[Schedule]


@dataclass(kw_only=True)
class FenceScanController:
    """
    Generate schedule for a fence scaning pattern
    """

    # TODO: the radar station computation capacity poses limit on the size of simutaneous `scan_range`, we should check/validate against it
    # TODO: should take radar/station `azimuth_deg`, `elevation_deg` limitation into account?
    # TODO: this is WIP

    tx_station: Station
    rx_stations: t.Sequence[Station]
    exp_datail: ExperimentDetail

    azimuth: Float_as_deg
    min_elevation: Float_as_deg
    pointings_per_cycle: int
    scan_range: npt.NDArray[Float64_as_m]

    def __post_init__(self):
        # TODO: update/adapt or remove?
        # self._total_duration_s = (self.end_time - self.start_time).total_seconds()
        # if self._total_duration_s < self.dwell_s:
        #     raise RuntimeError(
        #         f"The specified time range ({self.start_time.isoformat()} to {self.end_time.isoformat()}) "
        #         + f"cannot be smaller than the dwell ({self.dwell_s} sec)."
        #     )

        if len(self.scan_range) > 1:
            raise NotImplementedError(
                "Support for multiple pointings per control slice is not implemented yet."
            )
        pass

    def generate(
        self,
        start_time: Datetime_like,
        end_time: Datetime_like,
        scan_range: npt.NDArray[Float64_as_m] | None = None,
    ) -> FenceScanControllerOutput:

        # The logic of this function:
        # 0. the pointings are repetitive so we will generate one cycle of them and then repeat the cycle
        # 1. generate the a cycle of pointings of tx station
        # 2. repeat it to form the tx schedule
        # 3. from the single cycle of tx pointings, we convert it into ECEF location coord and extend them by the `scan_range`
        # 4. using the resultant location coords from previous step,
        #    we convert them to rx station pointings of a cycle in ECEF coord,
        #    and then further back to pointings in AzEl coord,
        #    and finally repeat them to form a rx schedule, for each rx station

        start_time_np = to_datetime64_us(start_time)
        end_time_np = to_datetime64_us(end_time)
        scan_range = scan_range if scan_range is not None else self.scan_range

        tx_slice_start_time = np.arange(start_time_np, end_time_np, self.exp_datail.slice_duration)
        tx_schedule_size = math.floor(
            (end_time_np - start_time_np) / self.exp_datail.slice_duration
        )

        tx_pointings_of_a_cycle = pointing_patterns.fence_pointing(
            azimuth=self.azimuth,
            min_elevation=self.min_elevation,
            pointings_per_cycle=self.pointings_per_cycle,
        )

        # repeat a cycle of pointings until it is at least the size of `tx_schedule_size`
        # then trim to exactly `tx_schedule_size` long
        tx_pointing: AzelrCoordinates_DegM = np.tile(
            tx_pointings_of_a_cycle,
            (tx_schedule_size + self.pointings_per_cycle - 1) // self.pointings_per_cycle,
        )[:, :tx_schedule_size]

        tx_schedule = Schedule(
            meta={self.exp_datail.id: self.exp_datail},
            start_time=tx_slice_start_time,
            exp_num=np.full(tx_schedule_size, self.exp_datail.id, dtype=np.int64),
            pointing_az=tx_pointing[0],
            pointing_el=tx_pointing[1],
        )

        rx_slice_start_time = tx_slice_start_time.repeat(len(scan_range))
        rx_schedule_size = tx_schedule_size * len(scan_range)
        rx_schedules: list[Schedule] = []
        tx_pointings_of_a_cycle_ecef: EcefCoordinates = azel_to_ecef(
            lat=self.tx_station.ecef_lat,
            lon=self.tx_station.ecef_lon,
            alt=self.tx_station.ecef_alt,
            az=tx_pointings_of_a_cycle[0],
            el=tx_pointings_of_a_cycle[1],
            degrees=True,
        )
        rx_pointing_loc_of_a_cycle_ecef: EcefCoordinates = (
            tx_pointings_of_a_cycle_ecef[:, :, np.newaxis] * scan_range[np.newaxis, np.newaxis, :]
            + self.tx_station.ecef[:, np.newaxis, np.newaxis]
        ).reshape((3, -1))

        for rx_station in self.rx_stations:
            rx_pointings_of_a_cycle_ecef: EcefCoordinates = (
                rx_pointing_loc_of_a_cycle_ecef - rx_station.ecef[:, np.newaxis]
            )
            rx_pointings_of_a_cycle_enu: EnuCoordinates = ecef_to_enu(
                lat=rx_station.ecef_lat,
                lon=rx_station.ecef_lon,
                alt=rx_station.ecef_alt,
                ecef=rx_pointings_of_a_cycle_ecef,
                degrees=True,
            )
            # TODO: `cart_to_sph` returns el in [-90, 90]. update `wrap_azimuths_elevations` to handle -ve el (by e.g. `el % 180`)?
            # TODO: update `wrap_azimuths_elevations` output a single ndarray of (3,n) ?
            rx_pointings_of_a_cycle: AzelrCoordinates_DegM = cart_to_sph(
                rx_pointings_of_a_cycle_enu, degrees=True
            )
            rx_pointings_of_a_cycle[0], rx_pointings_of_a_cycle[1] = wrap_azimuths_elevations(
                rx_pointings_of_a_cycle[0], rx_pointings_of_a_cycle[1]
            )

            # repeat a cycle of pointings until it is at least the size of `rx_schedule_size`
            # then trim to exactly `rx_schedule_size` long
            rx_pointing: AzelrCoordinates_DegM = np.tile(
                rx_pointings_of_a_cycle,
                (rx_schedule_size + self.pointings_per_cycle - 1) // self.pointings_per_cycle,
            )[:, :rx_schedule_size]

            rx_schedule = Schedule(
                meta={self.exp_datail.id: self.exp_datail},
                start_time=rx_slice_start_time,
                exp_num=np.full(rx_schedule_size, self.exp_datail.id, dtype=np.int64),
                pointing_az=rx_pointing[0],
                pointing_el=rx_pointing[1],
            )

            rx_schedules.append(rx_schedule)

        return FenceScanControllerOutput(tx_schedule, rx_schedules)
