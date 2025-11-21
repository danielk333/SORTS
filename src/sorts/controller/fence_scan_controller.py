from __future__ import annotations
import logging, math, typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import xarray as xr
from sorts import radar, scheduling
from sorts.const import min_datetime64_us
from sorts.radar import Station
from sorts.frames import enu_to_ecef, ecef_to_enu, sph_to_cart
from sorts.types import (
    Float_as_deg,
    Datetime64_us,
    Float64_as_m,
    EcefCoordinates,
    EnuCoordinates,
    Datetime_Like,
)
from sorts.utils import to_datetime64_us
from . import pointing_funcs
from .controller_base import ControllerBase

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ControllerState:
    start_time: Datetime64_us
    end_time: Datetime64_us
    tx_schedule_size: int
    tx_pointings_of_a_cycle: EnuCoordinates
    """NOTE: It may contain out of range pointings"""

    @classmethod
    def empty(cls) -> t.Self:
        return cls(
            start_time=min_datetime64_us,
            end_time=min_datetime64_us,
            tx_schedule_size=0,
            tx_pointings_of_a_cycle=np.empty((3, 0), dtype=np.float64),
        )


class FenceScanController(ControllerBase):
    """
    Generate schedule for a fence scaning pattern

    - The preferred way to create instances of this class is via its class methods (e.g. `TrackerController.from_space_object`).
    - This class serve as a frontend to the `State` type in this module
    """

    ControllerState = ControllerState
    """shortcut to module attribute"""

    def __init__(
        self,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        azimuth: Float_as_deg,
        min_elevation: Float_as_deg,
        pointings_per_cycle: int,
        scan_range: npt.NDArray[Float64_as_m],
        exp_detail: scheduling.ExperimentDetail,
        station_id_pairs: list[tuple[radar.StationId, radar.StationId]],
        state: ControllerState,
    ):
        """
        NOTE: This is intended as an internal constructor, please use the constructor methods to create instances.
        """

        self.tx_station = tx_station
        self.rx_stations = rx_stations
        self.azimuth = azimuth
        self.min_elevation = min_elevation
        self.pointings_per_cycle = pointings_per_cycle
        # TODO: probably make a new scan controller that can do optimal scan range selection based
        # on angle between the tx and rx beam (as the angle goes to 0 the range step goes to
        # infinity because the important parameter is the range overlap of the rx beam on the tx
        # beam)
        self.scan_range = scan_range
        self.exp_detail = exp_detail
        self.station_id_pairs = station_id_pairs

        self.state: ControllerState = state

    @classmethod
    def from_scan_spec(
        cls,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        azimuth: Float_as_deg,
        min_elevation: Float_as_deg,
        pointings_per_cycle: int,
        scan_range: npt.NDArray[Float64_as_m],
        exp_detail: scheduling.ExperimentDetail,
    ) -> t.Self:
        """A constructor method"""

        stn_pairs = [(tx_station.uid, rx_station.uid) for rx_station in rx_stations]

        ctrl = cls(
            tx_station=tx_station,
            rx_stations=rx_stations,
            azimuth=azimuth,
            min_elevation=min_elevation,
            pointings_per_cycle=pointings_per_cycle,
            scan_range=scan_range,
            exp_detail=exp_detail,
            station_id_pairs=stn_pairs,
            state=ControllerState.empty(),
        )

        return ctrl

    def get_experiment_detail(self) -> scheduling.ExperimentDetail:
        return self.exp_detail

    def get_experiment_id_station_id_pairs_map(self) -> scheduling.ExperimentIdStationIdPairsMap:
        return {self.exp_detail.id: self.station_id_pairs}

    def get_station_map(self) -> dict[radar.StationId, radar.Station]:
        stn_map: dict[radar.StationId, radar.Station] = {}

        stn_map[self.tx_station.uid] = self.tx_station
        stn_map.update(list([(stn.uid, stn) for stn in self.rx_stations]))

        return stn_map

    def _compute_controller_state(
        self, start_time: Datetime_Like, end_time: Datetime_Like
    ) -> ControllerState:
        """Do the computation and return the updated `state` property."""

        exp_detail = self.exp_detail

        start_time_np = to_datetime64_us(start_time)
        end_time_np = to_datetime64_us(end_time)
        tx_schedule_size = math.floor((end_time_np - start_time_np) / exp_detail.slice_duration)

        tx_pointings_of_a_cycle = sph_to_cart(
            pointing_funcs.fence_pattern(
                azimuth=self.azimuth,
                min_elevation=self.min_elevation,
                pointings_per_cycle=self.pointings_per_cycle,
            ),
            degrees=True,
        )

        state = ControllerState(
            start_time=start_time_np,
            end_time=end_time_np,
            tx_schedule_size=tx_schedule_size,
            tx_pointings_of_a_cycle=tx_pointings_of_a_cycle,
        )

        return state

    def _generate(self) -> scheduling.Schedule:
        """Generate the schedules."""

        # TODO: write schedule to disk generally and then chunk load it as needed in the actual
        # simulation, the general simulation pattern will be "1. propagate objects and generate
        # states and make schedule, 2. run simulation, 3. analyze results"

        # The logic of this function:
        # 1. repeat the cycle of tx pointings from state to form the tx schedule
        # 2. from the single cycle of tx pointings, we convert it into ECEF location coord and extend them by the `scan_range`
        # 3. using the resultant location coords from previous step,
        #    we convert them to rx station pointings of a cycle in ECEF coord,
        #    and then further back to pointings in ENU coord,
        #    and finally repeat them to form a rx schedule, for each rx station

        pointings_per_cycle = self.state.tx_pointings_of_a_cycle.shape[1]

        # NOTE: for `np.arange` 'stop param,
        #   - we subtract 'slice_duration' so that only full slice are included
        #   - and add `+1` so that slice with time range `('end_time - 'slice_duration', 'end_time')` is included
        tx_slice_start_time: npt.NDArray[Datetime64_us] = np.arange(
            self.state.start_time,
            self.state.end_time - self.exp_detail.slice_duration + 1,
            self.exp_detail.slice_duration,
        )

        # TODO: `tx_schedule_size` is a bit of a mismisnomer, as out-of-range entries might later be removed
        # repeat a cycle of pointings until it is at least the size of `tx_schedule_size`,
        # then trim to exactly `tx_schedule_size` long
        tx_pointing: EnuCoordinates = np.tile(
            self.state.tx_pointings_of_a_cycle,
            (self.state.tx_schedule_size + pointings_per_cycle - 1) // pointings_per_cycle,
        )[:, : self.state.tx_schedule_size]

        # mask tx values by min_elevation requirement
        tx_mask = pointing_funcs.create_mask_by_min_elevation(
            tx_pointing, self.tx_station.min_elevation
        )
        tx_slice_start_time_masked = tx_slice_start_time[tx_mask]
        tx_pointing_masked = tx_pointing[:, tx_mask]

        tx_sch = scheduling.from_ndarrays(
            start_time=tx_slice_start_time_masked,
            end_time=tx_slice_start_time_masked + self.exp_detail.slice_duration,
            exp_num=np.full(len(tx_slice_start_time_masked), self.exp_detail.id, dtype=np.int16),
            stn_num=np.full(len(tx_slice_start_time_masked), self.tx_station.uid, dtype=np.int16),
            simult_num=np.full(len(tx_slice_start_time_masked), 0, dtype=np.int16),
            pointing=tx_pointing_masked,
        )

        # TODO: `rx_schedule_size` is a bit of a mismisnomer, as out-of-range entries might later be removed
        rx_slice_start_time = tx_slice_start_time.repeat(len(self.scan_range))
        rx_schedule_size = self.state.tx_schedule_size * len(self.scan_range)
        rx_schs: list[scheduling.Schedule] = []
        tx_pointings_of_a_cycle_without_translation_ecef: EcefCoordinates = enu_to_ecef(
            lat=self.tx_station.ecef_lat,
            lon=self.tx_station.ecef_lon,
            alt=self.tx_station.ecef_alt,
            enu=self.state.tx_pointings_of_a_cycle,
            degrees=True,
        )
        rx_pointing_of_a_cycle_ecef: EcefCoordinates = (
            tx_pointings_of_a_cycle_without_translation_ecef[:, :, np.newaxis]
            * self.scan_range[np.newaxis, np.newaxis, :]
            + self.tx_station.ecef[:, np.newaxis, np.newaxis]
        ).reshape((3, -1))

        for rx_station in self.rx_stations:
            rx_pointings_of_a_cycle_without_translation_ecef: EcefCoordinates = (
                rx_pointing_of_a_cycle_ecef - rx_station.ecef[:, np.newaxis]
            )
            rx_pointings_of_a_cycle_enu: EnuCoordinates = ecef_to_enu(
                lat=rx_station.ecef_lat,
                lon=rx_station.ecef_lon,
                alt=rx_station.ecef_alt,
                ecef=rx_pointings_of_a_cycle_without_translation_ecef,
                degrees=True,
            )

            # repeat a cycle of pointings until it is at least the size of `rx_schedule_size`
            # then trim to exactly `rx_schedule_size` long
            rx_pointings_enu: EnuCoordinates = np.tile(
                rx_pointings_of_a_cycle_enu,
                (rx_schedule_size + pointings_per_cycle - 1) // pointings_per_cycle,
            )[:, :rx_schedule_size]
            rx_pointings_simult_num = np.arange(rx_schedule_size) % len(self.scan_range)

            # mask rx values by min_elevation requirement, and has a corresponding tx value
            rx_mask_by_min_elevation = pointing_funcs.create_mask_by_min_elevation(
                rx_pointings_enu, rx_station.min_elevation
            )
            rx_mask_by_tx_mask = np.isin(rx_slice_start_time, tx_slice_start_time_masked)
            rx_mask = np.logical_and(rx_mask_by_min_elevation, rx_mask_by_tx_mask)
            rx_slice_start_time_masked = rx_slice_start_time[rx_mask]
            rx_pointing_masked = rx_pointings_enu[:, rx_mask]
            rx_pointings_simult_num_masked = rx_pointings_simult_num[rx_mask]

            rx_sch = scheduling.from_ndarrays(
                start_time=rx_slice_start_time_masked,
                end_time=rx_slice_start_time_masked + self.exp_detail.slice_duration,
                exp_num=np.full(
                    len(rx_slice_start_time_masked), self.exp_detail.id, dtype=np.int16
                ),
                stn_num=np.full(len(rx_slice_start_time_masked), rx_station.uid, dtype=np.int16),
                simult_num=rx_pointings_simult_num_masked,
                pointing=rx_pointing_masked,
            )

            rx_schs.append(rx_sch)

        resultant_sch = xr.concat([tx_sch, *rx_schs], dim=scheduling._K.multi_index)
        resultant_sch = resultant_sch.sortby(scheduling._K.start_time)
        # TODO: re-eval if it is too brutal
        # there will be duplicates if the tx station is also a rx station, we drop the duplicates here
        resultant_sch = resultant_sch.drop_duplicates(scheduling._K.multi_index)
        output = resultant_sch

        return output

    def generate(self, start_time: Datetime_Like, end_time: Datetime_Like) -> scheduling.Schedule:
        """Generate the schedules."""

        self.state = self._compute_controller_state(start_time, end_time)
        sch = self._generate()

        return sch
