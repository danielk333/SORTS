from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
import spacecoords
from sorts import radar, scheduling
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.types import Datetime64_us, EnuCoordinates, Datetime_Like
from sorts.utils import to_datetime64_us
from .controller_base import ControllerBase
from sorts.simulation.funcs import InterpolatedPropagation

logger = logging.getLogger(__name__)


class TrackerController(ControllerBase):
    """
    Generate pointing schedule that tracks a space object.

    - The preferred way to create instances of this class is via its class methods (e.g. `TrackerController.from_space_object`).
    """

    def __init__(
        self,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        exp_detail: scheduling.ExperimentDetail,
        space_object: SpaceObject,
        epoch: Datetime64_us,
        station_id_pairs: list[tuple[radar.StationId, radar.StationId]],
        interpolated_propagation: InterpolatedPropagation,
    ):
        """
        NOTE: This is intended as an internal constructor, please use the constructor methods to create instances.
        """

        self.tx_station = tx_station
        self.rx_stations = rx_stations
        self.exp_detail = exp_detail
        self.space_object = space_object
        self.epoch = epoch
        self.station_id_pairs = station_id_pairs
        self.interpolated_propagation = interpolated_propagation

    @classmethod
    def from_space_object(
        cls,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        exp_detail: scheduling.ExperimentDetail,
        space_object: SpaceObject,
        epoch: Datetime_Like,
        interpolated_propagation: InterpolatedPropagation,
    ) -> t.Self:
        """A constructor method"""

        stn_pairs = [(tx_station.uid, rx_station.uid) for rx_station in rx_stations]

        ctrl = cls(
            tx_station=tx_station,
            rx_stations=rx_stations,
            exp_detail=exp_detail,
            space_object=space_object,
            epoch=to_datetime64_us(epoch),
            station_id_pairs=stn_pairs,
            interpolated_propagation=interpolated_propagation,
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

    # TODO: `start_time` and `end_time` are not used atm, remove or adj the logic
    def generate(
        self, start_time: Datetime_Like, end_time: Datetime_Like
    ) -> scheduling.ScheduleDataframe:
        """Generate the schedules."""

        loc_zenith = np.array([0, 0, 1], dtype=np.float64)

        # generate pointings
        tx_pointings: EnuCoordinates = self.tx_station.enu(self.interpolated_propagation.states[:3])

        tx_pointings_zenith_ang = spacecoords.linalg.vector_angle(
            loc_zenith, tx_pointings, degrees=True
        )
        tx_el_in_range_mask = tx_pointings_zenith_ang <= 90.0 - self.tx_station.min_elevation
        tx_pointings = tx_pointings[:, tx_el_in_range_mask]

        rxs_pointings: list[EnuCoordinates] = []
        rx_el_in_range_with_tx_masks: list[npt.NDArray[np.bool]] = []
        pure_rx_stations = [stn for stn in self.rx_stations if stn.uid != self.tx_station.uid]
        for rx_station in pure_rx_stations:
            rx_pointings: EnuCoordinates = rx_station.enu(self.interpolated_propagation.states[:3])

            rx_pointings_zenith_ang = spacecoords.linalg.vector_angle(
                loc_zenith, rx_pointings, degrees=True
            )
            rx_el_in_range_mask = rx_pointings_zenith_ang <= 90.0 - rx_station.min_elevation

            rx_el_in_range_with_tx_mask = np.logical_and(tx_el_in_range_mask, rx_el_in_range_mask)
            rx_el_in_range_with_tx_masks.append(rx_el_in_range_with_tx_mask)

            rx_pointings = rx_pointings[:, rx_el_in_range_with_tx_mask]
            rxs_pointings.append(rx_pointings)

        tx_sch_time = self.interpolated_propagation.times[tx_el_in_range_mask]
        tx_sch_len = len(tx_sch_time)

        tx_sch = scheduling.schedule_dataframe_from_ndarrays(
            exp_num=np.full(tx_sch_len, self.exp_detail.id, dtype=np.int16),
            stn_num=np.full(tx_sch_len, self.tx_station.uid, dtype=np.int16),
            simult_num=np.full(tx_sch_len, 0, dtype=np.int16),
            start_time=tx_sch_time,
            end_time=tx_sch_time + self.exp_detail.slice_duration,
            pointing_e=tx_pointings[0, :],
            pointing_n=tx_pointings[1, :],
            pointing_u=tx_pointings[2, :],
        )

        rx_schs: list[scheduling.ScheduleDataframe] = []
        for rx_stn, rx_mask, rx_pointings in zip(
            pure_rx_stations, rx_el_in_range_with_tx_masks, rxs_pointings
        ):
            rx_sch_time = self.interpolated_propagation.times[rx_mask]
            rx_sch_len = len(rx_sch_time)

            rx_schs.append(
                scheduling.schedule_dataframe_from_ndarrays(
                    exp_num=np.full(rx_sch_len, self.exp_detail.id, dtype=np.int16),
                    stn_num=np.full(rx_sch_len, rx_stn.uid, dtype=np.int16),
                    simult_num=np.full(rx_sch_len, 0, dtype=np.int16),
                    start_time=rx_sch_time,
                    end_time=rx_sch_time + self.exp_detail.slice_duration,
                    pointing_e=rx_pointings[0, :],
                    pointing_n=rx_pointings[1, :],
                    pointing_u=rx_pointings[2, :],
                )
            )

        resultant_sch = scheduling.ScheduleDataframe((pd.concat([tx_sch, *rx_schs])))
        resultant_sch = resultant_sch.sort_values(by=scheduling.ScheduleKey.start_time)

        return resultant_sch
