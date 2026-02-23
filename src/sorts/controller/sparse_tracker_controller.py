from __future__ import annotations
import logging, typing as t
import numpy as np
import pandas as pd
from sorts import types, radar, schedule
from sorts.utils import to_datetime64_us
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.types import (
    Datetime64_us,
    EnuCoordinates,
    Datetime_Like,
)
from sorts.simulation.types import Passage
from .controller_base import ControllerBase
from sorts.interpolation import Interpolator

logger = logging.getLogger(__name__)


class SparseTrackerController(ControllerBase):
    """
    Generate pointing schedule that tracks a space object.

    - The preferred way to create instances of this class is via its class methods (e.g. `TrackerController.from_space_object`).
    """

    def __init__(
        self,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        exp_detail: types.ExperimentDetail,
        space_object: SpaceObject,
        epoch: Datetime64_us,
        station_id_pairs: list[tuple[radar.StationId, radar.StationId]],
        points_per_passage: int,
        interpolator: Interpolator,
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
        self.points_per_passage = points_per_passage

        self.interpolator = interpolator

    @classmethod
    def from_space_object(
        cls,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        exp_detail: types.ExperimentDetail,
        space_object: SpaceObject,
        epoch: Datetime_Like,
        points_per_passage: int,
        interpolator: Interpolator,
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
            points_per_passage=points_per_passage,
            interpolator=interpolator,
        )

        return ctrl

    def get_experiment_detail(self) -> types.ExperimentDetail:
        return self.exp_detail

    def get_experiment_id_station_id_pairs_map(self) -> types.ExperimentIdStationIdPairsMap:
        return {self.exp_detail.id: self.station_id_pairs}

    def get_station_map(self) -> dict[radar.StationId, radar.Station]:
        stn_map: dict[radar.StationId, radar.Station] = {}

        stn_map[self.tx_station.uid] = self.tx_station
        stn_map.update(list([(stn.uid, stn) for stn in self.rx_stations]))

        return stn_map

    def generate(self, passages_of_spobj: list[Passage]) -> schedule.ScheduleDataframe:
        """Generate the schedules."""
        # early return for empty case
        if len(passages_of_spobj) == 0:
            return schedule.schedule_dataframe.empty()

        observation_times_relative = []
        observation_times = []
        min_time_needed = (
            self.points_per_passage * self.exp_detail.slice_duration / np.timedelta64(1, "s")
        )
        for ps in passages_of_spobj:
            pstart_time, pend_time = ps.time_range
            t0 = (pstart_time - ps.epoch) / np.timedelta64(1, "s")
            passage_time = (pend_time - pstart_time) / np.timedelta64(1, "s")
            if passage_time <= min_time_needed:
                continue

            relative_time_sampling = np.linspace(
                0.0, passage_time, num=self.points_per_passage + 2, endpoint=True
            )
            relative_time_sampling = relative_time_sampling[1:-1]

            observation_times_relative.append(t0 + relative_time_sampling)
            rel_us = (relative_time_sampling.copy() * 1e6).astype("timedelta64[us]")

            observation_times.append(pstart_time + rel_us)

        # TODO: this can be cleaned up quite a lot i feel like
        tx_sch_time_rel = np.concatenate(observation_times_relative)
        tx_sch_time = np.concatenate(observation_times)
        tx_sch_len = len(tx_sch_time)
        tx_pointings: EnuCoordinates = self.tx_station.enu(
            self.interpolator.get_state(tx_sch_time_rel)[:3, :]
        )
        tx_pointings = tx_pointings / np.linalg.norm(tx_pointings, axis=0)

        tx_sch = schedule.schedule_dataframe.from_ndarrays(
            exp_num=np.full(tx_sch_len, self.exp_detail.id, dtype=np.int16),
            stn_num=np.full(tx_sch_len, self.tx_station.uid, dtype=np.int16),
            simult_num=np.full(tx_sch_len, 0, dtype=np.int16),
            start_time=tx_sch_time,
            end_time=tx_sch_time + self.exp_detail.slice_duration,
            pointing_e=tx_pointings[0, :],
            pointing_n=tx_pointings[1, :],
            pointing_u=tx_pointings[2, :],
        )

        rx_schs: list[schedule.ScheduleDataframe] = []
        for rx_stn in self.rx_stations:
            rx_pointings: EnuCoordinates = rx_stn.enu(
                self.interpolator.get_state(tx_sch_time_rel)[:3, :]
            )
            rx_pointings = rx_pointings / np.linalg.norm(rx_pointings, axis=0)

            rx_schs.append(
                schedule.schedule_dataframe.from_ndarrays(
                    exp_num=np.full(tx_sch_len, self.exp_detail.id, dtype=np.int16),
                    stn_num=np.full(tx_sch_len, rx_stn.uid, dtype=np.int16),
                    simult_num=np.full(tx_sch_len, 0, dtype=np.int16),
                    start_time=tx_sch_time,
                    end_time=tx_sch_time + self.exp_detail.slice_duration,
                    pointing_e=rx_pointings[0, :],
                    pointing_n=rx_pointings[1, :],
                    pointing_u=rx_pointings[2, :],
                )
            )

        resultant_sch = schedule.ScheduleDataframe((pd.concat([tx_sch, *rx_schs])))
        resultant_sch = resultant_sch.sort_values(by=schedule.ScheduleKey.start_time)

        return resultant_sch
