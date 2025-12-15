from __future__ import annotations
import logging, typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import xarray as xr
from sorts import radar, scheduling
from sorts.utils import to_datetime64_us, to_timedelta64_us
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.types import (
    EcefStates,
    Datetime64_us,
    Timedelta64_us,
    EnuCoordinates,
    Datetime_Like,
)
from sorts.simulation.types import Passage
from .controller_base import ControllerBase
from sorts.interpolation import Interpolator

logger = logging.getLogger(__name__)


# TODO: this is probably a very general thing that can be just a
# standard type in `types`, we could then pass these around to e.g. the
# interpolator
# @dataclass
# class ControllerState:
#     spobj_time: npt.NDArray[Datetime64_us]
#     spobj_states: EcefStates
#
#     @classmethod
#     def empty(cls) -> t.Self:
#         return cls(
#             spobj_time=np.empty(0, dtype="datetime64[us]"),
#             spobj_states=np.empty((6, 0), dtype=np.float64),
#         )


class SparseTrackerController(ControllerBase):
    """
    Generate pointing schedule that tracks a space object.

    - The preferred way to create instances of this class is via its class methods (e.g. `TrackerController.from_space_object`).
    - This class serve as a frontend to the `State` type in this module
    """

    def __init__(
        self,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        exp_detail: scheduling.ExperimentDetail,
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
        exp_detail: scheduling.ExperimentDetail,
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

    def get_experiment_detail(self) -> scheduling.ExperimentDetail:
        return self.exp_detail

    def get_experiment_id_station_id_pairs_map(self) -> scheduling.ExperimentIdStationIdPairsMap:
        return {self.exp_detail.id: self.station_id_pairs}

    def get_station_map(self) -> dict[radar.StationId, radar.Station]:
        stn_map: dict[radar.StationId, radar.Station] = {}

        stn_map[self.tx_station.uid] = self.tx_station
        stn_map.update(list([(stn.uid, stn) for stn in self.rx_stations]))

        return stn_map

    def generate(self, passages_of_spobj: list[Passage]) -> scheduling.Schedule:
        """Generate the schedules."""
        # early return for empty case
        if len(passages_of_spobj) == 0:
            return scheduling.empty()

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

        # todo: this can be cleaned up quite a lot i feel like
        tx_sch_time_rel = np.concatenate(observation_times_relative)
        tx_sch_time = np.concatenate(observation_times)
        tx_sch_len = len(tx_sch_time)
        tx_pointings: EnuCoordinates = self.tx_station.enu(
            self.interpolator.get_state(tx_sch_time_rel)[:3, :]
        )
        tx_pointings = tx_pointings / np.linalg.norm(tx_pointings, axis=0)

        tx_sch = scheduling.from_ndarrays(
            start_time=tx_sch_time,
            end_time=tx_sch_time + self.exp_detail.slice_duration,
            exp_num=np.full(tx_sch_len, self.exp_detail.id, dtype=np.int16),
            stn_num=np.full(tx_sch_len, self.tx_station.uid, dtype=np.int16),
            simult_num=np.full(tx_sch_len, 0, dtype=np.int16),
            pointing=tx_pointings,
        )

        rx_schs: list[scheduling.Schedule] = []
        for rx_stn in self.rx_stations:
            rx_pointings: EnuCoordinates = rx_stn.enu(
                self.interpolator.get_state(tx_sch_time_rel)[:3, :]
            )
            rx_pointings = rx_pointings / np.linalg.norm(rx_pointings, axis=0)

            rx_schs.append(
                scheduling.from_ndarrays(
                    start_time=tx_sch_time,
                    end_time=tx_sch_time + self.exp_detail.slice_duration,
                    exp_num=np.full(tx_sch_len, self.exp_detail.id, dtype=np.int16),
                    stn_num=np.full(tx_sch_len, rx_stn.uid, dtype=np.int16),
                    simult_num=np.full(tx_sch_len, 0, dtype=np.int16),
                    pointing=rx_pointings,
                )
            )

        resultant_sch = xr.concat([tx_sch, *rx_schs], dim=scheduling._K.multi_index)
        resultant_sch = resultant_sch.sortby(scheduling._K.start_time)
        output = resultant_sch

        return output
