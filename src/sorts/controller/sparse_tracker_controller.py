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
    Float64_as_sec,
    EnuCoordinates,
    Datetime_Like,
    Timedelta_Like,
)
from sorts.simulation.funcs import find_simultaneous_passages
from .controller_base import ControllerBase

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class FromSpaceObjectParam:
    tx_station: Station
    rx_stations: t.Sequence[Station]
    exp_detail: scheduling.ExperimentDetail
    space_object: SpaceObject
    epoch: Datetime_Like
    points_per_passage: int


@dataclass(kw_only=True)
class ControllerState:
    spobj_time: npt.NDArray[Datetime64_us]
    spobj_states: EcefStates

    @classmethod
    def empty(cls) -> t.Self:
        return cls(
            spobj_time=np.empty(0, dtype="datetime64[us]"),
            spobj_states=np.empty((6, 0), dtype=np.float64),
        )


class SparseTrackerController(ControllerBase):
    """
    Generate pointing schedule that tracks a space object.

    - The preferred way to create instances of this class is via its class methods (e.g. `TrackerController.from_space_object`).
    - This class serve as a frontend to the `State` type in this module
    """

    FromSpaceObjectParam = FromSpaceObjectParam
    """shortcut to module attribute"""

    ControllerState = ControllerState
    """shortcut to module attribute"""

    def __init__(
        self,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        exp_detail: scheduling.ExperimentDetail,
        space_object: SpaceObject,
        epoch: Datetime64_us,
        station_id_pairs: list[tuple[radar.StationId, radar.StationId]],
        points_per_passage: int,
        state: ControllerState,
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

        self.state = state

    @classmethod
    def from_space_object(cls, param: FromSpaceObjectParam) -> t.Self:
        """A constructor method"""

        stn_pairs = [(param.tx_station.uid, rx_station.uid) for rx_station in param.rx_stations]

        ctrl = cls(
            tx_station=param.tx_station,
            rx_stations=param.rx_stations,
            exp_detail=param.exp_detail,
            space_object=param.space_object,
            epoch=to_datetime64_us(param.epoch),
            station_id_pairs=stn_pairs,
            points_per_passage=param.points_per_passage,
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
        self, start_time: Datetime_Like, end_time: Datetime_Like, slice_duration: Timedelta_Like
    ) -> ControllerState:
        """Do the computation and return the updated `state` property."""

        exp_detail: scheduling.ExperimentDetail = self.exp_detail

        # NOTE: for `np.arange` 'stop param,
        #   - we subtract 'slice_duration' so that only full slice are included
        #   - and add `+1` so that slice with time range `('end_time - 'slice_duration', 'end_time')` is included
        # TODO: use sampler method for this to ensure efficient propagator usage which has already
        # been defined for the other steps of the simulation
        time: npt.NDArray[Datetime64_us] = np.arange(
            to_datetime64_us(start_time),
            to_datetime64_us(end_time) - to_timedelta64_us(slice_duration) + 1,
            exp_detail.slice_duration,
        )
        dt: npt.NDArray[Timedelta64_us] = time - to_datetime64_us(self.epoch)
        dsec = t.cast(npt.NDArray[Float64_as_sec], dt.astype(np.float64) / 1e6)

        ecefs = self.space_object.get_state(dsec)

        state = ControllerState(spobj_time=time, spobj_states=ecefs)

        return state

    def _generate(self, start_time: Datetime_Like, end_time: Datetime_Like) -> scheduling.Schedule:
        """Generate the schedules."""

        passages_of_spobj = find_simultaneous_passages(
            dt=(self.state.spobj_time - self.epoch) / np.timedelta64(1, "s"),
            space_object=self.space_object,
            states=self.state.spobj_states[:3, ...],
            tx_station=self.tx_station,
            rx_stations=self.rx_stations,
            epoch=self.epoch,
        )

        # early return for empty case
        if len(passages_of_spobj) == 0:
            return scheduling.empty()

        tx_sch_index_list = []
        for ps in passages_of_spobj:
            start_time, end_time = ps.time_range
            passage_time = (end_time - start_time) / np.timedelta64(1, "s")
            relative_time_sampling = np.linspace(
                0.0, passage_time, num=self.points_per_passage + 2, endpoint=True
            )
            relative_time_sampling = relative_time_sampling[1:-1]
            # TODO: once the propagator sampling has been changed, use a interpolator here instead
            # at the cadence that the propagator currently uses
            pass_tx_index = np.empty((self.points_per_passage,), dtype=np.int64)
            for ind in range(self.points_per_passage):
                pass_tx_index[ind] = np.argmin(
                    np.abs(
                        (
                            relative_time_sampling[ind] * np.timedelta64(1, "s")
                            + start_time
                            - self.state.spobj_time
                        )
                        / np.timedelta64(1, "s")
                    )
                )
            tx_sch_index_list.append(pass_tx_index)

        tx_sch_index = np.concatenate(tx_sch_index_list)
        tx_sch_time = self.state.spobj_time[tx_sch_index]
        tx_sch_len = len(tx_sch_time)
        tx_pointings: EnuCoordinates = self.tx_station.enu(
            self.state.spobj_states[:3, tx_sch_index]
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
            rx_pointings: EnuCoordinates = rx_stn.enu(self.state.spobj_states[:3, tx_sch_index])
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

    def generate(self, start_time: Datetime_Like, end_time: Datetime_Like) -> scheduling.Schedule:
        """Generate the schedules."""

        self.state = self._compute_controller_state(
            start_time, end_time, self.exp_detail.slice_duration
        )
        sch = self._generate(start_time, end_time)

        return sch
