from __future__ import annotations
import logging, typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import xarray as xr
from sorts import radar, schedule
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
from sorts.utils import to_datetime64_us, to_timedelta64_us
from .controller_base import ControllerBase
from sorts import simulation

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ControllerSpec:
    tx_station: Station
    rx_stations: t.Sequence[Station]
    exp_detail: schedule.ExperimentDetail
    space_object: SpaceObject
    epoch: Datetime_Like
    station_id_pairs: list[tuple[radar.StationId, radar.StationId]]
    points_per_passage: int


@dataclass(kw_only=True)
class ControllerState:
    spobj_time: npt.NDArray[Datetime64_us]
    spobj_states: EcefStates


def generate_from_state(spec: ControllerSpec, state: ControllerState) -> schedule.Schedule:
    passages_of_spobj = simulation.find_simultaneous_passages(
        dt=(state.spobj_time - spec.epoch) / np.timedelta64(1, "s"),
        space_object=spec.space_object,
        states=state.spobj_states[:3, ...],
        tx_station=spec.tx_station,
        rx_stations=spec.rx_stations,
        epoch=spec.epoch,
    )

    tx_sch_index_list = []
    for ps in passages_of_spobj:
        start_time, end_time = ps["time_range"]
        passage_time = (end_time - start_time) / np.timedelta64(1, "s")
        relative_time_sampling = np.linspace(
            0.0, passage_time, num=spec.points_per_passage + 2, endpoint=True
        )
        relative_time_sampling = relative_time_sampling[1:-1]
        # TODO: once the propagator sampling has been changed, use a interpolator here instead
        # at the cadence that the propagator currently uses
        pass_tx_index = np.empty((spec.points_per_passage,), dtype=np.int64)
        for ind in range(spec.points_per_passage):
            pass_tx_index[ind] = np.argmin(
                np.abs(
                    (relative_time_sampling[ind] + start_time - state.spobj_time)
                    / np.timedelta64(1, "s")
                )
            )
        tx_sch_index_list.append(pass_tx_index)
    tx_sch_index = np.concatenate(tx_sch_index_list)
    tx_sch_time = state.spobj_time[tx_sch_index]
    tx_sch_len = len(tx_sch_time)
    tx_pointings: EnuCoordinates = spec.tx_station.enu(state.spobj_states[:3, tx_sch_index])
    tx_pointings = tx_pointings / np.linalg.norm(tx_pointings, axis=0)

    tx_sch = schedule.from_ndarrays(
        {
            "start_time": tx_sch_time,
            "end_time": tx_sch_time + spec.exp_detail["slice_duration"],
            "exp_num": np.full(tx_sch_len, spec.exp_detail["id"], dtype=np.int16),
            "stn_num": np.full(tx_sch_len, spec.tx_station.uid, dtype=np.int16),
            "simult_num": np.full(tx_sch_len, 0, dtype=np.int16),
            "pointing": tx_pointings,
        }
    )

    rx_schs: list[schedule.Schedule] = []
    for rx_stn in spec.rx_stations:
        rx_pointings: EnuCoordinates = rx_stn.enu(state.spobj_states[:3, tx_sch_index])
        rx_pointings = rx_pointings / np.linalg.norm(rx_pointings, axis=0)

        rx_schs.append(
            schedule.from_ndarrays(
                {
                    "start_time": tx_sch_time,
                    "end_time": tx_sch_time + spec.exp_detail["slice_duration"],
                    "exp_num": np.full(tx_sch_len, spec.exp_detail["id"], dtype=np.int16),
                    "stn_num": np.full(tx_sch_len, rx_stn.uid, dtype=np.int16),
                    "simult_num": np.full(tx_sch_len, 0, dtype=np.int16),
                    "pointing": rx_pointings,
                }
            )
        )

    resultant_sch = xr.concat([tx_sch, *rx_schs], dim=schedule._K.multi_index)
    resultant_sch = resultant_sch.sortby(schedule._K.start_time)
    output = resultant_sch

    return output


class SparseTrackerController(ControllerBase):
    """
    Generate pointing schedule that tracks a space object.

    - The preferred way to create instances of this class is via its class methods (e.g. `TrackerController.from_space_object`).
    - This class serve as a frontend to the `State` type in this module
    """

    def __init__(self, spec: ControllerSpec, state: ControllerState | None):
        """
        NOTE: This is intended as an internal constructor, please use the constructor methods to create instances.
        """

        self.spec: ControllerSpec = spec
        self.state: ControllerState | None = state

        self._cached_output: schedule.Schedule | None = None
        """A cache of the latest `Output`, handy for plotting"""

    @classmethod
    def from_space_object(
        cls,
        spobj: SpaceObject,
        epoch: Datetime_Like,
        tx_station: Station,
        rx_stations: t.Sequence[Station],
        exp_detail: schedule.ExperimentDetail,
    ) -> t.Self:
        """A constructor method"""

        stn_pairs = [(tx_station.uid, rx_station.uid) for rx_station in rx_stations]

        ctrl = cls(
            spec=ControllerSpec(
                tx_station=tx_station,
                rx_stations=rx_stations,
                exp_detail=exp_detail,
                space_object=spobj,
                epoch=epoch,
                station_id_pairs=stn_pairs,
                points_per_passage=3,
            ),
            state=None,
        )

        return ctrl

    def get_experiment_detail(self) -> schedule.ExperimentDetail:
        return self.spec.exp_detail

    def get_experiment_id_station_id_pairs_map(self) -> schedule.ExperimentIdStationIdPairsMap:
        return {self.spec.exp_detail["id"]: self.spec.station_id_pairs}

    def get_station_map(self) -> dict[radar.StationId, radar.Station]:
        stn_map: dict[radar.StationId, radar.Station] = {}

        stn_map[self.spec.tx_station.uid] = self.spec.tx_station
        stn_map.update(list([(stn.uid, stn) for stn in self.spec.rx_stations]))

        return stn_map

    def compute_ecef_states(
        self, start_time: Datetime_Like, end_time: Datetime_Like, slice_duration: Timedelta_Like
    ):
        """Do the computation then update the `state` property and return `self`."""

        if "spobj" not in self.spec:
            raise RuntimeError(
                "Cannot compute space object ECEF states without `spobj` in the `spec` prop."
            )
        if "epoch" not in self.spec:
            raise RuntimeError(
                "Cannot compute space object ECEF states without `epoch` in the `spec` prop."
            )

        exp_detail: schedule.ExperimentDetail = self.spec.exp_detail

        # NOTE: for `np.arange` 'stop param,
        #   - we subtract 'slice_duration' so that only full slice are included
        #   - and add `+1` so that slice with time range `('end_time - 'slice_duration', 'end_time')` is included
        # TODO: use sampler method for this to ensure efficient propagator usage which has already
        # been defined for the other steps of the simulation
        time: npt.NDArray[Datetime64_us] = np.arange(
            to_datetime64_us(start_time),
            to_datetime64_us(end_time) - to_timedelta64_us(slice_duration) + 1,
            exp_detail["slice_duration"],
        )
        dt: npt.NDArray[Timedelta64_us] = time - to_datetime64_us(self.spec.epoch)
        dsec = t.cast(npt.NDArray[Float64_as_sec], dt.astype(np.float64) / 1e6)

        ecefs = self.spec.space_object.get_state(dsec)

        self.state = ControllerState(spobj_time=time, spobj_states=ecefs)

        return self

    def generate(
        self, start_time: Datetime_Like | None = None, end_time: Datetime_Like | None = None
    ) -> schedule.Schedule:
        """
        Generate the schedules.
        `start_time` and `end_time` should be omitted if this instance is created from `TrackerController.from_ecef_states`
        """

        if start_time is not None and end_time is not None:
            self.compute_ecef_states(start_time, end_time, self.spec.exp_detail["slice_duration"])
            state = t.cast(ControllerState, self.state)
        elif self.state is None:
            raise RuntimeError(
                "Cannot generate without valid state property."
                + " Please either provide the `start_time` and `end_time` param"
                + " or ensure it is set correctly using methods like `compute_ecef_states` or proper constructors."
            )
        else:
            state = self.state

        output = generate_from_state(spec=self.spec, state=state)
        self._cached_output = output

        return output
