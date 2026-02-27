from __future__ import annotations
import typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts import types, radar
from sorts.types import TxRxTuple
from sorts.utils import to_datetime64_us
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.signals import hard_target_snr
from sorts.interpolation import Interpolator
from sorts.schedule import ScheduleDataframe, ScheduleKey
from sorts.simulation import Passage
from . import simulation_unit_state


@dataclass
class FromPassagesOverTxRxStationPairParam:
    id: str
    passages: list[Passage]
    spobj: SpaceObject
    spobj_interp: Interpolator
    tx_station: Station
    rx_station: Station
    tx_rx_pointing_pairs: pd.DataFrame # TODO: this is a tmp solution, should refactor this type and dataflow; # fmt: skip
    exp_detail_map: types.ExperimentDetailMap


# TODO: re-eval: `Station`` can be taken from `Passage`, but empty `list[Passage]` would be an issue in that case.
class SimulationUnit:
    """
    Contains all the params and results for a unit of simulation calculation.

    Notes about the state data:
    - It is stored in a private attribute `_state`
    - It can contain data for more than 1 passage
    - The dataset does not always contains all the columns,
      which ones are available depends on what calculation have been done.
    """

    def __init__(
        self,
        id: str,
        spobj: SpaceObject,
        spobj_interp: Interpolator,
        passages: list[Passage],
        tx_station: Station,
        rx_station: Station,
        exp_detail_map: types.ExperimentDetailMap,
        state: simulation_unit_state.SimulationUnitState,
    ):
        self.id = id

        self._state = state

        self.space_object = spobj
        self.space_object_interp = spobj_interp

        self.passages = passages

        self.tx_station = tx_station
        self.rx_station = rx_station
        self.exp_detail_map = exp_detail_map

        self.observations: list[Observation] = []

    # TODO: this param should maybe be expanded so the components are arguments, or a more
    # generalized units should be made: i think this might be too specialized as a data carrier?
    # will it be useful outside of this function call?
    @classmethod
    def from_passages_over_tx_rx_station_pair(
        cls, param: FromPassagesOverTxRxStationPairParam
    ) -> t.Self:
        # TODO: the logic inside this function is not super clear - it needs clarification
        _K = types.SimulationUnitKey
        id = param.id
        passages = param.passages
        spobj = param.spobj
        spobj_interp = param.spobj_interp

        # early return for empty cases
        # TODO: add test case for empty case?
        if len(passages) == 0 or len(param.tx_rx_pointing_pairs) == 0:
            return cls(
                id=id,
                spobj=spobj,
                spobj_interp=spobj_interp,
                passages=passages,
                tx_station=param.tx_station,
                rx_station=param.rx_station,
                exp_detail_map=param.exp_detail_map,
                state=simulation_unit_state.SimulationUnitState(
                    simulation_unit_state.empty_state()
                ),
            )

        state = param.tx_rx_pointing_pairs.set_index([_K.exp_num, _K.rx_simult_num, _K.time])

        return cls(
            id=id,
            spobj=spobj,
            spobj_interp=spobj_interp,
            passages=passages,
            tx_station=param.tx_station,
            rx_station=param.rx_station,
            exp_detail_map=param.exp_detail_map,
            state=simulation_unit_state.SimulationUnitState(state),
        )

    def simulate(self):
        """
        Run simulation calculations and update its state/data;
        Will populate the prop `observations`
        """

        _K = types.SimulationUnitKey

        if self.tx_station.wavelength is None:
            # TODO: remove this hack; see issues #25 for details
            raise RuntimeError(
                "A hack of injecting `frequency` into `tx_stn.frequency` is currently required for calling `hard_target_snr`"
            )

        epoch = to_datetime64_us(self.space_object.epoch)
        dsec = (
            (self._state.index.get_level_values(_K.time).to_numpy() - epoch)
            / np.timedelta64(1, "s")
        ) # fmt: skip
        spobj_states = self.space_object_interp.get_state(dsec)
        spobj_tx_enu = self.tx_station.enu(spobj_states)
        spobj_rx_enu = self.rx_station.enu(spobj_states)

        range_tx: npt.NDArray[types.Float64_as_m] = np.linalg.norm(spobj_tx_enu[:3, :], axis=0)
        range_rx: npt.NDArray[types.Float64_as_m] = np.linalg.norm(spobj_rx_enu[:3, :], axis=0)

        # TODO: can likely use assignment by slice/indexing instead of looping
        powers = np.array(
            [
                self.exp_detail_map[n].power
                for n in self._state.index.get_level_values(_K.exp_num).to_numpy()
            ],
            dtype=np.float64,
        )
        bandwidths = np.array(
            [
                self.exp_detail_map[n].bandwidth
                for n in self._state.index.get_level_values(_K.exp_num).to_numpy()
            ],
            dtype=np.float64,
        )
        rx_noise_temps = np.array(
            [
                self.exp_detail_map[n].noise_temp
                for n in self._state.index.get_level_values(_K.exp_num).to_numpy()
            ],
            dtype=np.float64,
        )

        self._state = simulation_unit_state.calc_gain(
            state=self._state,
            tx_stn=self.tx_station,
            rx_stn=self.rx_station,
            spobj_tx_enu=spobj_tx_enu,
            spobj_rx_enu=spobj_rx_enu,
        )

        snr = hard_target_snr(
            gain_tx=self._state[_K.gain_tx].to_numpy(),
            gain_rx=self._state[_K.gain_rx].to_numpy(),
            wavelength=self.tx_station.wavelength,
            power_tx=powers,
            range_tx_m=range_tx,
            range_rx_m=range_rx,
            diameter=self.space_object.d,
            bandwidth=bandwidths,
            rx_noise_temp=rx_noise_temps,
            radar_albedo=self.space_object.properties.get("radar_albedo", 1.0),
        )
        self._state[_K.snr] = snr

        self._state[_K.tx_range] = np.linalg.norm(spobj_tx_enu[:3, :], axis=0)

        self._state[_K.rx_range] = np.linalg.norm(spobj_rx_enu[:3, :], axis=0)

        self._state[_K.two_way_range] = range_tx + range_rx
        v_tx = np.sum(spobj_tx_enu[:3, :] * spobj_tx_enu[3:, :], axis=0) / range_tx
        v_rx = np.sum(spobj_rx_enu[:3, :] * spobj_rx_enu[3:, :], axis=0) / range_rx
        self._state[_K.two_way_range_rate] = v_tx + v_rx

        obss = self.get_observations()
        self.observations = obss

        return obss

    def get_observations(self) -> list[Observation]:
        obss: list[Observation] = []

        for passage in self.passages:
            obss.extend(Observation.from_passage(passage, self))

        return obss


ObservationStationScheduleIndexer = tuple[
    types.ExperimentId,
    radar.StationId,
    types.SimultaneousNum,
    npt.NDArray[types.Datetime64_us],
]
ObservationScheduleIndexer = TxRxTuple[
    ObservationStationScheduleIndexer, ObservationStationScheduleIndexer
]
ObservationStateIndexer = tuple[
    types.ExperimentId, types.SimultaneousNum, npt.NDArray[types.Datetime64_us]
]


# TODO: i dont understand why this does not seem to actually contain any data? everything seems to
# be in the simulation units? maybe parts of the simulations units could be moved here or vice versa
class Observation:
    def __init__(
        self,
        passage: Passage,
        sim_unit: SimulationUnit,
        exp_id: types.ExperimentId,
        simult_num: types.SimultaneousNum,
    ):
        # TODO: update for collecting passage and multi passage
        self.passage = passage
        self.sim_unit = sim_unit
        self.exp_id = exp_id
        self.simult_num = simult_num

    @classmethod
    def from_passage(cls, passage: Passage, sim_unit: SimulationUnit) -> list[t.Self]:
        _K = types.SimulationUnitKey

        state_slice = simulation_unit_state.filter_state_by_time_range(
            sim_unit._state, passage.time_range
        )

        unique_exp_id_simult_num_pairs: list[tuple[types.ExperimentId, types.SimultaneousNum]] = (
            state_slice.index.droplevel(_K.time).unique().to_list()
        )

        obss = [
            cls(passage=passage, sim_unit=sim_unit, exp_id=exp_id, simult_num=simult_num)
            for exp_id, simult_num in unique_exp_id_simult_num_pairs
        ]

        return obss

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Observation(",
                f"    time_range={self.passage.time_range}",
                f"    spobj_id={self.sim_unit.space_object.object_id}",
                f"    tx_stn_id={self.passage.tx_station.uid},"
                f"    rx_stn_id={self.passage.rx_stations[0].uid}",
                f"    exp_id={self.exp_id}, simult_num={self.simult_num}",
                ")",
            ]
        )

    def get_time_arr(self):
        _K = types.SimulationUnitKey

        time_arr = (
            simulation_unit_state.filter_state_by_time_range(
                self.sim_unit._state, self.passage.time_range
            )
            .index.get_level_values(_K.time)
            .to_numpy()
        )

        return time_arr

    def index_into_schedule_dataframe(
        self, df: ScheduleDataframe
    ) -> TxRxTuple[ScheduleDataframe, ScheduleDataframe]:
        """Returns subset of schedules, in `(tx_scheule, rx_schedule)` that corresponds to the observation"""

        tx_sch_obs = df[
            (df[ScheduleKey.exp_num] == self.exp_id)
            & (df[ScheduleKey.stn_num] == self.passage.tx_station.uid)
            & (df[ScheduleKey.start_time] >= self.passage.time_range[0])
            & (df[ScheduleKey.end_time] <= self.passage.time_range[1])
        ]

        rx_sch_obs = df[
            (df[ScheduleKey.exp_num] == self.exp_id)
            & (df[ScheduleKey.stn_num] != self.passage.tx_station.uid)
            & (df[ScheduleKey.start_time] >= self.passage.time_range[0])
            & (df[ScheduleKey.end_time] <= self.passage.time_range[1])
        ]

        return TxRxTuple(tx=tx_sch_obs, rx=rx_sch_obs)

    def get_state_slice(self) -> simulation_unit_state.SimulationUnitState:
        """Get the subset of `State` data the corresponds to the the observation"""

        sim_state_slice = simulation_unit_state.filter_state_by_time_range(
            self.sim_unit._state, self.passage.time_range
        )

        # NOTE: early return for empty case; `loc` method does not work with non-existent selection
        if len(sim_state_slice) == 0:
            return sim_state_slice

        sim_state_slice = sim_state_slice.loc[
            # NOTE: seems typing does not support passing tuple for MultiIndex yet
            (self.exp_id, self.simult_num, slice(None))  # type: ignore
        ]

        return sim_state_slice
