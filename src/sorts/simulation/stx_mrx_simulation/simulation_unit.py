from __future__ import annotations
import typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from sorts import types, radar, scheduling
from sorts.types import TxRxTuple
from sorts.utils import assert_class_attributes_equal_to, to_datetime64_us
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.signals import hard_target_snr
from sorts.interpolation import Interpolator
from sorts.scheduling import ExperimentDetailMap, Schedule
from sorts.simulation import Passage


CoordKey = t.Literal["multi_index", "time", "exp_num", "rx_simult_num", "enu", "e", "n", "u"]
DataKey = t.Literal[
    "tx_pointing",
    "rx_pointing",
    "exp_num",
    "gain_tx",
    "gain_rx",
    "snr",
    "tx_range",
    "rx_range",
    "two_way_range",
    "two_way_range_rate",
]
Key = t.Literal[DataKey, CoordKey]


# TODO: updated the name with tx/rx as suffix to prefix
class _K:
    """Internal helper for accessing string keys consistently"""

    multi_index: t.Final = "multi_index"
    time: t.Final = "time"
    exp_num: t.Final = "exp_num"
    rx_simult_num: t.Final = "rx_simult_num"
    enu: t.Final = "enu"
    e: t.Final = "e"
    n: t.Final = "n"
    u: t.Final = "u"
    tx_pointing: t.Final = "tx_pointing"
    rx_pointing: t.Final = "rx_pointing"
    gain_tx: t.Final = "gain_tx"
    gain_rx: t.Final = "gain_rx"
    snr: t.Final = "snr"
    tx_range: t.Final = "tx_range"
    rx_range: t.Final = "rx_range"
    two_way_range: t.Final = "two_way_range"
    two_way_range_rate: t.Final = "two_way_range_rate"


assert_class_attributes_equal_to(_K, t.get_args(Key))

_SK = scheduling._K
"""Internal helper for accessing string keys consistently"""

SimulationUnitState = t.NewType("SimulationUnitState", xr.Dataset)
"""
A xarray `Dataset` with:
  ```
  Dimensions:        (multi_index: n, enu: 3)
  Coordinates:
    * multi_index    (multi_index) object MultiIndex ('exp_num', 'rx_simult_num', 'time')
    * time           (multi_index) datetime64[us]
    * exp_num        (multi_index) int16
    * rx_simult_num  (multi_index) int16
    * enu            (enu) 'e' 'n' 'u'
  Data variables:
      tx_pointing    (enu, multi_index) float64
      rx_pointing    (enu, multi_index) float64
      gain_tx        (multi_index) float64
      gain_rx        (multi_index) float64
      snr            (multi_index) float64
      tx_range       (multi_index) float64
      rx_range       (multi_index) float64
      two_way_range  (multi_index) float64
      two_way_range_rate  (multi_index) float64
  ```
"""


def empty_state() -> SimulationUnitState:
    multi_index = pd.MultiIndex.from_arrays(
        [
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype="datetime64[us]"),
        ],
        names=(_K.exp_num, _K.rx_simult_num, _K.time),
    )

    state = xr.Dataset(
        coords={
            **xr.Coordinates.from_pandas_multiindex(multi_index, _K.multi_index),
            _K.enu: [_K.e, _K.n, _K.u],
        },
        data_vars={
            # NOTE: we used `.loc` instead of `reindex` here because we cannot get `reindex` working
            # TODO: investigate why `reindex` won't work
            #   not working: `tx_sch[_SK.pointing].reindex({_SK.multi_index: [(np.datetime64("2025-01-01 02:45:01", "us"), 0, 0), ...]})`
            _K.tx_pointing: (
                (_K.enu, _K.multi_index),
                np.empty((3, 0), dtype=np.float64),
            ),
            _K.rx_pointing: ((_K.enu, _K.multi_index), np.empty((3, 0), dtype=np.float64)),
        },
    )

    return SimulationUnitState(state)


def filter_state_by_time_range(
    state: SimulationUnitState, time_range: types.TimeRange_us
) -> SimulationUnitState:
    mask = (state[_K.time] >= time_range[0]) & (state[_K.time] <= time_range[1])
    state_masked = state[{_K.multi_index: mask}]

    return state_masked


def calc_gain(
    state: SimulationUnitState,
    tx_stn: Station,
    rx_stn: Station,
    spobj_tx_enu: types.EnuCoordinates,
    spobj_rx_enu: types.EnuCoordinates,
) -> SimulationUnitState:
    # will be populated to [tx_gain_arr, rx_gain_arr]
    gain_arr_list: list[npt.NDArray[np.float64]] = []

    for stn, spobj_stn_enu, pt_key in zip(
        [tx_stn, rx_stn],
        [spobj_tx_enu, spobj_rx_enu],
        [_K.tx_pointing, _K.rx_pointing],
    ):
        # early return for empty cases
        # NOTE: this is particularly needed because some `.gain` does not work with empty parameters (e.g. beam.parameters["pointing"])
        # TODO: add test case for empty case?
        if len(state[_K.multi_index]) == 0:
            gain_arr_list.append(np.empty(0, dtype=np.float64))

        elif stn.beam_parameters is None:
            # TODO: remove this hack; see issues #25 for details
            raise RuntimeError(
                "A hack of injecting `beam_parameters` into `tx_stn.beam_parameters` is currently required for gain calculation"
            )

        else:
            beam_parameters = stn.beam_parameters

            if "pointing" in stn.beam_parameters.keys:
                beam_parameters = stn.beam_parameters.replace_and_broadcast(
                    parameters=stn.beam_parameters,
                    new_parameters=dict(pointing=state[pt_key].to_numpy()),
                )

            gain_arr_list.append(stn.beam.gain(spobj_stn_enu[:3], beam_parameters))

    state[_K.gain_tx] = (_K.multi_index, gain_arr_list[0])
    state[_K.gain_rx] = (_K.multi_index, gain_arr_list[1])

    return state


@dataclass
class FromPassagesOverTxRxStationPairParam:
    id: str
    passages: list[Passage]
    spobj: SpaceObject
    spobj_interp: Interpolator
    tx_station: Station
    rx_station: Station
    schedule: Schedule
    exp_detail_map: ExperimentDetailMap


# TODO: re-eval: `Station`` can be taken from `Passage`, but empty `list[Passage]` would be an issue in that case.
class SimulationUnit:
    """
    Contains all the params and results for a unit of simulation calculation.

    Notes about the state data:
    - It is stored in a private attribute `_state`
    - It can contain data for more than 1 passage
    - The dataset does not always contains all the key defined in `DataKey`,
      which ones are available depends on what calculation have been done.
    """

    _K = _K
    """shortcut to module attribute"""

    # todo: my type checker is complaining over these shortcuts?
    # FromPassagesOverTxRxStationPairParam = FromPassagesOverTxRxStationPairParam
    """shortcut to module attribute"""

    def __init__(
        self,
        id: str,
        spobj: SpaceObject,
        spobj_interp: Interpolator,
        passages: list[Passage],
        tx_station: Station,
        rx_station: Station,
        exp_detail_map: ExperimentDetailMap,
        state: SimulationUnitState,
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

    # todo: this param should maybe be expanded so the components are arguments, or a more
    # generalized units should be made: i think this might be too specialized as a data carrier?
    # will it be useful outside of this function call?
    @classmethod
    def from_passages_over_tx_rx_station_pair(
        cls, param: FromPassagesOverTxRxStationPairParam
    ) -> t.Self:
        # TODO: the logic inside this function is not super clear - it needs clarification
        id = param.id
        passages = param.passages
        spobj = param.spobj
        spobj_interp = param.spobj_interp
        tx_station = param.tx_station
        rx_station = param.rx_station

        # early return for empty cases
        # NOTE: this is particularly needed because `.loc` will throw KeyError for non-existence keys
        # TODO: add test case for empty case?
        if (
            len(passages) == 0
            or not (param.schedule[_SK.stn_num] == tx_station.uid).any()
            or not (param.schedule[_SK.stn_num] == rx_station.uid).any()
        ):
            return cls(
                id=id,
                spobj=spobj,
                spobj_interp=spobj_interp,
                passages=passages,
                tx_station=param.tx_station,
                rx_station=param.rx_station,
                exp_detail_map=param.exp_detail_map,
                state=SimulationUnitState(empty_state()),
            )

        # NOTE: xarray simplify/collapse MultiIndex when filtering a level to an exact value,
        #   we filter on the top level "multi_index' with a tuple here to prevent it
        tx_schdata = param.schedule.loc[
            {_SK.multi_index: (slice(None), tx_station.uid, slice(None), slice(None))}
        ]
        rx_schdata = param.schedule.loc[
            {_SK.multi_index: (slice(None), rx_station.uid, slice(None), slice(None))}
        ]

        rx_time = rx_schdata[_SK.start_time].to_numpy()
        rx_exp_num = rx_schdata[_SK.exp_num].to_numpy()
        rx_simult_num = rx_schdata[_SK.simult_num].to_numpy()

        multi_index = pd.MultiIndex.from_arrays(
            [rx_exp_num, rx_simult_num, rx_time],
            names=(_K.exp_num, _K.rx_simult_num, _K.time),
        )

        tx_reindex_selector = xr.Coordinates.from_pandas_multiindex(
            pd.MultiIndex.from_arrays(
                [
                    rx_exp_num,
                    np.full(len(rx_time), param.tx_station.uid, dtype=np.int16),
                    np.full(len(rx_time), 0, dtype=np.int16),  # assuming single tx
                    rx_time,
                ],
                names=(_SK.exp_num, _SK.stn_num, _SK.simult_num, _SK.start_time),
            ),
            _SK.multi_index,
        )

        state = xr.Dataset(
            coords={
                **xr.Coordinates.from_pandas_multiindex(multi_index, _K.multi_index),
                _K.enu: [_K.e, _K.n, _K.u],
            },
            data_vars={
                # NOTE: we used `.loc` instead of `reindex` here because we cannot get `reindex` working
                # TODO: investigate why `reindex` won't work
                #   not working: `tx_sch[_SK.pointing].reindex({_SK.multi_index: [(np.datetime64("2025-01-01 02:45:01", "us"), 0, 0), ...]})`
                _K.tx_pointing: (
                    (_K.enu, _K.multi_index),
                    tx_schdata[_SK.pointing]
                    .loc[{_SK.multi_index: tx_reindex_selector[_SK.multi_index]}]
                    .to_numpy(),
                ),
                _K.rx_pointing: ((_K.enu, _K.multi_index), rx_schdata[_SK.pointing].to_numpy()),
            },
        )

        return cls(
            id=id,
            spobj=spobj,
            spobj_interp=spobj_interp,
            passages=passages,
            tx_station=param.tx_station,
            rx_station=param.rx_station,
            exp_detail_map=param.exp_detail_map,
            state=SimulationUnitState(state),
        )

    def simulate(self):
        """
        Run simulation calculations and update its state/data;
        Will populate the prop `observations`
        """

        if self.tx_station.wavelength is None:
            # TODO: remove this hack; see issues #25 for details
            raise RuntimeError(
                "A hack of injecting `frequency` into `tx_stn.frequency` is currently required for calling `hard_target_snr`"
            )

        epoch = to_datetime64_us(self.space_object.epoch)
        dsec = (self._state[_K.time] - epoch).to_numpy() / np.timedelta64(1, "s")
        spobj_states = self.space_object_interp.get_state(dsec)
        spobj_tx_enu = self.tx_station.enu(spobj_states)
        spobj_rx_enu = self.rx_station.enu(spobj_states)

        range_tx: npt.NDArray[types.Float64_as_m] = np.linalg.norm(spobj_tx_enu[:3, :], axis=0)
        range_rx: npt.NDArray[types.Float64_as_m] = np.linalg.norm(spobj_rx_enu[:3, :], axis=0)

        # TODO: can likely use assignment by slice/indexing instead of looping
        # TODO: do we need `pulse_lengths`?
        # TODO: do we need `ipps`?
        # TODO: do we need `duty_cycles`?
        powers = np.array(
            [self.exp_detail_map[n].power for n in self._state[_K.exp_num].to_numpy()],
            dtype=np.float64,
        )
        bandwidths = np.array(
            [self.exp_detail_map[n].bandwidth for n in self._state[_K.exp_num].to_numpy()],
            dtype=np.float64,
        )
        rx_noise_temps = np.array(
            [self.exp_detail_map[n].noise_temp for n in self._state[_K.exp_num].to_numpy()],
            dtype=np.float64,
        )

        self._state = calc_gain(
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
        self._state[_K.snr] = (_K.multi_index, snr)

        self._state[_K.tx_range] = (
            _K.multi_index,
            np.linalg.norm(spobj_tx_enu[:3, :], axis=0),
        )

        self._state[_K.rx_range] = (
            _K.multi_index,
            np.linalg.norm(spobj_rx_enu[:3, :], axis=0),
        )

        self._state[_K.two_way_range] = (
            _K.multi_index,
            range_tx + range_rx,
        )
        v_tx = np.sum(spobj_tx_enu[:3, :] * spobj_tx_enu[3:, :], axis=0) / range_tx
        v_rx = np.sum(spobj_rx_enu[:3, :] * spobj_rx_enu[3:, :], axis=0) / range_rx
        self._state[_K.two_way_range_rate] = (
            _K.multi_index,
            v_tx + v_rx,
        )

        obss = self.get_observations()
        self.observations = obss

        return obss

    def get_observations(self) -> list[Observation]:
        obss: list[Observation] = []

        for passage in self.passages:
            obss.extend(Observation.from_passage(passage, self))

        return obss


ObservationStationScheduleIndexer = tuple[
    scheduling.ExperimentId,
    radar.StationId,
    scheduling.SimultaneousNum,
    npt.NDArray[types.Datetime64_us],
]
ObservationScheduleIndexer = TxRxTuple[
    ObservationStationScheduleIndexer, ObservationStationScheduleIndexer
]
ObservationStateIndexer = tuple[
    scheduling.ExperimentId, scheduling.SimultaneousNum, npt.NDArray[types.Datetime64_us]
]


# TODO: i dont understand why this does not seem to actually contain any data? everything seems to
# be in the simulation units? maybe parts of the simulations units could be moved here or vice versa
class Observation:
    def __init__(
        self,
        passage: Passage,
        sim_unit: SimulationUnit,
        exp_id: scheduling.ExperimentId,
        simult_num: scheduling.SimultaneousNum,
    ):
        # todo: update for collecting passage and multi passage
        self.passage = passage
        self.sim_unit = sim_unit
        self.exp_id = exp_id
        self.simult_num = simult_num

    @classmethod
    def from_passage(cls, passage: Passage, sim_unit: SimulationUnit) -> list[t.Self]:
        state_slice = filter_state_by_time_range(sim_unit._state, passage.time_range)

        multi_index = t.cast(pd.MultiIndex, state_slice.indexes[_K.multi_index])

        unique_exp_id_simult_num_pairs: list[
            tuple[scheduling.ExperimentId, scheduling.SimultaneousNum]
        ] = (multi_index.droplevel(_K.time).unique().to_list())

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
        time_arr = filter_state_by_time_range(self.sim_unit._state, self.passage.time_range)[
            _K.time
        ].to_numpy()

        return time_arr

    def index_into_schedule(self, sch: Schedule) -> TxRxTuple[Schedule, Schedule]:
        """Returns subset of schedules, in `(tx_scheule, tx_schedule` that corresponds to the observation"""

        tx_sch_obs = scheduling.filter_by_time_range(sch, self.passage.time_range)
        tx_sch_obs = tx_sch_obs.loc[
            {
                _SK.multi_index: (
                    self.exp_id,
                    self.passage.tx_station.uid,
                    0,  # NOTE: we only support single simultaneous tx pointing
                    slice(None),
                )
            }
        ]

        rx_sch_obs = scheduling.filter_by_time_range(sch, self.passage.time_range)
        rx_sch_obs = rx_sch_obs.loc[
            {
                _SK.multi_index: (
                    self.exp_id,
                    self.passage.rx_stations[0].uid,
                    self.simult_num,
                    slice(None),
                )
            }
        ]

        return TxRxTuple(tx=tx_sch_obs, rx=rx_sch_obs)

    def get_state_slice(self) -> SimulationUnitState:
        """Get the subset of `State` data the corresponds to the the observation"""

        sim_state_slice = filter_state_by_time_range(self.sim_unit._state, self.passage.time_range)

        # NOTE: early return for empty case; `loc` method does not work with non-existent selection
        if len(sim_state_slice[_K.multi_index]) == 0:
            return sim_state_slice

        sim_state_slice = sim_state_slice.loc[
            {_K.multi_index: (self.exp_id, self.simult_num, slice(None))}
        ]

        return sim_state_slice
