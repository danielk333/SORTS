from __future__ import annotations
import typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from sorts import types, radar, schedule
from sorts.utils import assert_class_attributes_equal_to, to_datetime64_us
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.signals import hard_target_snr
from sorts.interpolation import Interpolator
from sorts.schedule import ExperimentDetailMap, Schedule
from sorts.simulation.types import Passage
from . import funcs

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

_SK = Schedule._K
"""Internal helper for accessing string keys consistently"""

# TODO: rename to just `State`?
StateData = t.NewType("StateData", xr.Dataset)
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


def empty_state_data() -> StateData:
    multi_index = pd.MultiIndex.from_arrays(
        [
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype="datetime64[us]"),
        ],
        names=(_K.exp_num, _K.rx_simult_num, _K.time),
    )

    state_data = xr.Dataset(
        coords={
            **xr.Coordinates.from_pandas_multiindex(multi_index, _K.multi_index),
            _K.enu: [_K.e, _K.n, _K.u],
        },
        data_vars={
            # NOTE: we used `.loc` instead of `reindex` here because we cannot get `reindex` working
            # TODO: investigate why `reindex` won't work
            #   not working: `tx_sch._data[_SK.pointing].reindex({_SK.multi_index: [(np.datetime64("2025-01-01 02:45:01", "us"), 0, 0), ...]})`
            _K.tx_pointing: (
                (_K.enu, _K.multi_index),
                np.empty((3, 0), dtype=np.float64),
            ),
            _K.rx_pointing: ((_K.enu, _K.multi_index), np.empty((3, 0), dtype=np.float64)),
        },
    )

    return StateData(state_data)


def filter_state_data_by_time_range(state: StateData, time_range: types.TimeRange_us) -> StateData:
    mask = (state[_K.time] >= time_range[0]) & (state[_K.time] <= time_range[1])
    state_masked = state[{_K.multi_index: mask}]

    return state_masked


# TODO: better naming
class FromPassagesOverTxRxStationPairParam(t.TypedDict):
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
    - It is stored in a private attribute `_state_data`
    - It can contain data for more than 1 passage
    - The dataset does not always contains all the key defined in `DataKey`,
      which ones are available depends on what calculation have been done.
    """

    _K = _K
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
        state_data: StateData,
    ):
        self.id = id

        self._state_data = state_data

        self.space_object = spobj
        self.space_object_interp = spobj_interp

        self.passages = passages

        self.tx_station = tx_station
        self.rx_station = rx_station
        self.exp_detail_map = exp_detail_map

        self.observations: list[Observation] = []

    # TODO: do not use 'Unpack' here
    @classmethod
    def from_passages_over_tx_rx_station_pair(
        cls, **kwargs: t.Unpack[FromPassagesOverTxRxStationPairParam]
    ) -> t.Self:
        id = kwargs["id"]
        passages = kwargs["passages"]
        spobj = kwargs["spobj"]
        spobj_interp = kwargs["spobj_interp"]
        tx_station = kwargs["tx_station"]
        rx_station = kwargs["rx_station"]

        # early return for empty cases
        # NOTE: this is particularly needed because `.loc` will throw KeyError for non-existence keys
        # TODO: add test case for empty case?
        if (
            len(passages) == 0
            or not (kwargs["schedule"]._data[_SK.stn_num] == tx_station.uid).any()
            or not (kwargs["schedule"]._data[_SK.stn_num] == rx_station.uid).any()
        ):
            return cls(
                id=id,
                spobj=spobj,
                spobj_interp=spobj_interp,
                passages=passages,
                tx_station=kwargs["tx_station"],
                rx_station=kwargs["rx_station"],
                exp_detail_map=kwargs["exp_detail_map"],
                state_data=StateData(empty_state_data()),
            )

        # NOTE: xarray simplify/collapse MultiIndex when filtering a level to an exact value,
        #   we filter on the top level "multi_index' with a tuple here to prevent it
        tx_schdata = kwargs["schedule"]._data.loc[
            {_SK.multi_index: (slice(None), tx_station.uid, slice(None), slice(None))}
        ]
        rx_schdata = kwargs["schedule"]._data.loc[
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
                    np.full(len(rx_time), kwargs["tx_station"].uid, dtype=np.int16),
                    np.full(len(rx_time), 0, dtype=np.int16),  # assuming single tx
                    rx_time,
                ],
                names=(_SK.exp_num, _SK.stn_num, _SK.simult_num, _SK.start_time),
            ),
            _SK.multi_index,
        )

        state_data = xr.Dataset(
            coords={
                **xr.Coordinates.from_pandas_multiindex(multi_index, _K.multi_index),
                _K.enu: [_K.e, _K.n, _K.u],
            },
            data_vars={
                # NOTE: we used `.loc` instead of `reindex` here because we cannot get `reindex` working
                # TODO: investigate why `reindex` won't work
                #   not working: `tx_sch._data[_SK.pointing].reindex({_SK.multi_index: [(np.datetime64("2025-01-01 02:45:01", "us"), 0, 0), ...]})`
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
            tx_station=kwargs["tx_station"],
            rx_station=kwargs["rx_station"],
            exp_detail_map=kwargs["exp_detail_map"],
            state_data=StateData(state_data),
        )

    def simulate(self):
        """
        Run simulation calculations and update its state/data;
        Will populate the prop `observations`
        """

        epoch = to_datetime64_us(self.space_object.epoch)
        dsec = (self._state_data[_K.time] - epoch).astype(np.float64) * 1e-6
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
            [self.exp_detail_map[n]["power"] for n in self._state_data[_K.exp_num].to_numpy()],
            dtype=np.float64,
        )
        bandwidths = np.array(
            [self.exp_detail_map[n]["bandwidth"] for n in self._state_data[_K.exp_num].to_numpy()],
            dtype=np.float64,
        )
        rx_noise_temps = np.array(
            [self.exp_detail_map[n]["noise_temp"] for n in self._state_data[_K.exp_num].to_numpy()],
            dtype=np.float64,
        )

        self._state_data = funcs.calc_gain(
            state_data=self._state_data,
            tx_stn=self.tx_station,
            rx_stn=self.rx_station,
            spobj_tx_enu=spobj_tx_enu,
            spobj_rx_enu=spobj_rx_enu,
        )

        snr = hard_target_snr(
            gain_tx=self._state_data[_K.gain_tx].to_numpy(),
            gain_rx=self._state_data[_K.gain_rx].to_numpy(),
            wavelength=self.tx_station.beam.wavelength,
            power_tx=powers,
            range_tx_m=range_tx,
            range_rx_m=range_rx,
            diameter=self.space_object.d,
            bandwidth=bandwidths,
            rx_noise_temp=rx_noise_temps,
            radar_albedo=self.space_object.parameters.get("radar_albedo", 1.0),
        )
        self._state_data[_K.snr] = (_K.multi_index, snr)

        self._state_data[_K.tx_range] = (
            _K.multi_index,
            np.linalg.norm(spobj_tx_enu[:3, :], axis=0),
        )

        self._state_data[_K.rx_range] = (
            _K.multi_index,
            np.linalg.norm(spobj_rx_enu[:3, :], axis=0),
        )

        self._state_data[_K.two_way_range] = (
            self._state_data[_K.tx_range] + self._state_data[_K.rx_range]
        )

        two_way_range_series = t.cast(pd.Series, self._state_data[_K.two_way_range].to_pandas())
        time_series = t.cast(pd.Series, self._state_data[_K.time].to_pandas())
        groupped_two_way_range_diff = two_way_range_series.groupby(
            level=[_K.exp_num, _K.rx_simult_num]
        ).diff()
        groupped_time_diff = time_series.groupby(level=[_K.exp_num, _K.rx_simult_num]).diff()
        self._state_data[_K.two_way_range_rate] = (
            _K.multi_index,
            groupped_two_way_range_diff
            / (groupped_time_diff / t.cast(t.Any, np.timedelta64(1, "s"))),
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
    schedule.ExperimentId,
    radar.StationId,
    schedule.SimultaneousNum,
    npt.NDArray[types.Datetime64_us],
]
ObservationScheduleIndexer = types.TxRxTuple[
    ObservationStationScheduleIndexer, ObservationStationScheduleIndexer
]
ObservationStateIndexer = tuple[
    schedule.ExperimentId, schedule.SimultaneousNum, npt.NDArray[types.Datetime64_us]
]


class Observation:
    def __init__(
        self,
        passage: Passage,
        sim_unit: SimulationUnit,
        exp_id: schedule.ExperimentId,
        simult_num: schedule.SimultaneousNum,
    ):
        self.passage = passage
        self.sim_unit = sim_unit
        self.exp_id = exp_id
        self.simult_num = simult_num

    @classmethod
    def from_passage(cls, passage: Passage, sim_unit: SimulationUnit) -> list[t.Self]:
        state_slice = filter_state_data_by_time_range(sim_unit._state_data, passage["time_range"])

        multi_index = t.cast(pd.MultiIndex, state_slice.indexes[_K.multi_index])

        unique_exp_id_simult_num_pairs: list[
            tuple[schedule.ExperimentId, schedule.SimultaneousNum]
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
                f"    time_range={self.passage["time_range"]}",
                f"    spobj_id={self.sim_unit.space_object.oid}, tx_stn_id={self.passage["tx_station"].uid}, rx_stn_id={self.passage["rx_station"].uid}",
                f"    exp_id={self.exp_id}, simult_num={self.simult_num}",
                ")",
            ]
        )

    def get_time_arr(self):
        time_arr = filter_state_data_by_time_range(
            self.sim_unit._state_data, self.passage["time_range"]
        )[_K.time].to_numpy()

        return time_arr

    def index_into_schedule(self, schedule: Schedule) -> types.TxRxTuple[Schedule, Schedule]:
        """Returns subset of schedules, in `(tx_scheule, tx_schedule` that corresponds to the observation"""

        tx_sch_obs = schedule.filter_by_time_range(self.passage["time_range"])
        tx_sch_obs = Schedule(
            tx_sch_obs._data.loc[
                {
                    _SK.multi_index: (
                        self.exp_id,
                        self.passage["tx_station"].uid,
                        0,  # NOTE: we only support single simultaneous tx pointing
                        slice(None),
                    )
                }
            ]
        )

        rx_sch_obs = schedule.filter_by_time_range(self.passage["time_range"])
        rx_sch_obs = Schedule(
            rx_sch_obs._data.loc[
                {
                    _SK.multi_index: (
                        self.exp_id,
                        self.passage["rx_station"].uid,
                        self.simult_num,
                        slice(None),
                    )
                }
            ]
        )

        return types.TxRxTuple(tx=tx_sch_obs, rx=rx_sch_obs)

    def get_state_slice(self) -> StateData:
        """Get the subset of `StateData` data the corresponds to the the observation"""

        sim_state_slice = filter_state_data_by_time_range(
            self.sim_unit._state_data, self.passage["time_range"]
        )

        # NOTE: early return for empty case; `loc` method does not work with non-existent selection
        if len(sim_state_slice[_K.multi_index]) == 0:
            return sim_state_slice

        sim_state_slice = sim_state_slice.loc[
            {_K.multi_index: (self.exp_id, self.simult_num, slice(None))}
        ]

        return sim_state_slice
