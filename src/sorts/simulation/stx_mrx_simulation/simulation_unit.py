from __future__ import annotations
import typing as t
from functools import reduce
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from sorts.utils import assert_class_attributes_equal_to, to_datetime64_us
from sorts.types import Float64_as_m
from sorts.space_object import SpaceObject
from sorts.signals import hard_target_snr
from sorts.interpolation import Interpolator
from sorts.schedule import Schedule
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
    "rx_range_rate",
]
AttrKey = t.Literal["tx_stn_id", "rx_stn_id"]
Key = t.Literal[DataKey, CoordKey, AttrKey]


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
    rx_range_rate: t.Final = "rx_range_rate"
    tx_stn_id: t.Final = "tx_stn_id"
    rx_stn_id: t.Final = "rx_stn_id"


assert_class_attributes_equal_to(_K, t.get_args(Key))

_SK = Schedule._K
"""Internal helper for accessing string keys consistently"""

# TODO: re-eval NewType vs just type alias
StateData = t.NewType("StateData", xr.Dataset)
"""
A xarray `Dataset` with:
  ```
  Dimensions:        (multi_index: n, enu: 3)
  Coordinates:
    * multi_index    (multi_index) object MultiIndex
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
      rx_range_rate  (multi_index) float64
  Attributes:
      tx_stn_id:   str
      rx_stn_id:   str
  ```
"""


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
        spobj: SpaceObject,
        spobj_interp: Interpolator,
        passages: list[Passage],
        tx_sch: Schedule,
        rx_sch: Schedule,
        state_data: StateData,
    ):
        self._state_data = state_data

        self.space_object = spobj
        self.space_object_interp = spobj_interp

        self.passages = passages

        self.tx_schedule = tx_sch
        self.rx_schedule = rx_sch
        self.tx_station = tx_sch.station
        self.rx_station = rx_sch.station

    # TODO: can derive the indexers inside this method instead of as param, now that we take passages as param
    @classmethod
    def from_passages_over_tx_rx_station_pair(
        cls,
        passages: list[Passage],
        spobj: SpaceObject,
        spobj_interp: Interpolator,
        tx_sch: Schedule,
        rx_sch: Schedule,
    ) -> t.Self:
        if len(passages) == 0:
            # TODO: return en empty instance would be better
            raise NotImplementedError()

        rx_time_mask: xr.DataArray = reduce(
            xr.ufuncs.logical_and,
            [
                (rx_sch._data[_SK.start_time] >= time_range[0])
                & (rx_sch._data[_SK.end_time] <= time_range[1])
                for time_range in [ps["time_range"] for ps in passages]
            ],
        )

        rx_time = rx_sch._data[_SK.start_time][rx_time_mask].to_numpy()
        rx_exp_num = rx_sch._data[_SK.exp_num][rx_time_mask].to_numpy()
        rx_simult_num = rx_sch._data[_SK.simult_num][rx_time_mask].to_numpy()

        multi_index = pd.MultiIndex.from_arrays(
            [rx_time, rx_exp_num, rx_simult_num],
            names=(_K.time, _K.exp_num, _K.rx_simult_num),
        )

        tx_sch_pointing_selector = list(
            zip(
                rx_time,
                rx_exp_num,
                np.full(len(rx_time), 0, dtype=np.int16),
            )
        )

        state_data = xr.Dataset(
            coords={
                _K.multi_index: multi_index,
                _K.enu: [_K.e, _K.n, _K.u],
            },
            data_vars={
                # NOTE: we used `.loc` instead of `reindex` here because we cannot get `reindex` working
                # TODO: investigate why `reindex` won't work
                #   not working: `tx_sch._data[_SK.pointing].reindex({_SK.multi_index: [(np.datetime64("2025-01-01 02:45:01", "us"), 0, 0), ...]})`
                _K.tx_pointing: (
                    (_K.enu, _K.multi_index),
                    tx_sch._data[_SK.pointing]
                    .loc[{_SK.multi_index: tx_sch_pointing_selector}]
                    .to_numpy(),
                ),
                _K.rx_pointing: (
                    (_K.enu, _K.multi_index),
                    rx_sch._data[_SK.pointing].loc[:, rx_time_mask].to_numpy(),
                ),
            },
            attrs={
                _K.tx_stn_id: tx_sch._data.attrs[_SK.stn_id],
                _K.rx_stn_id: rx_sch._data.attrs[_SK.stn_id],
            },
        )

        return cls(
            spobj=spobj,
            spobj_interp=spobj_interp,
            passages=passages,
            tx_sch=tx_sch,
            rx_sch=rx_sch,
            state_data=StateData(state_data),
        )

    def simulate(self):
        """Run simulation calculation and update its state/data"""

        epoch = to_datetime64_us(self.space_object.epoch)
        dsec = (self._state_data[_K.time] - epoch).astype(np.float64) * 1e-6
        spobj_states = self.space_object_interp.get_state(dsec)
        spobj_tx_enu = self.tx_station.enu(spobj_states)
        spobj_rx_enu = self.rx_station.enu(spobj_states)

        range_tx: npt.NDArray[Float64_as_m] = np.linalg.norm(spobj_tx_enu[:3, :], axis=0)
        range_rx: npt.NDArray[Float64_as_m] = np.linalg.norm(spobj_rx_enu[:3, :], axis=0)

        # TODO: can likely use assignment by slice/indexing instead of looping
        # TODO: do we need `pulse_lengths`?
        # TODO: do we need `ipps`?
        # TODO: do we need `duty_cycles`?
        powers = np.array(
            [
                self.tx_schedule._data.attrs[_SK.exp_detail_map][n]["power"]
                for n in self._state_data[_K.exp_num].to_numpy()
            ],
            dtype=np.float64,
        )
        bandwidths = np.array(
            [
                self.tx_schedule._data.attrs[_SK.exp_detail_map][n]["bandwidth"]
                for n in self._state_data[_K.exp_num].to_numpy()
            ],
            dtype=np.float64,
        )
        rx_noise_temps = np.array(
            [
                self.rx_schedule._data.attrs[_SK.exp_detail_map][n]["noise_temp"]
                for n in self._state_data[_K.exp_num].to_numpy()
            ],
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

        self._state_data[_K.rx_range_rate] = (
            _K.multi_index,
            np.sum(
                spobj_rx_enu[3:, :]
                * (spobj_rx_enu[:3, :] / np.linalg.norm(spobj_rx_enu[:3, :], axis=0)),
                axis=0,
            ),
        )
