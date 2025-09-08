from __future__ import annotations
import typing as t
from functools import reduce
import numpy as np
import numpy.typing as npt
import xarray as xr
from sorts.utils import assert_class_attributes_equal_to, to_datetime64_us
from sorts.types import Float64_as_m
from sorts.space_object import SpaceObject
from sorts.signals import hard_target_snr
from sorts.interpolation import Interpolator
from sorts.schedule import Schedule
from sorts.simulation.types import Passage
from . import funcs

CoordKey = t.Literal["time", "enu", "e", "n", "u"]
DataKey = t.Literal[
    "tx_pointing",
    "rx_pointing",
    "exp_num",
    "gain_tx",
    "gain_rx",
    "range_tx_m",
    "range_rx_m",
    "snr",
]
AttrKey = t.Literal["stn_id"]
Key = t.Literal[DataKey, CoordKey, AttrKey]


class _K:
    """Internal helper for accessing string keys consistently"""

    time: t.Final = "time"
    enu: t.Final = "enu"
    e: t.Final = "e"
    n: t.Final = "n"
    u: t.Final = "u"
    tx_pointing: t.Final = "tx_pointing"
    rx_pointing: t.Final = "rx_pointing"
    exp_num: t.Final = "exp_num"
    gain_tx: t.Final = "gain_tx"
    gain_rx: t.Final = "gain_rx"
    range_tx_m: t.Final = "range_tx_m"
    range_rx_m: t.Final = "range_rx_m"
    snr: t.Final = "snr"
    stn_id: t.Final = "stn_id"


assert_class_attributes_equal_to(_K, t.get_args(Key))

_SK = Schedule._K
"""Internal helper for accessing string keys consistently"""

# TODO: re-eval NewType vs just type alias
StateData = t.NewType("StateData", xr.Dataset)
"""
A xarray `Dataset` with:
  ```
  Dimensions:      (enu: 3, time: 750)
  Coordinates:
    * time         (time) datetime64[us]
    * enu          (enu) <U2 24B 'az' 'el' 'r'
  Data variables:
      tx_pointing  (enu, time)
      rx_pointing  (enu, time)
      exp_num      (time)
      gain_tx      (time)
      gain_rx      (time)
      snr          (time)
  Attributes:
      stn_id:   str
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
        time = rx_sch._data[_SK.start_time][rx_time_mask]

        state_data = xr.Dataset(
            coords={
                _K.time: (_K.time, time.to_numpy()),
                _K.enu: [_K.e, _K.n, _K.u],
            },
            data_vars={
                # we expand pointings from `tx_sch` here by re-indexing using `.loc[:, time]`
                # `xarray` allow it because `tx_sch` `start_time` is an unique index
                _K.tx_pointing: (
                    (_K.enu, _K.time),
                    tx_sch._data[_SK.pointing].loc[:, time].to_numpy(),
                ),
                # for pointings from `rx_sch`, we just apply the `rx_time_mask`
                _K.rx_pointing: (
                    (_K.enu, _K.time),
                    rx_sch._data[_SK.pointing].loc[:, rx_time_mask].to_numpy(),
                ),
                _K.exp_num: (_K.time, tx_sch._data[_SK.exp_num].loc[time].to_numpy()),
            },
            attrs={
                _K.stn_id: tx_sch._data.attrs[_SK.stn_id],
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

        size = len(self._state_data[_K.time])

        self.epoch = to_datetime64_us(self.space_object.epoch)
        self.dsec = (self._state_data[_K.time] - self.epoch).astype(np.float64) * 1e-6
        self.spobj_states = self.space_object_interp.get_state(self.dsec)
        self.spobj_tx_enu = self.tx_station.enu(self.spobj_states)
        self.spobj_rx_enu = self.rx_station.enu(self.spobj_states)

        self.range_tx: npt.NDArray[Float64_as_m] = np.linalg.norm(self.spobj_tx_enu[:3, :], axis=0)
        self.range_rx: npt.NDArray[Float64_as_m] = np.linalg.norm(self.spobj_rx_enu[:3, :], axis=0)

        self.snr = np.empty((size,), dtype=np.float64)
        self.powers = np.empty((size,), dtype=np.float64)

        # TODO: do we need `pulse_lengths`?
        # TODO: do we need `ipps`?
        # TODO: do we need `duty_cycles`?
        self.powers = np.array(
            [
                self.tx_schedule._data.attrs[_SK.exp_detail_map][n]["power"]
                for n in self._state_data[_K.exp_num].to_numpy()
            ],
            dtype=np.float64,
        )
        self.bandwidths = np.array(
            [
                self.tx_schedule._data.attrs[_SK.exp_detail_map][n]["bandwidth"]
                for n in self._state_data[_K.exp_num].to_numpy()
            ],
            dtype=np.float64,
        )
        self.rx_noise_temps = np.array(
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
            spobj_tx_enu=self.spobj_tx_enu,
            spobj_rx_enu=self.spobj_rx_enu,
        )

        snr = hard_target_snr(
            gain_tx=self._state_data[_K.gain_tx].to_numpy(),
            gain_rx=self._state_data[_K.gain_rx].to_numpy(),
            wavelength=self.tx_station.beam.wavelength,
            power_tx=self.powers,
            range_tx_m=self.range_tx,
            range_rx_m=self.range_rx,
            diameter=self.space_object.d,
            bandwidth=self.bandwidths,
            rx_noise_temp=self.rx_noise_temps,
            radar_albedo=self.space_object.parameters.get("radar_albedo", 1.0),
        )

        self._state_data[_K.snr] = (_K.time, snr)
