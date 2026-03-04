from __future__ import annotations
import typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts import types, radar, passage, schedule as sch
from sorts.types import TxRxTuple
from sorts.utils import to_datetime64_us
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.signals import hard_target_snr
from sorts.interpolation import Interpolator
from sorts.simulation import tx_rx_pair_state


@dataclass
class FromPassagesOverTxRxStationPairParam:
    id: str
    passages: list[passage.Passage]
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
        passages: list[passage.Passage],
        tx_station: Station,
        rx_station: Station,
        exp_detail_map: types.ExperimentDetailMap,
        state: tx_rx_pair_state.TxRxPairState,
    ):
        self.id = id

        self._state = state

        self.space_object = spobj
        self.space_object_interp = spobj_interp

        self.passages = passages

        self.tx_station = tx_station
        self.rx_station = rx_station
        self.exp_detail_map = exp_detail_map

    # TODO: this param should maybe be expanded so the components are arguments, or a more
    # generalized units should be made: i think this might be too specialized as a data carrier?
    # will it be useful outside of this function call?
    @classmethod
    def from_passages_over_tx_rx_station_pair(
        cls, param: FromPassagesOverTxRxStationPairParam
    ) -> t.Self:
        # TODO: the logic inside this function is not super clear - it needs clarification
        _K = tx_rx_pair_state.TxRxPairStateKey
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
                state=tx_rx_pair_state.TxRxPairState(tx_rx_pair_state.empty()),
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
            state=tx_rx_pair_state.TxRxPairState(state),
        )

    def simulate(self):
        """
        Run simulation calculations and update its state/data.
        """

        _K = tx_rx_pair_state.TxRxPairStateKey

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

        self._state = tx_rx_pair_state.calc_gain(
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
