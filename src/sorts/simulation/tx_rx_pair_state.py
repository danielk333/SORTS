from __future__ import annotations
import typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts import types, radar
from . import simulation_unit


TxRxPairState = t.NewType("TxRxPairState", pd.DataFrame)
"""
A pandas `DataFrame` with:
```
Index: MultiIndex('exp_num', 'rx_simult_num', 'time')
Cols:
    tx_pointing_e       float64
    tx_pointing_n       float64
    tx_pointing_u       float64
    rx_pointing_e       float64
    rx_pointing_n       float64
    rx_pointing_u       float64
    gain_tx             float64
    gain_rx             float64
    snr                 float64
    tx_range            float64
    rx_range            float64
    two_way_range       float64
    two_way_range_rate  float64
```
"""


def empty_state() -> TxRxPairState:
    _K = simulation_unit.SimulationUnitKey

    multi_index = pd.MultiIndex.from_arrays(
        [
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype="datetime64[us]"),
        ],
        names=(_K.exp_num, _K.rx_simult_num, _K.time),
    )

    state = pd.DataFrame(
        {
            # tx pointing enu
            _K.tx_pointing_e: np.empty(0, dtype=np.float64),
            _K.tx_pointing_n: np.empty(0, dtype=np.float64),
            _K.tx_pointing_u: np.empty(0, dtype=np.float64),
            # rx pointing enu
            _K.rx_pointing_e: np.empty(0, dtype=np.float64),
            _K.rx_pointing_n: np.empty(0, dtype=np.float64),
            _K.rx_pointing_u: np.empty(0, dtype=np.float64),
        },
        index=multi_index,
    )

    return TxRxPairState(state)


def filter_state_by_time_range(
    state: TxRxPairState, time_range: types.TimeRange_us
) -> TxRxPairState:
    _K = simulation_unit.SimulationUnitKey

    mask = (
        (state.index.get_level_values(_K.time) >= time_range[0])
        & (state.index.get_level_values(_K.time) <= time_range[1])
    ) # fmt: skip
    state_masked = state[mask]

    return state_masked


def calc_gain(
    state: TxRxPairState,
    tx_stn: radar.Station,
    rx_stn: radar.Station,
    spobj_tx_enu: types.EnuCoordinates,
    spobj_rx_enu: types.EnuCoordinates,
) -> TxRxPairState:
    _K = simulation_unit.SimulationUnitKey

    # will be populated to [tx_gain_arr, rx_gain_arr]
    gain_arr_list: list[npt.NDArray[np.float64]] = []

    for stn, spobj_stn_enu, pt_keys in zip(
        [tx_stn, rx_stn],
        [spobj_tx_enu, spobj_rx_enu],
        [
            [_K.tx_pointing_e, _K.tx_pointing_n, _K.tx_pointing_u],
            [_K.rx_pointing_e, _K.rx_pointing_n, _K.rx_pointing_u],
        ],
    ):
        # early return for empty cases
        # NOTE: this is particularly needed because some `.gain` does not work with empty parameters (e.g. beam.parameters["pointing"])
        # TODO: add test case for empty case?
        if len(state) == 0:
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
                    new_parameters=dict(pointing=state[pt_keys].T.to_numpy()),
                )

            gain_arr_list.append(stn.beam.gain(spobj_stn_enu[:3], beam_parameters))

    state[_K.gain_tx] = gain_arr_list[0]
    state[_K.gain_rx] = gain_arr_list[1]

    return state
