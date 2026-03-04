from __future__ import annotations
import typing as t, enum
import numpy as np
import numpy.typing as npt
import pandas as pd
from sorts import types, radar, passage


# TODO: remove key `multi_index`
# TODO: updated the name with tx/rx as suffix to prefix
class TxRxPairStateKey(enum.StrEnum):
    multi_index = "multi_index"
    time = "time"
    exp_num = "exp_num"
    rx_simult_num = "rx_simult_num"
    tx_pointing_e = "tx_pointing_e"
    tx_pointing_n = "tx_pointing_n"
    tx_pointing_u = "tx_pointing_u"
    rx_pointing_e = "rx_pointing_e"
    rx_pointing_n = "rx_pointing_n"
    rx_pointing_u = "rx_pointing_u"
    gain_tx = "gain_tx"
    gain_rx = "gain_rx"
    snr = "snr"
    tx_range = "tx_range"
    rx_range = "rx_range"
    two_way_range = "two_way_range"
    two_way_range_rate = "two_way_range_rate"


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


def empty() -> TxRxPairState:
    _K = TxRxPairStateKey

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


def filter_by_time_range(state: TxRxPairState, time_range: types.TimeRange_us) -> TxRxPairState:
    _K = TxRxPairStateKey

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
    _K = TxRxPairStateKey

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


def get_unique_exp_id_simult_num_pairs(
    state: TxRxPairState,
) -> list[tuple[types.ExperimentId, types.SimultaneousNum]]:
    _K = TxRxPairStateKey

    unique_exp_id_simult_num_pairs: list[tuple[types.ExperimentId, types.SimultaneousNum]] = (
        state.index.droplevel(_K.time).unique().to_list()
    )

    return unique_exp_id_simult_num_pairs


def filter_by_exp_id_simult_num(
    state: TxRxPairState, exp_id: types.ExperimentId, simult_num: types.SimultaneousNum
) -> TxRxPairState:
    _K = TxRxPairStateKey

    mask = (
        (state.index.get_level_values(_K.exp_num) >= exp_id)
        & (state.index.get_level_values(_K.rx_simult_num) <= simult_num)
    ) # fmt: skip
    state_masked = state[mask]

    return state_masked


def group_by_unique_exp_id_simult_num_pairs(
    state: TxRxPairState,
) -> dict[tuple[types.ExperimentId, types.SimultaneousNum], TxRxPairState]:
    unique_exp_id_simult_num_pairs = get_unique_exp_id_simult_num_pairs(state)

    state_groups = {
        (exp_id, simult_num): filter_by_exp_id_simult_num(state, exp_id, simult_num)
        for exp_id, simult_num in unique_exp_id_simult_num_pairs
    }

    return state_groups
