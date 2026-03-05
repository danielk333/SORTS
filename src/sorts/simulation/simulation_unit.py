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
