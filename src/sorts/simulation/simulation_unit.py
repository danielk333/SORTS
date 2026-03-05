from __future__ import annotations
from dataclasses import dataclass
from sorts import types, passage, schedule
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.interpolation import Interpolator


# TODO: the docstring is copied from `SimulationUnit`, and needs update
@dataclass
class FromPassagesOverTxRxStationPairParam:
    """
    Contains all the params and results for a unit of simulation calculation.

    Notes about the state data:
    - It is stored in a private attribute `_state`
    - It can contain data for more than 1 passage
    - The dataset does not always contains all the columns,
        which ones are available depends on what calculation have been done.
    """

    id: str
    passages: list[passage.Passage]
    spobj: SpaceObject
    spobj_interp: Interpolator
    tx_station: Station
    rx_station: Station
    tx_rx_pointing_pairs: schedule.TxRxPointingPairs # TODO: this is a tmp solution, should refactor this type and dataflow; # fmt: skip
    exp_detail_map: types.ExperimentDetailMap
