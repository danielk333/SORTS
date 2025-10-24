"""
Functions for core functionalities of this subpackage

- Intended to be imported as a whole module when consuming
"""

from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import sorts
from tqdm import tqdm
from sorts.interpolation import Interpolator
from sorts.radar import Station, StationId
from sorts.types import Float64_as_sec, EcefStates, Datetime64_us, EnuCoordinates
from sorts.schedule import Schedule
from sorts.simulation.types import Passage
from sorts.simulation import funcs
from .simulation_unit import SimulationUnit, Observation


if t.TYPE_CHECKING:
    from .simulation_unit import State
    from .simulation_unit import FromPassagesOverTxRxStationPairParam
    from .stx_mrx_simulation import Spec, SpaceObjectDsecSampler


logger = logging.getLogger(__name__)


# TODO: use `TxRxTuple` type for return value?
# TODO: can be combined with 'stx_mrx_simulation.funcs.find_passages'?
def group_passages_by_tx_rx_station_pair(
    passages: list[Passage],
) -> dict[tuple[StationId, StationId], list[Passage]]:
    groupped_passages: dict[tuple[StationId, StationId], list[Passage]] = {}

    for passage in passages:
        tx_station_id = passage["tx_station"].uid
        rx_station_id = passage["rx_station"].uid

        if (tx_station_id, rx_station_id) in groupped_passages:
            groupped_passages[(tx_station_id, rx_station_id)].append(passage)
        else:
            groupped_passages[(tx_station_id, rx_station_id)] = [passage]

    return groupped_passages


def find_passages(
    spec: Spec,
    spobjs_smpl_dsec: list[npt.NDArray[Float64_as_sec]],
    spobjs_smpl_states: list[EcefStates],
) -> list[list[Passage]]:
    """
    Find passages for each space objects over the simulation period.

    Returns a `list[Passage]` per space object.
    """

    passages_list: list[list[Passage]] = []

    for spobj, spobj_smpl_dsec, spobj_smpl_states in zip(
        spec["space_objects"],
        spobjs_smpl_dsec,
        spobjs_smpl_states,
    ):
        passages_of_spobj: list[Passage] = []

        for stn_id_pair in spec["station_id_pairs"]:
            tx_stn = spec["station_map"][stn_id_pair[0]]
            rx_stn = spec["station_map"][stn_id_pair[1]]

            passages_of_spobj.extend(
                funcs.find_passages(
                    dt=spobj_smpl_dsec,
                    space_object=spobj,
                    states=spobj_smpl_states,
                    tx_station=tx_stn,
                    rx_station=rx_stn,
                    epoch=spec["epoch"],
                )
            )

        passages_list.append(passages_of_spobj)

    return passages_list


def derive_simulation_unit_params(
    spec: Spec,
    passages_lists: list[list[Passage]],
    spobjs_interpolators: list[Interpolator],
) -> list[FromPassagesOverTxRxStationPairParam]:
    """
    Derive a list of param for the `from_passages_over_tx_rx_station_pair` constructor of `SimulationUnit`

    NOTE: Integers (casted to `str`) are used as `SimulationUnit`s' id
    """

    params: list[FromPassagesOverTxRxStationPairParam] = []

    for spobj, passages_of_a_spobj, spobj_states_interp in zip(
        spec["space_objects"], passages_lists, spobjs_interpolators
    ):
        groupped_passages = group_passages_by_tx_rx_station_pair(passages_of_a_spobj)

        for stn_id_pair, passages in groupped_passages.items():
            tx_stn = spec["station_map"][stn_id_pair[0]]
            rx_stn = spec["station_map"][stn_id_pair[1]]

            filtered_sch = spec["schedule"].filter_by_time_ranges(
                [ps["time_range"] for ps in passages]
            )

            params.append(
                {
                    "id": str(len(params)),
                    "passages": passages,
                    "spobj": spobj,
                    "spobj_interp": spobj_states_interp,
                    "tx_station": tx_stn,
                    "rx_station": rx_stn,
                    "schedule": filtered_sch,
                    "exp_detail_map": spec["exp_detail_map"],
                }
            )

    return params
