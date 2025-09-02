"""
Functions for core functionality of the subpackage

- Intended to be imported as a whole module when consuming
"""

from __future__ import annotations
import logging, typing as t
import numpy.typing as npt
import sorts
from tqdm import tqdm
from sorts.interpolation import Interpolator
from sorts.radar.tx_rx import StationId
from sorts.types import Float64_as_sec, EcefStates, Datetime64_us
from sorts.schedule_v2.schedule import Schedule, TimeRangeIndexer
from sorts.simulation_v2 import passage
from sorts.simulation_v2.simulation_unit import SimulationUnit
from sorts.simulation_v2.observation import ObservationIndexer

if t.type_check_only:
    from sorts.simulation_v2.stx_mrx_simulation.model import (
        Spec,
        SpaceObjectDsecSampler,
    )


logger = logging.getLogger(__name__)


def sample_and_propagate_pace_objects_states(
    sampler: SpaceObjectDsecSampler,
    spobjs: t.Sequence[sorts.SpaceObject],
    start_time: Datetime64_us,
    end_time: Datetime64_us,
) -> tuple[list[npt.NDArray[Float64_as_sec]], list[EcefStates]]:
    """
    Use the sampler to get the delta time of space object within the simulation `start_time` and `end_time`,
    then get the space object states at those delta time using the propagator in the space object.

    Returns a list of sampled delta seconds and a list of corresponding states.
    """

    spobjs_smpl_dsec: list[npt.NDArray[Float64_as_sec]] = []
    for spobj in tqdm(spobjs, total=len(spobjs)):
        spobjs_smpl_dsec.append(sampler(spobj.state, start_time, end_time))

    spobjs_smpl_states: list[EcefStates] = []
    for spobj, spobj_smpl_dsec in tqdm(zip(spobjs, spobjs_smpl_dsec), total=len(spobjs)):
        spobjs_smpl_states.append(spobj.get_state(spobj_smpl_dsec))

    return spobjs_smpl_dsec, spobjs_smpl_states


def derive_schedule_indexers_per_tx_rx_station_pair(
    passages: list[passage.Passage],
) -> dict[tuple[StationId, StationId], list[TimeRangeIndexer]]:
    """Derive a list of schedule indexer for each tx-rx station pair found in the give passages"""

    sch_indexers: dict[tuple[StationId, StationId], list[TimeRangeIndexer]] = {}

    for passage in passages:
        tx_station_id = passage["tx_station"].uid
        rx_station_id = passage["rx_station"].uid

        if (tx_station_id, rx_station_id) in sch_indexers:
            sch_indexers[(tx_station_id, rx_station_id)].append(passage["time_range"])
        else:
            sch_indexers[(tx_station_id, rx_station_id)] = [passage["time_range"]]

    return sch_indexers


def derive_observation_indexers(
    passage: passage.Passage,
    tx_sch: Schedule,
    rx_schs: t.Sequence[Schedule],
) -> list[ObservationIndexer]:
    """Derive an indexer for each observation"""

    _K = Schedule._K

    obs_indexers: list[ObservationIndexer] = []

    rx_schedule_map: dict[str, Schedule] = {sch._data.attrs[_K.stn_id]: sch for sch in rx_schs}

    tx_obs_idxers = tx_sch.filter_by_time_range(passage["time_range"]).get_indexer_per_measurement(
        is_split_simu=False
    )

    rx_sch = rx_schedule_map[passage["rx_station"].uid]
    rx_obs_idxers = rx_sch.filter_by_time_range(passage["time_range"]).get_indexer_per_measurement(
        is_split_simu=True
    )

    for tx_obs_idxer in tx_obs_idxers:
        for rx_obs_idxer in rx_obs_idxers:
            obs_indexers.append({"tx_indexer": tx_obs_idxer, "rx_indexer": rx_obs_idxer})

    return obs_indexers


def find_passages(
    spec: Spec,
    spobjs_smpl_dsec: list[npt.NDArray[Float64_as_sec]],
    spobjs_smpl_states: list[EcefStates],
) -> list[list[passage.Passage]]:
    """
    Find passages for each space objects over the simulation period.

    Returns a `list[Passage]` per space object.
    """

    passages_list: list[list[passage.Passage]] = []

    for spobj, spobj_smpl_dsec, spobj_smpl_states in zip(
        spec["space_objects"],
        spobjs_smpl_dsec,
        spobjs_smpl_states,
    ):
        passages_of_spobj: list[passage.Passage] = []

        for rx_station in spec["rx_stations"]:
            passages_of_spobj.extend(
                passage.find_passages(
                    dt=spobj_smpl_dsec,
                    space_object=spobj,
                    states=spobj_smpl_states,
                    tx_station=spec["tx_station"],
                    rx_station=rx_station,
                    epoch=spec["epoch"],
                )
            )

        passages_list.append(passages_of_spobj)

    return passages_list


# TODO: minor cleanup needed
#   - `enumerate` to get space_objects by `idx` can likely be simplified, with small adj in params/props
#   - the loops might be simplified a bit as well
def derive_simulation_units(
    spec: Spec,
    passages_lists: list[list[passage.Passage]],
    spobjs_interpolators: list[Interpolator],
):
    sim_units: list[SimulationUnit] = []

    for idx, (passages, spobj_states_interp) in enumerate(
        zip(passages_lists, spobjs_interpolators)
    ):
        indexers_dict = derive_schedule_indexers_per_tx_rx_station_pair(passages)

        for stn_id_pair, indexers in indexers_dict.items():
            rx_stn_idx = [stn.uid for stn in spec["rx_stations"]].index(stn_id_pair[1])

            sim_unit = SimulationUnit.from_passages_over_tx_rx_station_pair(
                indexers=indexers,
                spobj=spec["space_objects"][idx],
                spobj_interp=spobj_states_interp,
                tx_stn=spec["tx_station"],
                rx_stn=spec["rx_stations"][rx_stn_idx],
                tx_sch=spec["tx_schedule"],
                rx_sch=spec["rx_schedules"][rx_stn_idx],
            )
            sim_units.append(sim_unit)

    return sim_units
