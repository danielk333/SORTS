from __future__ import annotations
import logging, typing as t
import pandas as pd
from tqdm import tqdm
import sorts
from sorts import types, radar, schedule, controller, passage, simulation
from sorts.types import Datetime_Like
from sorts.radar import Station, StationId
from sorts.interpolated_propagation import InterpolatedPropagation
from sorts.simulation import tx_rx_pair_state

logger = logging.getLogger(__name__)


# TODO: this can be converted into a dataclass (or just be dissolved?)
# TODO: we need to enforce each station to has a unique id (`.uid` prop)
#   either in the simulation class or in related station getter like `get_radar`
class StxMrxSimulation:
    """
    NOTE: This is intended as an internal constructor, please use the constructor methods to create instances.
    """

    def __init__(
        self,
        station_map: dict[StationId, Station],
        station_id_pairs: t.Sequence[tuple[StationId, StationId]],
        schedule_db: schedule.ScheduleDb,
        exp_detail_map: types.ExperimentDetailMap,
        epoch: Datetime_Like,
        start_time: Datetime_Like,
        end_time: Datetime_Like,
        space_objects: t.Sequence[sorts.SpaceObject],
        interpolated_propagations: t.Sequence[InterpolatedPropagation],
        passages: list[passage.Passage],
        progress: bool = False,
    ):
        self.station_map = station_map
        self.station_id_pairs = station_id_pairs
        self.schedule_db = schedule_db
        self.exp_detail_map = exp_detail_map
        self.epoch = epoch
        self.start_time = start_time
        self.end_time = end_time
        self.space_objects = space_objects
        self.interpolated_propagations = interpolated_propagations
        self.progress = progress
        self.passages = passages

    @classmethod
    def from_controllers(
        cls,
        controllers: t.Sequence[controller.ControllerBase],
        schedule: schedule.ScheduleDb,
        epoch: Datetime_Like,
        start_time: Datetime_Like,
        end_time: Datetime_Like,
        space_objects: t.Sequence[sorts.SpaceObject],
        interpolated_propagations: t.Sequence[InterpolatedPropagation],
        passages: list[passage.Passage],
    ):
        """A constructor method"""
        # TODO: - the exp details are already computed outside? Should the `controllers` field be
        # removed? or this classmethod? or what?
        #
        # Notes from Hin, 2025-11-21:
        #   - both `exp_id_stn_id_pairs_map` and `ExperimentDetail` are currently owned by the controller;
        #   - the func `priority_scheduling` evolved to requires `exp_id_stn_id_pairs_map` at some point,
        #     and therefore it is sometimes found as an explicitly variable in simulation experiment file as well
        #   - we can re-work info flow later but this is needed atm

        schedule_db = schedule

        stn_map: dict[StationId, Station] = {}
        stn_id_pairs_set: set[tuple[StationId, StationId]] = set()
        exp_detail_map: types.ExperimentDetailMap = {}

        for ctrl in controllers:
            stn_map.update(ctrl.get_station_map())

            for pairs in ctrl.get_experiment_id_station_id_pairs_map().values():
                stn_id_pairs_set.update(pairs)

            exp_detail = ctrl.get_experiment_detail()
            exp_detail_map[exp_detail.id] = exp_detail

        return cls(
            station_map=stn_map,
            station_id_pairs=list(stn_id_pairs_set),
            schedule_db=schedule_db,
            exp_detail_map=exp_detail_map,
            epoch=epoch,
            start_time=start_time,
            end_time=end_time,
            space_objects=space_objects,
            interpolated_propagations=interpolated_propagations,
            passages=passages,
        )


# TODO: tmp; should be simplified
SimulationResult = t.NewType("SimulationResult", dict[int, list[tx_rx_pair_state.TxRxPairState]])
"""
`NewType` of `dict[int, list[tx_rx_pair_state.TxRxPairState]]`.
Indexed by space object index in spobj list (not `oid` of `SpaceObject`).
"""


def get_pointing_pairs_by_stn_id_pair_passages(
    stn_id_pair: tuple[radar.StationId, radar.StationId],
    passages: list[passage.Passage],
    schedule_db: schedule.ScheduleDb,
) -> schedule.TxRxPointingPairs:
    """Create `TxRxPointingPairs` from a list of `Passage`, sorted by time in ascending order."""

    tx_rx_pointing_pairs = pd.concat(
        [
            schedule_db.get_tx_rx_pointing_pairs(
                start_time=ps.time_range[0],
                end_time=ps.time_range[1],
                tx_stn_num=stn_id_pair[0],
                rx_stn_num=stn_id_pair[1],
            )
            for ps in passages
        ]
    )
    tx_rx_pointing_pairs = tx_rx_pointing_pairs.sort_values(
        by=simulation.TxRxPairStateKey.time, ascending=True
    )

    return schedule.TxRxPointingPairs(tx_rx_pointing_pairs)


def gather_tx_rx_pointing_pairs(
    passages: list[passage.Passage],
    schedule_db: schedule.ScheduleDb,
) -> dict[tuple[radar.StationId, radar.StationId], schedule.TxRxPointingPairs]:
    """
    Find the unique tx-rx station pairs among the `passages`,
    then for each pair, gather a `TxRxPointingPairs` from the schedule when the passages pass over the them.
    """

    passages_by_tx_rx_stn_pair = passage.group_passages_by_tx_rx_station_pair(passages)

    pointing_pairs_dict = {
        stn_id_pair: get_pointing_pairs_by_stn_id_pair_passages(
            stn_id_pair=stn_id_pair,
            passages=passages,
            schedule_db=schedule_db,
        )
        for stn_id_pair, passages in passages_by_tx_rx_stn_pair.items()
    }

    return pointing_pairs_dict


def gather_tx_rx_pair_state(
    passages: list[passage.Passage],
    schedule_db: schedule.ScheduleDb,
) -> dict[tuple[radar.StationId, radar.StationId], tx_rx_pair_state.TxRxPairState]:
    """
    Find the unique tx-rx station pairs among the `passages`,
    then for each pair, gather a `TxRxPairState` from the schedule when the passages pass over the them.
    """

    _K = tx_rx_pair_state.TxRxPairStateKey

    pointing_pairs_dict = gather_tx_rx_pointing_pairs(passages=passages, schedule_db=schedule_db)
    pair_state_dict = {
        key: tx_rx_pair_state.TxRxPairState(
            pointing_pairs.set_index([_K.exp_num, _K.rx_simult_num, _K.time])
        )
        for key, pointing_pairs in pointing_pairs_dict.items()
    }

    return pair_state_dict


def simulate(
    space_objects: t.Sequence[sorts.SpaceObject],
    interpolated_propagations: t.Sequence[InterpolatedPropagation],
    passages: list[passage.Passage],
    schedule_db: schedule.ScheduleDb,
    station_map: dict[StationId, Station],
    exp_detail_map: types.ExperimentDetailMap,
) -> list[list[tx_rx_pair_state.TxRxPairState]]:
    """
    Run a simulation for the list of space objects,
    over the specified passages and using the supplied propagations.

    `space_objects` and  `interpolated_propagations` should have the same length.

    Returns:
        A list of list of `TxRxPairState`.
        A list of `TxRxPairState` is generated for each space object.
    """

    sim_result: list[list[tx_rx_pair_state.TxRxPairState]] = []

    for spobj_idx in tqdm(range(len(space_objects)), desc="simulating"):
        sim_result.append([])

        pair_state_dict = gather_tx_rx_pair_state(
            # the same passage data is used for all perturbed objects
            passages=passages,
            schedule_db=schedule_db,
        )

        for stn_id_pair, pair_state in pair_state_dict.items():
            state = tx_rx_pair_state.simulate(
                state=pair_state,
                spobj=space_objects[spobj_idx],
                spobj_interp=interpolated_propagations[spobj_idx].interpolator,
                tx_station=station_map[stn_id_pair[0]],
                rx_station=station_map[stn_id_pair[1]],
                exp_detail_map=exp_detail_map,
            )

            sim_result[spobj_idx].append(state)

    return sim_result
