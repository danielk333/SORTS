from __future__ import annotations
import logging, typing as t
import numpy as np
from sorts import types, utils, space_object, radar, schedule, passage
from sorts.radar import Station, StationId
from sorts.interpolated_propagation import InterpolatedPropagation
from sorts.simulation import tx_rx_pair_state

logger = logging.getLogger(__name__)


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
        stn_id_pair: schedule.tx_rx_pointing_pairs.from_schedule_db_stn_id_pair_passages(
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
    space_object: space_object.SpaceObject,
    interpolated_propagation: InterpolatedPropagation,
    passages: list[passage.Passage],
    schedule_db: schedule.ScheduleDb,
    station_map: t.Mapping[StationId, Station],
    exp_detail_map: types.ExperimentDetailMap,
) -> list[tx_rx_pair_state.TxRxPairState]:
    """
    Run a simulation for the space object using the provided propagation, over the specified passages.
    """

    _K = tx_rx_pair_state.TxRxPairStateKey

    epoch = utils.to_datetime64_us(space_object.epoch)
    spobj_diameter = space_object.d
    # TODO: confirm with daniel if setting a default radar_albedo is okay
    spobj_radar_albedo = space_object.properties.get("radar_albedo", 1.0)

    txrx_state_dict = gather_tx_rx_pair_state(
        passages=passages,
        schedule_db=schedule_db,
    )

    sim_result: list[tx_rx_pair_state.TxRxPairState] = []
    for stn_id_pair, txrx_state in txrx_state_dict.items():
        dsec = (
            (txrx_state.index.get_level_values(_K.time).to_numpy() - epoch)
            / np.timedelta64(1, "s")
        ) # fmt: skip
        spobj_state = interpolated_propagation.interpolator.get_state(dsec)

        sim_result.append(
            tx_rx_pair_state.simulate(
                txrx_state=txrx_state,
                spobj_state=spobj_state,
                spobj_diameter=spobj_diameter,
                spobj_radar_albedo=spobj_radar_albedo,
                tx_station=station_map[stn_id_pair[0]],
                rx_station=station_map[stn_id_pair[1]],
                exp_detail_map=exp_detail_map,
            )
        )

    return sim_result
