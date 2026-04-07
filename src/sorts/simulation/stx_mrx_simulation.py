from __future__ import annotations
import logging, typing as t
import numpy as np
from sorts import types, utils, space_object, schedule, passage
from sorts.radar import Station, StationId
from sorts.interpolated_propagation import InterpolatedPropagation
from sorts.simulation import tx_rx_pair_state

logger = logging.getLogger(__name__)


def simulate(
    space_object: space_object.SpaceObject,
    interpolated_propagation: InterpolatedPropagation,
    passages: list[passage.Passage],
    sch: schedule.ScheduleDataframe,
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

    txrx_state_dict = tx_rx_pair_state.gather_from_passages_schedule_dataframe(
        passages=passages,
        sch=sch,
    )

    sim_result: list[tx_rx_pair_state.TxRxPairState] = []
    for stn_id_pair, txrx_state in txrx_state_dict.items():
        dsec = (txrx_state[_K.time].to_numpy() - epoch) / np.timedelta64(1, "s")
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


def simulate_with_schedule_db(
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

    pointing_pairs_dict = schedule_db.gather_pointing_pairs(passages)
    txrx_state_dict = {
        key: tx_rx_pair_state.from_tx_rx_pointing_pairs(pointing_pairs)
        for key, pointing_pairs in pointing_pairs_dict.items()
    }

    sim_result: list[tx_rx_pair_state.TxRxPairState] = []
    for stn_id_pair, txrx_state in txrx_state_dict.items():
        dsec = (txrx_state[_K.time].to_numpy() - epoch) / np.timedelta64(1, "s")
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
