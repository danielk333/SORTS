from __future__ import annotations
import logging
import pandas as pd
from sorts import radar, passage, schedule
from . import types, schedule_dataframe, schedule_db


logger = logging.getLogger(__name__)


def from_schedule_db_stn_id_pair_passages(
    schedule_db: schedule_db.ScheduleDb,
    stn_id_pair: tuple[radar.StationId, radar.StationId],
    passages: list[passage.Passage],
) -> types.TxRxPointingPairs:
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
        by=types.TxRxPointingPairsKey.time, ascending=True, ignore_index=True
    )

    return types.TxRxPointingPairs(tx_rx_pointing_pairs)


def from_schedule_dataframe_stn_id_pair_passages(
    sch: schedule_dataframe.ScheduleDataframe,
    stn_id_pair: tuple[radar.StationId, radar.StationId],
    passages: list[passage.Passage],
) -> types.TxRxPointingPairs:
    """
    Create `TxRxPointingPairs` from a list of `Passage`,
    sorted by `[time, rx_simult_num, exp_num]` in ascending order.
    """

    _K = types.TxRxPointingPairsKey

    tx_rx_pointing_pairs = pd.concat(
        [
            schedule_dataframe.get_tx_rx_pointing_pairs(
                sch=sch,
                start_time=ps.time_range[0],
                end_time=ps.time_range[1],
                tx_stn_num=stn_id_pair[0],
                rx_stn_num=stn_id_pair[1],
            )
            for ps in passages
        ]
    )
    tx_rx_pointing_pairs = tx_rx_pointing_pairs.sort_values(
        by=[_K.time, _K.rx_simult_num, _K.exp_num], ascending=True, ignore_index=True
    )

    return types.TxRxPointingPairs(tx_rx_pointing_pairs)


def gather_from_passages_schedule_dataframe(
    passages: list[passage.Passage],
    sch: schedule.ScheduleDataframe,
) -> dict[tuple[radar.StationId, radar.StationId], schedule.TxRxPointingPairs]:
    """
    Find the unique tx-rx station pairs among the `passages`,
    then for each pair, gather a `TxRxPointingPairs` from the schedule when the passages pass over the them.
    """

    passages_by_tx_rx_stn_pair = passage.group_passages_by_tx_rx_station_pair(passages)

    pointing_pairs_dict = {
        stn_id_pair: schedule.tx_rx_pointing_pairs.from_schedule_dataframe_stn_id_pair_passages(
            sch=sch,
            stn_id_pair=stn_id_pair,
            passages=passages,
        )
        for stn_id_pair, passages in passages_by_tx_rx_stn_pair.items()
    }

    return pointing_pairs_dict
