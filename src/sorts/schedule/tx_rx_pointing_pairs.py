from __future__ import annotations
import logging, typing as t, enum
import pandas as pd
from sorts import radar, passage
from .schedule_db import ScheduleDb


logger = logging.getLogger(__name__)


class TxRxPointingPairsKey(enum.StrEnum):
    exp_num = "exp_num"
    rx_simult_num = "rx_simult_num"
    time = "time"
    tx_pointing_e = "tx_pointing_e"
    tx_pointing_n = "tx_pointing_n"
    tx_pointing_u = "tx_pointing_u"
    rx_pointing_e = "rx_pointing_e"
    rx_pointing_n = "rx_pointing_n"
    rx_pointing_u = "rx_pointing_u"


TxRxPointingPairs = t.NewType("TxRxPointingPairs", pd.DataFrame)
"""
A pandas `Dataframe` with
```
Columns:
    exp_num        int16
    rx_simult_num  int16
    time           datetime64[us]
    tx_pointing_e  float64
    tx_pointing_n  float64
    tx_pointing_u  float64
    rx_pointing_e  float64
    rx_pointing_n  float64
    rx_pointing_u  float64
```
"""


def from_schedule_db_stn_id_pair_passages(
    stn_id_pair: tuple[radar.StationId, radar.StationId],
    passages: list[passage.Passage],
    schedule_db: ScheduleDb,
) -> TxRxPointingPairs:
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
        by=TxRxPointingPairsKey.time, ascending=True
    )

    return TxRxPointingPairs(tx_rx_pointing_pairs)
