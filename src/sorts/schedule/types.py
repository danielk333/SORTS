"""Shared types in this package."""

import enum
import logging, typing as t, enum
import pandas as pd


logger = logging.getLogger(__name__)


class ScheduleKey(enum.StrEnum):
    index = "index"  # type: ignore ; seems type checker might confuse this with the `index` method from `str`
    exp_num = "exp_num"
    stn_num = "stn_num"
    simult_num = "simult_num"
    start_time = "start_time"
    end_time = "end_time"
    pointing_e = "pointing_e"
    pointing_n = "pointing_n"
    pointing_u = "pointing_u"


class ScheduleValidationError(Exception):
    pass


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
