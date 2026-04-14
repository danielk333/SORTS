"""Shared types in this package."""

from __future__ import annotations
import logging, typing as t, enum
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas._typing as pdt
from sorts import types, utils, radar, passage, schedule


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
ScheduleDataframe = t.NewType("ScheduleDataframe", pd.DataFrame)
"""
A pandas `DataFrame` with:
```
Index: RangeIndex # without name or named as "index"
Cols:
    exp_num     np.int16
    stn_num     np.int16
    simult_num  np.int16
    start_time  datetime64[us]
    end_time    datetime64[us]
    pointing_e  np.float64
    pointing_n  np.float64
    pointing_u  np.float64
```

The keys are available as enum `ScheduleKey` for consistent access.
"""

scheduleDataframeDtypes: t.Final[dict[t.Hashable, pdt.Dtype]] = {
    ScheduleKey.exp_num: "int16",
    ScheduleKey.stn_num: "int16",
    ScheduleKey.simult_num: "int16",
    ScheduleKey.start_time: "datetime64[us]",
    ScheduleKey.end_time: "datetime64[us]",
    ScheduleKey.pointing_e: "float64",
    ScheduleKey.pointing_n: "float64",
    ScheduleKey.pointing_u: "float64",
}
"""
The dtypes of a `ScheduleDataframe` expressed in a python dict.
Useful for certain pandas IO methods.
"""


def validate(df: pd.DataFrame, allow_extra_cols=False) -> ScheduleDataframe:
    """
    Validate a pandas `DataFrame` against the definition of `ScheduleDataframe`.

    See the docs of `ScheduleDataframe` for its definition.

    Args:
        allow_extra_cols:
            If `True`, allow extra columns that is not in the definition of `ScheduleDataframe`. Defaults to `False.

    Returns:
        The original DataFrame casted into a `ScheduleDataframe`.

    Raises:
        `ScheduleValidationError`
    """

    _K = ScheduleKey

    if not (df.index.name is None or df.index.name == _K.index):
        raise ScheduleValidationError("DataFrame index name is not `None` or `'index'`")

    if allow_extra_cols:
        if not {key for key in _K if key != _K.index}.issubset(df.columns):
            raise ScheduleValidationError("One or more column is missing from the DataFrame")
    else:
        if not {key for key in _K if key != _K.index} == set(df.columns):
            raise ScheduleValidationError(
                "Columns not identical to the `ScheduleDataframe` definition"
            )

    if df.dtypes[_K.exp_num] != "int16":
        raise ScheduleValidationError("Column 'exp_num' is not numpy dtype 'int16'")
    if df.dtypes[_K.stn_num] != "int16":
        raise ScheduleValidationError("Column 'stn_num' is not numpy dtype 'int16'")
    if df.dtypes[_K.simult_num] != "int16":
        raise ScheduleValidationError("Column 'simult_num' is not numpy dtype 'int16'")

    if df.dtypes[_K.start_time] != "datetime64[us]":
        raise ScheduleValidationError("Column 'start_time' is not numpy dtype 'datetime64[us]'")
    if df.dtypes[_K.end_time] != "datetime64[us]":
        raise ScheduleValidationError("Column 'end_time' is not numpy dtype 'datetime64[us]'")

    if df.dtypes[_K.pointing_e] != "float64":
        raise ScheduleValidationError("Column 'pointing_e' is not numpy dtype 'float64'")
    if df.dtypes[_K.pointing_n] != "float64":
        raise ScheduleValidationError("Column 'pointing_n' is not numpy dtype 'float64'")
    if df.dtypes[_K.pointing_u] != "float64":
        raise ScheduleValidationError("Column 'pointing_u' is not numpy dtype 'float64'")

    return ScheduleDataframe(df)


def from_rows(rows: t.Sequence[t.Sequence[t.Any]]) -> ScheduleDataframe:
    """Create a `ScheduleDataframe` from rows of data."""

    # NOTE: Constructing `DataFrame` from `Series` seems to be the only safe way to ensure
    #       the datetime resolution is not overrided into `'ns'` from pandas's type infer attempt.

    if len(rows) == 0:
        cols = [[] for key in ScheduleKey if key != ScheduleKey.index]
    else:
        cols = list(zip(*rows))

    df = pd.DataFrame(
        {
            ScheduleKey.exp_num: pd.Series(cols[0], dtype=np.int16),
            ScheduleKey.stn_num: pd.Series(cols[1], dtype=np.int16),
            ScheduleKey.simult_num: pd.Series(cols[2], dtype=np.int16),
            ScheduleKey.start_time: pd.Series(cols[3], dtype="datetime64[us]"),
            ScheduleKey.end_time: pd.Series(cols[4], dtype="datetime64[us]"),
            ScheduleKey.pointing_e: pd.Series(cols[5], dtype=np.float64),
            ScheduleKey.pointing_n: pd.Series(cols[6], dtype=np.float64),
            ScheduleKey.pointing_u: pd.Series(cols[7], dtype=np.float64),
        }
    )

    return validate(df)


def from_series(
    exp_num: pd.Series,
    stn_num: pd.Series,
    simult_num: pd.Series,
    start_time: pd.Series,
    end_time: pd.Series,
    pointing_e: pd.Series,
    pointing_n: pd.Series,
    pointing_u: pd.Series,
) -> ScheduleDataframe:
    """Create an `ScheduleDataframe` from columns of pandas `Series`."""

    df = pd.DataFrame(
        {
            ScheduleKey.exp_num: exp_num,
            ScheduleKey.stn_num: stn_num,
            ScheduleKey.simult_num: simult_num,
            ScheduleKey.start_time: start_time,
            ScheduleKey.end_time: end_time,
            ScheduleKey.pointing_e: pointing_e,
            ScheduleKey.pointing_n: pointing_n,
            ScheduleKey.pointing_u: pointing_u,
        }
    )

    return validate(df)


def from_ndarrays(
    exp_num: npt.NDArray,
    stn_num: npt.NDArray,
    simult_num: npt.NDArray,
    start_time: npt.NDArray,
    end_time: npt.NDArray,
    pointing_e: npt.NDArray,
    pointing_n: npt.NDArray,
    pointing_u: npt.NDArray,
) -> ScheduleDataframe:
    """Create an empty `ScheduleDataframe` from columns of numpy `ndarray`."""

    df = from_series(
        exp_num=pd.Series(exp_num, dtype=np.int16),
        stn_num=pd.Series(stn_num, dtype=np.int16),
        simult_num=pd.Series(simult_num, dtype=np.int16),
        start_time=pd.Series(start_time, dtype="datetime64[us]"),
        end_time=pd.Series(end_time, dtype="datetime64[us]"),
        pointing_e=pd.Series(pointing_e, dtype=np.float64),
        pointing_n=pd.Series(pointing_n, dtype=np.float64),
        pointing_u=pd.Series(pointing_u, dtype=np.float64),
    )

    return validate(df)


def empty() -> ScheduleDataframe:
    """Create an empty `ScheduleDataframe`"""

    return from_rows([])


def filter_by_time_range(
    sch: ScheduleDataframe,
    start_time: types.Datetime64_us,
    end_time: types.Datetime64_us,
) -> ScheduleDataframe:
    _K = ScheduleKey

    mask = (
        (sch[_K.start_time] >= start_time)
        & (sch[_K.end_time] < end_time)
    ) # fmt: skip
    state_masked = sch[mask]

    return state_masked


def schedule_by_priority(schs: list[ScheduleDataframe], priorities: list[int]) -> ScheduleDataframe:
    """
    Generate a combined schedule for the specified schedules.
    which defaults to `self.combined_schedule_name`.

    Args:
        schedules: The list of schedules to combine.
        priorities: The list of priority corresponding to the table names.
            Must have the same length as the `schedules` param.

    Returns:
        The resultant schedule.
    """

    _K = ScheduleKey

    class _SK(enum.StrEnum):
        """additinoal string keys that is internal to this func."""

        priority = "priority"

    if len(schs) != len(priorities):
        raise RuntimeError("`priorities` does not have the same length as the `schedules` param.")

    # add priority column
    schs_with_pri = [sch.copy() for sch in schs]  # make shallow copies of the schs
    for sch, pri in zip(schs_with_pri, priorities):
        sch[_SK.priority] = pri

    # lump all the schedules into one
    master_sch = pd.concat(schs_with_pri, ignore_index=True)

    start_time_arr = master_sch[_K.start_time].to_numpy()
    end_time_arr = master_sch[_K.end_time].to_numpy()
    priority_arr = master_sch[_SK.priority].to_numpy()

    # TODO: re-eval if filtering operations can be optimized, e.g.:
    #     - losers_mask can be calculated only for overlapped rows?
    #     - `same_stn_mask`, `same_exp_mask` are symmetric, maybe some shortcut can be taken?

    overlap_mask = time_overlapped_mask(
        start_time=start_time_arr, end_time=end_time_arr, ignore_self_comparison=True
    )

    losers_mask = priority_loser_mask(priority=priority_arr)

    stn_num_arr = master_sch[_K.stn_num].to_numpy()
    same_stn_mask = stn_num_arr[:, np.newaxis] == stn_num_arr[np.newaxis, :]

    combined_mask = overlap_mask & same_stn_mask & losers_mask

    # flatten the mask by doing an "or" per row
    flattened_mask = np.any(combined_mask, axis=1)
    flattened_mask = t.cast(npt.NDArray[np.bool], flattened_mask)

    # all rows of the same time and exp are always removed togather
    #
    # we interpret `remove_togather_mask` as [nth row's impact to other rows, ...]
    # so the resultant impact is taken by collapsing withing a column (i.e. across rows)
    remove_togather_mask = same_exp_and_time_mask(
        exp_num=master_sch[_K.exp_num].to_numpy(), time=master_sch[_K.start_time].to_numpy()
    )
    flattened_mask = np.any(flattened_mask[:, np.newaxis] & remove_togather_mask, axis=0)
    flattened_mask = t.cast(npt.NDArray[np.bool], flattened_mask)

    resultant_sch = master_sch[~flattened_mask]
    resultant_sch = resultant_sch.drop(columns=[_SK.priority])  # remove priority column
    resultant_sch = resultant_sch.sort_values(
        by=[_K.start_time, _K.simult_num, _K.stn_num, _K.exp_num], ascending=True, ignore_index=True
    )

    return validate(resultant_sch)


def time_overlapped_mask(
    start_time: npt.NDArray[types.Datetime64_us],
    end_time: npt.NDArray[types.Datetime64_us],
    ignore_self_comparison=True,
) -> types.NDArray_NxN[np.bool]:
    """
    For each (`start_time`, `end_time`) row at equal index,
    check if it is overlapped with other rows in time.

    `start_time`, `end_time` must have equal length.

    Args:
        ignore_self_comparison:
            Self-comparison is always True/overlapped.
            It is therefore general not useful as hard-coded to `False` by default.
            Set this flag to `False` to disable such hard-coding.

    Returns:
        A (N, N) bool ndarray.
    """

    # inject new dimension to ndarray for broadcasting
    starts = start_time[:, np.newaxis]  # shape (N, 1)
    ends = end_time[np.newaxis, :]  # shape (1, N)

    # check conditions for overlap
    mask1 = starts < ends  # shape (N, N)
    mask2 = ends.T > starts.T  # shape (N, N)
    overlap_mask = mask1 & mask2

    if ignore_self_comparison:
        # remove self-comparison (i == j)
        np.fill_diagonal(overlap_mask, False)

    return overlap_mask


def priority_loser_mask(priority: npt.NDArray[np.int64]) -> types.NDArray_NxN[np.bool]:
    """
    For each `priorities` entry, compare its priority with other entries:

        - priority: lower number -> higher priority
        - entry order: lower number -> higher priority
    """

    # inject new dimension to ndarray for broadcasting
    # priorities
    pri_i = priority[:, np.newaxis]  # shape (N,1)
    pri_j = priority[np.newaxis, :]  # shape (1,N)

    # inject new dimension to ndarray for broadcasting
    # entry id
    rid_i = np.arange(len(priority))[:, np.newaxis]  # shape (N,1)
    rid_j = np.arange(len(priority))[np.newaxis, :]  # shape (1,N)

    # find losers of priorities. entry i loses to entry j if:
    # 1. j has lower priority number
    # 2. or same priority but lower rid
    losers_mask = (pri_i > pri_j) | ((pri_i == pri_j) & (rid_i > rid_j))

    return losers_mask


def same_exp_and_time_mask(
    exp_num: npt.NDArray[np.signedinteger],
    time: npt.NDArray[types.Datetime64_us],
) -> types.NDArray_NxN[np.bool]:
    """
    For each (`exp_num`, `time`) row at equal index,
    check if it has the same values as other rows.

    `exp_num`, `time` must have equal length.

    Returns:
        A (N, N) bool ndarray.
    """

    mask = exp_num[:, np.newaxis] == exp_num[np.newaxis, :]
    mask &= time[:, np.newaxis] == time[np.newaxis, :]

    return mask


def get_tx_rx_pointing_pairs(
    sch: ScheduleDataframe,
    start_time: types.Datetime_Like,
    end_time: types.Datetime_Like,
    tx_stn_num: int,
    rx_stn_num: int,
) -> TxRxPointingPairs:
    """Get pointing pairs from DB as specified by param."""

    _SK = ScheduleKey
    _PK = TxRxPointingPairsKey

    start_time_dt64 = utils.to_datetime64_us(start_time)
    end_time_dt64 = utils.to_datetime64_us(end_time)

    # get tx_sch by filtering by time and stn_num on input sch
    tx_sch = filter_by_time_range(sch, start_time_dt64, end_time_dt64)
    tx_sch = tx_sch[tx_sch[_SK.stn_num] == tx_stn_num]

    # get rx_sch by filtering by time and stn_num on input sch
    rx_sch = filter_by_time_range(sch, start_time_dt64, end_time_dt64)
    rx_sch = rx_sch[rx_sch[_SK.stn_num] == rx_stn_num]

    # prepare the cols and index of tx_sch with rx_sch, then join them
    tx_sch = tx_sch.rename(
        columns={
            _SK.pointing_e: _PK.tx_pointing_e,
            _SK.pointing_n: _PK.tx_pointing_n,
            _SK.pointing_u: _PK.tx_pointing_u,
        }
    )
    tx_sch = tx_sch.set_index([_SK.exp_num, _SK.start_time, _SK.end_time])
    tx_sch = tx_sch.drop(columns=[_SK.stn_num, _SK.simult_num])

    rx_sch = rx_sch.rename(
        columns={
            _SK.simult_num: _PK.rx_simult_num,
            _SK.pointing_e: _PK.rx_pointing_e,
            _SK.pointing_n: _PK.rx_pointing_n,
            _SK.pointing_u: _PK.rx_pointing_u,
        }
    )
    rx_sch = rx_sch.set_index([_SK.exp_num, _SK.start_time, _SK.end_time])
    rx_sch = rx_sch.drop(columns=[_SK.stn_num])

    df = tx_sch.join(rx_sch, how="inner")
    df = df.reset_index()  # put multiindex cols back to df body

    # rename col start_time to time, and drop col end_time
    df = df.rename(columns={_SK.start_time: _PK.time})
    df = df.drop(columns=[_SK.end_time])

    return TxRxPointingPairs(df)


def gather_pointing_pairs(
    passages: list[passage.Passage],
    sch: schedule.ScheduleDataframe,
) -> dict[tuple[radar.StationId, radar.StationId], TxRxPointingPairs]:
    """
    Find the unique tx-rx station pairs among the `passages`,
    then for each pair, gather a `TxRxPointingPairs` from the schedule when the passages pass over the them.

    The returned `TxRxPointingPairs` are sorted by time in ascending order.
    """

    passages_by_tx_rx_stn_pair = passage.group_passages_by_tx_rx_station_pair(passages)

    pointing_pairs_dict: dict[
        tuple[radar.StationId, radar.StationId], schedule.TxRxPointingPairs
    ] = {}
    for stn_id_pair, passages in passages_by_tx_rx_stn_pair.items():
        pairs_ls = [
            get_tx_rx_pointing_pairs(
                sch=sch,
                start_time=ps.time_range[0],
                end_time=ps.time_range[1],
                tx_stn_num=stn_id_pair[0],
                rx_stn_num=stn_id_pair[1],
            )
            for ps in passages
        ]

        pairs = TxRxPointingPairs(pd.concat(pairs_ls))
        pairs = pairs.sort_values(by=TxRxPointingPairsKey.time, ascending=True, ignore_index=True)

        pointing_pairs_dict[stn_id_pair] = pairs

    return pointing_pairs_dict
