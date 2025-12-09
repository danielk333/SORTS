from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from sorts.const import max_datetime64_us, min_datetime64_us
from sorts import utils
from . import schedule
from .schedule import Schedule


logger = logging.getLogger(__name__)


_SK = schedule._K

DsIntermediateVarKey = t.Literal[
    "cummax_start_time",
    "cummax_end_time",
    "is_overlaped",
]


class _IK:
    """Internal helper class for accessing string keys consistently"""

    cummax_start_time: t.Final = "cummax_start_time"
    cummax_end_time: t.Final = "cummax_end_time"
    is_overlaped: t.Final = "is_overlaped"


utils.assert_class_attributes_equal_to(_IK, t.get_args(DsIntermediateVarKey))


def _inject_intermediate_columns(sch: Schedule) -> Schedule:
    sch[_IK.cummax_start_time] = xr.full_like(
        sch[_SK.multi_index], np.datetime64("NaT"), dtype="datetime64[us]"
    )
    sch[_IK.cummax_end_time] = xr.full_like(
        sch[_SK.multi_index], np.datetime64("NaT"), dtype="datetime64[us]"
    )
    sch[_IK.is_overlaped] = xr.full_like(sch[_SK.multi_index], False, dtype=np.bool)

    return sch


def _propagate_cummax_start_time_cummax_end_time(merged_sch: Schedule) -> Schedule:
    # todo: why is this needed? need some more logic explanation
    merged_sch[_IK.cummax_start_time] = (
        t.cast(pd.Series, merged_sch[_IK.cummax_start_time].to_pandas())
        .groupby(level=[_SK.stn_num])
        .ffill()
    )
    merged_sch[_IK.cummax_end_time] = (
        t.cast(pd.Series, merged_sch[_IK.cummax_end_time].to_pandas())
        .groupby(level=[_SK.stn_num])
        .ffill()
    )

    return merged_sch


def _remove_entries_with_time_clash(merged_sch_: Schedule, incoming_sch: Schedule) -> Schedule:
    # NOTE:
    #   - the `allowed_start_time` has the previous `end_time`;
    #     for top values that have no corresponding pervious values, `min_datetime64_us` is used
    #   - the `allowed_end_time` has the next `start_time`;
    #     for bottom values that have no corresponding next values, `max_datetime64_us` is used
    allowed_start_time = (
        t.cast(pd.Series, merged_sch_[_IK.cummax_end_time].to_pandas())
        .groupby(level=_SK.stn_num)
        .shift(+1, fill_value=min_datetime64_us)
    )
    allowed_end_time = (
        t.cast(pd.Series, merged_sch_[_IK.cummax_start_time].to_pandas())
        .groupby(level=_SK.stn_num)
        .shift(-1, fill_value=max_datetime64_us)
    )

    # calc bool mask for overlapping just for the new/incoming entries
    is_over_allowed_start_time = merged_sch_[_SK.start_time].loc[
        {_SK.multi_index: incoming_sch[_SK.multi_index]}
    ] < allowed_start_time.reindex(incoming_sch[_SK.multi_index].to_pandas().index)

    is_over_allowed_end_time = merged_sch_[_SK.end_time].loc[
        {_SK.multi_index: incoming_sch[_SK.multi_index]}
    ] > allowed_end_time.reindex(incoming_sch[_SK.multi_index].to_pandas().index)

    # NOTE: we checked for is_overlaped instead of is_allowed
    #   so that it is safe agaisnt comparison with `NaT`, which always return false
    #   (and we assume `NaT` mean "no restructions" for both allowed_start_time and allowed_end_time)
    merged_sch_[_IK.is_overlaped].loc[{_SK.multi_index: incoming_sch[_SK.multi_index]}] = (
        is_over_allowed_start_time
    ) | is_over_allowed_end_time

    merged_sch_ = merged_sch_.loc[{_SK.multi_index: ~merged_sch_[_IK.is_overlaped]}]

    return merged_sch_


def _remove_entries_without_corresponding_tx(
    merged_sch: Schedule,
    incoming_sch: Schedule,
    exp_id_stn_id_pairs_map: schedule.ExperimentIdStationIdPairsMap,
) -> Schedule:
    exp_ids: npt.NDArray[np.int16] = np.unique(incoming_sch[_SK.exp_num].to_numpy())
    for exp_id in exp_ids:
        stn_pairs = exp_id_stn_id_pairs_map[exp_id]

        for tx_stn_num, rx_stn_num in stn_pairs:
            incoming_tx_entries = t.cast(
                pd.Series,
                incoming_sch.loc[{_SK.multi_index: (exp_id, tx_stn_num, slice(None), slice(None))}][
                    _SK.multi_index
                ].to_pandas(),
            )

            is_tx_dropped = ~incoming_tx_entries.isin(merged_sch[_SK.multi_index].to_pandas())
            dropped_tx_start_time_idx = (
                incoming_sch.loc[{_SK.multi_index: is_tx_dropped[is_tx_dropped == True].index}]
                .indexes[_SK.multi_index]
                .get_level_values(_SK.start_time)
            )

            # NOTE: we cannot filter the MultiIndex by levels 'start_time', 'exp_num', 'stn_num'
            #   directly using `(dropped_tx_start_time_idx, exp_id, rx_stn_num)` here
            #   because 'dropped_tx_start_time_idx' can contains entries from other exp or station, which will lead to KeyError.
            #
            #   Instead, we filter by first by levels 'exp_num', 'stn_num' here
            #   and followed by an intersection with 'dropped_tx_start_time_idx' later when dropping
            try:
                rx_start_time_of_the_exp_idx = (
                    merged_sch.loc[
                        {
                            _SK.multi_index: (
                                exp_id,
                                rx_stn_num,
                                slice(None),
                                slice(None),
                            )
                        }
                    ]
                    .indexes[_SK.multi_index]
                    .get_level_values(_SK.start_time)
                )
            except KeyError:
                # NOTE: it is possible that there is no entrise with multi_index = (ANY, exp_id, rx_stn_num, ANY)
                #   we can skip the dropping such case
                rx_start_time_of_the_exp_idx = None

            if rx_start_time_of_the_exp_idx is not None:
                # NOTE: cannot drop using `.drop_sel` directly for some reason, so we do a selection by `.loc` first
                dropping = merged_sch.loc[
                    {
                        _SK.multi_index: (
                            exp_id,
                            rx_stn_num,
                            slice(None),
                            rx_start_time_of_the_exp_idx.intersection(dropped_tx_start_time_idx),
                        )
                    }
                ]
                merged_sch = merged_sch.drop_sel({_SK.multi_index: dropping[_SK.multi_index]})
            else:
                continue

    return merged_sch


def _update_cummax_start_time_cummax_end_time(merged_sch: Schedule) -> Schedule:
    merged_sch[_IK.cummax_start_time] = (
        t.cast(pd.Series, merged_sch[_SK.start_time].to_pandas())
        .groupby(level=[_SK.stn_num])
        .cummax()
    )

    merged_sch[_IK.cummax_end_time] = (
        t.cast(pd.Series, merged_sch[_SK.end_time].to_pandas())
        .groupby(level=[_SK.stn_num])
        .cummax()
    )

    return merged_sch


def priority_scheduling(
    schs: t.Sequence[Schedule], exp_id_stn_id_pairs_map: schedule.ExperimentIdStationIdPairsMap
) -> Schedule:
    """
    Merge a sequence of schedule data for a single station into one,
    schedule with smaller index in the sequence is given priority over those with larger index.

    Note: It is assumed (and not checked) that each of the schedule itself does not contain overlapping entries.

    Known artifacts:
      There could be cases of dangling TX entries in the result
      (but not the inverse, i.e. there will not be any dangling RX entries).
    """

    # The logic of this function:
    # 1. prepare an empty schedule data as the merge result
    # 2. for each of the schedule passed in
    #   2.1. merge it into the merge result
    #   2.2. propagate `cummax_start_time`, `cummax_end_time` to new rows
    #   2.3. filter out new rows that have conflicts
    #   2.3. update `cummax_start_time`, `cummax_end_time`

    # early return for empty case
    if len(schs) == 0:
        logger.warning(
            f"An empty `Schedule` is being generated by `{priority_scheduling.__name__}`."
            + "This happens when an empty `Schedule` list is being passed as argument, which is usually not intended."
        )
        return schedule.empty()

    # init an empty dataset for a schedule and add some columns, will be used store merged schedule
    merged_sch = schedule.empty()
    if len(schs) > 0:
        merged_sch.attrs = schs[0].attrs
    merged_sch = _inject_intermediate_columns(merged_sch)
    for incoming_sch in schs:
        incoming_sch = _inject_intermediate_columns(incoming_sch)

        # merge and then sort the schedule
        # we use a "stable" sorting algo to retains relative order,
        # so the df will be in order of start_time, then priority after sorting
        # also note that attrs merged in the way that former schedule has higher priority than latter,
        # consistent with the func `priority_scheduling`
        merged_sch = xr.concat([merged_sch, incoming_sch], dim=_SK.multi_index)
        merged_sch = merged_sch.sortby(_SK.start_time)

        merged_sch = _propagate_cummax_start_time_cummax_end_time(merged_sch)
        merged_sch = _remove_entries_with_time_clash(merged_sch, incoming_sch)
        merged_sch = _remove_entries_without_corresponding_tx(
            merged_sch, incoming_sch, exp_id_stn_id_pairs_map
        )
        merged_sch = _update_cummax_start_time_cummax_end_time(merged_sch)

    merged_sch = merged_sch.drop_vars(
        [
            _IK.cummax_start_time,
            _IK.cummax_end_time,
            _IK.is_overlaped,
        ]
    )

    return merged_sch
