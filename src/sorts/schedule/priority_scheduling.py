from __future__ import annotations
import logging, typing as t
import numpy as np
import pandas as pd
import xarray as xr
from sorts.utils import assert_class_attributes_equal_to
from . import schedule_data_funcs
from .schedule_data_funcs import _K as _SK, ExperimentDetail

if t.TYPE_CHECKING:
    from .schedule import ScheduleData

logger = logging.getLogger(__name__)

max_datetime64_us = np.datetime64(np.iinfo(np.int64).max, "us")
min_datetime64_us = np.datetime64(
    np.iinfo(np.int64).min + 1, "us"
)  # +1 is needed, otherwise it will be NaT

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


assert_class_attributes_equal_to(_IK, t.get_args(DsIntermediateVarKey))


# TODO: remove?
def to_dataframe(ds: ScheduleData):
    """
    Convert schedule data in xarray dataset to pandas dataframe.

    A helper method for debugging.
    Includes extra intermediate columns used in function `priority_scheduling`.
    """

    empty_df = pd.DataFrame()

    df = pd.concat(
        t.cast(
            list[pd.DataFrame],
            [
                schedule_data_funcs.to_dataframe(ds),
                (
                    ds[_IK.cummax_start_time].transpose().to_pandas()
                    if _IK.cummax_start_time in ds
                    else empty_df
                ),
                (
                    ds[_IK.cummax_end_time].transpose().to_pandas()
                    if _IK.cummax_end_time in ds
                    else empty_df
                ),
                (
                    ds[_IK.is_overlaped].transpose().to_pandas()
                    if _IK.is_overlaped in ds
                    else empty_df
                ),
            ],
        ),
        axis=1,
        copy=False,
    )

    return df


def _inject_intermediate_columns(sch_data: ScheduleData) -> ScheduleData:
    """
    init `allowed_start_time`, `allowed_end_time`, `is_overlaped` fields in `sch_data`
    """

    sch_data[_IK.cummax_start_time] = xr.full_like(
        sch_data[_SK.multi_index], np.datetime64("NaT"), dtype="datetime64[us]"
    )
    sch_data[_IK.cummax_end_time] = xr.full_like(
        sch_data[_SK.multi_index], np.datetime64("NaT"), dtype="datetime64[us]"
    )
    sch_data[_IK.is_overlaped] = xr.full_like(sch_data[_SK.multi_index], False, dtype=np.bool)

    return sch_data


def _propagate_cummax_start_time_cummax_end_time(merged_sch_data: ScheduleData) -> ScheduleData:
    merged_sch_data[_IK.cummax_start_time] = (
        t.cast(pd.Series, merged_sch_data[_IK.cummax_start_time].to_pandas())
        .groupby(level=[_SK.stn_num])
        .ffill()
    )
    merged_sch_data[_IK.cummax_end_time] = (
        t.cast(pd.Series, merged_sch_data[_IK.cummax_end_time].to_pandas())
        .groupby(level=[_SK.stn_num])
        .ffill()
    )

    return merged_sch_data


def _remove_entries_with_time_clash(
    merged_sch_data: ScheduleData, incoming_sch_data: ScheduleData
) -> ScheduleData:
    # NOTE:
    #   - the `allowed_start_time` has the previous `end_time`;
    #     for top values that have no corresponding pervious values, `min_datetime64_us` is used
    #   - the `allowed_end_time` has the next `start_time`;
    #     for bottom values that have no corresponding next values, `max_datetime64_us` is used
    allowed_start_time = (
        t.cast(pd.Series, merged_sch_data[_IK.cummax_end_time].to_pandas())
        .groupby(level=_SK.stn_num)
        .shift(+1, fill_value=min_datetime64_us)
    )
    allowed_end_time = (
        t.cast(pd.Series, merged_sch_data[_IK.cummax_start_time].to_pandas())
        .groupby(level=_SK.stn_num)
        .shift(-1, fill_value=max_datetime64_us)
    )

    # calc bool mask for overlapping just for the new/incoming entries
    is_over_allowed_start_time = merged_sch_data[_SK.start_time].loc[
        {_SK.multi_index: incoming_sch_data[_SK.multi_index]}
    ] < allowed_start_time.reindex(incoming_sch_data[_SK.multi_index].to_pandas().index)

    is_over_allowed_end_time = merged_sch_data[_SK.end_time].loc[
        {_SK.multi_index: incoming_sch_data[_SK.multi_index]}
    ] > allowed_end_time.reindex(incoming_sch_data[_SK.multi_index].to_pandas().index)

    # NOTE: we checked for is_overlaped instead of is_allowed
    #   so that it is safe agaisnt comparison with `NaT`, which always return false
    #   (and we assume `NaT` mean "no restructions" for both allowed_start_time and allowed_end_time)
    merged_sch_data[_IK.is_overlaped].loc[{_SK.multi_index: incoming_sch_data[_SK.multi_index]}] = (
        is_over_allowed_start_time
    ) | is_over_allowed_end_time

    merged_sch_data = merged_sch_data.loc[{_SK.multi_index: ~merged_sch_data[_IK.is_overlaped]}]

    return merged_sch_data


def _remove_entries_without_corresponding_tx(
    merged_sch_data: ScheduleData, incoming_sch_data: ScheduleData
) -> ScheduleData:
    for exp_detail in incoming_sch_data.attrs[_SK.exp_detail_map].values():
        exp_detail: ExperimentDetail
        stn_pairs = exp_detail.get("stn_pairs")
        if stn_pairs is None:
            raise RuntimeError("stn_pairs not found in ExperimentDetail")

        for tx_stn_num, rx_stn_num in stn_pairs:
            is_tx_dropped_mask = ~np.isin(
                incoming_sch_data.loc[
                    {_SK.multi_index: (slice(None), slice(None), tx_stn_num, slice(None))}
                ][_SK.multi_index].to_numpy(),
                merged_sch_data[_SK.multi_index].to_numpy(),
            )

            dropped_tx_midx = incoming_sch_data[_SK.multi_index][is_tx_dropped_mask]
            try:
                corresponding_rx_to_drop = merged_sch_data.sel(
                    {
                        _SK.multi_index: (
                            # NOTE: `.tolist()` needed for MultiIndex level start_time, the value cannot be interpreted correctly otherwise for some reason
                            dropped_tx_midx[_SK.start_time].to_numpy().tolist(),
                            dropped_tx_midx[_SK.exp_num].to_numpy(),
                            rx_stn_num,
                            slice(None),
                        )
                    }
                )
                merged_sch_data = merged_sch_data.drop_sel(
                    {_SK.multi_index: corresponding_rx_to_drop[_SK.multi_index]}
                )

            except KeyError:
                # Do nothing when the selection returns no result.
                pass

    return merged_sch_data


def _update_cummax_start_time_cummax_end_time(merged_sch_data: ScheduleData) -> ScheduleData:
    merged_sch_data[_IK.cummax_start_time] = (
        t.cast(pd.Series, merged_sch_data[_SK.start_time].to_pandas())
        .groupby(level=[_SK.stn_num])
        .cummax()
    )

    merged_sch_data[_IK.cummax_end_time] = (
        t.cast(pd.Series, merged_sch_data[_SK.end_time].to_pandas())
        .groupby(level=[_SK.stn_num])
        .cummax()
    )

    return merged_sch_data


# TODO: go through the logic again, now that we have pandas MultiIndex backing the ScheduleData,
#   it might be possible to similify the slicing/alike logic;
#
#   maybe no need to `_update_allowed_start_time_allowed_end_time`,
#   and `allowed_start_time`, `allowed_end_time` can be kept ephemeral and passed ard as param?
def priority_scheduling(
    sch_datas: t.Sequence[ScheduleData],
) -> ScheduleData:
    """
    Merge a sequence of schedule data for a single station into one,
    schedule with smaller index in the sequence is given priority over those with larger index.

    Note: It is assumed (and not checked) that each of the schedule itself does not contain overlapping entries.
    """

    # The logic of this function:
    # 1. prepare an empty schedule data as the merge result
    # 2. for each of the schedule passed in
    #   2.1. merge it into the merge result
    #   2.2. populate `allowed_start_time`, `allowed_end_time` for new rows
    #   2.3. filter out new rows that conflict with `allowed_start_time`, `allowed_end_time`
    #   2.3. update `allowed_start_time`, `allowed_end_time`

    # init an empty dataset for a schedule and add some columns, will be used store merged schedule
    merged_sch_data = schedule_data_funcs.empty_data()
    if len(sch_datas) > 0:
        merged_sch_data.attrs = sch_datas[0].attrs
    merged_sch_data = _inject_intermediate_columns(merged_sch_data)
    for incoming_sch_data in sch_datas:
        incoming_sch_data = _inject_intermediate_columns(incoming_sch_data)

        # merge and then sort the schedule
        # we use a "stable" sorting algo to retains relative order,
        # so the df will be in order of start_time, then priority after sorting
        # also note that attrs merged in the way that former schedule has higher priority than latter,
        # consistent with the func `priority_scheduling`
        merged_sch_data.attrs = schedule_data_funcs.merge_attrs(
            [incoming_sch_data.attrs, merged_sch_data.attrs]
        )
        merged_sch_data = xr.concat([merged_sch_data, incoming_sch_data], dim=_SK.multi_index)
        merged_sch_data = merged_sch_data.sortby(_SK.start_time)

        merged_sch_data = _propagate_cummax_start_time_cummax_end_time(merged_sch_data)
        merged_sch_data = _remove_entries_with_time_clash(merged_sch_data, incoming_sch_data)
        merged_sch_data = _remove_entries_without_corresponding_tx(
            merged_sch_data, incoming_sch_data
        )
        merged_sch_data = _update_cummax_start_time_cummax_end_time(merged_sch_data)

    merged_sch_data = merged_sch_data.drop_vars(
        [
            _IK.cummax_start_time,
            _IK.cummax_end_time,
            _IK.is_overlaped,
        ]
    )

    return merged_sch_data
