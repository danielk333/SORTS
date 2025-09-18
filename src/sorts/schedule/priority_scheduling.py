from __future__ import annotations
import logging, typing as t
import numpy as np
import pandas as pd
import xarray as xr
from sorts.utils import assert_class_attributes_equal_to
from . import schedule_data_funcs
from .schedule_data_funcs import _K

if t.TYPE_CHECKING:
    from .schedule import ScheduleData

logger = logging.getLogger(__name__)

max_datetime64_us = np.datetime64(np.iinfo(np.int64).max, "us")
min_datetime64_us = np.datetime64(
    np.iinfo(np.int64).min + 1, "us"
)  # +1 is needed, otherwise it will be NaT

DsIntermediateVarKey = t.Literal["allowed_start_time", "allowed_end_time", "is_overlaped"]


class _IK:
    """Internal helper class for accessing string keys consistently"""

    allowed_start_time: t.Final = "allowed_start_time"
    allowed_end_time: t.Final = "allowed_end_time"
    is_overlaped: t.Final = "is_overlaped"


assert_class_attributes_equal_to(_IK, t.get_args(DsIntermediateVarKey))


def data_to_dataframe(ds: ScheduleData):
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
                    ds[_IK.allowed_start_time].transpose().to_pandas()
                    if _IK.allowed_start_time in ds
                    else empty_df
                ),
                (
                    ds[_IK.allowed_end_time].transpose().to_pandas()
                    if _IK.allowed_end_time in ds
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


# TODO: go through the logic again, now that we have pandas MultiIndex backing the ScheduleData,
#   it might be possible to similify the slicing/alike logic
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

    _SK = _K

    # define some const
    # +1 is needed for `min_datetime64_us`, otherwise it will be NaT
    max_datetime64_us = np.datetime64(np.iinfo(np.int64).max, "us")
    min_datetime64_us = np.datetime64(np.iinfo(np.int64).min + 1, "us")

    # init an empty dataset for a schedule and add some columns, will be used store merged schedule
    merged_sch_data = schedule_data_funcs.empty_data()
    if len(sch_datas) > 0:
        merged_sch_data.attrs = sch_datas[0].attrs
    merged_sch_data[_IK.allowed_start_time] = (
        _SK.multi_index,
        np.empty(0, "datetime64[us]"),
    )
    merged_sch_data[_IK.allowed_end_time] = (_SK.multi_index, np.empty(0, "datetime64[us]"))
    merged_sch_data[_IK.is_overlaped] = (_SK.multi_index, np.empty(0, np.bool))
    for sch_data in sch_datas:
        # init `allowed_start_time`, `allowed_end_time`, `is_overlaped` fields in `sch_data`
        sch_data[_IK.allowed_start_time] = xr.full_like(
            sch_data[_SK.multi_index], np.datetime64("NaT"), dtype="datetime64[us]"
        )
        sch_data[_IK.allowed_end_time] = xr.full_like(
            sch_data[_SK.multi_index], np.datetime64("NaT"), dtype="datetime64[us]"
        )
        sch_data[_IK.is_overlaped] = xr.full_like(sch_data[_SK.multi_index], False, dtype=np.bool)

        # merge and then sort the schedule
        # we use a "stable" sorting algo to retains relative order,
        # so the df will be in order of start_time, then priority after sorting
        # also note that attrs merged in the way that former schedule has higher priority than latter,
        # consistent with the func `priority_scheduling`
        merged_sch_data.attrs = schedule_data_funcs.merge_attrs(
            [sch_data.attrs, merged_sch_data.attrs]
        )
        merged_sch_data = xr.concat([merged_sch_data, sch_data], dim=_SK.multi_index)
        merged_sch_data = merged_sch_data.sortby(_SK.start_time)

        # populate `allowed_start_time`, `allowed_end_time` columns entries that have NaT values
        # - the `end_time` of entries which have non-NaT `allowed_start_time` will be the `allowed_start_time` of its next and ffill rows
        # - the `start_time` of entries which have non-NaT `allowed_end_time` will be the `allowed_end_time` of its previous and bfill rows
        allowed_start_time_mask = ~xr.ufuncs.isnat(merged_sch_data[_IK.allowed_start_time]).shift(
            {_SK.multi_index: 1}, fill_value=False
        )
        merged_sch_data[_IK.allowed_start_time].loc[allowed_start_time_mask] = (
            merged_sch_data[_SK.end_time]
            .shift({_SK.multi_index: 1}, fill_value=min_datetime64_us)
            .loc[allowed_start_time_mask]
        )
        merged_sch_data[_IK.allowed_start_time] = merged_sch_data[_IK.allowed_start_time].ffill(
            _SK.multi_index
        )

        allowed_end_time_mask = ~xr.ufuncs.isnat(merged_sch_data[_IK.allowed_end_time]).shift(
            {_SK.multi_index: -1}, fill_value=False
        )
        merged_sch_data[_IK.allowed_end_time].loc[allowed_end_time_mask] = (
            merged_sch_data[_SK.start_time]
            .shift({_SK.multi_index: -1}, fill_value=max_datetime64_us)
            .loc[allowed_end_time_mask]
        )
        merged_sch_data[_IK.allowed_end_time] = merged_sch_data[_IK.allowed_end_time].bfill(
            _SK.multi_index
        )

        # TODO: the `ffill`, `bfill` plus `shift` with `fill_value` should have left no `NaT`, investigate why it is not
        # NOTE: we fill in `min_datetime64_us`, `max_datetime64_us` for the remaining NaT in `allowed_start_time`, `allowed_end_time`
        #   so that resolved rows always have non NA values in that two column.
        #   (they are likely at the tops and bottoms)
        merged_sch_data[_IK.allowed_start_time] = merged_sch_data[_IK.allowed_start_time].fillna(
            min_datetime64_us
        )
        merged_sch_data[_IK.allowed_end_time] = merged_sch_data[_IK.allowed_end_time].fillna(
            max_datetime64_us
        )

        # remove rows (control slices) that have time clash
        # NOTE: we checked for is_overlaped instead of is_allowed
        #   so that it is safe agaisnt comparison with `NaT`, which always return false
        #   (and we assume `NaT` mean "no restructions" for both allowed_start_time and allowed_end_time)
        merged_sch_data[_IK.is_overlaped] = (
            merged_sch_data[_SK.start_time] < merged_sch_data[_IK.allowed_start_time]
        ) | (merged_sch_data[_SK.end_time] > merged_sch_data[_IK.allowed_end_time])
        merged_sch_data = merged_sch_data.loc[{_SK.multi_index: ~merged_sch_data[_IK.is_overlaped]}]

        # update `allowed_start_time`, `allowed_end_time` columns
        # - the `allowed_start_time` has the `end_time` of previous row
        # - the `allowed_end_time` has the `start_time` of next row
        merged_sch_data[_IK.allowed_start_time] = merged_sch_data[_SK.end_time].shift(
            {_SK.multi_index: 1}, fill_value=min_datetime64_us
        )
        merged_sch_data[_IK.allowed_end_time] = merged_sch_data[_SK.start_time].shift(
            {_SK.multi_index: -1}, fill_value=max_datetime64_us
        )

    merged_sch_data = merged_sch_data.drop_vars(
        [_IK.allowed_start_time, _IK.allowed_end_time, _IK.is_overlaped]
    )

    return merged_sch_data
