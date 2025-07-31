import logging, typing as t
import numpy as np
import pandas as pd
from sorts.schedule_v2 import Schedule, ExperimentDetail

logger = logging.getLogger(__name__)

max_datetime64_us = np.datetime64(np.iinfo(np.int64).max, "us")
min_datetime64_us = np.datetime64(0, "us")


# TODO: should we use a db like sqlite to enable larger than memory processing?
# TODO: add schedule validation?
# TODO: return the rows/index of dropped slice?
def _priority_scheduling_df(schs: t.Sequence[Schedule]):
    """
    Same as `priority_scheduling` but returns a pandas `DataFrame`.
    Used by `priority_scheduling` internally.
    """

    Cn = Schedule.Cn

    # The logic of this function:
    # 1. prepare an empty df as the merge result
    # 2. for each of the schema passed in
    #   2.1. merge it into the merge result df
    #   2.2. fill in the missing `cn_allowed_start_time`, `cn_allowed_end_time` for new rows
    #   2.3. filter out new rows that conflict with `cn_allowed_start_time`, `cn_allowed_end_time`
    #   2.3. update `cn_allowed_start_time`, `cn_allowed_end_time`

    # define some df column names
    cn_allowed_start_time = "allowed_start_time"
    cn_allowed_end_time = "allowed_end_time"
    cn_is_overlaped = "is_overlaped"

    # init an empty df for a schedule and add some columns, will be used store merged schedule
    merged_sch_df = Schedule.empty().to_dataframe()
    merged_sch_df[Cn.end_time] = np.empty(0, "datetime64[us]")
    merged_sch_df[cn_allowed_start_time] = np.empty(0, "datetime64[us]")
    merged_sch_df[cn_allowed_end_time] = np.empty(0, "datetime64[us]")
    merged_sch_df[cn_is_overlaped] = np.empty(0, np.bool)
    for sch in schs:
        sch_df = sch.to_dataframe()

        # merge and then sort the df
        # we use a "stable" sorting algo to retains relative order,
        # so the df will be in order of start_time, then priority after sorting
        merged_sch_df = pd.concat([merged_sch_df, sch_df])
        merged_sch_df = merged_sch_df.sort_values(Cn.start_time, kind="stable").reset_index(
            drop=True
        )  # TODO: re-eval if we should use start_time as index

        is_new_rows = merged_sch_df[cn_allowed_start_time].isna()

        # populate `cn_allowed_start_time`, `cn_allowed_end_time` columns
        # (rows from `sch_df` has null values in them after the merge).
        # `.isna().all()` check is needed because `.ffill()` will throw exception when all the values are NaT (not a time)
        if not merged_sch_df[cn_allowed_start_time].isna().all():
            merged_sch_df[cn_allowed_start_time] = merged_sch_df[cn_allowed_start_time].bfill()
        if not merged_sch_df[cn_allowed_end_time].isna().all():
            merged_sch_df[cn_allowed_end_time] = merged_sch_df[cn_allowed_end_time].ffill()

        # remove rows (control slices) that have time clash
        merged_sch_df[cn_is_overlaped] = (is_new_rows) & (
            (merged_sch_df[Cn.start_time] <= merged_sch_df[cn_allowed_start_time])
            | (merged_sch_df[Cn.end_time] >= merged_sch_df[cn_allowed_end_time])
        )
        merged_sch_df = merged_sch_df[~merged_sch_df[cn_is_overlaped]]

        # update `cn_allowed_start_time`, `cn_allowed_end_time` columns
        if (len(merged_sch_df)) > 0:
            merged_sch_df[cn_allowed_start_time] = merged_sch_df[Cn.end_time].shift(1)
            merged_sch_df.loc[merged_sch_df.index[0], cn_allowed_start_time] = min_datetime64_us
            merged_sch_df[cn_allowed_end_time] = merged_sch_df[Cn.start_time].shift(-1)
            merged_sch_df.loc[merged_sch_df.index[-1], cn_allowed_end_time] = max_datetime64_us

    merged_sch_df = merged_sch_df.reset_index(drop=True)

    return merged_sch_df


def priority_scheduling(schs: t.Sequence[Schedule], meta: dict[int, ExperimentDetail]) -> Schedule:
    """
    Merge a sequence of schedules for a single station into one,
    schedule with lower index in the sequence is given priority over those with higher index.

    Note: It is assumed (and not checked) that each of the schedule itself does not contain overlapping entries.
    """

    df = _priority_scheduling_df(schs)
    return Schedule.from_dataframe(df, meta=meta)
