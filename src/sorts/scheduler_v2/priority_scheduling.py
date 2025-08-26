import logging, typing as t
import numpy as np
import pandas as pd
from sorts import schedule_v2 as schedule
from sorts.schedule_v2 import ScheduleNdarrayDict, ExperimentDetail

logger = logging.getLogger(__name__)

max_datetime64_us = np.datetime64(np.iinfo(np.int64).max, "us")
min_datetime64_us = np.datetime64(
    np.iinfo(np.int64).min + 1, "us"
)  # +1 is needed, otherwise it will be NaT


# TODO: should we use a db like sqlite to enable larger than memory processing?
# TODO: add schedule validation?
# TODO: return the rows/index of dropped slice?
def priority_scheduling_df(sch_dfs: t.Sequence[pd.DataFrame]):
    """
    Same as `priority_scheduling` but takes and returns `Schedule` in pandas `DataFrame` form
    (see also `schedule.from_dataframe`, `schedule.to_dataframe`).

    Used by `priority_scheduling` internally.
    """

    cn = schedule.cn

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
    merged_sch_df = schedule.to_dataframe(schedule.empty_npardict())
    merged_sch_df[cn["end_time"]] = np.empty(0, "datetime64[us]")
    merged_sch_df[cn_allowed_start_time] = np.empty(0, "datetime64[us]")
    merged_sch_df[cn_allowed_end_time] = np.empty(0, "datetime64[us]")
    merged_sch_df[cn_is_overlaped] = np.empty(0, np.bool)
    for sch_df in sch_dfs:
        # merge and then sort the df
        # we use a "stable" sorting algo to retains relative order,
        # so the df will be in order of start_time, then priority after sorting
        merged_sch_df = pd.concat([merged_sch_df, sch_df])
        merged_sch_df = merged_sch_df.sort_values(cn["start_time"], kind="stable").reset_index(
            drop=True
        )  # TODO: re-eval if we should use start_time as index

        is_new_rows = merged_sch_df[cn_allowed_start_time].isna()

        # populate `cn_allowed_start_time`, `cn_allowed_end_time` columns
        # (rows from `sch_df` has NA values in them after the merge).
        # - the allowed_start_time NA chunks heads is filled by the end_time of previous row, then ffill the rest
        # - the allowed_end_time NA chunks tail is filled by the start_time of next row, then bfill the rest
        allowed_start_time_na_heads_mask = (
            merged_sch_df[cn_allowed_start_time].isna()
            & merged_sch_df[cn_allowed_start_time]
            .shift(1, fill_value=np.datetime64(0, "us"))
            .notna()
        )
        merged_sch_df.loc[allowed_start_time_na_heads_mask, cn_allowed_start_time] = (
            merged_sch_df.shift(1).loc[allowed_start_time_na_heads_mask, cn["end_time"]]
        )
        merged_sch_df[cn_allowed_start_time] = merged_sch_df[cn_allowed_start_time].ffill()

        allowed_end_time_na_tails_mask = (
            merged_sch_df[cn_allowed_end_time].isna()
            & merged_sch_df[cn_allowed_end_time]
            .shift(-1, fill_value=np.datetime64(0, "us"))
            .notna()
        )
        merged_sch_df.loc[allowed_end_time_na_tails_mask, cn_allowed_end_time] = (
            merged_sch_df.shift(-1).loc[allowed_end_time_na_tails_mask, cn["start_time"]]
        )
        merged_sch_df[cn_allowed_end_time] = merged_sch_df[cn_allowed_end_time].bfill()

        # remove rows (control slices) that have time clash
        # NOTE: we checked for is_overlaped instead of is_allowed
        #   so that it is safe agaisnt comparison with `NaT`, which always return false
        #   (and we assume `NaT` mean "no restructions" for both allowed_start_time and allowed_end_time)
        merged_sch_df[cn_is_overlaped] = (is_new_rows) & (
            (merged_sch_df[cn["start_time"]] < merged_sch_df[cn_allowed_start_time])
            | (merged_sch_df[cn["end_time"]] >= merged_sch_df[cn_allowed_end_time])
        )
        merged_sch_df = merged_sch_df[~merged_sch_df[cn_is_overlaped]]

        # update `cn_allowed_start_time`, `cn_allowed_end_time` columns
        # NOTE: we fill in `min_datetime64_us`, `max_datetime64_us` at the df top and end of allowed_start_time, allowed_end_time
        #   so that resolved rows always have non NA values in that two column,
        #   which we rely on atm to keep track on new new rows.
        # TODO: see if delaying `reset_index` can eliminate the need of setting `min_datetime64_us`, `max_datetime64_us`
        if (len(merged_sch_df)) > 0:
            merged_sch_df.loc[:, cn_allowed_start_time] = merged_sch_df[cn["end_time"]].shift(1)
            merged_sch_df.loc[merged_sch_df.index[0], cn_allowed_start_time] = min_datetime64_us
            merged_sch_df.loc[:, cn_allowed_end_time] = merged_sch_df[cn["start_time"]].shift(-1)
            merged_sch_df.loc[merged_sch_df.index[-1], cn_allowed_end_time] = max_datetime64_us

    merged_sch_df = merged_sch_df.reset_index(drop=True)

    return merged_sch_df


def priority_scheduling_npardict(schs: t.Sequence[ScheduleNdarrayDict]) -> ScheduleNdarrayDict:
    """
    Merge a sequence of schedules for a single station into one,
    schedule with lower index in the sequence is given priority over those with higher index.

    Note: It is assumed (and not checked) that each of the schedule itself does not contain overlapping entries.
    """
    logger.debug("resolving schedule")
    exp_detail_map: dict[int, ExperimentDetail] = {}
    for sch in reversed(schs):
        exp_detail_map.update(sch["exp_detail_map"])

    df = priority_scheduling_df([schedule.to_dataframe(sch) for sch in schs])

    logger.debug("priority_scheduling done")
    return schedule.from_dataframe(df, exp_detail_map=exp_detail_map)
