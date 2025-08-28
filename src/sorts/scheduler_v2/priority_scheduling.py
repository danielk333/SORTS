import logging, typing as t
import numpy as np
import pandas as pd
import xarray as xr
from sorts.schedule_v2 import (
    Schedule,
    ScheduleXrds,
    ScheduleKey,
    ScheduleNdarrayDict2,
    ExperimentDetail,
)

logger = logging.getLogger(__name__)

max_datetime64_us = np.datetime64(np.iinfo(np.int64).max, "us")
min_datetime64_us = np.datetime64(
    np.iinfo(np.int64).min + 1, "us"
)  # +1 is needed, otherwise it will be NaT

DsIntermediateVarKey = t.Literal["allowed_start_time", "allowed_end_time", "is_overlaped"]
DsVarKey = t.Literal[ScheduleKey, DsIntermediateVarKey]


def to_df(ds: xr.Dataset):
    """
    Convert schedule data in xarray dataset to pandas dataframe.
    A helper method for debugging.
    """

    # define some column names/keys
    keys: dict[DsVarKey, str] = {k: k for k in t.get_args(DsVarKey)}

    empty_df = pd.DataFrame()

    df = pd.concat(
        t.cast(
            list[pd.DataFrame],
            [
                ds[keys["end_time"]].transpose().to_pandas(),
                ds[keys["pointing"]].transpose().to_pandas(),
                (
                    ds[keys["allowed_start_time"]].transpose().to_pandas()
                    if keys["allowed_start_time"] in ds
                    else empty_df
                ),
                (
                    ds[keys["allowed_end_time"]].transpose().to_pandas()
                    if keys["allowed_end_time"] in ds
                    else empty_df
                ),
                (
                    ds[keys["is_overlaped"]].transpose().to_pandas()
                    if keys["is_overlaped"] in ds
                    else empty_df
                ),
            ],
        ),
        axis=1,
        copy=False,
    )

    return df


# TODO: add schedule validation?
def priority_scheduling(sch_datas: t.Sequence[ScheduleXrds]):
    # The logic of this function:
    # 1. prepare an empty schedule data as the merge result
    # 2. for each of the schedule passed in
    #   2.1. merge it into the merge result
    #   2.2. populate `allowed_start_time`, `allowed_end_time` for new rows
    #   2.3. filter out new rows that conflict with `allowed_start_time`, `allowed_end_time`
    #   2.3. update `allowed_start_time`, `allowed_end_time`

    # define some const
    # +1 is needed for `min_datetime64_us`, otherwise it will be NaT
    max_datetime64_us = np.datetime64(np.iinfo(np.int64).max, "us")
    min_datetime64_us = np.datetime64(np.iinfo(np.int64).min + 1, "us")

    # define some column names/keys
    keys: dict[DsVarKey, str] = {k: k for k in t.get_args(DsVarKey)}

    # init an empty dataset for a schedule and add some columns, will be used store merged schedule
    merged_sch_data = Schedule.empty().data
    merged_sch_data[keys["allowed_start_time"]] = (
        keys["start_time"],
        np.empty(0, "datetime64[us]"),
    )
    merged_sch_data[keys["allowed_end_time"]] = (keys["start_time"], np.empty(0, "datetime64[us]"))
    merged_sch_data[keys["is_overlaped"]] = (keys["start_time"], np.empty(0, np.bool))
    for sch_data in sch_datas:
        # init `allowed_start_time`, `allowed_end_time`, `is_overlaped` fields in `sch_data`
        sch_data[keys["allowed_start_time"]] = xr.full_like(
            sch_data[keys["start_time"]], np.datetime64("NaT"), dtype="datetime64[us]"
        )
        sch_data[keys["allowed_end_time"]] = xr.full_like(
            sch_data[keys["start_time"]], np.datetime64("NaT"), dtype="datetime64[us]"
        )
        sch_data[keys["is_overlaped"]] = xr.full_like(
            sch_data[keys["start_time"]], False, dtype=np.bool
        )

        # merge and then sort the schedule
        # we use a "stable" sorting algo to retains relative order,
        # so the df will be in order of start_time, then priority after sorting
        merged_sch_data = xr.concat([merged_sch_data, sch_data], dim=keys["start_time"])
        merged_sch_data = merged_sch_data.sortby(keys["start_time"])

        # populate `allowed_start_time`, `allowed_end_time` columns entries that have NaT values
        # - the `end_time` of entries which have non-NaT `allowed_start_time` will be the `allowed_start_time` of its next and ffill rows
        # - the `start_time` of entries which have non-NaT `allowed_end_time` will be the `allowed_end_time` of its previous and bfill rows
        allowed_start_time_mask = ~xr.ufuncs.isnat(
            merged_sch_data[keys["allowed_start_time"]]
        ).shift({keys["start_time"]: 1}, fill_value=False)
        merged_sch_data[keys["allowed_start_time"]].loc[allowed_start_time_mask] = (
            merged_sch_data[keys["end_time"]]
            .shift({keys["start_time"]: 1}, fill_value=min_datetime64_us)
            .loc[allowed_start_time_mask]
        )
        merged_sch_data[keys["allowed_start_time"]] = merged_sch_data[
            keys["allowed_start_time"]
        ].ffill(keys["start_time"])

        allowed_end_time_mask = ~xr.ufuncs.isnat(merged_sch_data[keys["allowed_end_time"]]).shift(
            {keys["start_time"]: -1}, fill_value=False
        )
        merged_sch_data[keys["allowed_end_time"]].loc[allowed_end_time_mask] = (
            merged_sch_data[keys["start_time"]]
            .shift({keys["start_time"]: -1}, fill_value=max_datetime64_us)
            .loc[allowed_end_time_mask]
        )
        merged_sch_data[keys["allowed_end_time"]] = merged_sch_data[keys["allowed_end_time"]].bfill(
            keys["start_time"]
        )

        # TODO: the `ffill`, `bfill` plus `shift` with `fill_value` should have left no `NaT`, investigate why it is not
        # NOTE: we fill in `min_datetime64_us`, `max_datetime64_us` for the remaining NaT in `allowed_start_time`, `allowed_end_time`
        #   so that resolved rows always have non NA values in that two column.
        #   (they are likely at the tops and bottoms)
        merged_sch_data[keys["allowed_start_time"]] = merged_sch_data[
            keys["allowed_start_time"]
        ].fillna(min_datetime64_us)
        merged_sch_data[keys["allowed_end_time"]] = merged_sch_data[
            keys["allowed_end_time"]
        ].fillna(max_datetime64_us)

        # remove rows (control slices) that have time clash
        # NOTE: we checked for is_overlaped instead of is_allowed
        #   so that it is safe agaisnt comparison with `NaT`, which always return false
        #   (and we assume `NaT` mean "no restructions" for both allowed_start_time and allowed_end_time)
        merged_sch_data[keys["is_overlaped"]] = (
            merged_sch_data[keys["start_time"]] < merged_sch_data[keys["allowed_start_time"]]
        ) | (merged_sch_data[keys["end_time"]] > merged_sch_data[keys["allowed_end_time"]])
        merged_sch_data = merged_sch_data.loc[
            {keys["start_time"]: ~merged_sch_data[keys["is_overlaped"]]}
        ]

        # update `allowed_start_time`, `allowed_end_time` columns
        # - the `allowed_start_time` has the `end_time` of previous row
        # - the `allowed_end_time` has the `start_time` of next row
        merged_sch_data[keys["allowed_start_time"]] = merged_sch_data[keys["end_time"]].shift(
            {keys["start_time"]: 1}, fill_value=min_datetime64_us
        )
        merged_sch_data[keys["allowed_end_time"]] = merged_sch_data[keys["start_time"]].shift(
            {keys["start_time"]: -1}, fill_value=max_datetime64_us
        )

    merged_sch_data = merged_sch_data.drop_vars(
        [keys["allowed_start_time"], keys["allowed_end_time"], keys["is_overlaped"]]
    )

    return merged_sch_data


# TODO: remove its usage, then remove this func
def priority_scheduling_npardict(schs: t.Sequence[ScheduleNdarrayDict2]) -> ScheduleNdarrayDict2:
    """
    Merge a sequence of schedules for a single station into one,
    schedule with lower index in the sequence is given priority over those with higher index.

    Note: It is assumed (and not checked) that each of the schedule itself does not contain overlapping entries.
    """
    logger.debug("resolving schedule")
    exp_detail_map: dict[int, ExperimentDetail] = {}
    for sch in reversed(schs):
        exp_detail_map.update(sch["exp_detail_map"])

    resultant_sch = priority_scheduling([Schedule.from_ndarrays_2(sch).data for sch in schs])

    logger.debug("priority_scheduling done")
    return resultant_sch.to_ndarrays_2()
