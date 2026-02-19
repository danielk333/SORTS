import sqlite3
import numpy as np
import pandas as pd
from sorts import scheduling
from sorts.scheduling import ScheduleKey


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()

    import pandas as pd

    pd.set_option("display.expand_frame_repr", False)


def empty_schedule_dataframe_test():
    df = scheduling.empty_schedule_dataframe()
    scheduling.validate_schedule_dataframe(df)

    return


def sql_db_round_trip_test():
    # db_conn = sqlite3.connect("./test.sqlite")  # use this if an actual db file is preferred
    db_conn = sqlite3.connect(":memory:")
    sdb = scheduling.ScheduleDb.empty(db_conn)

    df = scheduling.schedule_dataframe_from_rows(
        [
            [0,0,0, pd.Timestamp("2026-02-11 02:00:00.123456789", unit="us"), pd.Timestamp("2026-02-11 03:00:00.123456789", unit="us"), 0,0,0], # fmt: skip
            [0,1,2, pd.Timestamp("2026-02-11 02:00:00.123456789", unit="us"), pd.Timestamp("2026-02-11 03:00:00.123456789", unit="us"), 0.3,0.4,0.5], # fmt: skip
        ]
    )

    df_name = "test"
    sdb.add_dataframe(df, df_name)
    df_read = sdb.get_dataframe(df_name)

    assert df.equals(df_read)

    return


def schedule_by_priority_test():
    # db_conn = sqlite3.connect("./test.sqlite")  # use this if an actual db file is preferred
    db_conn = sqlite3.connect(":memory:")
    sdb = scheduling.ScheduleDb.empty(db_conn)

    sdb.add_dataframe(
        scheduling.schedule_dataframe_from_rows([
            [0,0,0, pd.Timestamp("2026-02-11 00:00:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:00:59.123456789", unit="us"), 0.1,0.2,0.3],
            [0,0,0, pd.Timestamp("2026-02-11 00:01:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:01:59.123456789", unit="us"), 0.1,0.2,0.3],
            [0,0,0, pd.Timestamp("2026-02-11 00:02:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:02:59.123456789", unit="us"), 0.1,0.2,0.3],
            [0,0,0, pd.Timestamp("2027-02-11 00:02:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:02:59.123456789", unit="us"), 0.1,0.2,0.3],
        ]), # fmt: skip
        "exp_00",
    )

    sdb.add_dataframe(
        scheduling.schedule_dataframe_from_rows([
            [1,0,0, pd.Timestamp("2026-02-11 00:01:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:01:59.123456789", unit="us"), 0.1,0.2,0.3],
            [1,1,0, pd.Timestamp("2026-02-11 00:01:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:01:59.123456789", unit="us"), 0.1,0.2,0.3],
        ]), # fmt: skip
        "exp_01",
    )

    sdb.add_dataframe(
        scheduling.schedule_dataframe_from_rows([
            [2,0,0, pd.Timestamp("2026-02-11 00:01:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:01:59.123456789", unit="us"), 0.1,0.2,0.3],
            [2,1,0, pd.Timestamp("2026-02-11 00:02:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:02:59.123456789", unit="us"), 0.1,0.2,0.3],
        ]), # fmt: skip
        "exp_02",
    )

    start_time = "2026-02-11"
    end_time = "2026-03-11"
    sdb.schedule_by_priority(
        ["exp_00", "exp_01", "exp_02"], [0, 1, 0], start_time=start_time, end_time=end_time
    )

    sdf = sdb.get_dataframe(sdb.combined_schedule_name)

    assert (
        1 not in sdf[ScheduleKey.exp_num].values
    ), "Entries with the same `exp_num` and `start_time` are not removed together."

    assert (
        2 in sdf[ScheduleKey.exp_num].values
    ), "Entries with the same `exp_num` but different `start_time` should not be removed together."

    assert (
        all(sdf[ScheduleKey.start_time] >= start_time)
        and all(sdf[ScheduleKey.end_time] <= end_time)
    ), "Entries outside of specified time range should not be included." # fmt: skip

    return
