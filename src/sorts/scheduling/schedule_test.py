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
    import sqlite3

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
    df_read = sdb.read_dataframe(df_name)

    assert df.equals(df_read)

    return
