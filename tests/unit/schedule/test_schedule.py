import sqlite3, typing as t
import numpy as np
import pandas as pd
from sorts import schedule
from sorts.schedule import schedule_dataframe, ScheduleDb, ScheduleKey


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()

    import pandas as pd

    pd.set_option("display.expand_frame_repr", False)


def test_empty_schedule_dataframe():
    df = schedule_dataframe.empty()
    schedule_dataframe.validate(df)

    return


def test_sql_db_round_trip():
    # db_conn = sqlite3.connect("./test.sqlite")  # use this if an actual db file is preferred
    db_conn = sqlite3.connect(":memory:")
    sdb = ScheduleDb.empty(db_conn)

    df = schedule_dataframe.from_rows(
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


def test_schedule_by_priority():
    # db_conn = sqlite3.connect("./test.sqlite")  # use this if an actual db file is preferred
    db_conn = sqlite3.connect(":memory:")
    sdb = ScheduleDb.empty(db_conn)

    sdb.add_dataframe(
        schedule_dataframe.from_rows([
            [0,0,0, pd.Timestamp("2026-02-11 00:00:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:00:59.123456789", unit="us"), 0.1,0.2,0.3],
            [0,0,0, pd.Timestamp("2026-02-11 00:01:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:01:59.123456789", unit="us"), 0.1,0.2,0.3],
            [0,0,0, pd.Timestamp("2026-02-11 00:02:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:02:59.123456789", unit="us"), 0.1,0.2,0.3],
            [0,0,0, pd.Timestamp("2027-02-11 00:02:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:02:59.123456789", unit="us"), 0.1,0.2,0.3],
        ]), # fmt: skip
        "exp_00",
    )

    sdb.add_dataframe(
        schedule_dataframe.from_rows([
            [1,0,0, pd.Timestamp("2026-02-11 00:01:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:01:59.123456789", unit="us"), 0.1,0.2,0.3],
            [1,1,0, pd.Timestamp("2026-02-11 00:01:00.123456789", unit="us"), pd.Timestamp("2026-02-11 00:01:59.123456789", unit="us"), 0.1,0.2,0.3],
        ]), # fmt: skip
        "exp_01",
    )

    sdb.add_dataframe(
        schedule_dataframe.from_rows([
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

    df = sdb.get_dataframe(sdb.combined_schedule_name)

    assert (
        1 not in df[ScheduleKey.exp_num].values
    ), "Entries with the same `exp_num` and `start_time` are not removed together."

    assert (
        2 in df[ScheduleKey.exp_num].values
    ), "Entries with the same `exp_num` but different `start_time` should not be removed together."

    assert (
        all(df[ScheduleKey.start_time] >= start_time)
        and all(df[ScheduleKey.end_time] < end_time)
    ), "Entries outside of specified time range should not be included." # fmt: skip

    return


def test_get_tx_rx_pointing_pairs():
    # db_conn = sqlite3.connect("./test.sqlite")  # use this if an actual db file is preferred
    db_conn = sqlite3.connect(":memory:")
    sdb = ScheduleDb.empty(db_conn)

    # tx rows
    tx_rows = [
        [0,0,0, np.datetime64("2026-02-11 00:00:00.123456", "us"), np.datetime64("2026-02-11 00:00:59.123456", "us"), 0.1,0.2,0.3],
        [0,0,0, np.datetime64("2026-02-11 00:02:00.123456", "us"), np.datetime64("2026-02-11 00:02:59.123456", "us"), 0.1,0.2,0.3],
    ] # fmt: skip

    # rx rows, without corresponding tx
    rx_rows_wo_tx = [
        [0,1,0, np.datetime64("2026-02-11 00:01:00.123456", "us"), np.datetime64("2026-02-11 00:01:59.123456", "us"), 0.1,0.2,0.3],
        [0,1,1, np.datetime64("2026-02-11 00:01:00.123456", "us"), np.datetime64("2026-02-11 00:01:59.123456", "us"), 0.1,0.2,0.3],
        [0,1,2, np.datetime64("2026-02-11 00:01:00.123456", "us"), np.datetime64("2026-02-11 00:01:59.123456", "us"), 0.1,0.2,0.3],
    ] # fmt: skip

    # rx rows, with corresponding tx
    rx_rows_w_tx =[
        [0,1,0, np.datetime64("2026-02-11 00:02:00.123456", "us"), np.datetime64("2026-02-11 00:02:59.123456", "us"), 0.1,0.2,0.3],
        [0,1,1, np.datetime64("2026-02-11 00:02:00.123456", "us"), np.datetime64("2026-02-11 00:02:59.123456", "us"), 0.1,0.2,0.3],
        [0,1,2, np.datetime64("2026-02-11 00:02:00.123456", "us"), np.datetime64("2026-02-11 00:02:59.123456", "us"), 0.1,0.2,0.3],
    ] # fmt: skip

    rows = [*tx_rows, *rx_rows_wo_tx, *rx_rows_w_tx]

    sdb.add_dataframe(
        schedule_dataframe.from_rows(rows),
        sdb.combined_schedule_name,
    )

    df = sdb.get_tx_rx_pointing_pairs(
        start_time="2026-02-01", end_time="2026-03-01", tx_stn_num=0, rx_stn_num=1
    )

    assert all(
        df.loc[t.cast(t.Any, (0, ["exp_num", "time"]))] == [rx_rows_w_tx[0][0], rx_rows_w_tx[0][3]]
    )
    assert all(df["rx_simult_num"] == [rx_rows_w_tx[0][2], rx_rows_w_tx[1][2], rx_rows_w_tx[2][2]])

    return


def test_schedule_by_priority_by_very_simple_case():
    sch_0 = schedule_dataframe.from_rows(
        [
            (0, 0, 0, "2026-02-11 00:00:00", "2026-02-11 00:00:59", 0.1, 0.2, 0.3),
            (0, 0, 0, "2026-02-11 00:02:00", "2026-02-11 00:02:59", 0.1, 0.2, 0.3),
        ]
    )
    sch_1 = schedule_dataframe.from_rows(
        [
            (0, 0, 0, "2026-02-11 00:00:30", "2026-02-11 00:00:59", 0.1, 0.2, 0.3),
        ]
    )
    result = schedule_dataframe.schedule_by_priority(schs=[sch_0, sch_1], priorities=[0, 1])

    assert pd.DataFrame.equals(result, sch_0)
    return


def test_time_overlapped_mask():
    time_range_str_list = [
        ("2026-02-11 00:00:00", "2026-02-11 00:00:59"),
        ("2026-02-11 00:00:30", "2026-02-11 00:00:59"),
        ("2026-02-11 00:02:00", "2026-02-11 00:02:59"),
    ]
    start_time = np.array(
        [trange[0] for trange in time_range_str_list],
        dtype="datetime64[us]",
    )
    end_time = np.array(
        [trange[1] for trange in time_range_str_list],
        dtype="datetime64[us]",
    )

    mask = schedule_dataframe.time_overlapped_mask(start_time=start_time, end_time=end_time)

    assert np.all(
        mask == [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ] # fmt: skip
    )
    return


def test_priority_loser_mask():
    priorities = np.array([0, 1, 0], dtype=np.int64)

    mask = schedule_dataframe.priority_loser_mask(priority=priorities)

    assert np.all(
        mask == [
            [False, False, False],
            [True, False, True],
            [True, False, False],
        ] # fmt: skip
    )
    return


def test_get_tx_rx_pointing_pairs_uses_inner_join():
    """`get_tx_rx_pointing_pairs` should use inner join over on columns `exp_num` `start_time` `end_time`."""

    _SK = schedule_dataframe.ScheduleKey
    _PK = schedule.TxRxPointingPairsKey

    df_input = schedule_dataframe.from_rows(
        [
            [0,0,0, pd.Timestamp("2026-02-11 01:00:00", unit="us"), pd.Timestamp("2026-02-11 02:00:00", unit="us"), 0,0,0], # fmt: skip
            [0,0,0, pd.Timestamp("2026-02-11 02:00:00", unit="us"), pd.Timestamp("2026-02-11 03:00:00", unit="us"), 0,0,0], # fmt: skip
            [0,1,0, pd.Timestamp("2026-02-11 02:00:00", unit="us"), pd.Timestamp("2026-02-11 03:00:00", unit="us"), 0,0,0], # fmt: skip
            [0,1,0, pd.Timestamp("2026-02-11 03:00:00", unit="us"), pd.Timestamp("2026-02-11 04:00:00", unit="us"), 0,0,0], # fmt: skip
        ]
    )

    df_result = schedule_dataframe.get_tx_rx_pointing_pairs(
        sch=df_input,
        start_time=df_input[_SK.start_time].min(),
        end_time=df_input[_SK.end_time].max(),
        tx_stn_num=0,
        rx_stn_num=1,
    )

    # assert the row at time == "2026-02-11 02:00:00" is the only one remains
    assert (df_result[_PK.time] == pd.Timestamp("2026-02-11 02:00:00", unit="us")).all()

    return
