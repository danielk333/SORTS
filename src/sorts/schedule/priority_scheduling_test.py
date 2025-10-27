import numpy as np
import xarray as xr
from sorts import schedule
from sorts.schedule import Schedule
from sorts.schedule.priority_scheduling import priority_scheduling


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()

    import pandas as pd

    pd.set_option("display.expand_frame_repr", False)


def priority_scheduling_interleaved_schedule_test():
    _SK = schedule._K

    # 30min long, 2hr intv
    sch_data_a = schedule.from_ndarrays(
        {
            _SK.start_time: np.arange(
                np.datetime64("2025-01-01", "s"),
                np.datetime64("2025-01-02", "s"),
                np.timedelta64(3600 * 2, "s"),
            ),
            _SK.end_time: np.arange(
                np.datetime64("2025-01-01 00:30:00", "s"),
                np.datetime64("2025-01-02 00:00:01", "s"),
                np.timedelta64(3600 * 2, "s"),
            ),
            _SK.exp_num: np.full(12, 0, dtype=np.int16),
            _SK.stn_num: np.full(12, 0, dtype=np.int16),
            _SK.simult_num: np.full(12, 0, dtype=np.int16),
            _SK.pointing: np.full((3, 12), 0.0, dtype=np.float64),
        },
    )
    sch_data_a = Schedule(sch_data_a)

    # 1hr long, 1hr intv
    sch_data_b = schedule.from_ndarrays(
        {
            _SK.start_time: np.arange(
                np.datetime64("2025-01-01", "s"),
                np.datetime64("2025-01-02", "s"),
                np.timedelta64(3600, "s"),
            ),
            _SK.end_time: np.arange(
                np.datetime64("2025-01-01 01:00:00", "s"),
                np.datetime64("2025-01-02 00:00:01", "s"),
                np.timedelta64(3600, "s"),
            ),
            _SK.exp_num: np.full(24, 1, dtype=np.int16),
            _SK.stn_num: np.full(24, 0, dtype=np.int16),
            _SK.simult_num: np.full(24, 0, dtype=np.int16),
            _SK.pointing: np.full((3, 24), 1.0, dtype=np.float64),
        }
    )
    sch_data_b = Schedule(sch_data_b)

    resultant_sch_data = priority_scheduling([sch_data_a, sch_data_b], {0: [(0, 0)], 1: [(0, 0)]})

    assert all(xr.ufuncs.equal(resultant_sch_data[_SK.exp_num], [0, 1] * 12))

    return


# TODO: add test case to ensure rx entries are removed when tx entries are removed.
#   this is related to the resolved error::
#
#   spobjs = [tracked_spobj]
#   Error "not all values found in index 'multi_index'",  at space_object.oid = 20; sim_unit.id = 6
#   ```
#   pd.MultiIndex.from_tuples(tx_sch_pointing_selector).isin(tx_sch._data.indexes["multi_index"])
#
#   :> np.argmax(~pd.MultiIndex.from_tuples(tx_sch_pointing_selector).isin(tx_sch._data.indexes["multi_index"]))
#   -> np.int64(22447)
#   tx_sch_pointing_selector[22447]
#   (np.datetime64('2025-01-01T02:49:16.530000'), np.int16(1), np.int16(0))
#   ```
#
#   also, need to ensure the existence of tx for one experiment
#   should not lead to retention of rx of another experiment, e.g. case like this:
#   ```
#   incoming_sch_data.loc[{_SK.multi_index: '2025-01-01 02:49:16.530000'}].to_dataframe()
#                                                 end_time      pointing  exp_num  stn_num  simult_num cummax_start_time cummax_end_time  is_overlaped                  start_time
#   exp_num stn_num simult_num enu
#   1       0       0          e   2025-01-01 02:49:16.540  3.420201e-01        1        0           0               NaT             NaT         False  2025-01-01 02:49:16.530000
#                              n   2025-01-01 02:49:16.540  2.094269e-17        1        0           0               NaT             NaT         False  2025-01-01 02:49:16.530000
#                              u   2025-01-01 02:49:16.540  9.396926e-01        1        0           0               NaT             NaT         False  2025-01-01 02:49:16.530000
#           1       0          e   2025-01-01 02:49:16.540  1.026060e+05        1        1           0               NaT             NaT         False  2025-01-01 02:49:16.530000
#                              n   2025-01-01 02:49:16.540  2.008874e-10        1        1           0               NaT             NaT         False  2025-01-01 02:49:16.530000
#                              u   2025-01-01 02:49:16.540  2.819078e+05        1        1           0               NaT             NaT         False  2025-01-01 02:49:16.530000
#           2       0          e   2025-01-01 02:49:16.540  1.428677e+04        1        2           0               NaT             NaT         False  2025-01-01 02:49:16.530000
#                              n   2025-01-01 02:49:16.540  1.000510e+05        1        2           0               NaT             NaT         False  2025-01-01 02:49:16.530000
#                              u   2025-01-01 02:49:16.540  2.817103e+05        1        2           0               NaT             NaT         False  2025-01-01 02:49:16.530000
#
#    ---
#
#    merged_sch_data.loc[{_SK.multi_index: '2025-01-01 02:49:16.530000'}].to_dataframe()
#                                                  end_time      pointing       cummax_start_time         cummax_end_time  ...  exp_num  stn_num  simult_num                  start_time
#    exp_num stn_num simult_num enu                                                                                        ...
#    0       0       0          e   2025-01-01 02:49:16.540 -1.137902e+06 2025-01-01 02:49:16.530 2025-01-01 02:49:16.540  ...        0        0           0  2025-01-01 02:49:16.530000
#                               n   2025-01-01 02:49:16.540  4.320081e+05 2025-01-01 02:49:16.530 2025-01-01 02:49:16.540  ...        0        0           0  2025-01-01 02:49:16.530000
#                               u   2025-01-01 02:49:16.540  7.027450e+05 2025-01-01 02:49:16.530 2025-01-01 02:49:16.540  ...        0        0           0  2025-01-01 02:49:16.530000
#            1       0          e   2025-01-01 02:49:16.540 -1.137902e+06 2025-01-01 02:49:16.530 2025-01-01 02:49:16.540  ...        0        1           0  2025-01-01 02:49:16.530000
#                               n   2025-01-01 02:49:16.540  4.320081e+05 2025-01-01 02:49:16.530 2025-01-01 02:49:16.540  ...        0        1           0  2025-01-01 02:49:16.530000
#                               u   2025-01-01 02:49:16.540  7.027450e+05 2025-01-01 02:49:16.530 2025-01-01 02:49:16.540  ...        0        1           0  2025-01-01 02:49:16.530000
#    1       2       0          e   2025-01-01 02:49:16.540  1.428677e+04                     NaT                     NaT  ...        1        2           0  2025-01-01 02:49:16.530000
#                               n   2025-01-01 02:49:16.540  1.000510e+05                     NaT                     NaT  ...        1        2           0  2025-01-01 02:49:16.530000
#                               u   2025-01-01 02:49:16.540  2.817103e+05                     NaT                     NaT  ...        1        2           0  2025-01-01 02:49:16.530000
#
#    ---
#
#    tx_schdata.loc[{_SK.multi_index: ('2025-01-01 02:49:16.530000', 1, 0, 0)}]  # will trigger not found error
#   ```

# TODO: add test case to ensure we do not shift among the index level `simult_num` and `stn_num`
#   when updating the `allowed_start_time`, `allowed_end_time`;
#   related func: '_update_allowed_start_time_allowed_end_time'
