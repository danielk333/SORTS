import numpy as np
import xarray as xr
from sorts.schedule import schedule_data_funcs
from sorts.schedule import Schedule
from sorts.schedule.priority_scheduling import priority_scheduling


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()

    import pandas as pd

    pd.set_option("display.expand_frame_repr", False)


def priority_scheduling_interleaved_schedule_test():
    _SK = Schedule._K

    # 30min long, 2hr intv
    sch_data_a = schedule_data_funcs.from_ndarrays(
        {
            _SK.exp_detail_map: {
                0: {
                    "id": 0,
                    "coh_int_bandwidth": 0,
                    "ipp": 0,
                    "pulse_length": 0,
                    "power": 0,
                    "bandwidth": 0,
                    "duty_cycle": 0,
                    "noise_temp": 0,
                    "slice_duration": np.timedelta64(3600, "s"),
                    "stn_pairs": [(0, 0)],
                }
            },
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

    # 1hr long, 1hr intv
    sch_data_b = schedule_data_funcs.from_ndarrays(
        {
            _SK.exp_detail_map: {
                1: {
                    "id": 1,
                    "coh_int_bandwidth": 0,
                    "ipp": 0,
                    "pulse_length": 0,
                    "power": 0,
                    "bandwidth": 0,
                    "duty_cycle": 0,
                    "noise_temp": 0,
                    "slice_duration": np.timedelta64(3600, "s"),
                    "stn_pairs": [(0, 0)],
                }
            },
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

    resultant_sch_data = priority_scheduling([sch_data_a, sch_data_b])

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

# TODO: add test case to ensure we do not shift among the index level `simult_num` and `stn_num`
#   when updating the `allowed_start_time`, `allowed_end_time`;
#   related func: '_update_allowed_start_time_allowed_end_time'
