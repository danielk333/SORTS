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
            _SK.stn_id: "stn_a",
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
            _SK.simult_num: np.full(12, 0, dtype=np.int16),
            _SK.pointing: np.full((3, 12), 0.0, dtype=np.float64),
        },
    )

    # 1hr long, 1hr intv
    sch_data_b = schedule_data_funcs.from_ndarrays(
        {
            _SK.stn_id: "stn_b",
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
            _SK.simult_num: np.full(24, 0, dtype=np.int16),
            _SK.pointing: np.full((3, 24), 1.0, dtype=np.float64),
        }
    )

    resultant_sch_data = priority_scheduling([sch_data_a, sch_data_b])

    assert all(xr.ufuncs.equal(resultant_sch_data[_SK.exp_num], [0, 1] * 12))
    assert resultant_sch_data.attrs[_SK.stn_id] == "stn_a"

    return
