import typing as t
import numpy as np
import xarray as xr
from sorts.schedule_v2.schedule import Schedule
from sorts.schedule_v2.priority_scheduling import priority_scheduling, DsVarKey


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()

    import pandas as pd

    pd.set_option("display.expand_frame_repr", False)


def priority_scheduling_interleaved_schedule_test():
    # define some column names/keys
    keys: dict[DsVarKey, str] = {k: k for k in t.get_args(DsVarKey)}

    # 1hr long, 2hr intv
    sch_a = Schedule.from_ndarrays(
        {
            "stn_id": "stn_a",
            "exp_detail_map": {
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
            "start_time": np.arange(
                np.datetime64("2025-01-01", "s"),
                np.datetime64("2025-01-02", "s"),
                np.timedelta64(3600 * 2, "s"),
            ),
            "end_time": np.arange(
                np.datetime64("2025-01-01 00:30:00", "s"),
                np.datetime64("2025-01-02 00:00:01", "s"),
                np.timedelta64(3600 * 2, "s"),
            ),
            "exp_num": np.full(12, 0, dtype=np.int64),
            "pointing": np.full((3, 12), 0.0, dtype=np.float64),
        }
    )

    # 1hr long, 1hr intv
    sch_b = Schedule.from_ndarrays(
        {
            "stn_id": "stn_b",
            "exp_detail_map": {
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
            "start_time": np.arange(
                np.datetime64("2025-01-01", "s"),
                np.datetime64("2025-01-02", "s"),
                np.timedelta64(3600, "s"),
            ),
            "end_time": np.arange(
                np.datetime64("2025-01-01 01:00:00", "s"),
                np.datetime64("2025-01-02 00:00:01", "s"),
                np.timedelta64(3600, "s"),
            ),
            "exp_num": np.full(24, 1, dtype=np.int64),
            "pointing": np.full((3, 24), 1.0, dtype=np.float64),
        }
    )

    resultant_sch_data = priority_scheduling([sch_a.data, sch_b.data])

    assert all(xr.ufuncs.equal(resultant_sch_data[keys["exp_num"]], [0, 1] * 12))
    assert resultant_sch_data.attrs[keys["stn_id"]] == "stn_a"

    return
