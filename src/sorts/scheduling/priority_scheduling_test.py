import numpy as np
import xarray as xr
from sorts import scheduling
from sorts.scheduling import Schedule
from sorts.scheduling.priority_scheduling import priority_scheduling


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()

    import pandas as pd

    pd.set_option("display.expand_frame_repr", False)


def priority_scheduling_interleaved_schedule_test():
    _SK = scheduling._K

    # 30min long, 2hr intv
    sch_a = scheduling.from_ndarrays(
        start_time=np.arange(
            np.datetime64("2025-01-01", "s"),
            np.datetime64("2025-01-02", "s"),
            np.timedelta64(3600 * 2, "s"),
        ),
        end_time=np.arange(
            np.datetime64("2025-01-01 00:30:00", "s"),
            np.datetime64("2025-01-02 00:00:01", "s"),
            np.timedelta64(3600 * 2, "s"),
        ),
        exp_num=np.full(12, 0, dtype=np.int16),
        stn_num=np.full(12, 0, dtype=np.int16),
        simult_num=np.full(12, 0, dtype=np.int16),
        pointing=np.full((3, 12), 0.0, dtype=np.float64),
    )
    sch_a = Schedule(sch_a)

    # 1hr long, 1hr intv
    sch_b = scheduling.from_ndarrays(
        start_time=np.arange(
            np.datetime64("2025-01-01", "s"),
            np.datetime64("2025-01-02", "s"),
            np.timedelta64(3600, "s"),
        ),
        end_time=np.arange(
            np.datetime64("2025-01-01 01:00:00", "s"),
            np.datetime64("2025-01-02 00:00:01", "s"),
            np.timedelta64(3600, "s"),
        ),
        exp_num=np.full(24, 1, dtype=np.int16),
        stn_num=np.full(24, 0, dtype=np.int16),
        simult_num=np.full(24, 0, dtype=np.int16),
        pointing=np.full((3, 24), 1.0, dtype=np.float64),
    )
    sch_b = Schedule(sch_b)

    resultant_sch = priority_scheduling([sch_a, sch_b], {0: [(0, 0)], 1: [(0, 0)]})

    assert all(xr.ufuncs.equal(resultant_sch[_SK.exp_num], [0, 1] * 12))

    return
