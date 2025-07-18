from copy import deepcopy
import numpy as np
from sorts.schedule_v2 import Schedule, ExperimentDetail


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def Schedule_dataframe_roundtrip_conversion_test():
    meta: dict[int, ExperimentDetail] = {
        0: ExperimentDetail(
            id=0,
            coh_int_bandwidth=1.0,
            ipp=1.0,
            pulse_length=1.0,
            power=1.0,
            bandwidth=1.0,
            duty_cycle=1.0,
            noise_temp=1.0,
            slice_duration=np.timedelta64(10_000, "us"),
        )
    }

    start_time = np.arange(
        np.datetime64("2025-06-01", "us"),
        np.datetime64("2025-06-02", "us"),
        np.timedelta64(int(1 * 3600 * 1e6), "us"),
    )
    sch_len = len(start_time)

    exp_num = np.full(sch_len, 0)
    pointing_az = np.linspace(0, 180, sch_len, dtype=np.float64)
    pointing_el = np.linspace(0, 90, sch_len, dtype=np.float64)

    # we create the schedule using copies of the field data, so the conversions will not mutate the originals
    sch = Schedule(
        meta=deepcopy(meta),
        start_time=start_time.copy(),
        exp_num=exp_num.copy(),
        pointing_az=pointing_az.copy(),
        pointing_el=pointing_el.copy(),
    )

    converted_sch = Schedule.from_dataframe(sch.as_dataframe(), sch.meta)

    assert meta == converted_sch.meta
    assert np.array_equal(start_time, converted_sch.start_time)
    assert np.array_equal(exp_num, converted_sch.exp_num)
    assert np.array_equal(pointing_az, converted_sch.pointing_az)
    assert np.array_equal(pointing_el, converted_sch.pointing_el)

    return
