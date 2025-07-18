import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sorts.radar.radars import get_radar
from sorts.schedule_v2 import ExperimentDetail
from sorts.controller_v2.fence_scan_controller_new import FenceScanController


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def FenceScanController_smoke_test():
    """
    Check if the `FenceScanController` can be initialize and generate schedule(s) with correct length in a basic settings.

    Note: only tx schedule is checked at the moment

    TODO: should also check if other fields are correct
    """

    start_time_np = np.datetime64("2025-06-30 00:00:00", "us")
    end_time_np = np.datetime64("2025-06-30 00:00:01", "us")
    slice_duration = np.timedelta64(10_000, "us")  # 10ms

    eiscat3d = get_radar("eiscat3d", "stage1-array")

    exp_detail_1 = ExperimentDetail(
        id=1,
        coh_int_bandwidth=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
        ipp=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
        pulse_length=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
        power=5000000.0,
        bandwidth=52.08333333333333,
        duty_cycle=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
        noise_temp=150.0,
        slice_duration=slice_duration,
    )

    fenceScanController = FenceScanController(
        tx_station=eiscat3d.tx[0],
        rx_station=[],
        exp_datail=exp_detail_1,
        azimuth=90,  # sweep from east to west
        min_elevation=30,
        pointings_per_cycle=40,
    )

    schs = fenceScanController.generate(Time(start_time_np), Time(end_time_np))
    expected_sch_len = round((end_time_np - start_time_np) / slice_duration)

    assert len(schs.tx_schedule.start_time) == expected_sch_len

    return
