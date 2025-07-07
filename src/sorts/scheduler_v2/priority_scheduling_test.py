import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sorts.propagator import SGP4
from sorts.space_object import SpaceObject
from sorts.radar.radars import get_radar
from sorts.types import Datetime64_us, Timedelta64_us, Float64_as_sec
from sorts.schedule_v2 import Schedule
from sorts.scheduler_v2.priority_scheduling import priority_scheduling
from sorts.controller_v2 import TrackerController
from sorts.simulation_v2 import ExperimentDetail


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()

    import pandas as pd

    pd.set_option("display.expand_frame_repr", False)


def priority_scheduling_smoke_test():
    epoch = Time(53005.0, format="mjd", scale="utc")  # 2004-01-01 00:00:00Z
    start_time = Time("2025-06-30 00:00:00")
    end_time = Time("2025-06-30 00:00:01")
    control_slice_duration = np.timedelta64(10_000, "us")  # 10ms

    eiscat3d = get_radar("eiscat3d", "stage1-array")

    spobj = SpaceObject(
        SGP4,
        propagator_options={"settings": {"out_frame": "ITRF"}},
        a=7200e3,
        e=0.02,
        i=75,
        raan=86,
        aop=0,
        mu0=60,
        epoch=epoch,
        parameters={"d": 0.1},
    )

    time_arr: npt.NDArray[Datetime64_us] = np.arange(
        start_time.to_value("datetime64").astype("datetime64[us]"),  # type: ignore
        end_time.to_value("datetime64").astype("datetime64[us]"),  # type: ignore
        control_slice_duration,
    )
    dt_arr: npt.NDArray[Timedelta64_us] = time_arr - epoch.to_value("datetime64").astype("datetime64[us]")  # type: ignore
    dsec_arr: npt.NDArray[Float64_as_sec] = dt_arr.astype(np.float64) / 1e6  # type: ignore

    ecefs = spobj.get_state(dsec_arr)

    controller = TrackerController(
        tx_station=eiscat3d.tx[0],
        rx_stations=[],
        time=time_arr,
        space_object_states=ecefs,
        exp_num=0,
        # azimuth_range=None,
        # elevation_range=None,
    )

    tx_sch_1, _rx_schs = controller.generate(use_cache=False)
    tx_sch_2, _rx_schs = controller.generate(use_cache=False)
    tx_sch_2.stt_tstmp_us = tx_sch_2.stt_tstmp_us + np.timedelta64(int(10e3 / 3), "us")
    tx_sch_2.exp_num = tx_sch_2.exp_num + 1

    exp_detail_map = {
        0: ExperimentDetail(
            coh_int_bandwidth=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            ipp=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            pulse_length=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            # power=5000000.0,
            power=5e8,  # TODO: tmp 100x higher for debugging; restore the value afterwards
            bandwidth=52.08333333333333,
            duty_cycle=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            noise_temp=150.0,
            slice_duration=np.timedelta64(10_000, "us"),  # 10ms
        ),
        1: ExperimentDetail(
            coh_int_bandwidth=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            ipp=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            pulse_length=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            # power=5000000.0,
            power=5e8,  # TODO: tmp 100x higher for debugging; restore the value afterwards
            bandwidth=52.08333333333333,
            duty_cycle=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            noise_temp=150.0,
            slice_duration=np.timedelta64(10_000, "us"),  # 10ms
        ),
    }

    merged_sch = priority_scheduling([tx_sch_1, tx_sch_2], exp_detail_map)

    print(merged_sch)
    return
