import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sorts.propagator import SGP4
from sorts.space_object import SpaceObject
from sorts.radar.radars import get_radar
from sorts.types import Datetime64_us, Timedelta64_us, Float64_as_sec
from sorts.schedule_v2 import ExperimentDetail
from sorts.scheduler_v2.priority_scheduling import priority_scheduling
from sorts.controller_v2 import TrackerController


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()

    import pandas as pd

    pd.set_option("display.expand_frame_repr", False)


def priority_scheduling_smoke_test():
    epoch = Time(53005.0, format="mjd", scale="utc")  # 2004-01-01 00:00:00Z

    # this start and end time pair should return empty schedule
    # start_time = Time("2025-06-30 00:00:00")
    # end_time = Time("2025-06-30 00:00:01")

    # this start and end time pair should return non-empty schedule
    start_time = Time("2025-01-01 04:04:00")
    end_time = Time("2025-01-01 04:04:01")

    sch_1_slice_duration = np.timedelta64(10_000, "us")  # 10ms
    sch_2_slice_duration = np.timedelta64(10_000 // 3, "us")  # 33.3...ms

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

    time_arr_1: npt.NDArray[Datetime64_us] = np.arange(
        start_time.to_value("datetime64").astype("datetime64[us]"),  # type: ignore
        end_time.to_value("datetime64").astype("datetime64[us]"),  # type: ignore
        sch_1_slice_duration,
    )
    dt_arr_1: npt.NDArray[Timedelta64_us] = time_arr_1 - epoch.to_value("datetime64").astype("datetime64[us]")  # type: ignore
    dsec_arr_1: npt.NDArray[Float64_as_sec] = dt_arr_1.astype(np.float64) / 1e6  # type: ignore
    ecefs_1 = spobj.get_state(dsec_arr_1)

    exp_detail_0 = ExperimentDetail(
        id=0,
        coh_int_bandwidth=1.0,
        ipp=1.0,
        pulse_length=1.0,
        power=5e8,
        bandwidth=52.08333333333333,
        duty_cycle=1.0,
        noise_temp=150.0,
        slice_duration=sch_1_slice_duration,
    )

    controller_1 = TrackerController(
        tx_station=eiscat3d.tx[0],
        rx_stations=[],
        time=time_arr_1,
        space_object_states=ecefs_1,
        exp_detail=exp_detail_0,
    )

    time_arr_2: npt.NDArray[Datetime64_us] = np.arange(
        start_time.to_value("datetime64").astype("datetime64[us]"),  # type: ignore
        end_time.to_value("datetime64").astype("datetime64[us]"),  # type: ignore
        sch_2_slice_duration,
    )
    dt_arr_2: npt.NDArray[Timedelta64_us] = time_arr_2 - epoch.to_value("datetime64").astype("datetime64[us]")  # type: ignore
    dsec_arr_2: npt.NDArray[Float64_as_sec] = dt_arr_2.astype(np.float64) / 1e6  # type: ignore
    ecefs_2 = spobj.get_state(dsec_arr_2)

    exp_detail_1 = ExperimentDetail(
        id=1,
        coh_int_bandwidth=1.0,
        ipp=1.0,
        pulse_length=1.0,
        power=5e8,
        bandwidth=52.08333333333333,
        duty_cycle=1.0,
        noise_temp=150.0,
        slice_duration=sch_2_slice_duration,
    )

    controller_2 = TrackerController(
        tx_station=eiscat3d.tx[0],
        rx_stations=[],
        time=time_arr_2,
        space_object_states=ecefs_2,
        exp_detail=exp_detail_1,
    )

    tx_sch_1, _rx_schs = controller_1.generate()
    tx_sch_2, _rx_schs = controller_2.generate()

    merged_sch = priority_scheduling([tx_sch_1, tx_sch_2], {0: exp_detail_0, 1: exp_detail_1})
    merged_sch_df = merged_sch.to_dataframe()

    return
