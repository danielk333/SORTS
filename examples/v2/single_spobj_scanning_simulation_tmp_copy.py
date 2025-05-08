"""
A temp copy of `examples/v2/single_spobj_scanning_simulation.py`,
it is used a testing ground for refactored code.

Will be removed afterwards.
"""

import typing as t
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time
import sorts
from sorts import _v2 as sortsV2

# TODO: might be if `epoch`, `start_time`, `end_time` can be integrated into some config or dataclass ?
epoch = t.cast(
    datetime, Time(53005.0, format="mjd", scale="utc").to_datetime(timezone=timezone.utc)
)  # 2004-01-01 00:00:00Z
start_time = t.cast(
    datetime,
    Time("2004-01-01 00:00:00Z", format="iso", scale="utc").to_datetime(timezone=timezone.utc),
)
end_time = t.cast(
    datetime,
    Time("2004-01-01 00:10:00Z", format="iso", scale="utc").to_datetime(timezone=timezone.utc),
)  # 600 sec after start time

eiscat3d = sorts.get_radar("eiscat3d", "stage1-array")

exp_detail = sortsV2.detection_config.ExperimentDetail(
    coh_int_bandwidth=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    ipp=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    pulse_length=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    power=5000000.0,
    bandwidth=52.08333333333333,
    duty_cycle=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    noise_temp=150.0,
)
exp_num_map: dict[int, sortsV2.detection_config.ExperimentDetail] = {0: exp_detail}

fence_scan_controller = sortsV2.controller.FenceScanController(
    tx_station=eiscat3d.tx[0],
    rx_station=eiscat3d.rx[0],
    azimuth_deg=90,
    min_elevation_deg=30,
    dwell_s=0.1,
    num=40,
    start_time=start_time,
    end_time=end_time,
    exp_num=0,
)

(tx_schedule, rx_schedule) = fence_scan_controller.generate(start_time, end_time)

sim = sortsV2.Simulation(
    epoch=epoch,
    start_time=start_time,
    end_time=end_time,
    detection_config=sortsV2.detection_config.StxSrx(
        tx_station=eiscat3d.tx[0],
        tx_schedule=tx_schedule,
        rx_station=eiscat3d.rx[0],
        rx_schedule=rx_schedule,
        exp_num_map=exp_num_map,
    ),
    space_objects=[
        sorts.SpaceObject(
            sorts.propagator.SGP4,
            propagator_options={"settings": {"out_frame": "ITRF"}},
            a=7200e3,
            e=0.02,
            i=75,
            raan=86,
            aop=0,
            mu0=60,
            epoch=Time(epoch),
            parameters={"d": 0.1},
        )
    ],
    space_objects_dt_sampler_s=lambda orbit, start_time, end_time: sorts.equidistant_sampling(
        orbit=orbit,
        start_t=(start_time - epoch).total_seconds(),
        end_t=(end_time - epoch).total_seconds(),
        max_dpos=1e3,
    ),
    exp_num_map={
        0: sortsV2.detection_config.ExperimentDetail(
            coh_int_bandwidth=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            ipp=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            pulse_length=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            power=5000000.0,
            bandwidth=52.08333333333333,
            duty_cycle=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
            noise_temp=150.0,
        )
    },
)

# sim.run()
obs, sch_dt_s_arr_pass_mask = sim.calculate_observations()
# target_pass_obj = pass_arr[0][0]


##
# do some plottings
##
fig, axs = plt.subplots(2, 2)

sch_dt_s_arr = (sim.detection_config.rx_schedule.stt_tstmp_us - np.datetime64(sim.epoch)).astype(
    "timedelta64[us]"
).astype(np.float64) / 1e6
sch_dt_s_arr_pass = sch_dt_s_arr[sch_dt_s_arr_pass_mask]

axs[0, 0].plot(
    sim.detection_config.rx_schedule.stt_tstmp_us[sch_dt_s_arr_pass_mask],
    np.log10(np.clip(obs.snr, a_min=1, a_max=None)) * 10,
    "r",
)

# interpolation functions for secondary x-axis
datetimef = mdates.date2num(sim.detection_config.rx_schedule.stt_tstmp_us[sch_dt_s_arr_pass_mask])
# NOTE: `fill_value="extrapolate"` triggers error but is actually okay
datetimef_to_timedelta = interp1d(datetimef, sch_dt_s_arr_pass, fill_value="extrapolate")  # type: ignore
timedelta_to_datetimef = interp1d(sch_dt_s_arr_pass, datetimef, fill_value="extrapolate")  # type: ignore

# add secondary x-axis
axs[0, 0].secondary_xaxis("top", functions=(datetimef_to_timedelta, timedelta_to_datetimef))

axs[0, 1].plot(
    sim.detection_config.rx_schedule.stt_tstmp_us,
    sim.detection_config.rx_schedule.pointing_az,
    "r",
)
axs[0, 1].plot(
    sim.detection_config.rx_schedule.stt_tstmp_us,
    sim.detection_config.rx_schedule.pointing_el,
    "g",
)

plt.show()
