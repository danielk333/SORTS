"""
following the same params as in `examples/examples/simulate_scanning_v2.py`

NOTE: WIP; this is currently a testing ground for refactored code
TODO: complete it and clean up
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

# TODO: switch to normal named imports; these are tmp alias until `_v2` becomes the default namespace
Simulation = sortsV2.simulation.Simulation
SimulationParam = sortsV2.simulation.SimulationParam

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

exp_detail = sortsV2.detection_systems.ExperimentDetail(
    coh_int_bandwidth=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    ipp=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    pulse_length=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    power=5000000.0,
    bandwidth=52.08333333333333,
    duty_cycle=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    noise_temp=150.0,
)
exp_num_map: dict[int, sortsV2.detection_systems.ExperimentDetail] = {0: exp_detail}

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

detection_system = sortsV2.detection_systems.StxMrxRadarSystem(
    {
        "tx_station": eiscat3d.tx[0],
        "tx_schedule": tx_schedule,
        "rx_stations": [eiscat3d.rx[0]],
        "rx_schedules": [rx_schedule],
        "exp_num_map": exp_num_map,
    }
)

sim = Simulation(
    SimulationParam(
        epoch=epoch,
        start_time=start_time,
        end_time=end_time,
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
        # space_objects_dt_interpolator_s=sorts.interpolation.Legendre8,
        space_objects_dt_interpolator_s=sorts.interpolation.Linear,
        find_passes_time_ranges=detection_system.find_passes_time_ranges,
        get_schedule_mask_by_time_range=detection_system.get_schedule_mask_by_time_range,
        calculate_observation=detection_system.calculate_observation,
    )
)

# sim.run()
obss = sim.calculate_observations()

obs = obss[0][0]
rx_sch_pass_mask = detection_system.param.rx_schedules[0].create_mask_by_time_range(obs.time_range)
rx_sch_pass = detection_system.param.rx_schedules[0].filter_by_mask(rx_sch_pass_mask)

##
# do some plottings
##
fig, axs = plt.subplots(2, 2)

sch_dt_s_arr = (
    detection_system.param.rx_schedules[0].stt_tstmp_us - np.datetime64(sim.param.epoch)
).astype("timedelta64[us]").astype(np.float64) / 1e6
sch_dt_s_arr_pass = sch_dt_s_arr[rx_sch_pass_mask]

axs[0, 0].plot(
    rx_sch_pass.stt_tstmp_us,
    np.log10(np.clip(obs.snr, a_min=1, a_max=None)) * 10,
    "r",
)

# interpolation functions for secondary x-axis
datetimef = mdates.date2num(rx_sch_pass.stt_tstmp_us)
# NOTE: `fill_value="extrapolate"` triggers error but is actually okay
datetimef_to_timedelta = interp1d(datetimef, sch_dt_s_arr_pass, fill_value="extrapolate")  # type: ignore
timedelta_to_datetimef = interp1d(sch_dt_s_arr_pass, datetimef, fill_value="extrapolate")  # type: ignore

# add secondary x-axis
axs[0, 0].secondary_xaxis("top", functions=(datetimef_to_timedelta, timedelta_to_datetimef))

axs[0, 1].plot(
    detection_system.param.rx_schedules[0].stt_tstmp_us,
    detection_system.param.rx_schedules[0].pointing_az,
    "r",
)
axs[0, 1].plot(
    detection_system.param.rx_schedules[0].stt_tstmp_us,
    detection_system.param.rx_schedules[0].pointing_el,
    "g",
)

plt.show()
