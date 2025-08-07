"""
following the same params as in `examples/examples/simulate_scanning_v2.py`

NOTE: WIP; this is currently a testing ground for refactored code
TODO: complete it and clean up
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time
import sorts
from sorts import _v2 as sortsV2
from sorts.utils import to_pydatetime

# TODO: switch to normal named imports; these are tmp alias until `_v2` becomes the default namespace
StxMrxSimulation = sortsV2.simulation.StxMrxSimulation
StxMrxSimulationParam = sortsV2.simulation.Spec

# TODO: might be if `epoch`, `start_time`, `end_time` can be integrated into some config or dataclass ?
epoch = Time(53005.0, format="mjd", scale="utc")  # 2004-01-01 00:00:00Z
start_time = Time("2004-01-01 00:00:00Z", format="iso", scale="utc")
end_time = Time("2004-01-01 00:10:00Z", format="iso", scale="utc")  # 600 sec after start time

eiscat3d = sorts.get_radar("eiscat3d", "stage1-array")

exp_num_map: dict[int, sortsV2.schedule.ExperimentDetail] = {
    0: {
        "id": 0,
        "coh_int_bandwidth": 1.0,
        "ipp": 1.0,
        "pulse_length": 1.0,
        "power": 5000000.0,
        "bandwidth": 52.08333333333333,
        "duty_cycle": 1.0,
        "noise_temp": 150.0,
        "slice_duration": np.timedelta64(10_000, "us"),  # 10ms
    }
}

tx_station: sorts.Station = eiscat3d.tx[0]
tx_station.uid = ("eiscat3d", "stage1-array", "tx", "0")
rx_station: sorts.Station = eiscat3d.rx[0]
rx_station.uid = ("eiscat3d", "stage1-array", "rx", "0")

fence_scan_controller = sortsV2.controller.FenceScanController.from_scan_spec(
    tx_station=tx_station,
    rx_stations=[rx_station],
    exp_detail=exp_num_map[0],
    azimuth=90,  # sweep from east to west
    min_elevation=30,
    pointings_per_cycle=40,
    scan_range=np.array([300e3], dtype=np.float64),
)

(tx_schedule, rx_schedules) = fence_scan_controller.generate(start_time, end_time)

sim = StxMrxSimulation.from_spec(
    {
        "tx_station": tx_station,
        "tx_schedule": tx_schedule,
        "rx_stations": [rx_station],
        "rx_schedules": rx_schedules,
        "exp_num_map": exp_num_map,
        "epoch": epoch,
        "start_time": start_time,
        "end_time": end_time,
        "space_objects": [
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
        "spobj_dsec_sampler": lambda orbit, start_time, end_time: sorts.equidistant_sampling(
            orbit=orbit,
            start_t=(to_pydatetime(start_time) - to_pydatetime(epoch)).total_seconds(),
            end_t=(to_pydatetime(end_time) - to_pydatetime(epoch)).total_seconds(),
            max_dpos=1e3,
        ),
        # space_object_interpolator_class=sorts.interpolation.Legendre8,
        "space_object_interpolator_class": sorts.interpolation.Linear,
    }
)

# sim.run()
obss = sim.calculate_observations()

obs = obss[0]
rx_sch_pass_mask = sim.state["rx_schedules"][0].create_mask_by_time_range(
    obs["experiment_passage"]["time_range"]
)
rx_sch_pass = sim.state["rx_schedules"][0].filter_by_mask(rx_sch_pass_mask)

##
# do some plottings
##
fig, axs = plt.subplots(2, 2)

sch_dt_s_arr = (sim.state["rx_schedules"][0].start_time - np.datetime64(sim.state["epoch"])).astype(
    "timedelta64[us]"
).astype(np.float64) / 1e6
sch_dt_s_arr_pass = sch_dt_s_arr[rx_sch_pass_mask]

axs[0, 0].plot(
    rx_sch_pass.start_time,
    np.log10(np.clip(obs["snr"], a_min=1, a_max=None)) * 10,
    "r",
)

# interpolation functions for secondary x-axis
datetimef = mdates.date2num(rx_sch_pass.start_time)
# NOTE: `fill_value="extrapolate"` triggers error but is actually okay
datetimef_to_timedelta = interp1d(datetimef, sch_dt_s_arr_pass, fill_value="extrapolate")  # type: ignore
timedelta_to_datetimef = interp1d(sch_dt_s_arr_pass, datetimef, fill_value="extrapolate")  # type: ignore

# add secondary x-axis
axs[0, 0].secondary_xaxis("top", functions=(datetimef_to_timedelta, timedelta_to_datetimef))

axs[0, 1].plot(
    sim.state["rx_schedules"][0].start_time,
    sim.state["rx_schedules"][0].pointing_az,
    "r",
)
axs[0, 1].plot(
    sim.state["rx_schedules"][0].start_time,
    sim.state["rx_schedules"][0].pointing_el,
    "g",
)

plt.show()
