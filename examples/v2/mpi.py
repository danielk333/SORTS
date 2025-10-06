import pickle, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import numpy.typing as npt
from astropy.time import Time
import logging
import sorts
from sorts import (
    interpolation,
    population,
    propagator,
    space_object,
    radar,
    controller,
    schedule,
    simulation,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("starting example")

# 15min runtime
start_time = Time("2025-01-01 02:45:00")
# start_time = Time("2025-01-01 02:59:59")
end_time = Time("2025-01-01 03:00:00")
control_slice_duration = np.timedelta64(10_000, "us")  # 10ms

eiscat3d = sorts.get_radar("eiscat3d", "stage1-array")
# eiscat3d = get_radar("nostra", "example1")
# TODO: these patching of station prop should be integrated into codebase
tx_station: radar.Station = eiscat3d.tx[0]
tx_station.uid = 0
rx_station_0: radar.Station = eiscat3d.rx[0]
rx_station_0.uid = 1
rx_station_1: radar.Station = eiscat3d.rx[1]
rx_station_1.uid = 2

stn_num_map = {
    0: "eiscat3d, stage1-array, tx, 0",
    1: "eiscat3d, stage1-array, rx, 0",
    2: "eiscat3d, stage1-array, rx, 1",
}

tracked_spobj = space_object.SpaceObject(
    oid=-1,
    propagator=propagator.SGP4,
    propagator_options={"settings": {"out_frame": "ITRF"}},
    a=7200e3,
    e=0.02,
    i=75,
    raan=86,
    aop=0,
    mu0=60,
    epoch=start_time,
    parameters={"d": 0.1},
)

catalog_fpath = default = (
    Path(__file__).parent / ".." / ".." / "local_data" / "celn_20090501_00.sim"
)
_spobj_pop = population.master_catalog(
    catalog_fpath,
    propagator=propagator.SGP4,
    propagator_options={"settings": {"in_frame": "TEME", "out_frame": "ITRF"}},
)
rand_seed = 120389
# TODO: reduce the filter size to more sensible value
spobj_pop = population.master_catalog_factor(_spobj_pop, treshhold=5.0, seed=rand_seed)
# spobjs = [tracked_spobj, *[spobj_pop.get_object(i) for i in range(spobj_pop.shape[0])]]
# spobjs = [tracked_spobj, *[spobj_pop.get_object(i) for i in range(spobj_pop.shape[0])][0:21]]
spobjs = [tracked_spobj, spobj_pop.get_object(20)]


def dsec_sampler(orbit, start_time, end_time):
    return np.arange(0, (end_time - start_time) / np.timedelta64(1, "s"), 120, dtype=np.float64)


tracker_ctrl = controller.TrackerController.from_space_object(
    spobj=tracked_spobj,
    epoch=start_time,
    tx_station=tx_station,
    rx_stations=[rx_station_0, rx_station_1],
    exp_detail={
        "id": 0,
        "coh_int_bandwidth": 1.0,
        "ipp": 1.0,
        "pulse_length": 1.0,
        "power": 5000000.0,
        "bandwidth": 52.08333333333333,
        "duty_cycle": 1.0,
        "noise_temp": 150.0,
        "slice_duration": control_slice_duration,
        "stn_pairs": [(0, 1), (0, 2)],
    },
)

fence_scan_ctrl = controller.FenceScanController.from_scan_spec(
    tx_station=tx_station,
    rx_stations=[rx_station_0, rx_station_1],
    exp_detail={
        "id": 1,
        "coh_int_bandwidth": 1.0,
        "ipp": 1.0,
        "pulse_length": 1.0,
        "power": 5000000.0,
        "bandwidth": 52.08333333333333,
        "duty_cycle": 1.0,
        "noise_temp": 150.0,
        "slice_duration": control_slice_duration,
        "stn_pairs": [(0, 1), (0, 2)],
    },
    azimuth=90,  # sweep from east to west
    min_elevation=30,
    pointings_per_cycle=40,
    # scan_range=np.linspace(300e3, 1000e3, num=10, dtype=np.float64),
    scan_range=np.array([300e3], dtype=np.float64),
)

tracker_sch = tracker_ctrl.generate(start_time, end_time)
fence_sch = fence_scan_ctrl.generate(start_time, end_time)

exp_detail_map = {
    tracker_ctrl.spec["exp_detail"]["id"]: tracker_ctrl.spec["exp_detail"],
    fence_scan_ctrl.spec["exp_detail"]["id"]: fence_scan_ctrl.spec["exp_detail"],
}


master_sch = schedule.Schedule.priority_scheduling([tracker_sch, fence_sch])


sim = simulation.StxMrxSimulation(
    spec={
        "tx_station": tx_station,
        "rx_stations": [rx_station_0, rx_station_1],
        "schedule": master_sch,
        "exp_detail_map": exp_detail_map,
        "epoch": start_time,
        "start_time": start_time,
        "end_time": end_time,
        "space_objects": spobjs,
        "dsec_sampler": dsec_sampler,
        # "interpolator_class": interpolation.Legendre8,
        "interpolator_class": interpolation.Linear,
    }
)

calc_start_time = time.perf_counter()
obss, sim_units = sim.mpi_run(Path(__file__).parent / ".." / ".." / "local_data" / "mpi")
# obss, sim_units = sim.run()  # or, do not use non-mpi for debugging
calc_time = time.perf_counter() - calc_start_time

print(f"len(obss): {len(obss)}")
for idx, obs in enumerate(obss):
    print(f"obs: {idx}")
    print(obs.passage)
    print(obs.get_state_slice())

exit()
