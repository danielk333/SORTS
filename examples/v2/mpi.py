import pickle, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import numpy.typing as npt
from astropy.time import Time
import logging
import sorts
from sorts.interpolation import Legendre8, Linear
from sorts.population import master_catalog, master_catalog_factor
from sorts.propagator import SGP4
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.radar.radars import get_radar
from sorts.controller import TrackerController, FenceScanController
from sorts.schedule import Schedule
from sorts.simulation import StxMrxSimulation

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("starting example")

# 15min runtime
start_time = Time("2025-01-01 02:45:00")
# start_time = Time("2025-01-01 02:59:59")
end_time = Time("2025-01-01 03:00:00")
control_slice_duration = np.timedelta64(10_000, "us")  # 10ms

eiscat3d = get_radar("eiscat3d", "stage1-array")
# eiscat3d = get_radar("nostra", "example1")
# TODO: these patching of station prop should be integrated into codebase
tx_station: Station = eiscat3d.tx[0]
tx_station.uid = "eiscat3d, stage1-array, tx, 0"
rx_station_0: Station = eiscat3d.rx[0]
rx_station_0.uid = "eiscat3d, stage1-array, rx, 0"
rx_station_1: Station = eiscat3d.rx[1]
rx_station_1.uid = "eiscat3d, stage1-array, rx, 1"


tracked_spobj = SpaceObject(
    oid=-1,
    propagator=SGP4,
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
_spobj_pop = master_catalog(
    catalog_fpath,
    propagator=SGP4,
    propagator_options={"settings": {"in_frame": "TEME", "out_frame": "ITRF"}},
)
rand_seed = 120389
# TODO: reduce the filter size to more sensible value
spobj_pop = master_catalog_factor(_spobj_pop, treshhold=5.0, seed=rand_seed)
# spobjs = [tracked_spobj, *[spobj_pop.get_object(i) for i in range(spobj_pop.shape[0])]]
spobjs = [tracked_spobj]


def dsec_sampler(orbit, start_time, end_time):
    return np.arange(0, (end_time - start_time) / np.timedelta64(1, "s"), 120, dtype=np.float64)


tracker_ctrl = TrackerController.from_space_object(
    spobj=tracked_spobj,
    epoch=start_time,
    tx_station=eiscat3d.tx[0],
    # rx_stations=eiscat3d.rx[0:1],
    rx_stations=eiscat3d.rx[0:2],
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
    },
)

fence_scan_ctrl = FenceScanController.from_scan_spec(
    tx_station=eiscat3d.tx[0],
    # rx_stations=eiscat3d.rx[0:1],
    rx_stations=eiscat3d.rx[0:2],
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
    },
    azimuth=90,  # sweep from east to west
    min_elevation=30,
    pointings_per_cycle=40,
    # scan_range=np.linspace(300e3, 1000e3, num=10, dtype=np.float64),
    scan_range=np.array([300e3], dtype=np.float64),
)

tracker_schs = tracker_ctrl.generate(start_time, end_time)
fence_schs = fence_scan_ctrl.generate(start_time, end_time)

exp_detail_map = {
    tracker_ctrl.spec["exp_detail"]["id"]: tracker_ctrl.spec["exp_detail"],
    fence_scan_ctrl.spec["exp_detail"]["id"]: fence_scan_ctrl.spec["exp_detail"],
}


tx_master_sch = Schedule.priority_scheduling([tracker_schs.tx_schedule, fence_schs.tx_schedule])

rx_master_schs = [
    Schedule.priority_scheduling(rx_schs)
    for rx_schs in zip(tracker_schs.rx_schedules, fence_schs.rx_schedules)
]

start_time_entries_diff = rx_master_schs[0]._data.loc[
    {"start_time": ~rx_master_schs[0]._data["start_time"].isin(tx_master_sch._data["start_time"])}
]

assert len(start_time_entries_diff["start_time"]) == 0

sim = StxMrxSimulation(
    spec={
        "tx_schedule": tx_master_sch,
        "rx_schedules": rx_master_schs,
        "exp_detail_map": exp_detail_map,
        "epoch": start_time,
        "start_time": start_time,
        "end_time": end_time,
        "space_objects": spobjs,
        # "space_objects": [
        #     o for i, o in enumerate(spobjs) if i in [0, 4, 5, 17]
        # ],  # just picked a few from the whole list for now
        "dsec_sampler": dsec_sampler,
        # "interpolator_class": Legendre8,
        "interpolator_class": Linear,
    }
)

calc_start_time = time.perf_counter()
obss, sim_units = sim.mpi_run(Path(__file__).parent / ".." / ".." / "local_data" / "mpi")
calc_time = time.perf_counter() - calc_start_time

print(f"len(obss): {len(obss)}")
for idx, obs in enumerate(obss):
    print(f"obs: {idx}")
    print(obs.passage)
    print(obs.get_state_slice())

exit()
