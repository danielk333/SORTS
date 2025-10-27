import pickle, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import numpy.typing as npt
from pathlib import Path
from astropy.time import Time
import logging
import sorts
from sorts import equidistant_sampling
from sorts.interpolation import Legendre8, Linear
from sorts.population import master_catalog, master_catalog_factor
from sorts.propagator import SGP4
from sorts.space_object import SpaceObject
from sorts.radar import Station
from sorts.radar.radars import get_radar
from sorts.utils import to_datetime64_us, to_pydatetime
from sorts.controller import TrackerController, FenceScanController
from sorts.schedule import ScheduleOld
from sorts.simulation import StxMrxSimulation

# import for plottings
from IPython.display import display
import pandas as pd
import bokeh.plotting as bp
from sorts import plots

# disable pandas table wrapping
pd.set_option("display.expand_frame_repr", False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("example")
logger.info("starting example")

# 93min runtime
# the `control_slice_duration` is much longer than normal, practical radar `control_slice_duration`
# for easier debugging, inspection of scheduling/schedules
# start_time = Time("2025-01-01 02:45:00")
# end_time = Time("2025-01-01 06:15:00")
# control_slice_duration = np.timedelta64(int(60 * 1e6), "us")

# 115min runtime
# start_time = Time("2025-01-01 02:45:00")
# end_time = Time("2025-01-01 03:00:00")
# control_slice_duration = np.timedelta64(10_000, "us")  # 10ms

start_time = Time("2025-01-01 02:45:00")
end_time = Time("2025-01-01 03:45:00")
control_slice_duration = np.timedelta64(10_000, "us")  # 10ms

# same as above, but use more realistic 10ms `control_slice_duration`
# start_time = Time("2025-01-01 02:45:00")
# end_time = Time("2025-01-01 06:15:00")
# control_slice_duration = np.timedelta64(10_000, "us")  # 10ms

eiscat3d = get_radar("eiscat3d", "stage1-array")
# eiscat3d = get_radar("nostra", "example1")
# TODO: these patching of station prop should be integrated into codebase
tx_station: Station = eiscat3d.tx[0]
tx_station.uid = 0
rx_station_0: Station = eiscat3d.rx[0]
rx_station_0.uid = 1
rx_station_1: Station = eiscat3d.rx[1]
rx_station_1.uid = 2

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

catalog_fpath = Path(__file__).parent / ".." / ".." / "local_data" / "celn_20090501_00.sim"
_spobj_pop = master_catalog(
    catalog_fpath,
    propagator=SGP4,
    propagator_options={"settings": {"in_frame": "TEME", "out_frame": "ITRF"}},
)
rand_seed = 120389
# TODO: reduce the filter size to more sensible value
spobj_pop = master_catalog_factor(_spobj_pop, treshhold=5.0, seed=rand_seed)
spobjs = [tracked_spobj, *[spobj_pop.get_object(i) for i in range(spobj_pop.shape[0])]]


# we can also use a lambda function, but we cannot pickle the whole simulation in that case
#  (python's pickle does not support lambda function)
# def dsec_sampler(orbit, start_time, end_time):
#     return sorts.equidistant_sampling(
#         orbit=orbit,
#         start_t=(to_pydatetime(start_time) - to_pydatetime(epoch)).total_seconds(),
#         end_t=(to_pydatetime(end_time) - to_pydatetime(epoch)).total_seconds(),
#         max_dpos=1e3,
#     )
def dsec_sampler(orbit, start_time, end_time):
    return np.arange(0, (end_time - start_time) / np.timedelta64(1, "s"), 120, dtype=np.float64)


tracker_ctrl = TrackerController.from_space_object(
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
    },
)

fence_scan_ctrl = FenceScanController.from_scan_spec(
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
    },
    azimuth=90,  # sweep from east to west
    min_elevation=30,
    pointings_per_cycle=40,
    # scan_range=np.linspace(300e3, 1000e3, num=10, dtype=np.float64),
    scan_range=np.array([300e3], dtype=np.float64),
)

tracker_sch = tracker_ctrl.generate(start_time, end_time)
fence_sch = fence_scan_ctrl.generate(start_time, end_time)
master_sch = ScheduleOld.priority_scheduling(
    [tracker_sch, fence_sch],
    {
        **tracker_ctrl.get_experiment_id_station_id_pairs_map(),
        **fence_scan_ctrl.get_experiment_id_station_id_pairs_map(),
    },
)


output_folder = Path(__file__).parent / ".." / ".." / "local_data"
pickle_fpath = (
    output_folder
    / f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}-{Path(__file__).name}.pickle'
)

sim = StxMrxSimulation.from_controllers(
    spec={
        "controllers": [tracker_ctrl, fence_scan_ctrl],
        "schedule": master_sch,
        "epoch": start_time,
        "start_time": start_time,
        "end_time": end_time,
        "space_objects": spobjs,
        # "space_objects": [
        #     o for i, o in enumerate(spobjs) if i in [0, 4, 5, 17]
        # ],  # just picked a few from the whole list for now
        "dsec_sampler": dsec_sampler,
        "interpolator_class": Linear,
    }
)

calc_start_time = time.perf_counter()
obss = sim.run()
calc_time = time.perf_counter() - calc_start_time

with open(pickle_fpath, "wb") as f:
    pickle.dump(
        {
            "obss": obss,
            # "sim": sim, # TODO: picking the whole sim is not working: seems `dsec_sampler` is causing issues
            "calc_time": calc_time,
        },
        f,
    )
