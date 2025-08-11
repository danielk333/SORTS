import pickle, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import numpy.typing as npt
from pathlib import Path
from astropy.time import Time
import sorts
from sorts import equidistant_sampling
from sorts.interpolation import Legendre8, Linear
from sorts.population import master_catalog, master_catalog_factor
from sorts.propagator import SGP4
from sorts.space_object import SpaceObject
from sorts.radar.tx_rx import Station
from sorts.radar.radars import get_radar
from sorts.types import Datetime64_us, Timedelta64_us, Float64_as_sec
from sorts.utils import to_datetime64_us, to_pydatetime
from sorts.controller_v2 import tracker_controller
from sorts.controller_v2.tracker_controller import TrackerController
from sorts.controller_v2.fence_scan_controller import FenceScanController
from sorts import schedule_v2 as schedule
from sorts.schedule_v2 import Schedule, ExperimentDetail
from sorts.scheduler_v2.priority_scheduling import priority_scheduling
from sorts.simulation_v2 import StxMrxSimulation
from sorts.simulation_v2.observation import list_to_dataframe

# import for plottings
from IPython.display import display
import pandas as pd
import bokeh.plotting as bp
from sorts import plots

# disable pandas table wrapping
pd.set_option("display.expand_frame_repr", False)


epoch = Time(53005.0, format="mjd", scale="utc")  # 2004-01-01 00:00:00Z

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
# TODO: these patching of station prop should be integrated into codebase
tx_station: Station = eiscat3d.tx[0]
tx_station.uid = ("eiscat3d", "stage1-array", "tx", "0")
rx_station_0: Station = eiscat3d.rx[0]
rx_station_0.uid = ("eiscat3d", "stage1-array", "rx", "0")
rx_station_1: Station = eiscat3d.rx[1]
rx_station_1.uid = ("eiscat3d", "stage1-array", "rx", "1")


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
    epoch=epoch,
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
spobjs = [tracked_spobj, *[spobj_pop.get_object(i) for i in range(spobj_pop.shape[0])]]


# we can also use a lambda function, but we cannot pickle the whole simulation in that case
#  (python's pickle does not support lambda function)
def dsec_sampler(orbit, start_time, end_time):
    return sorts.equidistant_sampling(
        orbit=orbit,
        start_t=(to_pydatetime(start_time) - to_pydatetime(epoch)).total_seconds(),
        end_t=(to_pydatetime(end_time) - to_pydatetime(epoch)).total_seconds(),
        max_dpos=1e3,
    )


tracker_ctrl = TrackerController.from_space_object(
    spobj=tracked_spobj,
    epoch=epoch,
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
    fence_scan_ctrl.state["exp_detail"]["id"]: fence_scan_ctrl.state["exp_detail"],
}

tx_master_sch = priority_scheduling(
    [tracker_schs.tx_schedule, fence_schs.tx_schedule], exp_detail_map
)

rx_master_schs = [
    priority_scheduling(rx_schs, exp_detail_map)
    for rx_schs in zip(tracker_schs.rx_schedules, fence_schs.rx_schedules)
]

output_folder = Path(__file__).parent / ".." / ".." / "local_data"
pickle_fpath = (
    output_folder
    / f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}-{Path(__file__).name}.pickle'
)

sim = StxMrxSimulation.from_spec(
    {
        "tx_station": tx_station,
        "tx_schedule": tx_master_sch,
        "rx_stations": [rx_station_0, rx_station_1],
        "rx_schedules": rx_master_schs,
        "exp_num_map": exp_detail_map,
        "epoch": epoch,
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
            "sim": sim,
            "calc_time": calc_time,
        },
        f,
    )
