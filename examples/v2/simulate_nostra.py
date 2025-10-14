import time, logging, argparse
from pathlib import Path
import numpy as np
from astropy.time import Time, TimeDelta
import sorts
from sorts import (
    interpolation,
    population,
    propagator,
    radar,
    controller,
    schedule,
    simulation,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("starting example")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-master_catalog",
    default=Path(__file__).parent / ".." / ".." / "local_data" / "celn_20090501_00.sim",
)
parser.add_argument(
    "-output_folder", default=Path(__file__).parent / ".." / ".." / "local_data" / "simulate_nostra"
)
# parser.add_argument("-c", "-clobber", action="store_true") # TODO: implement
args = parser.parse_args()

catalog_fpath = Path(args.master_catalog)
output_folder = Path(args.output_folder)


start_time = Time("2004-01-01 00:00:00Z", format="iso", scale="utc")
# end_time = start_time + TimeDelta(6 * 10, format="sec")
end_time = start_time + TimeDelta(6 * 10 * 60, format="sec")
control_slice_duration = np.timedelta64(10_000, "us")  # 10ms

eiscat3d = sorts.get_radar("eiscat3d", "stage1-array")

tx_station: radar.Station = eiscat3d.tx[0]
tx_station.uid = 0
rx_station_0: radar.Station = eiscat3d.rx[0]
rx_station_0.uid = 1
_pop = population.master_catalog(
    catalog_fpath,
    propagator=propagator.SGP4,
    propagator_options={"settings": {"in_frame": "TEME", "out_frame": "ITRF"}},
)
rand_seed = 120389


def dsec_sampler(orbit, start_time, end_time):
    return np.arange(0, (end_time - start_time) / np.timedelta64(1, "s"), 120, dtype=np.float64)


pop = population.master_catalog_factor(_pop, treshhold=1.0, seed=rand_seed)
print(f"true population size: {len(pop)}")
pop.delete(slice(100, None))
space_objects = [obj for obj in pop]
print(f"clamped population size: {len(space_objects)}")

tracker_ctrl = controller.TrackerController.from_space_object(
    spobj=space_objects[0],
    epoch=start_time,
    tx_station=tx_station,
    rx_stations=[rx_station_0],
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

fence_scan_ctrl = controller.FenceScanController.from_scan_spec(
    tx_station=tx_station,
    rx_stations=[rx_station_0],
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
    scan_range=np.linspace(300e3, 1000e3, num=10, dtype=np.float64),
)

tracker_sch = tracker_ctrl.generate(start_time, end_time)
fence_sch = fence_scan_ctrl.generate(start_time, end_time)

exp_detail_map = {
    tracker_ctrl.spec["exp_detail"]["id"]: tracker_ctrl.spec["exp_detail"],
    fence_scan_ctrl.spec["exp_detail"]["id"]: fence_scan_ctrl.spec["exp_detail"],
}

master_sch = schedule.Schedule.priority_scheduling([tracker_sch, fence_sch])

sim = simulation.StxMrxSimulation.from_controllers(
    spec={
        "controllers": [tracker_ctrl, fence_scan_ctrl],
        "schedule": master_sch,
        "epoch": start_time,
        "start_time": start_time,
        "end_time": end_time,
        "space_objects": space_objects,
        "dsec_sampler": dsec_sampler,
        # "interpolator_class": interpolation.Legendre8,
        "interpolator_class": interpolation.Linear,
    }
)


calc_start_time = time.perf_counter()
obss, sim_units = sim.mpi_run(output_folder)
# obss, sim_units = sim.run()  # or, do not use non-mpi for debugging
calc_time = time.perf_counter() - calc_start_time

print(f"len(obss): {len(obss)}")
for idx, obs in enumerate(obss):
    print(f"obs: {idx}")
    print(obs.passage)
    print(obs.get_state_slice())

exit()
