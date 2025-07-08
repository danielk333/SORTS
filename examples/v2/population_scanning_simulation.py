"""
based on `examples/v2/single_spobj_scanning_simulation.py`,
but modified to work with population of space objects

NOTE: WIP; this is currently a testing ground for refactored code
TODO: complete it and clean up
"""

import pickle, time, typing as t, argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time
import pyvista as pv
from pyvista import examples
import sorts
from sorts import _v2 as sortsV2

# TODO: switch to normal named imports; these are tmp alias until `_v2` becomes the default namespace
StxMrxSimulation = sortsV2.simulation.StxMrxSimulation
StxMrxSimulationParam = sortsV2.simulation.StxMrxSimulationParam

parser = argparse.ArgumentParser()
parser.add_argument(
    "-master_catalog",
    default=Path(__file__).parent / ".." / ".." / "local_data" / "celn_20090501_00.sim",
)
parser.add_argument("-output_folder", default=Path(__file__).parent / ".." / ".." / "local_data")
parser.add_argument("-c", "-clobber", action="store_true")
args = parser.parse_args()

catalog_fpath = Path(args.master_catalog)
output_folder = Path(args.output_folder)
pickle_fpath = output_folder / f"{Path(__file__).name}.pickle"

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

exp_detail = sortsV2.simulation.ExperimentDetail(
    id=0,
    coh_int_bandwidth=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    ipp=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    pulse_length=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    # power=5000000.0,
    power=5e8,  # TODO: tmp 100x higher for debugging; restore the value afterwards
    bandwidth=52.08333333333333,
    duty_cycle=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    noise_temp=150.0,
    slice_duration=np.timedelta64(10_000, "us"),  # 10ms
)
exp_num_map: dict[int, sortsV2.simulation.ExperimentDetail] = {0: exp_detail}

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

_pop = sorts.population.master_catalog(
    catalog_fpath,
    propagator=sorts.propagator.SGP4,
    propagator_options={"settings": {"in_frame": "TEME", "out_frame": "ITRF"}},
)
rand_seed = 120389
# TODO: reduce the filter size to more sensible value
pop = sorts.population.master_catalog_factor(_pop, treshhold=5.0, seed=rand_seed)
space_objects = [pop.get_object(i) for i in range(pop.shape[0])]
print(f"true population size: {len(space_objects)}")
space_objects_slice = slice(0, 100)  # take only 100 items
# space_objects_slice = slice(0,500)  # take only 500 items
# space_objects_slice = slice(None, None)  # take all
space_objects = space_objects[space_objects_slice]
print(f"clamped population size: {len(space_objects)}")


sim = StxMrxSimulation(
    StxMrxSimulationParam(
        tx_station=eiscat3d.tx[0],
        tx_schedule=tx_schedule,
        rx_stations=[eiscat3d.rx[0]],
        rx_schedules=[rx_schedule],
        exp_num_map=exp_num_map,
        epoch=epoch,
        start_time=start_time,
        end_time=end_time,
        space_objects=space_objects,
        space_objects_dt_sampler_s=lambda orbit, start_time, end_time: sorts.equidistant_sampling(
            orbit=orbit,
            start_t=(start_time - epoch).total_seconds(),
            end_t=(end_time - epoch).total_seconds(),
            max_dpos=1e3,
        ),
        # space_objects_dt_interpolator_s=sorts.interpolation.Legendre8,
        space_objects_dt_interpolator_s=sorts.interpolation.Linear,
    )
)

if Path(pickle_fpath).is_file():
    with open(pickle_fpath, "rb") as f:
        saved_data = pickle.load(f)
        obss: list[list[sortsV2.simulation.Observation]] = saved_data["obss"]
        calc_time = saved_data["calc_time"]
        spobjs_states_interps = saved_data["spobjs_states_interps"]
else:
    calc_start_time = time.perf_counter()
    obss = sim.calculate_observations()
    calc_time = time.perf_counter() - calc_start_time
    spobjs_states_interps = sim._spobjs_states_interps

    with open(pickle_fpath, "wb") as f:
        pickle.dump(
            {
                "obss": obss,
                "calc_time": calc_time,
                "spobjs_states_interps": spobjs_states_interps,
            },
            f,
        )
        print(f"calculate_observations took {calc_time} sec")


##
# some matplotlib plottings
##

# find the index of the space objects which has non-empty observation list
nonempty_obss_idx_ls = [
    x[0] for x in filter(lambda x: len(x[1]) != 0, enumerate(obss[space_objects_slice]))
]
print(f"space object with observations: {nonempty_obss_idx_ls}")
target_spobj_idx = nonempty_obss_idx_ls[0]

obs = obss[target_spobj_idx][0]
rx_sch_pass_mask = sim.param.rx_schedules[0].create_mask_by_time_range(obs.time_range)
rx_sch_pass = sim.param.rx_schedules[0].filter_by_mask(rx_sch_pass_mask)

spobjs_states_interp = spobjs_states_interps[target_spobj_idx]

fig, axs = plt.subplots(2, 2)

sch_dt_s_arr = (sim.param.rx_schedules[0].stt_tstmp_us - np.datetime64(sim.param.epoch)).astype(
    "timedelta64[us]"
).astype(np.float64) / 1e6
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
    sim.param.rx_schedules[0].stt_tstmp_us,
    sim.param.rx_schedules[0].pointing_az,
    "r",
)
axs[0, 1].plot(
    sim.param.rx_schedules[0].stt_tstmp_us,
    sim.param.rx_schedules[0].pointing_el,
    "g",
)

plt.show()

##
# some vtk plottings
#
# TODO: clean up the plottings
##
plotter = pv.Plotter()
# dt_s_arr_1_yr = np.arange(0, 60 * 60 * 24 * 365, 60 * 60 * 24, dtype=np.float64)
dt_s_arr_path = np.arange(-3600, 3600, 60, dtype=np.float64)
obs_splines: list[pv.PolyData] = []
path_splines: list[pv.PolyData] = []
for target_spobj_idx in nonempty_obss_idx_ls[:]:
    # for target_spobj_idx in [nonempty_obss_idx_ls[1]]: # plot just 1 spobj for debugging
    spobj = space_objects[target_spobj_idx]
    spobjs_states_interp = spobjs_states_interps[target_spobj_idx]

    path_pts = spobj.get_state(dt_s_arr_path)[:3].T
    spline = pv.Spline(path_pts, 1000)  # generate a spline with n interpolation points
    path_splines.append(spline)

    obs_pts = spobjs_states_interp.get_state(sch_dt_s_arr_pass)[:3].T
    spline = pv.Spline(obs_pts, 100)  # generate a spline with n interpolation points
    obs_splines.append(spline)


# refs:
# - https://docs.pyvista.org/api/examples/_autosummary/pyvista.examples.planets.load_earth.html
# - https://docs.pyvista.org/examples/99-advanced/planets.html
# - https://docs.pyvista.org/examples/00-load/create_spline#:~:text=The%20spline%20can%20also%20be%20plotted%20as%20a%20plain%20line

earth = examples.planets.load_earth(radius=6371e3)
# earth.rotate_z(angle=180, inplace=True)
texture = examples.load_globe_texture()
image_path = examples.planets.download_stars_sky_background(load=False)
plotter.add_mesh(earth, texture=texture)
for spline in path_splines:
    plotter.add_mesh(spline, color="b", line_width=2)
for spline in obs_splines:
    plotter.add_mesh(spline, color="r", line_width=5)
plotter.show_grid()  # type: ignore
plotter.show()
