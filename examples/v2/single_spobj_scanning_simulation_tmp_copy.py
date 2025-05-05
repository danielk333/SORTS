"""
A temp copy of `examples/v2/single_spobj_scanning_simulation.py`,
it is used a testing ground for refactored code.

Will be removed afterwards.
"""

import typing as t, os, pickle
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import pyant
import sorts
from sorts.schedule_v2 import Schedule
from sorts.calculations import ExperimentDetail, Observation
from sorts.detection_config import SimpleStxSrx, calculate_simple_stx_srx_observations

epoch = Time(53005.0, format="mjd", scale="utc")
simulation_end_dt = 600.0  # in seconds
eiscat3d = sorts.get_radar("eiscat3d", "stage1-array")

spobj = sorts.SpaceObject(
    sorts.propagator.SGP4,
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

spobj_smpl_dt_arr = sorts.equidistant_sampling(
    orbit=spobj.state,
    start_t=0,
    end_t=simulation_end_dt,
    max_dpos=1e3,
)

exp_detail = ExperimentDetail(
    coh_int_bandwidth=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    ipp=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    pulse_length=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    power=5000000.0,
    bandwidth=52.08333333333333,
    duty_cycle=1.0,  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    noise_temp=150.0,
)
exp_num = 0
exp_num_map: dict[int, ExperimentDetail] = {0: exp_detail}

# TODO: feels like we could use a tracker controller here
#   (`src/sorts/controller_v2/tracker_controller.py`),
#   but it requires a `Pass` object which is not that convenient to use
#     - a `Pass` object feels like sth internal to a simulation and not sth known in prior/externally
spobj_smpl_states = spobj.get_state(spobj_smpl_dt_arr)

pass_arr = eiscat3d.find_passes(spobj_smpl_dt_arr, spobj_smpl_states, cache_data=True)
target_pass_obj: sorts.Pass = pass_arr[0][0][0]

scan = sorts.radar.scans.Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)
scan_dt_arr = np.arange(0, simulation_end_dt, scan.dwell())
scanner_ctrl = sorts.controller.Scanner(
    eiscat3d,
    scan,
    t=scan_dt_arr,
)
scan_dt_arr_pass_mask = np.logical_and(
    scan_dt_arr >= target_pass_obj.t[0], scan_dt_arr <= target_pass_obj.t[-1]
)
scan_dt_arr_pass = scan_dt_arr[scan_dt_arr_pass_mask]
sch_total_rows = len(scan_dt_arr_pass)

# space object in tx, rx station coordinate and the pointings of tx, rx station,
# all under the delta times of a pass
# note that `tx_azelr_deg_pass`, `rx_azelr_deg_pass` init to 1.0,
spobj_tx_enu_pass = eiscat3d.tx[0].enu(spobj.get_state(scan_dt_arr_pass))[:, 0]
spobj_tx_enu_pass = eiscat3d.rx[0].enu(spobj.get_state(scan_dt_arr_pass))[:, 0]
tx_azelr_deg_pass = eiscat3d.tx[0].enu(spobj.get_state(scan_dt_arr_pass))[:, 0]
# TODO: vectorize
tx_azelr_deg_pass = np.full((3, sch_total_rows), 1.0, dtype=np.float64)
rx_azelr_deg_pass = np.full((3, sch_total_rows), 1.0, dtype=np.float64)

for dt_idx, (radar, _meta) in enumerate(scanner_ctrl.generator(scan_dt_arr_pass)):
    tx_azelr_deg_pass[:2, dt_idx] = [
        radar.tx[0].beam.azimuth[0],
        radar.tx[0].beam.elevation[0],
    ]
    rx_azelr_deg_pass[:2, dt_idx] = [
        radar.rx[0].beam.azimuth[0],
        radar.rx[0].beam.elevation[0],
    ]

scan_pass_stt_tstmp_us = (scan_dt_arr_pass * 1e6).astype("timedelta64[us]") + np.datetime64(
    t.cast(datetime, epoch.to_datetime(timezone=timezone.utc))
)

tx_schedule = Schedule(
    stt_tstmp_us=scan_pass_stt_tstmp_us,
    exp_num=np.full(sch_total_rows, exp_num),
    pointing_az=tx_azelr_deg_pass[0],
    pointing_el=tx_azelr_deg_pass[1],
    coh_int_bandwidth=np.full(sch_total_rows, exp_detail.coh_int_bandwidth),
    ipp=np.full(sch_total_rows, exp_detail.ipp),
    pulse_length=np.full(sch_total_rows, exp_detail.pulse_length),
)

rx_schedule = Schedule(
    stt_tstmp_us=scan_pass_stt_tstmp_us,
    exp_num=np.full(sch_total_rows, exp_num),
    pointing_az=rx_azelr_deg_pass[0],
    pointing_el=rx_azelr_deg_pass[1],
    coh_int_bandwidth=np.full(sch_total_rows, exp_detail.coh_int_bandwidth),
    ipp=np.full(sch_total_rows, exp_detail.ipp),
    pulse_length=np.full(sch_total_rows, exp_detail.pulse_length),
)

dcfg = SimpleStxSrx(
    stt_tstmp_us=scan_pass_stt_tstmp_us,
    tx_station=eiscat3d.tx[0],
    rx_station=eiscat3d.rx[0],
    tx_schedule=tx_schedule,
    rx_schedule=rx_schedule,
    exp_num_map=exp_num_map,
)

obs = calculate_simple_stx_srx_observations(
    dcfg=dcfg,
    space_object=spobj,
    epoch=t.cast(datetime, epoch.to_datetime(timezone=timezone.utc)),
)

target_data_fpath = (
    Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "simulate_scanning_v2__saves.pickle"
)
if not os.path.isfile(target_data_fpath):
    raise RuntimeError(
        f'saved data from example "simulate_scanning_v2.py" is required, please ensure the file {target_data_fpath} exists'
    )
with open(target_data_fpath, "rb") as f:
    example_result = pickle.load(f)

# we target the `[1st_result][tx station 0][rx station 0][0th pass]`
target = example_result["datas"][0][0][0][0]

max_snr_diff_ratio = (max(obs.snr) / max(target["snr"])) - 1
assert max_snr_diff_ratio < 1e-6  # TODO: remove

max_snr_dt_diff_ratio = (
    scan_dt_arr_pass[np.argmax(obs.snr)] / target["t"][np.argmax(target["snr"])]
) - 1
assert max_snr_dt_diff_ratio < 1e-6  # TODO: remove

# for comparison with `target["tx_k"]`, `target["rx_k"]`
tx_k = pyant.coordinates.sph_to_cart(
    np.stack(
        [
            dcfg.tx_schedule.pointing_az,
            dcfg.tx_schedule.pointing_el,
            np.full(len(dcfg.stt_tstmp_us), 1.0, dtype=np.float64),
        ],
        axis=0,
    ),
    degrees=True,
)
rx_k = pyant.coordinates.sph_to_cart(
    np.stack(
        [
            dcfg.rx_schedule.pointing_az,
            dcfg.rx_schedule.pointing_el,
            np.full(len(dcfg.stt_tstmp_us), 1.0, dtype=np.float64),
        ],
        axis=0,
    ),
    degrees=True,
)

# ASK: why do they differ significantly?
result_tx_k = np.array_equal(target["tx_k"], tx_k)
result_rx_k = np.array_equal(target["tx_k"], rx_k)

# do some plottings
fig, ax = plt.subplots()
ax.plot(scan_dt_arr_pass, obs.snr, "r")
ax.plot(scan_dt_arr_pass, target["snr"], "b")
plt.show()
