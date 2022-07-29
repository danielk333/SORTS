#!/usr/bin/env python

'''
=====================================
Using scans on alternative parameters
=====================================

This example
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import pyant
import sorts

from sorts.radar.scans import Fence
from sorts.radar import RadarController
from sorts import radar_controls
from sorts.targets import SpaceObject
from sorts.targets.propagator.pysgp4 import SGP4
from sorts import Profiler
from sorts.transformations import frames

Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)

radar = sorts.radars.tsdr_phased_fence

radar.tx[0].beam.phase_steering = 30.0

fig, axes = plt.subplots(2,2,figsize=(10,6),dpi=80)
axes = axes.flatten()
for i in range(4):
    pyant.plotting.gain_heatmap(
        radar.tx[0].beam, 
        resolution=901, 
        min_elevation=30.0, 
        ax=axes[i],
        ind = {
            "pointing":i,
        },
    )
    axes[i].set_title(f'Panel {i}: {int(radar.tx[0].beam.phase_steering)} deg steering')

radar.tx[0].beam.phase_steering = 0.0

for station in radar.tx + radar.rx:
    station.min_elevation = 0.0

scan = Fence(azimuth=0, num=100, dwell=0.1, min_elevation=30)
end_t = 1000

p = Profiler()
logger = sorts.profiling.get_logger('scanning')

objs = [
    SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = 7200e3, 
        e = 0.02, 
        i = 75, 
        raan = 86,
        aop = 0,
        mu0 = 60,
        epoch = 53005.0,
        parameters = dict(
            d = 0.1,
        ),
    ),
]


class PhasedTSDR(RadarController):
    def __init__(self, logger=logger, profiler=p):
        super().__init__(logger=logger, profiler=p)

    def generate_controls(self, t, radar, scan, max_points=3600):
        # create controls
        controls = radar_controls.RadarControls(radar, self, scheduler=None, priority=0)  # the controls structure is defined as a dictionnary of subcontrols
        controls.meta["scan"] = scan
        controls.set_time_slices(t, scan.dwell(), max_points=max_points)

        # phase steering
        els = scan.pointing(t)
        
        # compute local normal of each station (ecef)
        for station in radar.tx + radar.rx:
            station_id = radar.get_station_id(station)
            controls.add_property_control("phase_steering", station, els[1,:])

        RadarController.coh_integration(controls, radar, scan.dwell())

        return controls

p.start('total')

radar_ctrl = PhasedTSDR()
t = np.arange(0, end_t, scan.dwell())

p.start('PhasedTSDR:generate_controls')
controls = radar_ctrl.generate_controls(t, radar, scan, max_points=1000)
p.stop('PhasedTSDR:generate_controls')

p.start('PhasedTSDR:get_radar_states')
radar_states = radar.control(controls)
p.stop('PhasedTSDR:get_radar_states')

# print results
fig = plt.figure(figsize=(15,15))
axes = [
    [
        fig.add_subplot(221, projection='3d'),
        fig.add_subplot(222),
    ],
    [
        fig.add_subplot(223),
        fig.add_subplot(224),
    ],
]

sorts.plotting.grid_earth(axes[0][0])
for tx in radar.tx:
    axes[0][0].plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in radar.rx:
    axes[0][0].plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')


for period_id in range(radar_states.n_periods):
    for ti in range(len(radar_states.t[period_id])):
        tx = radar.tx[0]
        tx.beam.phase_steering = radar_states.property_controls[period_id]["tx"]["phase_steering"][0][ti]

        point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*1000e3 + tx.ecef[:,None]
        axes[0][0].plot([tx.ecef[0], point_tx[0,ti]], [tx.ecef[1], point_tx[1,ti]], [tx.ecef[2], point_tx[2,ti]], 'r-', alpha=0.15)

plt.show()

datas = []
for ind in range(len(objs)):
    print(f'Temporal points obj {ind}: {len(t)}')
    data = radar.compute_measurements(
        radar_states, 
        objs[ind], 
        logger=logger, 
        profiler=p, 
        max_dpos=100e3, 
        snr_limit=False,
        parallelization=False, 
        save_states=True, 
        n_processes=8)

    datas.append(data)

p.stop('total')

#print(p.fmt(normalize='total'))
print(p)



for ind in range(len(objs)):
    data = datas[ind]

    axes[0][0].plot(data["states"][0], data["states"][1], data["states"][2], "+b")

    for pass_data in data["pass_data"]:
        txi = 0
        rxi = 0

        if pass_data is not None:
            states = pass_data["states"]
            measurements = pass_data["measurements"]

            detection = measurements['detection']
            snr = measurements['snr']
            ranges = measurements['range']
            range_rates = measurements['range_rate']
            t = measurements['t_measurements']

            SNRdB = 10*np.log10(snr)
            det_inds = SNRdB > 10.0

            axes[0][0].plot(states[0], states[1], states[2], "-r")

            axes[0][1].plot(t/3600.0, ranges*1e-3, '-', label=f'obj{ind}')
            axes[1][0].plot(t/3600.0, range_rates*1e-3, '-')
            axes[1][1].plot(t/3600.0, SNRdB, '-')

            # detections
            axes[0][1].plot(t[det_inds]/3600.0, ranges[det_inds]*1e-3, '.r')
            axes[1][0].plot(t[det_inds]/3600.0, range_rates[det_inds]*1e-3, '.r')
            axes[1][1].plot(t[det_inds]/3600.0, SNRdB[det_inds], '.r')
            # axes[1][1].set_ylim([0, None])

font_ = 18
axes[0][1].set_xlabel('Time [h]', fontsize=font_)
axes[1][0].set_xlabel('Time [h]', fontsize=font_)
axes[1][1].set_xlabel('Time [h]', fontsize=font_)

axes[0][1].set_ylabel('Two way range [km]', fontsize=font_)
axes[1][0].set_ylabel('Two way range rate [km/s]', fontsize=font_)
axes[1][1].set_ylabel('SNR [dB]', fontsize=font_)

#axes[0][1].legend()

dr = 3000e3
axes[0][0].set_xlim([radar.tx[0].ecef[0]-dr, radar.tx[0].ecef[0]+dr])
axes[0][0].set_ylim([radar.tx[0].ecef[1]-dr, radar.tx[0].ecef[1]+dr])
axes[0][0].set_zlim([radar.tx[0].ecef[2]-dr, radar.tx[0].ecef[2]+dr])

plt.show()

# Performances :
# new implementation
# -------------------------------------- Performance analysis --------------------------------------
#  Name                                                  |   Executions | Mean time     | Total time
# -------------------------------------------------------+--------------+---------------+--------------
#  equidistant_sampling                                  |            1 | 1.23262e-04 s | 0.00 %
#  get_state                                             |            1 | 2.31135e+00 s | 22.91 %
#  find_passes                                           |            1 | 5.55754e-03 s | 0.06 %
#  Measurements:Measure:Initialization:get_object_states |           36 | 2.89154e-02 s | 10.32 %
#  Measurements:Measure:Initialization:find_passes       |           36 | 8.03656e-04 s | 0.29 %
#  Measurements:Measure:Initialization                   |            1 | 1.08658e+00 s | 10.77 %
#  Measurements:Measure                                  |            1 | 7.74711e+00 s | 76.79 %
#  observe_passes                                        |            1 | 7.77166e+00 s | 77.03 %
#  total                                                 |            1 | 1.00888e+01 s | 100.00 %
# --------------------------------------------------------------------------------------------------

# old implementation
# --------------------------------------- Performance analysis --------------------------------------
#  Name                                                   |   Executions | Mean time     | Total time
# --------------------------------------------------------+--------------+---------------+--------------
#  equidistant_sampling                                   |            1 | 9.56059e-05 s | 0.00 %
#  get_state                                              |            1 | 2.34274e+00 s | 14.91 %
#  find_passes                                            |            1 | 4.39820e-02 s | 0.28 %
#  Obs.Param.:calculate_observation:get_state             |            1 | 1.22822e-01 s | 0.78 %
#  Obs.Param.:calculate_observation:enus,range,range_rate |            1 | 1.16563e-03 s | 0.01 %
#  Obs.Param.:calculate_observation:observable_filter     |         6698 | 3.11674e-06 s | 0.13 %
#  Obs.Param.:calculate_observation:snr-step:gain         |         6698 | 1.89960e-03 s | 80.98 %
#  Obs.Param.:calculate_observation:snr-step:snr          |         6698 | 2.80090e-05 s | 1.19 %
#  Obs.Param.:calculate_observation:snr-step:rcs,filter   |         6698 | 1.77821e-05 s | 0.76 %
#  Obs.Param.:calculate_observation:snr-step              |         6698 | 1.95321e-03 s | 83.26 %
#  Obs.Param.:calculate_observation:generator             |            1 | 1.31975e+01 s | 83.99 %
#  Obs.Param.:calculate_observation                       |            1 | 1.33250e+01 s | 84.81 %
#  observe_passes                                         |            1 | 1.33255e+01 s | 84.81 %
#  total                                                  |            1 | 1.57124e+01 s | 100.00 %
# ---------------------------------------------------------------------------------------------------
