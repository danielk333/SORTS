#!/usr/bin/env python

'''
=============================
Optimizing with interpolation
=============================

This example showcases the use of the ``sorts.common.interpolation`` module to optimize the 
computational performances of the ``measurement`` simulation module.
'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt


import sorts
eiscat3d = sorts.radars.eiscat3d_interp

from sorts.radar.controllers import Scanner
from sorts import SpaceObject
from sorts.common.profiling import Profiler
from sorts.radar.scans import Fence
from sorts.common.interpolation import Legendre8
from sorts.radar.measurements import measurement

from sorts.targets.propagator import SGP4

# try:
#     pth = pathlib.Path(__file__).parent.resolve()
# except NameError:
#     pth = pathlib.Path('.').parent.resolve()
# pth = pth / 'data' / 'orekit-data-master.zip'


# if not pth.is_file():
#     sorts.propagator.Orekit.download_quickstart_data(pth, verbose=True)

Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        in_frame='GCRS',
        out_frame='ITRS',
    ),
)

# set simulation time to 6 hours
end_t = 3600*6

# setup scanning sequence
scan = Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)


# intializes the logging/profiling module
p = Profiler()
logger = sorts.profiling.get_logger('scanning')

# create a space object over the radar system
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
            A = 1.0,
        ),
    ),
]
for obj in objs: print(obj)

# intialization of the controller
controller = Scanner(profiler=p, logger=logger)
t = np.arange(0, end_t, scan.dwell())

# generate scanning controls and compute corresponding radar states
controls = controller.generate_controls(t, eiscat3d, scan)
radar_states = eiscat3d.control(controls)

p.start('total')

# space object propagation time
t = np.arange(0.0, end_t, 30.0)

datas = []
passes = []
states = []
for ind in range(len(objs)):
    print(f'Temporal points obj {ind}: {len(t)}')
    
    # propagate states
    p.start('get_state')
    states += [objs[ind].get_state(t)]
    p.stop('get_state')

    # initialization of the interpolation class
    interpolator = Legendre8(states[ind], t)

    # look for passes inside the computed states
    p.start('find_passes')
    passes += [eiscat3d.find_passes(t, states[ind], cache_data=False)] 
    p.stop('find_passes')

    # compute measurements of passes without using the state interpolator
    p.start('observe_passes')
    data = eiscat3d.observe_passes(passes[ind], radar_states, objs[ind], snr_limit=False, parallelization=True, logger=logger)
    p.stop('observe_passes')

    # compute measurements of passes using the state interpolator
    p.start('observe_passes_interpolator')
    data = eiscat3d.observe_passes(passes[ind], radar_states, objs[ind], interpolator=interpolator, snr_limit=False, parallelization=True)
    p.stop('observe_passes_interpolator')

    datas.append(data)

p.stop('total')
print(p.fmt(normalize='total'))

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

# plot earth/radar stations on the 3d subplot
sorts.plotting.grid_earth(axes[0][0])
for tx in eiscat3d.tx:
    axes[0][0].plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in eiscat3d.rx:
    axes[0][0].plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

# plot radar scanning scheme over the first period (all other periods are identical) 
ax = sorts.plotting.plot_beam_directions(radar_states.pdirs[period_id], eiscat3d, ax=axes[0][0], logger=logger, profiler=p, zoom_level=0.95, azimuth=10, elevation=10)

# plot measurement results
for ind in range(len(objs)):
    for pi in range(len(passes[ind][0][0])):
        ps = passes[ind][0][0][pi]
        dat = datas[ind][0][0][pi]
        
        # plot pass
        axes[0][0].plot(states[ind][0,ps.inds], states[ind][1,ps.inds], states[ind][2,ps.inds], '-')

        if dat is not None:
            # get max snr measurements for each radar control time slice
            max_snr_measurements = measurement.get_max_snr_measurements(dat, copy=False)["measurements"]
            SNRdB = 10*np.log10(max_snr_measurements['snr'])
            det_inds = SNRdB > 10.0

            axes[0][1].plot(max_snr_measurements['t_measurements']/3600.0, max_snr_measurements['range']*1e-3, '-', label=f'obj{ind}-pass{pi}')
            axes[1][0].plot(max_snr_measurements['t_measurements']/3600.0, max_snr_measurements['range_rate']*1e-3, '-')
            axes[1][1].plot(max_snr_measurements['t_measurements']/3600.0, SNRdB, '-')

            axes[0][1].plot(max_snr_measurements['t_measurements'][det_inds]/3600.0, max_snr_measurements['range'][det_inds]*1e-3, '.r')
            axes[1][0].plot(max_snr_measurements['t_measurements'][det_inds]/3600.0, max_snr_measurements['range_rate'][det_inds]*1e-3, '.r')
            axes[1][1].plot(max_snr_measurements['t_measurements'][det_inds]/3600.0, SNRdB[det_inds], '.r')
            axes[1][1].set_ylim([0, None])

font_ = 18
axes[0][1].set_xlabel('Time [h]', fontsize=font_)
axes[1][0].set_xlabel('Time [h]', fontsize=font_)
axes[1][1].set_xlabel('Time [h]', fontsize=font_)

axes[0][1].set_ylabel('Two way range [km]', fontsize=font_)
axes[1][0].set_ylabel('Two way range rate [km/s]', fontsize=font_)
axes[1][1].set_ylabel('SNR [dB]', fontsize=font_)

axes[0][1].legend()

dr = 600e3
axes[0][0].set_xlim([eiscat3d.tx[0].ecef[0]-dr, eiscat3d.tx[0].ecef[0]+dr])
axes[0][0].set_ylim([eiscat3d.tx[0].ecef[1]-dr, eiscat3d.tx[0].ecef[1]+dr])
axes[0][0].set_zlim([eiscat3d.tx[0].ecef[2]-dr, eiscat3d.tx[0].ecef[2]+dr])

plt.show()