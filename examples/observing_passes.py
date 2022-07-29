#!/usr/bin/env python

'''
================================
Observing a set of passes
================================

This example showcases the creation of a simple radar scheduler and observation of object passes using the 
observation simulation feature of the SORTS Toolbox.

Two objects are propagated at the same time and only the first one is tracked. A simple scheduler implementation is 
added to log the azimuth and elevation of the stations during the tracking of the object, and finally, both objects 
measurements are simulated using the radar states resulting from the tracking controller.
'''
import pathlib
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

import sorts
eiscat3d = sorts.radars.eiscat3d
from sorts.radar.controllers import Tracker
from sorts.targets import SpaceObject
from sorts.common import interpolation, profiling
from sorts.radar import RadarSchedulerBase


# propagator initialization
from sorts.targets.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)
prop = Prop_cls(**Prop_opts)

p = profiling.Profiler()
logger = profiling.get_logger()

# space object initialization
objs = [
    SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = 7200e3, 
        e = 0.1, 
        i = 75, 
        raan = 79,
        aop = 0,
        mu0 = mu0,
        epoch = 53005.0,
        parameters = dict(
            d = 1.0,
        ),
    )
    for mu0 in [62.0, 61.9]
]
for obj in objs: print(obj)

# create propagation time array
t = sorts.equidistant_sampling(
    orbit = objs[0].state, 
    start_t = 0, 
    end_t = 250, 
    max_dpos=1e3,
)

# propagate states of both objects
logger.info(f'Temporal points: {len(t)}')
states0 = objs[0].get_state(t)
states1 = objs[1].get_state(t)
t_slice = 0.1

# set cache_data = True to save the data in local coordinates 
# for each pass inside the Pass instance, setting to false saves RAM
passes0 = eiscat3d.find_passes(t, states0, cache_data=False) 
passes1 = eiscat3d.find_passes(t, states1, cache_data=False)

#just create a controller for observing 10 points of the first pass
ps = passes0[0][0][0]
use_inds = np.arange(0, len(ps.inds), len(ps.inds)//10)

# simple scheduler implementation to log radar azimuth and elevation
class MyScheduler(RadarSchedulerBase):
    ''' Simple example scheduler '''
    def __init__(self, radar, t0, scheduler_period, logger=None, profiler=None):
        super().__init__(radar=radar, t0=t0, scheduler_period=scheduler_period, logger=logger, profiler=profiler)

    def run(self, controls, control_id):
        ''' The scheduler in this example only takes the i^th control into account '''
        if self.logger is not None: self.logger.info("Running scheduler")

        control = controls[control_id] 
        final_control_sequence = control.copy()

        # extract scheduler dara
        data = np.empty((control.n_periods,), dtype=object)
        for period_id in range(control.n_periods):
            data_tmp = np.ndarray((len(final_control_sequence.t[period_id]), len(self.radar.rx)*2 + 1))
            data_tmp[:,0] = final_control_sequence.t[period_id]

            names = []
            targets = []
            pdirs = final_control_sequence.get_pdirs(period_id)
            for ti in range(len(final_control_sequence.t[period_id])):
                names.append(control.meta['controller_type'])
                targets.append(control.meta['target'])

                for ri, rx in enumerate(self.radar.rx):
                    rx.point_ecef(pdirs["rx"][ri, 0, :, ti])
                    data_tmp[ti,1+ri*2] = rx.beam.azimuth
                    data_tmp[ti,2+ri*2] = rx.beam.elevation

            data_tmp = data_tmp.T.tolist() + [names, targets]
            data_tmp = list(map(list, zip(*data_tmp)))
            data[period_id] = data_tmp

        final_control_sequence.meta["scheduler_data"] = data
        return final_control_sequence


# create scheduler instance with a 10s schedule period
scheduler = MyScheduler(eiscat3d, 0.0, 10.0, logger=logger, profiler=p)

# create the controller and generate the controls in sync with the radar scheduler
e3d_tracker = Tracker()
controls = e3d_tracker.generate_controls(
    t[ps.inds[use_inds]], 
    eiscat3d, 
    t[ps.inds[use_inds]], 
    states0[:3,ps.inds[use_inds]], 
    t_slice=t_slice, 
    states_per_slice=1, 
    interpolator=interpolation.Linear, 
    scheduler=scheduler)

# update meta and run scheduler
controls.meta['target'] = 'Cool object 1'
controls.meta['controller_type'] = 'Tracker'
controls = scheduler.run([controls], 0)

# print schedule
print("\n Scheduler results")
for period_id in range(controls.n_periods):
    rx_head = [f'rx{i} {co}' for i in range(len(scheduler.radar.rx)) for co in ['az', 'el']]
    sched_tab = tabulate(controls.meta["scheduler_data"][period_id], headers=["t [s]"] + rx_head + ['Controller', 'Target'])
    print("Scheduler results for scheduler period ", period_id)
    print(sched_tab)
    print("")

# control radar to get radar states
radar_states = eiscat3d.control(controls)

# simulate radar observations
p.start('parallelization')
data0 = eiscat3d.observe_passes(passes0, radar_states, objs[0], snr_limit=False, parallelization=True, interpolator=interpolation.Linear)
p.stop('parallelization')

p.start('no parallelization')
data1 = eiscat3d.observe_passes(passes1, radar_states, objs[1], snr_limit=False, parallelization=False, interpolator=interpolation.Linear)
p.stop('no parallelization')
print(p)
#create a tdm file example
# pth = pathlib.Path(__file__).parent / 'data' / 'test_tdm.tdm'
# print(f'Writing TDM data to: {pth}')

# dat = data0[0][0][0]
# sorts.io.write_tdm(
#     pth,
#     dat['t'],
#     dat['range'],
#     dat['range_rate'],
#     np.ones(dat['range'].shape),
#     np.ones(dat['range_rate'].shape),
#     freq=eiscat3d.tx[0].beam.frequency,
#     tx_ecef=eiscat3d.tx[0].ecef,
#     rx_ecef=eiscat3d.rx[0].ecef,
#     tx_name="EISCAT 3D Skiboten",
#     rx_name="EISCAT 3D Skiboten",
#     oid="Some cool space object",
#     tdm_type="track",
# )

# plot results
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

for tx in eiscat3d.tx:
    axes[0][0].plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]], 'or')
for rx in eiscat3d.rx:
    axes[0][0].plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]], 'og')

for pi in range(len(passes0[0][0])):
    dat = data0[0][0][pi]
    dat2 = data1[0][0][pi]

    if dat is not None:
        measurements = dat["measurements"]
        axes[0][0].plot(states0[0,passes0[0][0][pi].inds], states0[1,passes0[0][0][pi].inds], states0[2,passes0[0][0][pi].inds], '-', label=f'pass-{pi}')
        axes[0][1].plot(measurements['t_measurements']/3600.0, measurements['range'], '-', label=f'pass-{pi}')
        axes[1][0].plot(measurements['t_measurements']/3600.0, measurements['range_rate'], '-', label=f'pass-{pi}')
        axes[1][1].plot(measurements['t_measurements']/3600.0, 10*np.log10(measurements['snr']), '-', label=f'pass-{pi}')

    if dat2 is not None:
        measurements = dat2["measurements"]
        axes[0][0].plot(states1[0,passes1[0][0][pi].inds], states1[1,passes1[0][0][pi].inds], states1[2,passes1[0][0][pi].inds], '-', label=f'obj2 pass-{pi}')
        axes[0][1].plot(measurements['t_measurements']/3600.0, measurements['range'], '-', label=f'obj2 pass-{pi}')
        axes[1][0].plot(measurements['t_measurements']/3600.0, measurements['range_rate'], '-', label=f'obj2 pass-{pi}')
        axes[1][1].plot(measurements['t_measurements']/3600.0, 10*np.log10(measurements['snr']), '-', label=f'obj2 pass-{pi}')

axes[0][1].legend()
plt.show()