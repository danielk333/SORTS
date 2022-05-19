#!/usr/bin/env python

'''
Simulate scanning for objects with simulation class
======================================================
'''
import pathlib
import configparser

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle

import sorts
eiscat3d = sorts.radars.eiscat3d_interp

from sorts.radar.scheduler import StaticList, ObservedParameters
from sorts.radar.controllers import Scanner
from sorts import SpaceObject, Simulation
from sorts import MPI_single_process, MPI_action, iterable_step, store_step, cached_step
from sorts.radar.scans import Fence

from sorts.targets.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)


try:
    base_pth = pathlib.Path(__file__).parents[1].resolve()
except NameError:
    base_pth = pathlib.Path('.').parents[1].resolve()

config = configparser.ConfigParser(interpolation=None)
config.read([base_pth / 'example_config.conf'])
simulation_root = pathlib.Path(config.get('simulation_helper_scanning.py', 'simulation_root'))

if not simulation_root.is_absolute():
    simulation_root = base_pth / simulation_root.relative_to('.')



end_t = 600.0
scan = Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)

objs = [
    SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = a, 
        e = 0.1, 
        i = 75, 
        raan = 79,
        aop = 0,
        mu0 = 60,
        epoch = 53005.0,
        parameters = dict(
            d = 0.3,
        ),
    ) for a in np.linspace(7200e3, 8400e3, 4)
]

class ObservedScanning(StaticList, ObservedParameters):
    pass

scan_sched = Scanner(eiscat3d, scan)
scan_sched.t = np.arange(0, end_t, scan.dwell())

scheduler = ObservedScanning(
    radar = eiscat3d, 
    controllers = [scan_sched], 
)

class Scanning(Simulation):
    def __init__(self, objs, *args, **kwargs):
        self.objs = objs

        super().__init__(*args, **kwargs)

        # These steps will be run at 'step.run()'
        self.steps['propagate'] = self.get_states       # 'propagate' determines name of stored stage
        self.steps['passes'] = self.find_passes
        self.steps['observe'] = self.observe_passes


    @store_step(store=['states', 't'], iterable=True)
    @MPI_action(action='gather', iterable=True, root=0)
    @iterable_step(iterable='objs', MPI=True)
    @cached_step(caches='h5')
    def get_states(self, index, item, **kw):
        t = sorts.equidistant_sampling(
            orbit = item.state, 
            start_t = self.scheduler.controllers[0].t.min(), 
            end_t = self.scheduler.controllers[0].t.max(), 
            max_dpos=1e3,
        )
        state = item.get_state(t)
        return state, t


    @store_step(store='passes', iterable=True)
    @MPI_action(action='gather', iterable=True, root=0)
    @iterable_step(iterable=['states', 't'], MPI=True)
    @cached_step(caches='passes')
    def find_passes(self, index, item, **kw):
        state, t = item
        passes = scheduler.radar.find_passes(t, state, cache_data = False)
        return passes


    @store_step(store='obs_data', iterable=True)
    @MPI_action(action='gather-clear', iterable=True, root=0)
    @iterable_step(iterable='passes', MPI=True)
    @cached_step(caches='pickle')
    def observe_passes(self, index, item, **kw):
        data = scheduler.observe_passes(item, space_object = self.objs[index], snr_limit=True)
        return data


    def save_passes(self, path, passes):
        with open(path, 'wb') as h:
            pickle.dump(passes, h)


    def load_passes(self, path):
        with open(path, 'rb') as h:
            passes = pickle.load(h)
        return passes


    @MPI_single_process(process_id = 0)
    def plot(self, txi=0, rxi=0):

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
        for tx in self.scheduler.radar.tx:
            axes[0][0].plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
        for rx in self.scheduler.radar.rx:
            axes[0][0].plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

        for radar, meta in scan_sched(np.arange(0,scan.cycle(),scan.dwell())):
            for tx in radar.tx:
                point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*scan_sched.r.max() + tx.ecef
                axes[0][0].plot([tx.ecef[0], point_tx[0]], [tx.ecef[1], point_tx[1]], [tx.ecef[2], point_tx[2]], 'r-', alpha=0.15)

                for rx in radar.rx:
                    pecef = rx.pointing_ecef/np.linalg.norm(rx.pointing_ecef, axis=0)

                    for ri in range(pecef.shape[1]):
                        point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*scan_sched.r[ri] + tx.ecef
                        point = pecef[:,ri]*np.linalg.norm(rx.ecef - point_tx) + rx.ecef

                        axes[0][0].plot([rx.ecef[0], point[0]], [rx.ecef[1], point[1]], [rx.ecef[2], point[2]], 'g-', alpha=0.05)

        for ind in range(len(objs)):
            for pi in range(len(self.passes[ind][txi][rxi])):
                ps = self.passes[ind][txi][rxi][pi]
                dat = self.obs_data[ind][txi][rxi][pi]
                
                axes[0][0].plot(self.states[ind][0,ps.inds], self.states[ind][1,ps.inds], self.states[ind][2,ps.inds], '-')

                if dat is not None:
                    axes[0][1].plot(dat['t']/3600.0, dat['range']*1e-3, '-', label=f'obj{ind}-pass{pi}')
                    axes[1][0].plot(dat['t']/3600.0, dat['range_rate']*1e-3, '-')
                    axes[1][1].plot(dat['t']/3600.0, 10*np.log10(dat['snr']), '-')


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

        



sim = Scanning(
    objs = objs,
    scheduler = scheduler,
    root = simulation_root,
)
# sim.delete('test') #to delete the test branch
# sim.branch('test', empty=True) #to create an empty branch
sim.checkout('test')

sim.profiler.start('total')

sim.run()

sim.profiler.stop('total')
sim.logger.always('\n'+sim.profiler.fmt(normalize='total'))


sim.plot()

plt.show()


######################

# sim.run('propagate')
# sim.run('passes')
# for ind, freq in enumerate([1.2e6, 2.4e6]):
#     sim.checkout('master')
#     sim.branch(f'f{ind}')
#     sim.scheduler.radar.tx[0].beam.frequency = freq
#     sim.scheduler.radar.rx[0].beam.frequency = freq
#     sim.run('observe')

#################
