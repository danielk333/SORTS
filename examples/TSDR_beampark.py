#!/usr/bin/env python

'''
TSDR beam park simulation
===========================
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle

import sorts
import pyant

# radar = sorts.radars.tsdr
radar = sorts.radars.tsdr_fence

from sorts.scheduler import StaticList, ObservedParameters
from sorts.controller import Scanner
from sorts import SpaceObject, Simulation
from sorts import MPI_single_process, MPI_action, iterable_step, store_step, cached_step
from sorts.radar.scans import Beampark
from sorts.population import master_catalog, master_catalog_factor

from sorts.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)

# pyant.plotting.gain_heatmap(radar.tx[0].beam, resolution=300, min_elevation=80.0)
# plt.show()


end_t = 600.0
scan = Beampark(azimuth=radar.tx[0].beam.azimuth, elevation=radar.tx[0].beam.elevation)

# master_path = '/home/danielk/IRF/IRF_GITLAB/SORTSpp/master/celn_20090501_00.sim'
master_path = '/processed/projects/AO9884_RDPP/simulations/master/celn_20090501_00.sim'

pop = master_catalog(
    master_path,
    propagator = Prop_cls,
    propagator_args = Prop_opts,
)
pop = master_catalog_factor(pop, treshhold = 0.1)

# pop.delete(slice(100,None,None)) #DELETE ALL BUT THE FIRST 100

class ObservedScanning(StaticList, ObservedParameters):
    pass

scan_sched = Scanner(radar, scan)
scan_sched.t = np.arange(0, end_t, scan.dwell())

scheduler = ObservedScanning(
    radar = radar, 
    controllers = [scan_sched], 
)

class Scanning(Simulation):
    def __init__(self, population, *args, **kwargs):
        self.population = population
        self.inds = list(range(len(population)))

        super().__init__(*args, **kwargs)

        if self.logger is not None:
            self.logger.always(f'Population of size {len(population)} objects loaded.')

        self.steps['propagate'] = self.get_states
        self.steps['passes'] = self.find_passes
        self.steps['observe'] = self.observe_passes


    @store_step(store=['states', 't'], iterable=True)
    @MPI_action(action='gather', iterable=True, root=0)
    @iterable_step(iterable='inds', MPI=True, log=True)
    @cached_step(caches='h5')
    def get_states(self, index, item):
        obj = self.population.get_object(item)
        t = sorts.equidistant_sampling(
            orbit = obj.orbit, 
            start_t = self.scheduler.controllers[0].t.min(), 
            end_t = self.scheduler.controllers[0].t.max(), 
            max_dpos=1e3,
        )
        state = obj.get_state(t)
        return state, t


    @store_step(store='passes', iterable=True)
    @MPI_action(action='gather', iterable=True, root=0)
    @iterable_step(iterable=['states', 't'], MPI=True, log=True)
    @cached_step(caches='pickle')
    def find_passes(self, index, item):
        state, t = item
        passes = scheduler.radar.find_passes(t, state, cache_data = False)
        return passes


    @store_step(store='obs_data', iterable=True)
    @MPI_action(action='gather-clear', iterable=True, root=0)
    @iterable_step(iterable='passes', MPI=True, log=True)
    @cached_step(caches='pickle')
    def observe_passes(self, index, item):
        data = scheduler.observe_passes(item, space_object = self.population.get_object(index), snr_limit=False)
        return data


    @MPI_single_process(process_id = 0)
    def plot(self, ind=0, txi=0, rxi=0):

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

        for radar, meta in scan_sched(np.arange(0,scan.dwell(),scan.dwell())):
            for tx in radar.tx:
                point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*scan_sched.r.max()
                if len(point_tx.shape) > 1:
                    point_tx += tx.ecef[:,None]
                else:
                    point_tx += tx.ecef
                    point_tx = point_tx.reshape(3,1)
                for ti in range(point_tx.shape[1]):
                    axes[0][0].plot([tx.ecef[0], point_tx[0,ti]], [tx.ecef[1], point_tx[1,ti]], [tx.ecef[2], point_tx[2,ti]], 'r-', alpha=0.15)

                for rx in radar.rx:
                    pecef = rx.pointing_ecef/np.linalg.norm(rx.pointing_ecef, axis=0)

                    for ri in range(pecef.shape[1]):
                        point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*scan_sched.r[ri]
                        if len(point_tx.shape) > 1:
                            point_tx += tx.ecef[:,None]
                        else:
                            point_tx += tx.ecef
                        point = pecef[:,ri]*np.linalg.norm(rx.ecef - point_tx[:,0]) + rx.ecef

                        axes[0][0].plot([rx.ecef[0], point[0]], [rx.ecef[1], point[1]], [rx.ecef[2], point[2]], 'g-', alpha=0.05)

        
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
        axes[0][0].set_xlim([self.scheduler.radar.tx[0].ecef[0]-dr, self.scheduler.radar.tx[0].ecef[0]+dr])
        axes[0][0].set_ylim([self.scheduler.radar.tx[0].ecef[1]-dr, self.scheduler.radar.tx[0].ecef[1]+dr])
        axes[0][0].set_zlim([self.scheduler.radar.tx[0].ecef[2]-dr, self.scheduler.radar.tx[0].ecef[2]+dr])

        



sim = Scanning(
    population = pop,
    scheduler = scheduler,
    # root = '/home/danielk/IRF/E3D_PA/sorts_v4_tests/sim_tsdr',
    root = '/processed/projects/AO9884_RDPP/simulations/sim3',
    logger=True, 
    profiler=True,
)
# sim.delete('test')
# sim.branch('test', empty=True)
# sim.checkout('test')

sim.profiler.start('total')


sim.run()

sim.profiler.stop('total')
sim.logger.always('\n'+sim.profiler.fmt(normalize='total'))


sim.plot()

plt.show()

## EX

# sim.run('propagate')
# sim.run('passes')
# for ind, freq in enumerate([1.2e6, 2.4e6]):
#     sim.checkout('master')
#     sim.branch(f'f{ind}')
#     sim.scheduler.radar.tx[0].beam.frequency = freq
#     sim.scheduler.radar.rx[0].beam.frequency = freq
#     sim.run('observe')

##


