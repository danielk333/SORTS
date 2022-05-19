#!/usr/bin/env python

'''
Example stare and chase scheduler
==================================
'''

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from astropy.time import Time, TimeDelta

import sorts

radar = sorts.radars.eiscat3d_interp

poptions = dict(
    settings = dict(
        in_frame='GCRS',
        out_frame='ITRS',
    ),
)
epoch = Time(53005.0, format='mjd')

scan = sorts.scans.Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)

profiler = sorts.profiling.Profiler()
logger = sorts.profiling.get_logger('StareAndChase')

obj = sorts.SpaceObject(
    sorts.propagator.SGP4,
    propagator_options = poptions,
    a = 7200e3, 
    e = 0.02, 
    i = 75, 
    raan = 86,
    aop = 0,
    mu0 = 60,
    epoch = epoch,
    parameters = dict(
        d = 0.1,
    ),
)


class StareAndChase(sorts.scheduler.ObservedParameters):
    def __init__(self, radar, scan, epoch, profiler=None, logger=None, **kwargs):
        super().__init__(
            radar=radar, 
            logger=logger, 
            profiler=profiler,
        )
        self.end_time = kwargs.get('end_time', 3600.0*24.0)
        self.timeslice = kwargs.get('timeslice', 0.1)
        self.max_predict_time = kwargs.get('max_predict_time', 30*60.0)
        self.epoch = epoch
        self.scan = scan
        self.tracking_object = None
        self.tracker = None
        self.update(None)

    def update(self, space_object, start_track=None):
        self.tracking_object = space_object

        if self.tracking_object is not None:
            dt = (self.tracking_object.epoch - self.epoch).to_value('sec')

            if start_track is None:
                self.states_t = np.arange(dt, dt + self.max_predict_time, self.timeslice)
            else:
                self.states_t = np.arange(start_track, start_track + self.max_predict_time, self.timeslice)

            if self.logger is not None:
                self.logger.info(f'StareAndChase:update:propagating {len(self.states_t)} steps')
            if self.profiler is not None:
                self.profiler.start('StareAndChase:update:propagating')

            self.states = self.tracking_object.get_state(self.states_t - dt)

            if self.profiler is not None:
                self.profiler.stop('StareAndChase:update:propagating')
            if self.logger is not None:
                self.logger.info(f'StareAndChase:update:propagating complete')

            self.passes = self.radar.find_passes(self.states_t, self.states, cache_data = False)

        self.scanner = sorts.controllers.Scanner(
            self.radar,
            self.scan,
            t_slice = self.timeslice,
            t = np.arange(0, self.end_time, self.timeslice), 
            profiler=self.profiler, 
            logger=self.logger,
        )
        self.controllers = [self.scanner]
        if self.tracking_object is not None:
            selection = self.passes[0][0][0].inds

            if start_track is not None:
                selection = selection[self.states_t[selection] >= start_track]

            self.tracker = sorts.controller.Tracker(
                radar = self.radar, 
                t=self.states_t[selection], 
                t0=0.0, 
                t_slice = self.timeslice,
                ecefs = self.states[:3,selection],
                return_copy=True,
            )
            self.tracker.meta['target'] = f'Object {self.tracking_object.oid}'

            #remove all scanning during tracking
            self.scanner.t = self.scanner.t[np.logical_or(self.scanner.t < self.tracker.t.min(), self.scanner.t > self.tracker.t.max())]

            self.controllers.append(self.tracker)



    def get_controllers(self):
        return self.controllers



    def generate_schedule(self, t, generator):
        header = ['time', 'TX-az', 'TX-el', 'controller', 'target']

        data = []
        for ind, (radar, meta) in enumerate(generator):
            row = [
                t[ind],
                radar.tx[0].beam.azimuth,
                radar.tx[0].beam.elevation,
                meta['controller_type'].__name__,
                meta.get('target', ''),
            ]
            data.append(row)

        return data, header

profiler.start('total')

scheduler = StareAndChase(
    radar = radar, 
    scan = scan, 
    epoch = epoch,
    timeslice = 0.1, 
    end_time = 3600.0*6,
    logger = logger,
    profiler = profiler,
)

#to simulate a stare and chase, we need to figure out the initial orbit determination errors
try:
    pth = pathlib.Path(__file__).parent / 'data'
except NameError:
    import os
    pth = 'examples' + os.path.sep + 'data' + os.path.sep

print(f'\nUsing "{pth}" as cache for LinearizedCoded errors.')
err = sorts.measurement_errors.LinearizedCodedIonospheric(radar.tx[0], seed=123, cache_folder=pth)

variables = ['x','y','z','vx','vy','vz']
deltas = [1e-4]*3 + [1e-6]*3 + [1e-2]

#see if the object is detected by the scan

logger.info(f'ScanDetections:equidistant_sampling')
profiler.start('ScanDetections:equidistant_sampling')

_t = sorts.equidistant_sampling(
    orbit = obj.state, 
    start_t = 0, 
    end_t = scheduler.end_time, 
    max_dpos=1e3,
)

profiler.stop('ScanDetections:equidistant_sampling')

logger.info(f'ScanDetections:get_state')
profiler.start('ScanDetections:get_state')

states = obj.get_state(_t)

profiler.stop('ScanDetections:get_state')

logger.info(f'ScanDetections:find_passes')
profiler.start('ScanDetections:find_passes')

passes = radar.find_passes(_t, states, cache_data = True)

#lets just look at the first pass
passes[0][0] = passes[0][0][:1]

profiler.stop('ScanDetections:find_passes')

#Create a list of the same pass at all rx stations
rx_passes = []
for p_tx0_rx in passes[0]:
    if len(p_tx0_rx)>0: rx_passes.append(p_tx0_rx[0])

datas = []

logger.info(f'ScanDetections:observe_passes')
profiler.start('ScanDetections:observe_passes')

#observe one pass from all rx stations, including measurement Jacobian
for rxi in range(len(radar.rx)):
    #the Jacobean is stacked as [r_measurements, v_measurements]^T so we stack the measurement covariance equally
    data, J_rx = scheduler.calculate_observation_jacobian(
        rx_passes[rxi], 
        space_object=obj, 
        variables=variables, 
        deltas=deltas, 
        snr_limit=True,
        save_states=True, 
    )
    datas.append(data) #create a rx-list of data

    #now we get the expected standard deviations
    r_stds_tx = err.range_std(data['range'], data['snr'])
    v_stds_tx = err.range_rate_std(data['snr'])

    #Assume uncorrelated errors = diagonal covariance matrix
    Sigma_m_diag_tx = np.r_[r_stds_tx**2, v_stds_tx**2]

    #we simply append the results on top of each other for each station
    if rxi > 0:
        J = np.append(J, J_rx, axis=0)
        Sigma_m_diag = np.append(Sigma_m_diag, Sigma_m_diag_tx, axis=0)
    else:
        J = J_rx
        Sigma_m_diag = Sigma_m_diag_tx

    print(f'Range errors std [m] (rx={rxi}):')
    print(r_stds_tx)
    print(f'Velocity errors std [m/s] (rx={rxi}):')
    print(v_stds_tx)

profiler.stop('ScanDetections:observe_passes')

#diagonal matrix inverse is just element wise inverse of the diagonal
Sigma_m_inv = np.diag(1.0/Sigma_m_diag)

#For a thorough derivation of this formula:
#see Fisher Information Matrix of a MLE with Gaussian errors and a Linearized measurement model
#without a prior since this is IOD
Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J)

print(f'\nLinear orbit estimator covariance [SI-units] (shape={Sigma_orb.shape}):')

header = ['']+variables

list_sig = (Sigma_orb).tolist()
list_sig = [[var] + row for row,var in zip(list_sig, header[1:])]

print(tabulate(list_sig, header, tablefmt="simple"))

#sample IOD covariance

np.random.seed(3487)

#first detection is IOD state
init_epoch = epoch + TimeDelta(datas[0]['t'][0], format='sec')
init_orb = np.random.multivariate_normal(datas[0]['states'][:,0], Sigma_orb)

init_orb = sorts.frames.convert(
    init_epoch, 
    init_orb, 
    in_frame='ITRS', 
    out_frame='GCRS',
)
true_orb = sorts.frames.convert(
    init_epoch, 
    datas[0]['states'][:,0], 
    in_frame='ITRS', 
    out_frame='GCRS',
)

init_object = sorts.SpaceObject(
    sorts.propagator.SGP4,
    propagator_options = poptions,
    x = init_orb[0],
    y = init_orb[1],
    z = init_orb[2],
    vx = init_orb[3],
    vy = init_orb[4],
    vz = init_orb[5],
    epoch = init_epoch,
    parameters = dict(
        d = 0.1,
    ),
)
true_object = sorts.SpaceObject(
    sorts.propagator.SGP4,
    propagator_options = poptions,
    x = true_orb[0],
    y = true_orb[1],
    z = true_orb[2],
    vx = true_orb[3],
    vy = true_orb[4],
    vz = true_orb[5],
    epoch = init_epoch,
    parameters = dict(
        d = 0.1,
    ),
)

print('True object')
print(true_object)

print('Init object')
print(init_object)

#explore the time lag parameter space: figure out at what time lags it stops working
#- use first detection point vs use max detection point vs use all detection points
#- pick a few objects (there is a reduced master file we used before)


#introduce lag
computation_time = 10 #seconds after last time

chase_schdeule_time = datas[0]['t'][-1] + computation_time
staring = True

table_time = 5.0 #s, /timeslice = rows
table_size = int(table_time//scheduler.timeslice)
start_print = datas[0]['t'][0] - 10.0

print(f'Time for scheduler to inject chase: {chase_schdeule_time} sec')

scheduler.update(init_object, start_track=chase_schdeule_time)

profiler.stop('total')
print('\n' + profiler.fmt(normalize='total'))

stare_and_chase_datas = scheduler.observe_passes(passes, space_object = obj, snr_limit=True)
sorts.plotting.observed_parameters(stare_and_chase_datas[0][0])

sorts.plotting.observed_parameters([datas[0]])
sorts.plotting.local_passes([rx_passes[0]])

plt.show()
