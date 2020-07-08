#!/usr/bin/env python

'''Main simulation handler in the form of a class using the capabilities of the entire toolbox.

'''

#Standard python
import shutil
import time
import glob
import types
import copy

#Packages
import numpy as np

from mpi4py import MPI
import h5py
from tqdm import tqdm
import scipy

# SORTS imports


comm = MPI.COMM_WORLD


# class Simulation:
#     '''Main simulation handler class.

#     # TODO: Write docstring

#     '''
#     def __init__(self,
#                 radar,
#                 population,
#                 root,
#                 scheduler = schlib.dynamic_scheduler,
#                 simulation_name = 'SORTS++ Simulation',
#             ):

#         #check that mjd0 is the same for every object

#         _test_mjd0 = population['mjd0'][0]
#         for mjd0 in population['mjd0']:
#             assert (_test_mjd0 - mjd0)*3600.0*24.0 < 0.01, 'Population Epoch needs to be synchronized to run simulation'

#         #root has no trailing slash
#         if root[-1] == '/':
#             root = root[:-1]

#         self.radar = radar
#         self.population = population
#         self.root = root
#         self.scheduler = scheduler
#         self.scheduler_data = {}
#         self.name = simulation_name

#         if comm.rank == 0:
#             if not os.path.exists(root):
#                 os.makedirs(root)
#             if not os.path.exists(self._logs_folder):
#                 os.makedirs(self._logs_folder)
#         comm.barrier()

#         self.logger = logs.setup_logging(
#             root = self._logs_folder,
#             file_level = logging.INFO,
#             term_level = logging.INFO,
#             parallel = comm.rank,
#             logfile = True,
#         )

#         self.set_version('master')

#         self.parameters = None
#         self.observation_parameters()
#         self.simulation_parameters(_reset = True)

#         self._config_radar_scans()

#         self.catalogue = Catalogue(population)
        
#         self.__t0 = 0.0
#         self.__t1 = 0.0
#         self.__t = 0.0

#         self.__init_time = time.time()

#         self.__my_objects = []
#         for thrid in range(comm.size):
#             self.__my_objects.append(range(thrid, len(self.population), comm.size))
        
#         self.logger.info('<< Simulation initialized >>')
#         self.logger.info('<< {} >>'.format(self.root,))

#         logs.record_time_diff('sim_init')


#     def __make_folder(self, _path):
#         if not os.path.exists(_path):
#             self.logger.info("Making path: {}".format(_path))
#             os.makedirs(_path)


#     def set_version(self, version):
#         self.version = version
#         if comm.rank == 0:
#             self.__make_folder(self.root + '/' + version)
#             self.__make_folder(self._orbits_folder)
#             self.__make_folder(self._detections_folder)
#             self.__make_folder(self._prior_folder)
#             self.__make_folder(self._tracklet_folder)
#             self.__make_folder(self._plots_folder)

#         self._set_barrier('Set version "{}"'.format(version))


#     @property
#     def _out_folder(self):
#         return self.root + '/' + self.version

#     @property
#     def _logs_folder(self):
#         return self.root + '/logs'

#     @property
#     def _plots_folder(self):
#         return self._out_folder + '/plots'

#     @property
#     def _orbits_folder(self):
#         return self._out_folder + '/orbits'

#     @property
#     def _detections_folder(self):
#         return self._out_folder + '/detections'

#     @property
#     def _prior_folder(self):
#         return self._out_folder + '/prior'

#     @property
#     def _tracklet_folder(self):
#         return self._out_folder + '/tracklets'


#     def _set_barrier(self, msg):
#         self.logger.info('{} <barrier>'.format(msg))
#         comm.barrier()
#         self.logger.info('{} </barrier>'.format(msg))



#     def _collect_catalogue(self, thread):
#         if comm.rank == thread:
#             for thr_id in range(comm.size):
#                 if thr_id != thread:
#                     for ind in self.__my_objects[thr_id]:
#                         if self.catalogue._known[ind]:
#                             self.catalogue._maintinence[ind] = comm.recv(source=thr_id, tag=ind)
#                         else:
#                             self.catalogue._detections[ind] = comm.recv(source=thr_id, tag=ind)
#         else:
#             for ind in self.__my_objects[comm.rank]:
#                 if self.catalogue._known[ind]:
#                     comm.send(self.catalogue._maintinence[ind], dest=thread, tag=ind)
#                 else:
#                     comm.send(self.catalogue._detections[ind], dest=thread, tag=ind)
        
#         if len(self.catalogue.tracklets) > 0:
#             tr_index = np.array(
#                 [_trl['index'] for _trl in self.catalogue.tracklets],
#                 dtype=np.int64,
#             )

#             if comm.rank == thread:
#                 for thr_id in range(comm.size):
#                     if thr_id != thread:
#                         for ind in self.__my_objects[thr_id]:
#                             _iters = np.argwhere(tr_index == ind).flatten()
#                             for tr_ind in _iters:
#                                 self.catalogue.tracklets[tr_ind] = comm.recv(source=thr_id, tag=tr_ind)

#             else:
#                 for ind in self.__my_objects[comm.rank]:
#                     _iters = np.argwhere(tr_index == ind).flatten()
#                     for tr_ind in _iters:
#                         comm.send(self.catalogue.tracklets[tr_ind], dest=thread, tag=tr_ind)

#         if comm.rank == thread:
#             for thr_id in range(comm.size):
#                 if thr_id == thread:
#                     continue
#                 self.catalogue.priors = np.append(self.catalogue.priors, comm.recv(source=thr_id, tag=thr_id))
#         else:
#             comm.send(self.catalogue.priors, dest=thread, tag=comm.rank)

#         self._set_barrier('Collect catalogue to thread {}'.format(thread))


#     def save(self, MPI_synch = True):
#         if comm.size > 1 and MPI_synch:
#             self._collect_catalogue(thread=0)

#         if comm.rank == 0:
#             self.catalogue.save(self._out_folder + '/catalogue_data.h5')
#             self.parameters.save(self._out_folder + '/obs_parameters_data.h5')
#             self._save_sim_params(self._out_folder + '/sim_parameters_data.h5')

#         self._set_barrier('Saving simulation data')


#     def check_load(self):
#         _check = True
#         _check = _check and os.path.isfile(self._out_folder + '/catalogue_data.h5')
#         _check = _check and os.path.isfile(self._out_folder + '/obs_parameters_data.h5')
#         _check = _check and os.path.isfile(self._out_folder + '/sim_parameters_data.h5')
#         return _check


#     def load(self):
#         self.catalogue.load(self._out_folder + '/catalogue_data.h5')
#         self.parameters.load(self._out_folder + '/obs_parameters_data.h5')
#         self._load_sim_params(self._out_folder + '/sim_parameters_data.h5')


#     def simulation_movie(self, **kwargs):
#         '''Create an animation of the current simulation. Needs the information from a schedule.
            
#             List of keyword arguments:
#                 * t0
#                 * t1
#                 * dt


#         Contains a 3d animation of objects and radar beams, a table of scheduler function values and a pointing direction azimuth elevation diagram with underlying detection rates
#         '''

#         d_az = kwargs.get('camera_rate', 0.2)
#         fps = kwargs.get('fps', 20)

#         beam_len = kwargs.get('scan_range', 1000e3)

#         t0 = kwargs.get('t0', 0.0)
#         t1 = kwargs.get('t1', self.__t*3600.0*0.01)
#         dt = kwargs.get('dt', self.parameters.coher_int_t)

#         t = np.arange(t0, t1, dt, dtype=np.float64)

#         plt.style.use('dark_background')

#         fig = plt.figure(figsize=(15,15))
#         ax = fig.add_subplot(111, projection='3d')
#         plothelp.draw_earth_grid(ax, alpha = 0.2, color='white')

#         ax.grid(False)
#         plt.axis('off')

#         data_list = np.empty( (3, len(t), len(self.population)), dtype=np.float64 )
#         point_dir = np.empty( (3, len(t)), dtype=np.float64 )
#         scan_dir = np.empty( (3, len(t), len(self.radar._tx)), dtype=np.float64 )

#         for ind, space_o in enumerate(self.population.object_generator())
#             data_list[:,:, ind] = space_o.get_orbit(t)

#         m_times = []
#         for tracklet in self.catalogue.tracklets:
#             for _t in tracklet['t']:
#                 m_times += (_t, tracklet['index'])
#         m_times.sort(key=lambda x: x[0])
#         m_times = np.array(m_times)
#         _mtim_keep = np.empty(m_times.shape, dtype=np.int64)
#         for ind, _t in enumerate(t):
#             _mtim_keep[ind] = np.argmin(np.abs(m_times - _t))
#             point_dir[:, ind] = data_list[:,:, _mtim_keep[ind]]

#         for txi, tx in enumerate(self.radar._tx):
#             for ind, _t in enumerate(t):
#                 scan_dir[:, ind, txi] = tx.get_scan(_t).local_pointing(_t)
#                 az, el, _ = coord.cart_to_azel(scan_dir[:, ind, txi])
#                 scan_dir[:, ind, txi] = coord.azel_ecef(tx.lat, tx.lon, tx.alt, az, el)



#         traj_len = kwargs.get('trajectory', 20)

#         def run(ti):
#             titl.set_text('Simulation t=%.4f min' % (t_curr/60.0))

#             print('Updating plot {}'.format(ti))
#             ax_traj_list = []
#             ax_point_list = []
#             for ind in range(len(self.population)):
#                  ax_traj_list[ind].set_data(
#                     data_list[0, (ti-traj_len):ti, ind],
#                     data_list[1, (ti-traj_len):ti, ind],
#                     )
#                 ax_traj_list[ind].set_3d_properties(
#                     data_list[2, (ti-traj_len):ti, ind],
#                     )
#                 ax_traj_list[ind].figure.canvas.draw()

#                 if t_ind[tri] < dat.shape[1]:
#                     print('- point ', tri)
#                     ax_point_list[tri].set_data(
#                         dat[0,t_ind[tri]],
#                         dat[1,t_ind[tri]],
#                         )
#                     ax_point_list[tri].set_3d_properties(
#                         dat[2,t_ind[tri]],
#                         )
#                     ax_point_list[tri].figure.canvas.draw()

#             for tri,tr_ax in enumerate(ax_traj_list):
#                 if len(tr_ax.get_xydata()) > 0:
#                     if tri not in data_list:
#                         ax_traj_list[tri].set_data([],[])
#                         ax_traj_list[tri].set_3d_properties([])
#                         ax_traj_list[tri].figure.canvas.draw()

#                         ax_point_list[tri].set_data([],[])
#                         ax_point_list[tri].set_3d_properties([])
#                         ax_point_list[tri].figure.canvas.draw()


            
#             ax_tx_scan.set_data(
#                 [tx.ecef[0],tx.ecef[0] + k0[0]*beam_len],
#                 [tx.ecef[1],tx.ecef[1] + k0[1]*beam_len],
#                 )
#             ax_tx_scan.set_3d_properties([tx.ecef[2],tx.ecef[2] + k0[2]*beam_len])
#             ax_tx_scan.figure.canvas.draw()

#             #radar beams
#             ax_txb.set_data(
#                 [tx.ecef[0],point[0]],
#                 [tx.ecef[1],point[1]],
#                 )
#             ax_txb.set_3d_properties([tx.ecef[2],point[2]])
#             ax_txb.figure.canvas.draw()
#             for rxi in range(len(radar._rx)):
#                 print('- reciv ', rxi)
#                 ax_rxb_list[rxi].set_data(
#                     [radar._rx[rxi].ecef[0],point[0]],
#                     [radar._rx[rxi].ecef[1],point[1]],
#                     )
#                 ax_rxb_list[rxi].set_3d_properties([radar._rx[rxi].ecef[2],point[2]])
#                 ax_rxb_list[rxi].figure.canvas.draw()
#             print('returning axis')
#             return ax_traj_list, ax_txb, ax_rxb_list, ax_tx_scan



#         dets_ax = fig.add_axes([.65, .05, .33, .25], facecolor='k')
#         dets_ax_az, = dets_ax.plot([], [], '.r', alpha = 1)
#         dets_ax.set_ylabel('Azimuth', color='r')
#         dets_ax2 = dets_ax.twinx()  # instantiate a second axes that shares the same x-axis

#         dets_ax2.set_ylabel('sin', color='c')  # we already handled the x-label with ax1
#         dets_ax_el, = dets_ax2.plot([], [], color='c')
#         dets_ax2.tick_params(axis='y', labelcolor='c')


#         #traj
#         ax_traj_list = []
#         ax_point_list = []
#         for ind in range(len(self.population)):
#             ax_traj, = ax.plot([],[],[],alpha=0.5,color="white")
#             ax_traj_list.append(ax_traj)

#             ax_point, = ax.plot([],[],[],'.',alpha=1,color="yellow")
#             ax_point_list.append(ax_point)

#         ax_txbs = []
#         ax_tx_scans = []
#         ax_rxbs = []

#         for tx in self.radar._tx:
#             #radar beams
#             ax_txb, = ax.plot(
#                 [tx.ecef[0],tx.ecef[0]],
#                 [tx.ecef[1],tx.ecef[1]],
#                 [tx.ecef[2],tx.ecef[2]],
#                 alpha=1,color="green",
#                 )  
#             ax_txbs += [ax_txb]

#             ax_tx_scan, = ax.plot(
#                 [tx.ecef[0],tx.ecef[0]],
#                 [tx.ecef[1],tx.ecef[1]],
#                 [tx.ecef[2],tx.ecef[2]],
#                 alpha=1,color="yellow",
#                 )
#             ax_tx_scans += [ax_tx_scan]
#             ax_rxb_list = []
#             for rx in radar._rx:
#                 ax_rxb, = ax.plot(
#                     [rx.ecef[0],rx.ecef[0]],
#                     [rx.ecef[1],rx.ecef[1]],
#                     [rx.ecef[2],rx.ecef[2]],
#                     alpha=1,color="green",
#                     )
#                 ax_rxb_list.append(ax_rxb)
#             ax_rxbs += [ax_rxb_list]


#         box_c = kwargs.get('box_center', np.zeros((3,)))
#         delta = kwargs.get('box_width', 1500e3)
#         ax.set_xlim([box_c[0] - delta,box_c[0] + delta])
#         ax.set_ylim([box_c[1] - delta,box_c[1] + delta]) 
#         ax.set_zlim([box_c[2] - delta,box_c[2] + delta]) 

#         titl = fig.text(0.5,0.94,'',size=22,horizontalalignment='center')
        
#         t0_exec = time.time()

#         print('setup done, starting anim')
#         ani = animation.FuncAnimation(fig, run, range(len(t)),
#             blit=kwargs.get('blit', True),
#             interval=1.0e3/float(fps),
#             repeat=True,
#         )

#         print('Anim done, writing movie')

#         Writer = animation.writers['ffmpeg']
#         writer = Writer(metadata=dict(artist='SORTS++'),fps=fps)
#         ani.save(kwargs.get('path', self._out_folder + '/sim_movie.mp4'), writer=writer)

#         print('showing plot')
#         plt.show()


#     @logs.class_log_call('{1|level} set to {2|handle}')
#     def set_log_level(self, level, handle=''):
#         '''Set the log level of simulation class internal logger
#         '''
#         if len(handle) == 0:
#             self.set_terminal_level(level)
#             self.set_logfile_level(level)
#         elif handle == 'terminal':
#             self.set_terminal_level(level)
#         elif handle == 'file':
#             self.set_logfile_level(level)
#         else:
#             raise TypeError('Handle not recognised: {}, please use "terminal", "file" or "" for both'.format(handle))


#     @logs.class_log_call('{1|level} set to terminal')
#     def set_terminal_level(self, level):
#         '''Set the log level of simulation class internal logger stream handle
#         '''
#         if level is None:
#             self.logger.handlers[1].disabled = True
#         else:
#             self.logger.handlers[1].setLevel(level)


#     @logs.class_log_call('{1|level} set to logifle')
#     def set_logfile_level(self, level):
#         '''Set the log level of simulation class internal logger file handle
#         '''
#         if level is None:
#             self.logger.handlers[0].disabled = True
#         else:
#             self.logger.handlers[0].setLevel(level)


#     def _construct_scan_controler(self, txi):
#         _p = self.parameters

#         on_time = _p.SST_slices * _p.coher_int_t
#         off_time = _p.Interleaving_slices * _p.interleaving_time_slice

#         _tx = self.radar._tx[txi]
#         scan_n = len(_tx.extra_scans)

#         if scan_n > 0:

#             _tx.extra_scans_coherr_bw = 1.0 / (_p.T_pulse * _p.N_IPP)
#             _tx.standard_scan_coherr_bw = 1.0 / (_p.T_pulse_scan * _p.N_IPP)
            
#             def controler(tx, t):
#                 t = t % (on_time+off_time)
#                 if t <= on_time:
#                     tx.coh_int_bandwidth = tx.standard_scan_coherr_bw
#                     return tx.scan
#                 else:
#                     tx.coh_int_bandwidth = tx.extra_scans_coherr_bw
#                     scan_i = int((t - on_time)/off_time*scan_n)
#                     return tx.extra_scans[ scan_i ]
#             return controler
#         else:
#             def controler(tx,t):
#                 return tx.scan

#         return controler


#     def _config_radar_scans(self):
#         for txi, tx in enumerate(self.radar._tx):
#             if tx.extra_scans is not None:
#                 tx.scan_controler = self._construct_scan_controler(self, txi)


#     def simulation_parameters(self, **kwargs):
#         '''Calculate and set the necessary simulation parameters.

#         The simulation default parameters can be found by looking at the source code to this function.

#         :param dict kwargs: Simulation parameters to set before re-calculating and saving simulation meta-data. Keyword arguments not in the list of supported parameters will be ignored.

#         **Keyword arguments:**

#             * max_dpos [float]: Description
#             * tracklet_noise [bool]: Description
#             * auto_synchronize [bool]: Determines if threads should be automatically synchronized after state changing commands (like run_observations)
#             * pass_dt [float]: Description

#         '''
#         _reset = kwargs.setdefault('_reset', False)

#         if _reset:
#             self._tracklet_noise = kwargs.setdefault('tracklet_noise', True)
#             self._max_dpos = kwargs.setdefault('max_dpos', 50.0e3)
#             self._auto_synchronize = kwargs.setdefault('auto_synchronize', True)
#             self._pass_dt = kwargs.setdefault('pass_dt', None)
#         else:
#             self._tracklet_noise = kwargs.setdefault('tracklet_noise', self._tracklet_noise)
#             self._max_dpos = kwargs.setdefault('max_dpos', self._max_dpos)
#             self._auto_synchronize = kwargs.setdefault('auto_synchronize', self._auto_synchronize)
#             self._pass_dt = kwargs.setdefault('pass_dt', self._pass_dt)


#     def _save_sim_params(self, fname):
#         with h5py.File(fname,"w") as hf:
#             hf.attrs['_tracklet_noise'] = self._tracklet_noise
#             hf.attrs['_max_dpos'] = self._max_dpos
#             hf.attrs['_auto_synchronize'] = self._auto_synchronize
#             hf.attrs['__t0'] = self.__t0
#             hf.attrs['__t1'] = self.__t1
#             hf.attrs['__t'] = self.__t

#     def _load_sim_params(self, fname):
#         with h5py.File(fname,"r") as hf:
#             self._tracklet_noise = hf.attrs['_tracklet_noise']
#             self._max_dpos = hf.attrs['_max_dpos']
#             self._auto_synchronize = hf.attrs['_auto_synchronize']
#             self.__t0 = hf.attrs['__t0']
#             self.__t1 = hf.attrs['__t1']
#             self.__t = hf.attrs['__t']

#     def observation_parameters(self, **kwargs):
#         '''Calculate and set the necessary observation parameters. If just a subset of parameter is supplied the others keep their old values.

#         The observation default parameters can be found by looking at the source code to this function.

#         :param dict kwargs: Observation parameters to set before re-calculating and saving observation meta-data.

#         **Keyword arguments:**

#             * duty_cycle [float]: Description
#             * SST_fraction [float]: Description
#             * tracking_fraction [float]: Description
#             * interleaving_time_slice [float]: Description
#             * SST_time_slice [float]: Description
#             * IPP [float]: Description
#             * scan_during_interleaved [bool]: Description

#         '''

#         if self.parameters is None:
#             self.parameters = ObservationParameters(
#                 duty_cycle = kwargs.setdefault('duty_cycle', 0.25),
#                 SST_f = kwargs.setdefault('SST_fraction', 1.0),
#                 tracking_f = kwargs.setdefault('tracking_fraction', 0.5),
#                 coher_int_t = kwargs.setdefault('SST_time_slice', 0.1),
#                 IPP = kwargs.setdefault('IPP', 10e-3),
#                 interleaving_time_slice = kwargs.setdefault('interleaving_time_slice', 0.4),
#                 scan_during_interleaved = kwargs.setdefault('scan_during_interleaved', False),
#             )
#         else:
#             if 'SST_fraction' in kwargs:
#                 kwargs['SST_f'] = kwargs['SST_fraction']
#                 del kwargs['SST_fraction']

#             if 'tracking_fraction' in kwargs:
#                 kwargs['tracking_f'] = kwargs['tracking_fraction']
#                 del kwargs['tracking_fraction']

#             if 'SST_time_slice' in kwargs:
#                 kwargs['coher_int_t'] = kwargs['SST_time_slice']
#                 del kwargs['SST_time_slice']
            
#             self.parameters.calculate_parameters(**kwargs)

#         self.parameters.configure_radar_to_observation(self.radar, mode = None)

#         if self.parameters.scan_during_interleaved:
#             self._config_radar_scans()


#     def _sim_time(self):
#         return self.__t


#     def status(self, fout = None):
#         '''Print summary status of the simulation.
#         '''
#         logs.record_time_diff('sim_status')
#         logs.logg_time_record(logs.exec_times, self.logger)

#         if comm.rank == 0:

#             pop_len = float(len(self.population))

#             known_n = np.sum( self.catalogue._known )
#             unknown_n = self.catalogue.size - known_n
#             maint_objs = np.sum(1 for i in self.catalogue._maintinence if i is not None)
#             det_objs = np.sum(1 for i in self.catalogue._detections if i is not None)


#             status_lines = []
#             status_lines.append('--- Simulation status --- ')
#             status_lines.append('--- {}: {} '.format(self.name, self.version))
#             status_lines.append('--- ----------------- --- ')
#             status_lines.append('-> Orbit determinations : %s' % (self._orbits_folder,))
#             status_lines.append('-> Tracklets            : %s' % (self._tracklet_folder,))
#             status_lines.append('-> Prior information    : %s' % (self._prior_folder,))
#             status_lines.append('-> Population           : %s' % (self.population.name,) )
#             status_lines.append('-> Population size      : %i' % (len(self.population),) )
#             status_lines.append('-> Radar system         : %s' % (self.radar.name,) )
#             status_lines.append('-> Radar recivers       : %i' % (len(self.radar._rx),) )
#             for ri,rx in enumerate(self.radar._rx):
#                 status_lines.append('-> Radar reciver %i      : %s' % (ri,rx.name,) )
#             status_lines.append('-> Radar transmitters   : %i' % (len(self.radar._tx),) )
#             for ri,tx in enumerate(self.radar._tx):
#                 status_lines.append('-> Radar transmitter %i  : %s' % (ri,tx.name,) )
#                 status_lines.append('-> Transmitter %i scan   : %s' % (ri,tx.scan.name,) )
#                 status_lines.append('-> Scan information     : %s' % (tx.scan.info(),) )
#             status_lines.append('-> Current time [h]     : %.2f' % (self.__t,) )
#             status_lines.append('-> Population detected  : %.2f %%' % (det_objs/pop_len*100.0,) )
#             status_lines.append('-> Population maintained: %.2f %%' % (maint_objs/pop_len*100.0,) )
#             status_lines.append('-> Execution time       : %.3f h' % ((time.time() - self.__init_time)/3600.0,))

#             status_lines.append('-> Number of tracks     : %i' % (len(self.catalogue.tracks),) )
#             status_lines.append('-> Number of tracklets  : %i' % (len(self.catalogue.tracklets),) )

#             status_lines.append('-> Total %i of %i known objects maintained'%(maint_objs, known_n ))
#             status_lines.append('-> Total %i of %i unknown objects discovered'%(det_objs, unknown_n ))

#             status_lines.append( str(self.parameters) )

#             if fout is not None:
#                 file = open(self._out_folder + '/status_' + fout + '.txt',"w")
#                 for line in status_lines:
#                     file.write(line+'\n')
#                 file.close()
            
            
#             self.logger.info('\n'.join(status_lines))


#     def clear_simulation(self):
#         '''Clear current version folder of all files.
#         '''
#         self.logger.always('Clearing version: ' + self.version)

#         if comm.rank == 0:
#             files = glob.glob(self._out_folder + '/*')
#             for f in files:
#                 self.logger.info('DELETE: {}'.format(f) )
#                 if os.path.isdir(f):
#                     shutil.rmtree(f)
#                 else:
#                     os.remove(f)

#         self._set_barrier('Clear simulation')
#         self.set_version(self.version)



#     def checkout_simulation(self, reference_version):
#         '''Checkout a copy of the given simulation version and replace the current version with it.
#         '''
        
#         self.logger.always('Checking out simulation:'
#             + '\n From: ' + reference_version
#             + '\n To  : ' + self.version
#         )

#         reference_version

#         if comm.rank == 0:
#             rm_files = glob.glob(self._out_folder + '/*')
#             for f in rm_files:
#                 self.logger.info('DELETE: {}'.format(f) )
#                 if os.path.isdir(f):
#                     shutil.rmtree(f)
#                 else:
#                     os.remove(f)

#             files = glob.glob(self.root + '/' + reference_version + '/*')
#             for src in files:
#                 dst = self._out_folder + '/' + src.split('/')[-1]
#                 self.logger.info('Copy: {} -> {}'.format(src, dst))
#                 if os.path.isdir(src):
#                     if os.path.isdir(dst):
#                         shutil.rmtree(dst)
#                     shutil.copytree(src, dst)
#                 else:
#                     shutil.copy2(src, dst)

#         self._set_barrier('Checkout simulation')
#         self.load()


#     def branch_simulation(self, new_version):
#         '''Branch a copy of the current simulation to a new version.
#         '''
        
#         self.logger.always('Branching simulation:'
#             + '\n From: ' + self.version
#             + '\n To  : ' + new_version
#         )

#         _old_version = self.version
#         _old_path = self._out_folder
#         self.save() #generate cache

#         self.set_version(new_version)

#         if comm.rank == 0:
#             files = glob.glob(_old_path + '/*')
#             for src in files:
#                 dst = self._out_folder + '/' + src.split('/')[-1]
#                 self.logger.info('Copy: {} -> {}'.format(src, dst))
#                 if os.path.isdir(src):
#                     if os.path.isdir(dst):
#                         shutil.rmtree(dst)
#                     shutil.copytree(src, dst)
#                 else:
#                     shutil.copy2(src, dst)

#         self._set_barrier('Branch simulation')

#     def plot_beams(self):
#         '''Plot all beam-patterns of all transmitters and receivers.
#         '''
#         if comm.rank == 0:
#             for tx in self.radar._tx:
#                 antenna.plot_gain_heatmap(tx.beam, res=100, min_el = 75.)
#             for rx in self.radar._rx:
#                 antenna.plot_gain_heatmap(rx.beam, res=100, min_el = 75.)


#     def plot_radar(self, save_folder):
#         '''Plot radar configuration, includes beam pattern, geographical location and scan.
#         '''
#         if comm.rank == 0:
#             radar_config.plot_radar(self.radar, save_folder = save_folder)


#     def _clear_folder(self, folder):
#         if comm.rank == 0:
#             files = glob.glob(folder + '/*')
#             for f in files:
#                 os.remove(f)
#         self._set_barrier('Cleared "{}"'.format(folder))


#     def clear_detections(self):
#         '''Delete all files in "detections" folder. Branch specific.
#         '''
#         self._clear_folder(self._detections_folder)


#     def clear_tracklets(self):
#         '''Delete all files in "tracklets" folder. Branch specific.
#         '''
#         self._clear_folder(self._tracklet_folder)


#     def clear_prior(self):
#         '''Delete all files in "prior" folder. Branch specific.
#         '''
#         self._clear_folder(self._prior_folder)


#     def clear_plots(self):
#         '''Delete all files in "plots" folder. Branch specific.
#         '''
#         self._clear_folder(self._plots_folder)


#     def clear_logs(self):
#         '''Delete all files in "logs" folder. Affects entire Simulation.
#         '''
#         self._clear_folder(self._logs_folder)


#     def clear_orbits(self):
#         '''Delete all files in "orbits" folder. Branch specific.
#         '''
#         self._clear_folder(self._orbits_folder)


#     def list(self):
#         '''List all available methods.
#         '''
#         lst = dir(self)
#         lst = [attr for attr in lst if attr[0] != '_']
#         for attr in lst:
#             _attr = getattr(self, attr)
#             if type(_attr) == types.MethodType:
#                 self.logger.always('Method name: ' + attr)


#     @logs.class_log_call('running simulation: t0 + {1|t} h')
#     def run_observation(self, t):
#         self.__t0 = self.__t
#         self.__t1 = self.__t0 + t
#         self.__t = self.__t1

#         objs_to_iter = len(self.__my_objects[comm.rank])
        
#         self.__init_time_runscan = time.time()
#         self.__time_last = time.time()
#         if objs_to_iter > 1:
#             self.__time_stamps_runscan = np.empty( (objs_to_iter-1,) )
#         else:
#             self.__time_stamps_runscan = np.empty( (objs_to_iter,) )
#         self.__time_stamps_runscan[0] = self.__init_time_runscan

#         oid_iter = 0
#         logs.record_time_diff('run_scan_start')
#         for oid in self.__my_objects[comm.rank]:
#             oid_iter += 1
#             if oid_iter == 1:
#                 est_left = 0
#             else:
#                 self.__time_stamps_runscan[oid_iter-2] = time.time() - self.__time_last
#                 iter_time_est = np.mean( self.__time_stamps_runscan[:(oid_iter-1)] )/3600.0
#                 est_left = iter_time_est*(objs_to_iter - oid_iter)

#             est_elap = (time.time() - self.__init_time_runscan)/3600.0
#             self.__time_last = time.time()
            
            
#             self.logger.info("Thread {:<3} at obj {:<6}: Time elapsed: {:<6.2f} h, Estimated time left: {:<6.2f} h ({:<6} objects)".format(comm.rank, oid, est_elap, est_left,objs_to_iter - oid_iter))

#             s_obj = self.population.get_object(oid)

#             if not self.catalogue._known[oid]:
#                 self.logger.debug(" ==== Scanning for object ==== \n {} \n ======================".format(str(s_obj)))
                
#                 # get_iods
#                 if self.parameters.scanning_f > 0 or (self.parameters.scan_during_interleaved and self.parameters.SST_f < 1.0):

#                     self.parameters.configure_radar_to_observation(self.radar, mode = 'scan')

#                     try:
#                         detections = simulate_scan.get_detections(
#                             obj = s_obj,
#                             radar = self.radar,
#                             t0 = self.__t0*3600.0,
#                             t1 = self.__t1*3600.0,
#                             max_dpos = self._max_dpos,
#                             logger = self.logger,
#                             pass_dt = self._pass_dt,
#                         )
#                     except Exception as err:
#                         self.logger.error('Simulate scanning failed for object\n {}'.format(str(s_obj)))
#                         self.logger.error(err)
#                         continue

#                     if self.catalogue._detections[oid] is None:
#                         if np.sum([len(tx_det['tm']) for tx_det in detections]) > 0:
#                             self.catalogue._detections[oid] = detections

#                     else:
#                         for txi, tx_det in enumerate(detections):
#                             n_dets = len(tx_det['tm'])
                            
#                             for field in self.catalogue._det_fields:
#                                self.catalogue._detections[oid][txi][field] += tx_det[field]

#             else:
#                 self.logger.debug(" ====== Tracking object ====== \n {} \n ======================".format(str(s_obj)))
                
#                 self.parameters.configure_radar_to_observation(self.radar, mode = 'track')

#                 try:
#                     passes = simulate_tracking.get_passes(
#                         o=s_obj,
#                         radar = self.radar,
#                         t0 = self.__t0*3600.0,
#                         t1 = self.__t1*3600.0,
#                         max_dpos = self._max_dpos,
#                         logger = self.logger,
#                     )
#                 except Exception as err:
#                     self.logger.error('Simulate tracking failed for object\n {}'.format(str(s_obj)))
#                     self.logger.error(err)
#                     continue
#                 #passes['t'][tx_index][pass_index][rise time, fall time]
#                 #passes['snr'][tx_index][pass_index][rx_index][peak snr, peak snr time]

#                 pass_n = np.sum([len(x) for x in passes['snr'] ])

#                 if self.catalogue._maintinence[oid] is None:
#                     if pass_n > 0:
#                         self.catalogue._maintinence[oid] = passes
#                 else:
#                     for txi in range(len(self.radar._tx)):
#                         if len(passes['t'][txi]) > 0:
#                             self.catalogue._maintinence[oid]['t'  ][txi] += passes['t'  ][txi]
#                             self.catalogue._maintinence[oid]['snr'][txi] += passes['snr'][txi]

#             logs.record_time_diff('run_scan_loop')

#         self.save(MPI_synch = True)
#         if self._auto_synchronize:
#             if comm.size > 1:
#                 self.load()


#     def print_maintenance(self):
#         if comm.rank == 0:
#             self.logger.always('<< PRINTING MAINTENANCE >>')
#             det_obj_n = 0
#             _maintained = [ind for ind, passes in enumerate(self.catalogue._maintinence) if passes is not None]
#             if len(_maintained) == 0:
#                 self.logger.info('No stored maintenance passes to print.')
#             else:
#                 for ind in _maintained:
#                     passes = self.catalogue._maintinence[ind]
#                     t   = passes['t']
#                     snr = passes['snr']
#                     for txi in range(len(t)):
#                             n_dets = len(t[txi])
#                             det_obj_n += n_dets
                            
#                             self.logger.always('Object index {}: Total of {} passes for TX {} --'.format(ind, n_dets, txi,) )

#                             for det_idx in range(n_dets):
#                                 SNR_l = [ x[0] for x in snr[txi][det_idx] ]
#                                 SNR_tl = [ x[1] for x in snr[txi][det_idx] ]
#                                 self.logger.always("-- pass {}: Horizon rise {:.2f} h, fall {:.2f} h, peak snr {:.2f} dB at {:.2f} h".format(
#                                         det_idx,
#                                         t[txi][det_idx][0]/3600.0,
#                                         t[txi][det_idx][1]/3600.0,
#                                         10.0*np.log10(np.max( SNR_l )),
#                                         SNR_tl[np.argmax(SNR_l)]/3600.0,
#                                     ))

#                 if det_obj_n != 0:
#                     self.logger.always('Total {:.2f} % of objects available for maintenance an average of {:.2f} times per object.'.format(
#                             float(len(_maintained))/float(len(self.population))*100.0,
#                             float(det_obj_n)/float(len(_maintained)),
#                         ))
#                 else:
#                     self.logger.always('Total {:.2f} % of objects available for maintenance'.format(
#                             float(len(_maintained))/float(len(self.population))*100.0,
#                         ))


#     def print_detections(self):
#         if comm.rank == 0:
#             self.logger.always('<< PRINTING DETECTIONS >>')
#             det_obj_n = 0
#             _detected = [ind for ind, dets in enumerate(self.catalogue._detections) if dets is not None]
#             if len(_detected) == 0:
#                 self.logger.info('No stored detections to print.')
#             else:
#                 for ind in _detected:
#                     detections = self.catalogue._detections[ind]
#                     for txi,tx in enumerate(detections):
#                         n_dets = len(tx["tm"])
#                         det_obj_n += n_dets
#                         self.logger.always('Object index {}: Total of {} detections for TX {} --'.format(ind, n_dets, txi,) )
                        
#                         _best_det = [
#                             10.0*np.log10(np.max(tx["snr"][det_idx]))
#                             for det_idx in range(n_dets)
#                         ]
#                         _first_det = [
#                             tx["tm"][det_idx]
#                             for det_idx in range(n_dets)
#                         ]

#                         _best_ind = np.argmax(_best_det)
#                         _first_ind = np.argmin(_first_det)
                    
#                         self.logger.always("-- Best detection {}: {:.2f} h With SNR {:.2f} dB".format(
#                             _best_ind,
#                             tx["tm"][_best_ind]/3600.0,
#                             _best_det[_best_ind],
#                         ))
#                         self.logger.always("-- First detection {}: {:.2f} h With SNR {:.2f} dB".format(
#                             _first_ind,
#                             tx["tm"][_first_ind]/3600.0,
#                             _best_det[_first_ind],
#                         ))
#                 if det_obj_n != 0:
#                     self.logger.always('Total {:.2f} % of objects detected an average of {:.2f} times per detected object.'.format(
#                             float(len(_detected))/float(len(self.population))*100.0,
#                             float(det_obj_n)/float(len(_detected)),
#                         ))
#                 else:
#                     self.logger.always('Total {:.2f} % of objects detected.'.format(
#                             float(len(_detected))/float(len(self.population))*100.0,
#                         ))

#     def set_scheduler_args(self, **kwargs):
#         self.scheduler_args = kwargs

    
#     @logs.class_log_call('running scheduler')
#     def run_scheduler(self, t0 = None, t1 = None):
#         if comm.rank == 0:
#             logs.record_time_diff('scheduler_start')
            
#             if t0 is None:
#                 t0 = self.__t0
#             if t1 is None:
#                 t1 = self.__t

#             N_det = np.sum(1 for x in self.catalogue._detections if x is not None)
#             N_maint = np.sum(1 for x in self.catalogue._maintinence if x is not None)
            
#             if N_det > 0 or N_maint > 0:

#                 self.catalogue = self.scheduler(
#                     catalogue = self.catalogue,
#                     radar = self.radar,
#                     parameters = self.parameters,
#                     t0 = t0*3600.0,
#                     t1 = t1*3600.0,
#                     **self.scheduler_args
#                 )

#             else:
#                 self.logger.error('No detection or maintenance data to schedule.')

#             logs.record_time_diff('scheduler')

#         self.save(MPI_synch = False)
#         if self._auto_synchronize:
#             if comm.size > 1:
#                 self.load()



#     def print_tracklets(self):
#         if comm.rank == 0:
#             self.logger.always('<< PRINTING TRACKLETS >>')
#             header_lst = self.catalogue._tracklet_format.keys()
#             _sep = '  |  '

#             row = ['{!r:<14}']*len(header_lst)
#             header = ['{:<14}']*len(header_lst)
#             header = [header[ind].format(nm) for ind, nm in enumerate(header_lst)]
#             header = _sep.join(header)

#             self.logger.always(header)
#             self.logger.always('-'*len(header))
#             for ti, tracklet in enumerate(self.catalogue.tracklets):
#                 _row = []
#                 for ind, field in enumerate(header_lst):
#                     dat = tracklet[field]
#                     if isinstance(dat, list) or isinstance(dat, np.ndarray):
#                         _row.append(row[ind].format(len(dat)))
#                     else:
#                         _row.append(row[ind].format(dat))
#                 self.logger.always(_sep.join(_row))

#     def print_tracks(self):
#         if comm.rank == 0:
#             self.logger.always('<< PRINTING TRACKS >>')
#             header_lst = self.catalogue.tracks.dtype.names
#             _sep = '  |  '

#             row = []
#             header = ['{:<14}']*len(header_lst)
#             for field in header_lst:
#                 if np.issubdtype(self.catalogue.tracks.dtype[field], np.inexact):
#                     row.append('{:<14.3f}')
#                 elif np.issubdtype(self.catalogue.tracks.dtype[field], np.bool_):
#                     row.append('{!r:<14}')
#                 else:
#                     row.append('{:<14}')

#             header = [header[ind].format(nm) for ind, nm in enumerate(header_lst)]
#             header = _sep.join(header)
#             self.logger.always(header)
#             self.logger.always('-'*len(header))
#             for ti, track in enumerate(self.catalogue.tracks):
#                 _row = [row[ind].format(track[field]) for ind, field in enumerate(header_lst)]
#                 self.logger.always(_sep.join(_row))


#     def plots(self):
#         if comm.rank == 0:
#             self.catalogue.plots(save_folder = self._plots_folder)
#             self.plot_radar(save_folder = self._plots_folder)
#             plt.close('all')


#     def generate_tracklets(self):
#         if len(self.catalogue.tracklets) > 0:
#             self.logger.info('<< Starting tracklet generation >>')

#             logs.record_time_diff('generate_tracklets')
#             tr_index = np.array(
#                 [_trl['index'] for _trl in self.catalogue.tracklets],
#                 dtype=np.int64,
#             )
#             _my_len = len(self.__my_objects[comm.rank])
#             _cnt = 0

#             for ind in self.__my_objects[comm.rank]:
#                 _cnt += 1
#                 _iters = np.argwhere(tr_index == ind).flatten()
#                 for tid in _iters:
#                     tracklet = self.catalogue.tracklets[tid]
#                     track = self.catalogue.tracks[tracklet['track_id']]

#                     logs.record_time_diff('tracklet_calc_write')

#                     s_obj = self.population.get_object(ind)

#                     self.logger.info('Tracklet {:<6}: object {:<6} ({:<6} left) mode {:<8}'.format(tid, ind, _my_len-_cnt, track['type']))

#                     #configure radar to correct mode, regardless of discovery method, we always need control to track
#                     self.parameters.configure_radar_to_observation(self.radar, mode = 'track')

#                     data, tracklet_fnames, errs = simulate_tracklet.create_tracklet(
#                         o = s_obj,
#                         radar = self.radar,
#                         t_obs = tracklet['t'],
#                         hdf5_out = True,
#                         ccsds_out = True,
#                         dname = self._tracklet_folder,
#                         noise = self._tracklet_noise,
#                     )

#                     tracklet['fnames'] += tracklet_fnames
#         else:
#             self.logger.error('No tracklet data has been generated')

#         self.save(MPI_synch = True)
#         if self._auto_synchronize:
#             if comm.size > 1:
#                 self.load()


#     def generate_priors(self, frame_transformation, frame_options = {}, **kwargs):
#         if len(self.catalogue.tracklets) > 0:
#             self.logger.info('<< Starting prior generation >>')

#             self.logger.info('Deleting possible previous priors.')
#             self.catalogue.priors = np.empty((0,), dtype=self.catalogue.priors.dtype)

#             logs.record_time_diff('start_generate_priors')
#             for ind in self.__my_objects[comm.rank]:
                
#                 object_tracks = self.catalogue.tracks[self.catalogue.tracks['index'] == ind]
                
#                 #remove rows without tracklets
#                 object_tracks = object_tracks[object_tracks['tracklet']]

#                 #sort according to start time
#                 start_times = object_tracks['t0']
#                 _sort = np.argsort(start_times)
#                 object_tracks = object_tracks[_sort]

#                 s_obj = self.population.get_object(ind)
#                 s_obj_true = s_obj.copy()
#                 _prior_generated = False

#                 for track in object_tracks:
#                     logs.record_time_diff('generate_prior')

#                     tracklet = self.catalogue.tracklets[track['tracklet_index']]

#                     if len(tracklet['t']) <= 1:
#                         continue

#                     if len(tracklet['fnames']) < 3:
#                         self.logger.debug('Not enough tracklets to make OEM: {}'.format(len(tracklet['fnames'])))
#                         continue
#                     fnames = [fname + '.h5' for fname in tracklet['fnames']]
#                     paths = sources.Path.from_list(fnames, 'file')
#                     for path in paths:
#                         self.logger.debug('Loading tracklet data from: {}'.format(path))

#                     source_list = sources.SourceCollection(paths = paths)
#                     source_list.sort(key=lambda x: x.meta['fname'])

#                     _start_dates = []
#                     for src in source_list:
#                         _start_dates += [np.min(src.data['date'])]
#                     _start_dates = np.array(_start_dates)

#                     _start = np.argmin(_start_dates)
#                     _sd_max_r = []
#                     _sd_max_v = []
#                     for src in source_list:
#                         src.data = src.data[src.data['date'] > _start_dates[_start]]
#                         _sd_max_r.append(src.data['r_sd'][0])
#                         _sd_max_v.append(src.data['v_sd'][0])

#                     _sd_max_r = np.max(_sd_max_r)
#                     _sd_max_v = np.max(_sd_max_v)

#                     if 'tracklet_truncate' in kwargs:
#                         for src in source_list:
#                             src.data = src.data[kwargs['tracklet_truncate']]

#                     mjd0 = dpt.npdt2mjd(_start_dates[_start])
#                     date0 = _start_dates[_start]

#                     _dt = (mjd0 - s_obj.mjd0)*3600.0*24.0

#                     self.logger.info('perturbing first state: sd = {} m'.format(_sd_max_r))
#                     self.logger.info('perturbing first state: sd = {} m/s'.format(_sd_max_v))

#                     _start00 = s_obj.get_state( _dt )
                    
#                     self.logger.warning('INITIAL STATE ESTIMATE NOT IMPLEMENTED, using true state plus a normal 1km, 10m/s perturbation as initial guess')

#                     working_orb = False
#                     while not working_orb:
#                         _start0 = _start00.copy()

#                         _start0[:3,0] += np.random.randn(3)*1e3  #*_sd_max_r
#                         _start0[3:,0] += np.random.randn(3)*10.0  #*_sd_max_v

#                         _start0[:,0] = _perform_transform(
#                             _start0[:,0], 
#                             frame_transformation, 
#                             frame_options, 
#                             jd_ut1 = dpt.mjd_to_jd(mjd0),
#                         )

#                         _kep0 = dpt.cart2kep(_start0, M_cent=space_object.M_e, radians=False)
#                         if _kep0[1] < 1.0:
#                             working_orb = True

#                     variables_orb = ['x', 'y', 'z', 'vx', 'vy', 'vz']

#                     start0 = np.empty((1,), dtype=[(name, 'float64') for name in variables_orb])
#                     for dim, var in enumerate(variables_orb):
#                         start0[var][0] = _start0[dim]

#                     self.logger.info('Generating prior {} from tracklet index {}'.format(
#                         ind, track['tracklet_index'],
#                     ))

#                     input_data_tracklets = {
#                         'sources': source_list,
#                         'Model': orbit_determination.RadarPair,
#                         'date0': date0,
#                         'params': {
#                             'A': 1.0,
#                             'm': 1.0,
#                             'C_D': 2.3,
#                             'C_R': 1.4,
#                         },
#                     }

#                     find_map = orbit_determination.OptimizeLeastSquares(
#                         data = input_data_tracklets,
#                         variables = variables_orb,
#                         start = start0,
#                         prior = None,
#                         propagator = self.population.propagator(**self.population.propagator_options),
#                         method = 'Nelder-Mead',
#                         options = dict(
#                             maxiter = 10000,
#                             disp = False,
#                         ),
#                     )
#                     try:
#                         estimated_map = find_map.run()
#                     except:
#                         self.logger.error('Prior generation failed, skipping')
#                         continue

#                     #residuals = find_map.residuals()

#                     state_prior = np.empty((6,), dtype=np.float64)
#                     state_prior[0] = estimated_map.MAP['x'][0]
#                     state_prior[1] = estimated_map.MAP['y'][0]
#                     state_prior[2] = estimated_map.MAP['z'][0]
#                     state_prior[3] = estimated_map.MAP['vx'][0]
#                     state_prior[4] = estimated_map.MAP['vy'][0]
#                     state_prior[5] = estimated_map.MAP['vz'][0]
                    
#                     s_obj.mjd0 = mjd0
#                     s_obj.update(
#                         x = estimated_map.MAP['x'][0]*1e-3,
#                         y = estimated_map.MAP['y'][0]*1e-3,
#                         z = estimated_map.MAP['z'][0]*1e-3,
#                         vx = estimated_map.MAP['vx'][0]*1e-3,
#                         vy = estimated_map.MAP['vy'][0]*1e-3,
#                         vz = estimated_map.MAP['vz'][0]*1e-3,
#                     )

#                     _t = tracklet['t'] - _dt
#                     _t = np.sort(_t[_t > 0])
#                     _t_us = np.round(_t.copy()*1e6).astype('int64')
#                     ecefs = s_obj.get_state(_t)

#                     ecefs_true = s_obj_true.get_state(_t + _dt)

#                     err_r = np.linalg.norm(ecefs_true[:3, :] - ecefs[:3, :], axis=0)
#                     err_v = np.linalg.norm(ecefs_true[3:, :] - ecefs[3:, :], axis=0)

#                     self.logger.info('Prior estimation true error')
#                     self.logger.info('mu = {} m, sd = {} m'.format(np.mean(err_r), np.std(err_r)))
#                     self.logger.info('mu = {} m/s, sd = {} m/s'.format(np.mean(err_v), np.std(err_v)))

#                     self.catalogue.add_prior(
#                         index = ind,
#                         state = state_prior,
#                         cov = np.eye(6)*1e4,
#                         date = date0,
#                     )
                    
#                     t0 = dpt.jd_to_unix(dpt.mjd_to_jd(mjd0))

#                     oid = int(self.catalogue._oids[ind])
#                     _fname = '/{}_init.oem'.format(oid)

#                     ccsds_write.write_oem(
#                         t0 + _t,
#                         ecefs.T,
#                         oid=oid,
#                         fname=self._prior_folder + _fname,
#                     )
#                     tracklet['is_prior'] = True
#                     _prior_generated = True
#                     break

#                 if not _prior_generated:
#                     self.logger.warn('No tracklet could be used as prior, object cannot be added to catalogue')

#             self.logger.info('<< Prior generation complete >>')

#         self.save(MPI_synch = True)
#         if self._auto_synchronize:
#             if comm.size > 1:
#                 self.load()


#     def run_orbit_determination(self, frame_transformation, frame_options = {}, **kwargs):

#         max_zenith_error = kwargs['max_zenith_error']
#         steps=kwargs.get('steps',100000)
#         error_samp = kwargs.get('error_samp', steps//10)
#         error_dt = kwargs.get('error_dt', 60.0)
#         error_t = np.arange(0.0, (self.__t + 24.0)*3600.0, error_dt)

#         prop = self.population.propagator(**self.population.propagator_options)

#         variables = kwargs.get('variables', ['x', 'y', 'z', 'vx', 'vy', 'vz', 'A', 'm'])
#         variables_orb = ['x', 'y', 'z', 'vx', 'vy', 'vz']

#         params = {
#             'C_D': 2.3,
#             'm': 1.0,
#             'A': 0.1,
#             'C_R': 1.0,
#         }

#         _params = kwargs.get('params', None)
        
#         if params is None:
#             params.update(_params)

#         params_unknown = params.copy()
#         for var in variables:
#             if var in params_unknown:
#                 del params_unknown[var]

#         self.logger.info('Parameters:')
#         for key, item in params.items():
#             self.logger.info('{:<5}: {}'.format(key, item))

#         variables_known = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'C_D']

#         dt_known = [(name, 'float64') for name in variables_known]
#         dt_unknown = [(name, 'float64') for name in variables]
#         dt_all = [(name, 'float64') for name in set(variables + variables_known)]
#         step = kwargs.get('step', None)
#         if step is None:
#             step = np.empty((1,), dtype=dt_all)
#             for var in ['x', 'y', 'z']:
#                 if var in variables + variables_known:
#                     step[var] = 1.0
#             for var in ['vx', 'vy', 'vz']:
#                 if var in variables + variables_known:
#                     step[var] = 0.1
#             if 'A' in variables + variables_known:
#                 step['A'] = 0.1
#             if 'C_D' in variables + variables_known:
#                 step['C_D'] = 0.01
#             if 'm' in variables + variables_known:
#                 step['m'] = 0.1
#         step0 = step.copy()

#         if len(self.catalogue.tracklets) > 0:
#             self.logger.info('<< Starting orbit determinations >>')

#             logs.record_time_diff('start_sequencial_OD')
#             for ind in self.__my_objects[comm.rank]:

#                 error_inds = None
#                 OD_errors = {
#                     'r': np.empty((error_samp, 0), dtype=np.float64),
#                     'v': np.empty((error_samp, 0), dtype=np.float64),
#                 }

#                 obj_folder = self._orbits_folder + '/' + str(ind)

#                 if self.catalogue._known[ind]:
#                     state0 = np.zeros((1,), dtype=dt_known)
#                     use_variables = variables_known
#                 else:
#                     state0 = np.zeros((1,), dtype=dt_unknown)
#                     use_variables = variables

#                 object_tracks = self.catalogue.tracks[self.catalogue.tracks['index'] == ind]

#                 #remove rows without tracklets
#                 object_tracks = object_tracks[object_tracks['tracklet']]
#                 _save = np.full((len(object_tracks,)), True, np.bool)
#                 for obid, obtr in enumerate(object_tracks[:]):
#                     if self.catalogue.tracklets[obtr['tracklet_index']]['is_prior']:
#                         _save[obid] = False
#                 object_tracks = object_tracks[_save]
#                 #sort according to start time
#                 start_times = object_tracks['t0']
#                 _sort = np.argsort(start_times)
#                 object_tracks = object_tracks[_sort]

#                 priors = self.catalogue.priors
#                 prior = priors[priors['index']==ind]
#                 if len(prior) == 0:
#                     self.logger.info('No prior avalible, skipping')
#                     continue

#                 s_obj = self.population.get_object(ind)
#                 _states0 = s_obj.get_state(error_t)

#                 self.__make_folder(obj_folder)

#                 pr_date0 = prior['date'][0]
#                 date0 = pr_date0

#                 prior_cov = prior['cov'][0].copy()
#                 prior_mu = np.empty((len(variables_orb),), dtype=np.float64)

#                 _prior_generated = False

#                 for dim, var in enumerate(variables_orb):
#                     state0[0][var] = prior[var][0]
#                     prior_mu[dim] = prior[var][0]
#                 if self.catalogue._known[ind]:
#                     go_params = params
#                     for var, dt in dt_known:
#                         if var in variables_orb:
#                             continue
#                         state0[0][var] = self.population.objs[ind][var]
#                 else:
#                     go_params = params_unknown
#                     for var, dt in dt_unknown:
#                         if var in variables_orb:
#                             continue
#                         state0[0][var] = params[var]

#                 data_times = []
#                 fail_time = []

#                 for tri, track in enumerate(object_tracks):
#                     plt.close('all')
#                     logs.record_time_diff('orbit_determination')

#                     date0_ts = (dpt.npdt2mjd(date0) - s_obj.mjd0)*3600.0*24.0
#                     self.logger.info('Date for prior @ {:.3f} h'.format(date0_ts/3600.0))
#                     self.logger.info('Prior mean: ' + ('{:<10.2f} | '*len(prior_mu)).format(*prior_mu.tolist()))
#                     self.logger.info('Prior cov:')

#                     if np.linalg.det(prior_cov) < 1e-9:
#                         self.logger.warning('Singular prior covariance: adding noise to diagonal')
#                         for dim in range(len(variables_orb)):
#                             prior_cov[dim, dim] += np.abs(np.random.randn(1)*1e-3)

#                     for row in prior_cov:
#                         self.logger.info(('{:<10.2f} | '*len(row)).format(*row.tolist()))

#                     step = step0.copy()


#                     tracklet = self.catalogue.tracklets[track['tracklet_index']]
#                     if tri == len(object_tracks) - 1:
#                         next_tracklet = None
#                     else:
#                         next_tracklet = self.catalogue.tracklets[object_tracks[tri+1]['tracklet_index']]

#                     _end_t = np.max(tracklet['t'])
#                     _max_ind = np.argmin(np.abs(error_t - _end_t)) - 1
#                     if error_inds is None:
#                         error_inds = slice(None, _max_ind)
#                     else:
#                         error_inds = slice(_prev_ind, _max_ind)
#                     _prev_ind = _max_ind


#                     fnames = []
#                     for _track in object_tracks[:(tri+1)]:
#                         _tracklet = self.catalogue.tracklets[_track['tracklet_index']]
#                         fnames += [fname + '.h5' for fname in _tracklet['fnames']]

#                     paths = sources.Path.from_list(fnames, 'file')
#                     for path in paths:
#                         self.logger.debug('Loading tracklet data from: {}'.format(path))
#                     self.logger.info('Loading data from {} tracklets'.format(len(paths)))

#                     source_list = sources.SourceCollection(paths = paths)
#                     source_list.sort(key=lambda x: x.meta['fname'])

#                     if 'tracklet_truncate' in kwargs:
#                         for src in source_list:
#                             src.data = src.data[kwargs['tracklet_truncate']]

#                     input_data_tracklets = {
#                         'sources': source_list,
#                         'Model': orbit_determination.RadarPair,
#                         'date0': date0,
#                         'params': go_params,
#                     }


#                     find_map = orbit_determination.OptimizeLeastSquares(
#                         data = input_data_tracklets,
#                         variables = use_variables,
#                         start = state0,
#                         propagator = self.population.propagator(**self.population.propagator_options),
#                         method = 'Nelder-Mead',
#                         options = dict(
#                             maxiter = 10000,
#                             disp = False,
#                         ),
#                     )
                    
#                     estimated_map = find_map.run()

                    
#                     prior_dilute = kwargs.get('prior_dilute', np.diag([1e3, 1e3, 1e3, 10.0, 10.0, 10.0]))

#                     prior_dists = [ #these can be any combination of variables and any scipy continues variable
#                         {
#                             'variables': copy.copy(variables_orb),
#                             'distribution': 'multivariate_normal',
#                             'params': {
#                                 'mean': prior_mu,
#                                 'cov': prior_cov + prior_dilute,
#                             },
#                         },
#                     ]
#                     if 'C_D' in use_variables:
#                         prior_dists.append(
#                             {
#                                 'variables': ['C_D'],
#                                 'distribution': 'uniform',
#                                 'params': {
#                                     'loc': 0.1,
#                                     'scale': 4.0,
#                                 },
#                             },
#                         )

#                     MCMCtrace = orbit_determination.MCMCLeastSquares(
#                         MPI = False,
#                         data = input_data_tracklets,
#                         variables = use_variables,
#                         start = estimated_map.MAP,
#                         prior = None,
#                         propagator = prop,
#                         method = 'SCAM',
#                         steps = steps,
#                         step = step,
#                         tune = kwargs.get('tune',3000),
#                     )
#                     results = MCMCtrace.run()

#                     self.logger.info('MCMC MAP error')
#                     _est_map = np.empty((6,), dtype=np.float64)
#                     for dim, var in enumerate(variables_orb):
#                         _est_map[dim] = results.MAP[var][0]
#                     _true_map = s_obj.get_state([(dpt.npdt2mjd(results.date) - s_obj.mjd0)*3600.0*24.0])
#                     _true_map[:, 0] = _perform_transform(
#                         _true_map[:, 0], 
#                         frame_transformation, 
#                         frame_options, 
#                         jd_ut1 = dpt.mjd_to_jd(dpt.npdt2mjd(results.date)),
#                     )
#                     self.logger.info('{} m, {} m/s'.format(np.linalg.norm(_true_map[:3,0] - _est_map[:3]), np.linalg.norm(_true_map[3:,0] - _est_map[3:])))



#                     _dtus = int(_end_t*1e6)
#                     _dts = np.float64(_dtus)*1e-6

#                     #print('date0',date0)
#                     #print('_end_t',_end_t)
#                     #print('_dts',_dts)
#                     #print('date0_ts', date0_ts)

#                     fout_tmp = obj_folder + '/{}_{}_'.format(ind, tri)
#                     results.save(fout_tmp + 'orbit_determination.h5')
#                     orbit_determination.plot_trace(results, fout=fout_tmp + 'trace')
#                     orbit_determination.plot_scatter_trace(results, fout=fout_tmp + 'scatter')
#                     orbit_determination.plot_autocorrelation(results, fout=fout_tmp + 'autocorr')

#                     prior_mu = np.empty((len(use_variables),), dtype=np.float64)
#                     for dim, var in enumerate(use_variables):
#                         prior_mu[dim] = results.MAP[0][var]
#                     prior_cov = results.covariance_mat(variables=use_variables)

#                     #print(prior_mu)
#                     #for dim, var in enumerate(use_variables):
#                     #    print('{}: {}'.format(var, prior_cov[dim,:]))

#                     _get_subsamp = np.random.randint(0, high=steps, size=error_samp)
#                     _samps = np.empty((len(use_variables), error_samp), dtype=np.float64)
#                     for sampi in range(error_samp):
#                         for dim, var in enumerate(use_variables):
#                             _samps[dim, sampi] = results.trace[_get_subsamp[sampi]][var]
#                     #_samps = np.random.multivariate_normal(
#                     #    mean=prior_mu,
#                     #    cov=prior_cov,
#                     #    size=error_samp,
#                     #).T

#                     _errs = np.empty((error_samp, len(error_t[error_inds]), 2), dtype=np.float64)
#                     _new_trace = np.empty((len(variables_orb), error_samp), dtype=np.float64)
#                     _next_tracklet = np.empty((error_samp,), dtype=np.float64)

#                     if next_tracklet is not None:
#                         _next_dt = np.min(next_tracklet['t'])
#                         _tv = np.array(error_t[error_inds].tolist() + [_dts, _next_dt], dtype=np.float64)
#                         _tmp_states = s_obj.get_state([_dts, _next_dt])
                        
#                         _prior_true_state = _tmp_states[:, 0]
#                         _prior_true_state = _perform_transform(
#                             _prior_true_state, 
#                             frame_transformation, 
#                             frame_options, 
#                             jd_ut1 = dpt.mjd_to_jd(s_obj.mjd0 + _dts/(3600.0*24.0)),
#                         )

#                         _true_prior_obj = self.population.get_object(ind)
#                         _true_prior_obj.mjd0 = s_obj.mjd0 + date0_ts/(3600.0*24.0)
#                         update_kw = {}
#                         for ati, attr in enumerate(variables_orb):
#                                 update_kw[attr] = _prior_true_state[ati]*1e-3
#                         _true_prior_obj.update(**update_kw)


#                         _next_true_state = _tmp_states[:, 1]
#                     else:
#                         _tv = np.array(error_t[error_inds].tolist() + [_dts], dtype=np.float64)

#                     self.logger.warning('ADDING DEFAULT PROPERTIES NOT YET IMPLEMENTED')
#                     for sid in tqdm(range(error_samp)):
#                         _samp = _samps[:,sid]
#                         update_kw = {}

#                         s_obj_pert = self.population.get_object(ind)
#                         s_obj_pert.mjd0 = s_obj.mjd0 + date0_ts/(3600.0*24.0)

#                         for ati, attr in enumerate(use_variables):
#                             if attr in variables_orb:
#                                 update_kw[attr] = _samp[ati]*1e-3
#                             else:
#                                 setattr(s_obj_pert, attr, _samp[ati])

#                         s_obj_pert.update(**update_kw)

#                         _states = s_obj_pert.get_state(_tv - date0_ts)

#                         if next_tracklet is not None:
#                             _tmp_point1 = _states[:3, -1] - self.radar._tx[0].ecef
#                             _tmp_point2 = _next_true_state[:3] - self.radar._tx[0].ecef
#                             _next_tracklet[sid] = coord.angle_deg(_tmp_point1, _tmp_point2)
#                             _states = _states[:, :-1]

#                         print('DEBUG:')
#                         print(_states[:, -1] - )

#                         _new_trace[:, sid] = _states[:, -1]

#                         _new_trace[:, sid] = _perform_transform(
#                             _new_trace[:, sid], 
#                             frame_transformation, 
#                             frame_options, 
#                             jd_ut1 = dpt.mjd_to_jd(s_obj.mjd0 + _dts/(3600.0*24.0)),
#                         )

#                         _errs_vec = _states0[:, error_inds] - _states[:, :-1]
#                         _errs[sid, :, 0] = np.linalg.norm(_errs_vec[:3,:], axis=0)
#                         _errs[sid, :, 1] = np.linalg.norm(_errs_vec[3:,:], axis=0)
                        

#                     date0 = dpt.mjd2npdt(s_obj.mjd0) + np.timedelta64(_dtus, 'us')

#                     _mu = np.mean(_new_trace, axis=1)

#                     if next_tracklet is not None:
#                         _mu_err = _new_trace.copy()
#                         for ddim in range(6):
#                             _mu_err[ddim, :] -= _prior_true_state[ddim]
#                         _mu_err_r = np.linalg.norm(_mu_err[:3,:], axis=0)
#                         _mu_err_v = np.linalg.norm(_mu_err[3:,:], axis=0)

#                         self.logger.info('New-trace error vs true:')
#                         self.logger.info('mu {} m, sd {} m'.format(np.mean(_mu_err_r), np.std(_mu_err_r)))
#                         self.logger.info('mu {} m/s, sd {} m/s'.format(np.mean(_mu_err_v), np.std(_mu_err_v)))

#                     prior_mu = np.empty((len(variables_orb),), dtype=np.float64)
#                     for dim, var in enumerate(variables_orb):
#                         prior_mu[dim] = _mu[dim]
#                         state0[var] = _mu[dim]
#                     for dim, var in enumerate(use_variables):
#                         if var not in variables_orb:
#                             state0[var] = results.MAP[var][0]
#                     prior_cov = np.cov(_new_trace)

#                     OD_errors['r'] = np.append(OD_errors['r'], _errs[:, :, 0], axis=1)
#                     OD_errors['v'] = np.append(OD_errors['v'], _errs[:, :, 1], axis=1)

#                     if next_tracklet is not None:
#                         _mean_err_next = np.mean(_next_tracklet)
                        
#                         if _mean_err_next > max_zenith_error:
#                             self.logger.info('Object index {} lost: {} deg off zenith error at {} deg tolerance'.format(ind, _mean_err_next, max_zenith_error))

#                             fail_time = [ _next_dt ]

#                             break
#                         else:
#                             self.logger.info('Object index {} kept: {} deg off zenith error'.format(ind, _mean_err_next))
#                             data_times += tracklet['t'].tolist()


#                 _tv = error_t[_max_ind:]
#                 _errs = np.empty((error_samp, len(_tv), 2), dtype=np.float64)
#                 for sid in tqdm(range(error_samp)):
#                     _samp = _samps[:,sid]
#                     update_kw = {}

#                     s_obj_pert = s_obj.copy()
#                     s_obj_pert.mjd0 = dpt.npdt2mjd(results.date)

#                     for ati, attr in enumerate(use_variables):
#                         if attr in variables_orb:
#                             update_kw[attr] = _samp[ati]*1e-3
#                         else:
#                             setattr(s_obj_pert, attr, _samp[ati])
                            
#                     s_obj_pert.update(**update_kw)

#                     _states = s_obj_pert.get_state(_tv - (s_obj_pert.mjd0 - s_obj.mjd0)*3600.0*24.0)

#                     _errs_vec = _states0[:, _max_ind:] - _states
#                     _errs[sid, :, 0] = np.linalg.norm(_errs_vec[:3,:], axis=0)
#                     _errs[sid, :, 1] = np.linalg.norm(_errs_vec[3:,:], axis=0)
                
#                 OD_errors['r'] = np.append(OD_errors['r'], _errs[:, :, 0], axis=1)
#                 OD_errors['v'] = np.append(OD_errors['v'], _errs[:, :, 1], axis=1)
                
#                 CI = kwargs.get('CI', 0.95)
#                 CI_alpha = (1.0 - CI)*0.5

#                 _mean_r = np.mean(OD_errors['r'], axis=0)
#                 _CI_rp = np.zeros(_mean_r.shape, dtype=_mean_r.dtype)
#                 _CI_rm = np.zeros(_mean_r.shape, dtype=_mean_r.dtype)
#                 for tid in range(len(error_t)):
#                     _sortr = np.sort(OD_errors['r'][:,tid])
#                     _CI_rm[tid] = _sortr[int(error_samp*CI_alpha)]
#                     _CI_rp[tid] = _sortr[int(error_samp*(1.0-CI_alpha))]

#                 _mean_v = np.mean(OD_errors['v'], axis=0)
#                 _CI_vp = np.zeros(_mean_v.shape, dtype=_mean_v.dtype)
#                 _CI_vm = np.zeros(_mean_v.shape, dtype=_mean_v.dtype)
#                 for tid in range(len(error_t)):
#                     _sortr = np.sort(OD_errors['v'][:,tid])
#                     _CI_vm[tid] = _sortr[int(error_samp*CI_alpha)]
#                     _CI_vp[tid] = _sortr[int(error_samp*(1.0-CI_alpha))]

#                 fontsize = 22

#                 fig = plt.figure(figsize=(15,15))
#                 ax1 = fig.add_subplot(211)
#                 ax1.set_title(kwargs.get('title', 'Orbit determination errors'), fontsize=fontsize)
#                 for _t in data_times:
#                     plt.axvline(x=_t/3600.0, color='g', alpha=0.25)
#                 for _t in fail_time:
#                     plt.axvline(x=_t/3600.0, color='r', alpha=0.25)

#                 ax1.fill_between(error_t/3600.0, _CI_rm, _CI_rp, 
#                     facecolor='b', 
#                     alpha=0.2,
#                     label='{:.1f} % error CI'.format(CI*100),
#                 )

#                 ax1.semilogy(error_t/3600.0, _mean_r, '-k', alpha=1.0, label='Mean error')

#                 ax1.legend()
#                 ax1.set_xlabel('Time [h]', fontsize=fontsize)
#                 ax1.set_ylabel('Position error [m]', fontsize=fontsize)

#                 ax2 = fig.add_subplot(212)
#                 for _t in data_times:
#                     plt.axvline(x=_t/3600.0, color='g', alpha=0.25)
#                 for _t in fail_time:
#                     plt.axvline(x=_t/3600.0, color='r', alpha=0.25)

#                 ax2.fill_between(error_t/3600.0, _CI_vm, _CI_vp, 
#                     facecolor='b', 
#                     alpha=0.2,
#                     label='{:.1f} % error CI'.format(CI*100),
#                 )

#                 ax2.semilogy(error_t/3600.0, _mean_v, '-k', alpha=1.0, label='Mean error')
#                 ax2.legend()
#                 ax2.set_xlabel('Time [h]', fontsize=fontsize)
#                 ax2.set_ylabel('Velocity error [m/s]', fontsize=fontsize)

#                 fig.savefig(obj_folder + '/OD_errors.png', bbox_inches='tight')


# def _perform_transform(state, frame_transformation, frame_options = {}, **kwargs):

#     if isinstance(frame_transformation, str):
#         jd_ut1 = kwargs['jd_ut1']

#         frame_options.setdefault('xp', 0.0)
#         frame_options.setdefault('yp', 0.0)

#         if frame_transformation == 'ITRF':
#             state = tle.TEME_to_ITRF(state, jd_ut1, **frame_options)
#         elif frame_transformation == 'TEME':
#             state = tle.ITRF_to_TEME(state, jd_ut1, **frame_options)
#         else:
#             raise ValueError('Tranformation {} not recognized'.format(frame_transformation))
#     elif isinstance(frame_transformation, None):
#         pass
#     else:
#         state = frame_transformation(state, jd_ut1, **frame_options)

#     return state