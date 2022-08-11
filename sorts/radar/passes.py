#!/usr/bin/env python

'''Encapsulates a fundamental component of tracking space objects: a 
pass over a geographic location. Also provides convenience functions 
for finding passes given states and stations and sorting structures 
of passes in particular ways. 
'''
import datetime

import numpy as np
import pyorb
import pyant

import multiprocessing

#Local import
from ..transformations import frames
from .signals import hard_target_snr

class Pass:
    ''' 
    Encapsulates a space object *pass* over a geographical location. 

    The :class:`Pass` contains all the functions and attributes needed to store
    and run computations over the sequence of states which belong to the pass.

    Parameters
    ----------
    t : numpy.ndarray of float (N,)
        Time points associated with the space object's states (s). 
    enu : numpy.ndarray of float (6, N)
        Space object's states in the station local *ENU* coordinate frame (m).
    inds : numpy.ndarray of int, default=None
        Indices of the space object's states within the larger state dataset.
    cache : bool, default=True
        If True, ``start``, ``end``, ``range``, ``range_rate`` and ``zenit_angle``
        will only be computed once and the result will be stored, therefore increasing
        RAM usage.

        If False, the computations of those attributes will be computed at each 
        call, therefore increasing computation time.
    stations : list of :class:`sorts.Station<sorts.radar.system.station.Station>`, default=None
        Radar :class:`sorts.Stations<sorts.radar.system.station.Station>` associated with the radar
        pass. Usually, the :class:`Pass` is defined as the set of states within the 
        combined FOV of :class:`sorts.Stations<sorts.radar.system.station.Station>` 
        within ``stations``.  
    
    Examples
    --------
    As a simple example, consider an array ``states`` of space object's states propagated 
    using a :ref:`propagator`.
    
    >>> states.shape
    (6, 150)
    >>> t_states.shape
    (150)

    The creation of a new :class:`Pass` object holding the first 10 states of the ``states`` 
    array can be achieved by running:

    >>> inds = np.arange(0, 9)
    >>> pass1 = Pass(
    ...     t_states[inds], 
    ...     radar.tx.enu(states[:, inds]), # convert states to ENU in the TX reference frame
    ...     inds,
    ...     cache=False,
    ...     stations=[radar.tx[0], radar.rx[0]])

    The new :class:`Pass` object will be associated with the 1st :ref:`station_tx` station and the first
    :ref:`station_rx` station of the radar system (see :ref:`radar` for more information about
    the creation of radar systems).
    
    '''

    def __init__(self, t, enu, inds=None, cache=True, stations=None):
        ''' Default :class:`Pass` class constructor. '''

        self.inds = inds
        ''' Indices of space object's states constituing the :class:`Pass`. '''

        self.t = t
        ''' Time points associated with the space object's states (s). '''

        self.enu = enu
        ''' Space object's states in the station local *ENU* coordinate frame (m). '''

        self.cache = cache
        ''' If True, the results of ``start``, ``end``, ``range``, ``range_rate`` and 
        ``zenit_angle`` will be cached. '''

        self._stations = stations
        ''' Radar :ref:`station` associated with the :class:`Pass` '''

        self.snr = None
        ''' Optimal Signal-to-Noise Ratio (SNR) computed from the states constituing 
        the :class:`Pass`. '''

        self._start = None
        ''' Start time of the :class:`Pass` '''

        self._end = None
        ''' End time of the :class:`Pass` '''

        self._range = None
        ''' Object *range* from the stations at each time point. '''

        self._range_rate = None
        ''' Object *range rate* from the stations at each time point. '''

        self._zenith_angle = None
        ''' Object *zenith angle* with respect to the stations at each time point. '''


    def __str__(self):
        ''' Implementation of __str__. '''
        str_ = 'Pass '
        if self._stations is not None:
            str_ += f'Station {self._stations} | '
        str_ += f'Rise {str(datetime.timedelta(seconds=self.start()))} ({(self.end() - self.start())/60.0:.1f} min) {str(datetime.timedelta(seconds=self.end()))} Fall'
        
        return str_


    def __repr__(self):
        ''' Implementation of __repr__. '''
        return str(self)


    def calculate_snr(
        self, 
        tx, 
        rx,
        diameter,
        parallelization=True,
        n_processes=16,
    ):
        '''Uses the :code:`signals.hard_target_snr` function to calculate the optimal 
        SNR curve of a target during the pass.
    
        The **optimal SNR** curve corresponds to the SNR measured when the object
        is exactly in the line-of-sight of the radar (i.e. the gain is maximum). 
        The SNR's are returned from the function but also stored in the property 
        :code:`self.snr`. 
        
        Parameters
        ----------
        tx : :class:`TX<sorts.radar.system.station.TX>` 
            :class:`TX<sorts.radar.system.station.TX>` station observing the pass.
        rx : :class:`RX<sorts.radar.system.station.RX>`
            :class:`RX<sorts.radar.system.station.RX>` station observing the pass.
        diameter : float
            Space object diameter (m).
        parallelization : bool, default=True
            If ``True``, the computations will be run using the ``multiprocessing`` 
            capabilities of python.
        n_processes : int, default=16
            Number of parallel processes during computations (only used when 
            ``parallelization`` is ``True``).

        Returns
        -------
        numpy.ndarrat of float (N,) 
            Vector of optimal SNR values measured by the receiver during the pass.

        Examples
        --------
        Consider a space object (Kepler propagator) passing over the **EISCAT_3D** radar system:

        >>> import sorts
        >>> import matplotlib.pyplot as plt
        >>> radar = sorts.radars.eiscat3d
        >>> Prop_cls = sorts.propagator.Kepler
        >>> Prop_opts = dict(
        ...     settings = dict(
        ...         out_frame='ITRS',
        ...         in_frame='TEME',
        ...     ),
        ... )
        >>> 
        >>> # Object
        >>> space_object = sorts.SpaceObject(
        ...         Prop_cls,
        ...         propagator_options = Prop_opts,
        ...         a = 7000e3, 
        ...         e = 0.0,
        ...         i = 78,
        ...         raan = 86,
        ...         aop = 0, 
        ...         mu0 = 50,
        ...         epoch = 53005.0,
        ...         parameters = dict(
        ...             d = 0.1,
        ...         ),
        ...     )
        >>> print(space_object)
        Space object 1: <Time object: scale='utc' format='mjd' value=53005.0>:
        a    : 7.0000e+06   x : -7.9830e+05
        e    : 0.0000e+00   y : 4.5663e+06
        i    : 7.8000e+01   z : 5.2451e+06
        omega: 0.0000e+00   vx: -1.4093e+03
        Omega: 8.6000e+01   vy: -5.6962e+03
        anom : 5.0000e+01   vz: 4.7445e+03
        Parameters: C_D=2.3, m=1.0, C_R=1.0, d=0.1
    
        The states are propagated over a time period of 1 day:

        >>> t_states = sorts.equidistant_sampling(
        ...     orbit=space_object.state, 
        ...     start_t=0, 
        ...     end_t=3600.0, 
        ...     max_dpos=10e3)
        >>> object_states = space_object.get_state(t_states)

        The passes can then be found in the state array as follows:

        >>> radar_passes = radar.find_passes(t_states, object_states, cache_data=True) 
        >>> radar_passes 
        [[[Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, 
        <sorts.radar.system.station.RX object at 0x7f1f904860c0>] | Rise 0:04:05.161251 (4.4 min) 0:08:30.200441 Fall], 
        [Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object 
        at 0x7f1f903ca340>] | Rise 0:04:05.161251 (4.3 min) 0:08:23.574462 Fall], [Pass Station [<sorts.radar.system.
        station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object at 0x7f1f903eebc0>] | Rise 0:04:05.
        161251 (4.1 min) 0:08:10.322502 Fall]]]

        To compute the SNR over all the :class:`RX<sorts.radar.system.station.RX>` stations, run:
        
        >>> snr = np.ndarray((len(radar.rx),), dtype=object)
        >>> for rxi in range(len(radar.rx)):
        ...     snr[rxi] = radar_passes[0][rxi][0].calculate_snr(radar.tx[0], radar.rx[rxi], 0.1, parallelization=True, n_processes=16)
        
        Finally, the results can be plotted as follow:

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> fmt = ["-r", "-g", "-b"]
        >>> for rxi in range(len(radar.rx)):
        ...     ax.plot(t_states[radar_passes[0][rxi][0].inds], 10*np.log10(snr[rxi]), fmt[rxi], label=f"Rx {rxi}") 
        >>> ax.set_xlabel("$t$ [$s$]")
        >>> ax.set_ylabel("$SNR$ [$-$]")
        >>> ax.grid()
        >>> ax.legend()
        >>> plt.show()

        Yiedlding: 

        .. figure:: ../../../../figures/passes_example_snr.png
        '''
        if self.stations != 2:
            raise Exception("SNR can only be computed for TX-RX pairs")

        # compute the number of states being handled by each process
        n_time_points = len(self.t)
        n_max_points_per_period = int(n_time_points/(n_processes-1))
        if n_max_points_per_period == 0:
            n_processes = n_time_points
            n_max_points_per_period = 1
            n_points_last_period = 1
        else:
            n_points_last_period = n_time_points%((n_processes-1)*n_max_points_per_period)

        # compute range and enu states
        ranges = self.range()
        enus = self.enu

        # intialization of the SNR array
        self.snr = np.empty((n_time_points,), dtype=np.float64)

        def process_function(pid, n_points, return_dict):
            ''' Multiprocessing function. '''
            # compute gain
            # TODO change implementation when pyant has a vectorized compute gain method
            tx_gain = np.zeros((n_points,), dtype=float)
            rx_gain = np.zeros((n_points,), dtype=float)

            for i in range(n_points):
                ti = n_max_points_per_period*pid + i
                tx.beam.point(enus[0][:3,ti])
                rx.beam.point(enus[1][:3,ti])

                tx_gain[i] = tx.beam.gain(enus[0][:3,ti])
                rx_gain[i] = rx.beam.gain(enus[1][:3,ti])

            # compute snr
            return_dict[pid] = hard_target_snr(
                tx_gain,
                rx_gain,
                rx.wavelength,
                tx.power,
                ranges[0][n_max_points_per_period*pid:n_max_points_per_period*pid + n_points],
                ranges[1][n_max_points_per_period*pid:n_max_points_per_period*pid + n_points],
                diameter=diameter,
                bandwidth=tx.coh_int_bandwidth,
                rx_noise_temp=rx.noise_temperature,
            )

        # multiprocessing manager
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        process_subgroup = []

        # set up and run computations
        for pid in range(n_processes):
            if pid < n_processes-1:
                n_points = n_max_points_per_period
            else:
                n_points = n_points_last_period

            if parallelization is True:
                # create new process
                process = multiprocessing.Process(target=process_function, args=(pid, n_points, return_dict)) 
                process_subgroup.append(process)
                process.start()
            else:
                snr = dict()
                process_function(pid, n_points, snr)
                self.snr[n_max_points_per_period*pid:n_max_points_per_period*pid + n_points] = snr[pid]

        # retreive computation results
        if parallelization is True:
            for pid in range(n_processes):
                process_subgroup[pid].join()

            id_start = 0
            for pid in range(n_processes):
                data = return_dict[pid]
                id_end = id_start + len(data)

                self.snr[id_start:id_end] = data
                id_start = id_end

        return self.snr


    @property
    def stations(self):
        ''' Returns the number of stations that can observe the pass. '''
        if self._stations is not None:
            if isinstance(self._stations, list) or isinstance(self._stations, np.ndarray):
                return len(self._stations)
            else:
                return 1
        else:
            return 1


    def start(self):
        ''' Returns the start time of the pass (uses cached value after first call if :code:`self.cache=True`). '''
        if self.cache:
            if self._start is None:
                self._start = self.t.min()
            return self._start
        else:
            return self.t.min()


    def end(self):
        ''' Returns the ending time of the pass (uses cached value after first call if :code:`self.cache=True`). '''
        if self.cache:
            if self._end is None:
                self._end = self.t.max()
            return self._end
        else:
            return self.t.max()


    @staticmethod
    def calculate_range(enu):
        ''' Returns the ``ranges`` associated with a set of *ENU* states.

        The computation of the range relies on the computation of the norm of the 
        ``Pass.enu`` states. Therefore, the results will correspond to the consecutive
        ranges between the space object and the reference station in which frame the
        states are defined. 

        Parameters:
        -----------
        enu : numpy.ndarray (6, N)
            Space object's states in the *ENU* frame of the reference station.

        Returns
        -------
        ranges : numpy.ndarray
            Ranges of the object at each time point.

        Examples
        --------
        Consider a space object (Kepler propagator) passing over the **EISCAT_3D** radar system:

        >>> import sorts
        >>> import matplotlib.pyplot as plt
        >>> radar = sorts.radars.eiscat3d
        >>> Prop_cls = sorts.propagator.Kepler
        >>> Prop_opts = dict(
        ...     settings = dict(
        ...         out_frame='ITRS',
        ...         in_frame='TEME',
        ...     ),
        ... )
        >>> 
        >>> # Object
        >>> space_object = sorts.SpaceObject(
        ...         Prop_cls,
        ...         propagator_options = Prop_opts,
        ...         a = 7000e3, 
        ...         e = 0.0,
        ...         i = 78,
        ...         raan = 86,
        ...         aop = 0, 
        ...         mu0 = 50,
        ...         epoch = 53005.0,
        ...         parameters = dict(
        ...             d = 0.1,
        ...         ),
        ...     )
        >>> print(space_object)
        Space object 1: <Time object: scale='utc' format='mjd' value=53005.0>:
        a    : 7.0000e+06   x : -7.9830e+05
        e    : 0.0000e+00   y : 4.5663e+06
        i    : 7.8000e+01   z : 5.2451e+06
        omega: 0.0000e+00   vx: -1.4093e+03
        Omega: 8.6000e+01   vy: -5.6962e+03
        anom : 5.0000e+01   vz: 4.7445e+03
        Parameters: C_D=2.3, m=1.0, C_R=1.0, d=0.1
    
        The states are propagated over a time period of 1 day:

        >>> t_states = sorts.equidistant_sampling(
        ...     orbit=space_object.state, 
        ...     start_t=0, 
        ...     end_t=3600.0, 
        ...     max_dpos=10e3)
        >>> object_states = space_object.get_state(t_states)

        The passes can then be found in the state array as follows:

        >>> radar_passes = radar.find_passes(t_states, object_states, cache_data=True) 
        >>> radar_passes 
        [[[Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, 
        <sorts.radar.system.station.RX object at 0x7f1f904860c0>] | Rise 0:04:05.161251 (4.4 min) 0:08:30.200441 Fall], 
        [Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object 
        at 0x7f1f903ca340>] | Rise 0:04:05.161251 (4.3 min) 0:08:23.574462 Fall], [Pass Station [<sorts.radar.system.
        station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object at 0x7f1f903eebc0>] | Rise 0:04:05.
        161251 (4.1 min) 0:08:10.322502 Fall]]]

        To compute and plot the range over all the :class:`RX<sorts.radar.system.station.RX>` stations, run:

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> fmt = ["-r", "-g", "-b"]
        >>> for rxi in range(len(radar.rx)):
        ...     enu = radar.rx[rxi].enu(object_states[:, radar_passes[0][rxi][0].inds])
        ...     range_ = radar_passes[0][rxi][0].calculate_range(enu)
        ...     ax.plot(t_states[radar_passes[0][rxi][0].inds], range_, fmt[rxi], label=f"Rx {rxi}")
        >>> ax.set_xlabel("$t$ [$s$]")
        >>> ax.set_ylabel("$R_{rx}$ [$m$]")
        >>> ax.grid()
        >>> ax.legend()
        >>> plt.show()

        Yiedlding: 

        .. figure:: ../../../../figures/passes_example_range.png
        '''
        return np.linalg.norm(enu[:3,:], axis=0)


    @staticmethod
    def calculate_range_rate(enu):
        ''' Returns the ``range-rates`` associated with a set of *ENU* states.
        
        The *range-rate* corresponds to the projection of the velocity vector in the ENU
        frame of reference along the range vector (pointing from the *station* to the *object*).

        Therefore, the range-rate will be negative if the object is getting closer to the station 
        and positive otherwise.

        Parameters:
        -----------
        enu : numpy.ndarray (6, N)
            Space object's states in the *ENU* frame of the reference station.

        Returns
        -------
        range_rates : numpy.ndarray
            Range-rates of the object at each time point.

        Examples
        --------
        Consider a space object (Kepler propagator) passing over the **EISCAT_3D** radar system:

        >>> import sorts
        >>> import matplotlib.pyplot as plt
        >>> radar = sorts.radars.eiscat3d
        >>> Prop_cls = sorts.propagator.Kepler
        >>> Prop_opts = dict(
        ...     settings = dict(
        ...         out_frame='ITRS',
        ...         in_frame='TEME',
        ...     ),
        ... )
        >>> 
        >>> # Object
        >>> space_object = sorts.SpaceObject(
        ...         Prop_cls,
        ...         propagator_options = Prop_opts,
        ...         a = 7000e3, 
        ...         e = 0.0,
        ...         i = 78,
        ...         raan = 86,
        ...         aop = 0, 
        ...         mu0 = 50,
        ...         epoch = 53005.0,
        ...         parameters = dict(
        ...             d = 0.1,
        ...         ),
        ...     )
        >>> print(space_object)
        Space object 1: <Time object: scale='utc' format='mjd' value=53005.0>:
        a    : 7.0000e+06   x : -7.9830e+05
        e    : 0.0000e+00   y : 4.5663e+06
        i    : 7.8000e+01   z : 5.2451e+06
        omega: 0.0000e+00   vx: -1.4093e+03
        Omega: 8.6000e+01   vy: -5.6962e+03
        anom : 5.0000e+01   vz: 4.7445e+03
        Parameters: C_D=2.3, m=1.0, C_R=1.0, d=0.1
    
        The states are propagated over a time period of 1 day:

        >>> t_states = sorts.equidistant_sampling(
        ...     orbit=space_object.state, 
        ...     start_t=0, 
        ...     end_t=3600.0, 
        ...     max_dpos=10e3)
        >>> object_states = space_object.get_state(t_states)

        The passes can then be found in the state array as follows:

        >>> radar_passes = radar.find_passes(t_states, object_states, cache_data=True) 
        >>> radar_passes 
        [[[Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, 
        <sorts.radar.system.station.RX object at 0x7f1f904860c0>] | Rise 0:04:05.161251 (4.4 min) 0:08:30.200441 Fall], 
        [Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object 
        at 0x7f1f903ca340>] | Rise 0:04:05.161251 (4.3 min) 0:08:23.574462 Fall], [Pass Station [<sorts.radar.system.
        station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object at 0x7f1f903eebc0>] | Rise 0:04:05.
        161251 (4.1 min) 0:08:10.322502 Fall]]]

        To compute and plot the *range rate* over all the :class:`RX<sorts.radar.system.station.RX>` stations, run:

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> fmt = ["-r", "-g", "-b"]
        >>> for rxi in range(len(radar.rx)):
        ...     enu = radar.rx[rxi].enu(object_states[:, radar_passes[0][rxi][0].inds])
        ...     range_rate = radar_passes[0][rxi][0].calculate_range_rate(enu)
        ...     ax.plot(t_states[radar_passes[0][rxi][0].inds], range_rate, fmt[rxi], label=f"Rx {rxi}")
        >>> ax.set_xlabel("$t$ [$s$]")
        >>> ax.set_ylabel("$v_r_{rx}$ [$m/s$]")
        >>> ax.grid()
        >>> ax.legend()
        >>> plt.show()

        Yiedlding: 

        .. figure:: ../../../../figures/passes_example_range_rate.png
        '''
        return np.sum(enu[3:,:]*(enu[:3,:]/np.linalg.norm(enu[:3,:], axis=0)), axis=0)


    @staticmethod
    def calculate_zenith_angle(enu, radians=False):
        ''' Returns the ``zenith angle`` of all the states within the :class:`Pass`.
        
        The *zenith angle* corresponds to angle between the local vertical of the reference
        station and the position vector (pointing from the *station* to the *object*).

        Parameters:
        -----------
        enu : numpy.ndarray (6, N)
            Space object's states in the *ENU* frame of the reference station.
        radians : bool, default=False
            If ``True``, the angles will be expressed in rafians. If not, all the
            angles will be in degrees.

        Returns
        -------
        zenith_angle : numpy.ndarray
            Zenith angles of the object at each time point.

        Examples
        --------
        Consider a space object (Kepler propagator) passing over the **EISCAT_3D** radar system:

        >>> import sorts
        >>> import matplotlib.pyplot as plt
        >>> radar = sorts.radars.eiscat3d
        >>> Prop_cls = sorts.propagator.Kepler
        >>> Prop_opts = dict(
        ...     settings = dict(
        ...         out_frame='ITRS',
        ...         in_frame='TEME',
        ...     ),
        ... )
        >>> 
        >>> # Object
        >>> space_object = sorts.SpaceObject(
        ...         Prop_cls,
        ...         propagator_options = Prop_opts,
        ...         a = 7000e3, 
        ...         e = 0.0,
        ...         i = 78,
        ...         raan = 86,
        ...         aop = 0, 
        ...         mu0 = 50,
        ...         epoch = 53005.0,
        ...         parameters = dict(
        ...             d = 0.1,
        ...         ),
        ...     )
        >>> print(space_object)
        Space object 1: <Time object: scale='utc' format='mjd' value=53005.0>:
        a    : 7.0000e+06   x : -7.9830e+05
        e    : 0.0000e+00   y : 4.5663e+06
        i    : 7.8000e+01   z : 5.2451e+06
        omega: 0.0000e+00   vx: -1.4093e+03
        Omega: 8.6000e+01   vy: -5.6962e+03
        anom : 5.0000e+01   vz: 4.7445e+03
        Parameters: C_D=2.3, m=1.0, C_R=1.0, d=0.1
    
        The states are propagated over a time period of 1 day:

        >>> t_states = sorts.equidistant_sampling(
        ...     orbit=space_object.state, 
        ...     start_t=0, 
        ...     end_t=3600.0, 
        ...     max_dpos=10e3)
        >>> object_states = space_object.get_state(t_states)

        The passes can then be found in the state array as follows:

        >>> radar_passes = radar.find_passes(t_states, object_states, cache_data=True) 
        >>> radar_passes 
        [[[Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, 
        <sorts.radar.system.station.RX object at 0x7f1f904860c0>] | Rise 0:04:05.161251 (4.4 min) 0:08:30.200441 Fall], 
        [Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object 
        at 0x7f1f903ca340>] | Rise 0:04:05.161251 (4.3 min) 0:08:23.574462 Fall], [Pass Station [<sorts.radar.system.
        station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object at 0x7f1f903eebc0>] | Rise 0:04:05.
        161251 (4.1 min) 0:08:10.322502 Fall]]]

        To compute and plot the *zenith angle* over all the :class:`RX<sorts.radar.system.station.RX>` stations, run:

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> fmt = ["-r", "-g", "-b"]
        >>> for rxi in range(len(radar.rx)):
        ...     enu = radar.rx[rxi].enu(object_states[:, radar_passes[0][rxi][0].inds])
        ...     zenith_angle = radar_passes[0][rxi][0].calculate_zenith_angle(enu)
        ...     ax.plot(t_states[radar_passes[0][rxi][0].inds], zenith_angle, fmt[rxi], label=f"Rx {rxi}")
        >>> ax.set_xlabel("$t$ [$s$]")
        >>> ax.set_ylabel("$\\theta_{rx}$ [$deg$]")
        >>> ax.grid()
        >>> ax.legend()
        >>> plt.show()

        Yiedlding: 

        .. figure:: ../../../../figures/passes_example_zenith_angle.png
        '''
        return pyant.coordinates.vector_angle(np.array([0,0,1], dtype=np.float64), enu[:3,:], radians=radians)


    def get_range(self):
        ''' Returns the ``ranges`` associated with a set of *ENU* states.

        The computation of the range relies on the computation of the norm of the 
        ``Pass.enu`` states. Therefore, the results will correspond to the consecutive
        ranges between the space object and the reference station in which frame the
        states are defined. 

        Parameters:
        -----------
        None

        Returns
        -------
        ranges : numpy.ndarray
            Ranges of the object at each time point.

        Example
        -------
        See :attr:`Pass.range` to get a example.
        '''
        if self.stations > 1:
            return [Pass.calculate_range(enu) for enu in self.enu]
        else:
            return Pass.calculate_range(self.enu)


    def range(self):
        ''' Returns the ``ranges`` of all the states within the :class:`Pass`.

        The computation of the range relies on the computation of the norm of the 
        ``Pass.enu`` states. Therefore, the results will correspond to the consecutive
        ranges between the space object and the reference station in which frame the
        states are defined. 

        .. note::
            if :attr:`Pass.cache` is True, *range* will only be computed once on the
            first call of the function. Any future call will return the cached values.

        Parameters:
        -----------
        None

        Returns
        -------
        ranges : numpy.ndarray
            Ranges of the object at each time point.

        Examples
        --------
        Consider a space object (Kepler propagator) passing over the **EISCAT_3D** radar system:

        >>> import sorts
        >>> import matplotlib.pyplot as plt
        >>> radar = sorts.radars.eiscat3d
        >>> Prop_cls = sorts.propagator.Kepler
        >>> Prop_opts = dict(
        ...     settings = dict(
        ...         out_frame='ITRS',
        ...         in_frame='TEME',
        ...     ),
        ... )
        >>> 
        >>> # Object
        >>> space_object = sorts.SpaceObject(
        ...         Prop_cls,
        ...         propagator_options = Prop_opts,
        ...         a = 7000e3, 
        ...         e = 0.0,
        ...         i = 78,
        ...         raan = 86,
        ...         aop = 0, 
        ...         mu0 = 50,
        ...         epoch = 53005.0,
        ...         parameters = dict(
        ...             d = 0.1,
        ...         ),
        ...     )
        >>> print(space_object)
        Space object 1: <Time object: scale='utc' format='mjd' value=53005.0>:
        a    : 7.0000e+06   x : -7.9830e+05
        e    : 0.0000e+00   y : 4.5663e+06
        i    : 7.8000e+01   z : 5.2451e+06
        omega: 0.0000e+00   vx: -1.4093e+03
        Omega: 8.6000e+01   vy: -5.6962e+03
        anom : 5.0000e+01   vz: 4.7445e+03
        Parameters: C_D=2.3, m=1.0, C_R=1.0, d=0.1
    
        The states are propagated over a time period of 1 day:

        >>> t_states = sorts.equidistant_sampling(
        ...     orbit=space_object.state, 
        ...     start_t=0, 
        ...     end_t=3600.0, 
        ...     max_dpos=10e3)
        >>> object_states = space_object.get_state(t_states)

        The passes can then be found in the state array as follows:

        >>> radar_passes = radar.find_passes(t_states, object_states, cache_data=True) 
        >>> radar_passes 
        [[[Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, 
        <sorts.radar.system.station.RX object at 0x7f1f904860c0>] | Rise 0:04:05.161251 (4.4 min) 0:08:30.200441 Fall], 
        [Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object 
        at 0x7f1f903ca340>] | Rise 0:04:05.161251 (4.3 min) 0:08:23.574462 Fall], [Pass Station [<sorts.radar.system.
        station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object at 0x7f1f903eebc0>] | Rise 0:04:05.
        161251 (4.1 min) 0:08:10.322502 Fall]]]

        To compute and plot the range over all the :class:`RX<sorts.radar.system.station.RX>` stations, run:

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> fmt = ["-r", "-g", "-b"]
        >>> for rxi in range(len(radar.rx)):
        ...     ax.plot(t_states[radar_passes[0][rxi][0].inds], radar_passes[0][rxi][0].range()[1], fmt[rxi], label=f"Rx {rxi}")
        >>> ax.set_xlabel("$t$ [$s$]")
        >>> ax.set_ylabel("$R_{rx}$ [$m$]")
        >>> ax.grid()
        >>> ax.legend()
        >>> plt.show()

        Yiedlding: 

        .. figure:: ../../../../figures/passes_example_range.png
        '''
        if self.cache:
            if self._range is None:
                self._range = self.get_range()
            return self._range
        else:
            return self.get_range()


    def get_range_rate(self):
        ''' Computes the ``range-rates`` of all the states within the :class:`Pass`.
        
        The *range-rate* corresponds to the projection of the velocity vector in the ENU
        frame of reference along the range vector (pointing from the *station* to the *object*).

        Therefore, the range-rate will be negative if the object is getting closer to the station 
        and positive otherwise.

        Parameters:
        -----------
        enu : numpy.ndarray (6, N)
            Space object's states in the *ENU* frame of the reference station.

        Returns
        -------
        range_rates : numpy.ndarray
            Range-rates of the object at each time point.

        Example
        -------
        See :attr:`Pass.range_rate` to get a example.
        '''
        if self.stations > 1:
            return [Pass.calculate_range_rate(enu) for enu in self.enu]
        else:
            return Pass.calculate_range_rate(self.enu)


    def range_rate(self):
        ''' Returns the ``range-rates`` of all the states within the :class:`Pass`.
        
        The *range-rate* corresponds to the projection of the velocity vector in the ENU
        frame of reference along the range vector (pointing from the *station* to the *object*).

        Therefore, the range-rate will be negative if the object is getting closer to the station 
        and positive otherwise. 

        .. note::
            if :attr:`Pass.cache` is True, *range-rates* will only be computed once on the
            first call of the function. Any future call will return the cached values.

        Parameters:
        -----------
        enu : numpy.ndarray (6, N)
            Space object's states in the *ENU* frame of the reference station.

        Returns
        -------
        range_rates : numpy.ndarray
            Range-rates of the object at each time point.

        Examples
        --------
        Consider a space object (Kepler propagator) passing over the **EISCAT_3D** radar system:

        >>> import sorts
        >>> import matplotlib.pyplot as plt
        >>> radar = sorts.radars.eiscat3d
        >>> Prop_cls = sorts.propagator.Kepler
        >>> Prop_opts = dict(
        ...     settings = dict(
        ...         out_frame='ITRS',
        ...         in_frame='TEME',
        ...     ),
        ... )
        >>> 
        >>> # Object
        >>> space_object = sorts.SpaceObject(
        ...         Prop_cls,
        ...         propagator_options = Prop_opts,
        ...         a = 7000e3, 
        ...         e = 0.0,
        ...         i = 78,
        ...         raan = 86,
        ...         aop = 0, 
        ...         mu0 = 50,
        ...         epoch = 53005.0,
        ...         parameters = dict(
        ...             d = 0.1,
        ...         ),
        ...     )
        >>> print(space_object)
        Space object 1: <Time object: scale='utc' format='mjd' value=53005.0>:
        a    : 7.0000e+06   x : -7.9830e+05
        e    : 0.0000e+00   y : 4.5663e+06
        i    : 7.8000e+01   z : 5.2451e+06
        omega: 0.0000e+00   vx: -1.4093e+03
        Omega: 8.6000e+01   vy: -5.6962e+03
        anom : 5.0000e+01   vz: 4.7445e+03
        Parameters: C_D=2.3, m=1.0, C_R=1.0, d=0.1
    
        The states are propagated over a time period of 1 day:

        >>> t_states = sorts.equidistant_sampling(
        ...     orbit=space_object.state, 
        ...     start_t=0, 
        ...     end_t=3600.0, 
        ...     max_dpos=10e3)
        >>> object_states = space_object.get_state(t_states)

        The passes can then be found in the state array as follows:

        >>> radar_passes = radar.find_passes(t_states, object_states, cache_data=True) 
        >>> radar_passes 
        [[[Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, 
        <sorts.radar.system.station.RX object at 0x7f1f904860c0>] | Rise 0:04:05.161251 (4.4 min) 0:08:30.200441 Fall], 
        [Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object 
        at 0x7f1f903ca340>] | Rise 0:04:05.161251 (4.3 min) 0:08:23.574462 Fall], [Pass Station [<sorts.radar.system.
        station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object at 0x7f1f903eebc0>] | Rise 0:04:05.
        161251 (4.1 min) 0:08:10.322502 Fall]]]

        To compute and plot the range rate over all the :class:`RX<sorts.radar.system.station.RX>` stations, run:

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> fmt = ["-r", "-g", "-b"]
        >>> for rxi in range(len(radar.rx)):
        ...     ax.plot(t_states[radar_passes[0][rxi][0].inds], radar_passes[0][rxi][0].range_rate()[1], fmt[rxi], label=f"Rx {rxi}")
        >>> ax.set_xlabel("$t$ [$s$]")
        >>> ax.set_ylabel("$v_r_{rx}$ [$m/s$]")
        >>> ax.grid()
        >>> ax.legend()
        >>> plt.show()

        Yiedlding: 

        .. figure:: ../../../../figures/passes_example_range_rate.png
        '''
        if self.cache:
            if self._range_rate is None:
                self._range_rate = self.get_range_rate()
            return self._range_rate
        else:
            return self.get_range_rate()


    def get_zenith_angle(self, radians=False):
        ''' Computes the ``zenith angle`` of all the states within the :class:`Pass`.
        
        The *zenith angle* corresponds to angle between the local vertical of the reference
        station and the position vector (pointing from the *station* to the *object*).

        Parameters:
        -----------
        None

        Returns
        -------
        zenith_angle : numpy.ndarray
            Range-rates of the object at each time point.

        Example
        -------
        See :attr:`Pass.zenith_angle` to get a example.
        '''
        if self.stations > 1:
            return [
                Pass.calculate_zenith_angle(enu, radians=radians)
                for enu in self.enu
            ]
        else:
            return Pass.calculate_zenith_angle(self.enu, radians=radians)


    def zenith_angle(self, radians=False):
        ''' Returns the ``zenith angle`` of all the states within the :class:`Pass`.
        
        The *zenith angle* corresponds to angle between the local vertical of the reference
        station and the position vector (pointing from the *station* to the *object*).

        .. note::
            if :attr:`Pass.cache` is True, *zenith angle* will only be computed once on the
            first call of the function. Any future call will return the cached values.


        Parameters:
        -----------
        None

        Returns
        -------
        zenith_angle : numpy.ndarray
            zenith angle of the object at each time point.

        Examples
        --------
        Consider a space object (Kepler propagator) passing over the **EISCAT_3D** radar system:

        >>> import sorts
        >>> import matplotlib.pyplot as plt
        >>> radar = sorts.radars.eiscat3d
        >>> Prop_cls = sorts.propagator.Kepler
        >>> Prop_opts = dict(
        ...     settings = dict(
        ...         out_frame='ITRS',
        ...         in_frame='TEME',
        ...     ),
        ... )
        >>> space_object = sorts.SpaceObject(
        ...         Prop_cls,
        ...         propagator_options = Prop_opts,
        ...         a = 7000e3, 
        ...         e = 0.0,
        ...         i = 78,
        ...         raan = 86,
        ...         aop = 0, 
        ...         mu0 = 50,
        ...         epoch = 53005.0,
        ...         parameters = dict(
        ...             d = 0.1,
        ...         ),
        ...     )
        >>> print(space_object)
        Space object 1: <Time object: scale='utc' format='mjd' value=53005.0>:
        a    : 7.0000e+06   x : -7.9830e+05
        e    : 0.0000e+00   y : 4.5663e+06
        i    : 7.8000e+01   z : 5.2451e+06
        omega: 0.0000e+00   vx: -1.4093e+03
        Omega: 8.6000e+01   vy: -5.6962e+03
        anom : 5.0000e+01   vz: 4.7445e+03
        Parameters: C_D=2.3, m=1.0, C_R=1.0, d=0.1
    
        The states are propagated over a time period of 1 day:

        >>> t_states = sorts.equidistant_sampling(
        ...     orbit=space_object.state, 
        ...     start_t=0, 
        ...     end_t=3600.0, 
        ...     max_dpos=10e3)
        >>> object_states = space_object.get_state(t_states)

        The passes can then be found in the state array as follows:

        >>> radar_passes = radar.find_passes(t_states, object_states, cache_data=True) 
        >>> radar_passes 
        [[[Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, 
        <sorts.radar.system.station.RX object at 0x7f1f904860c0>] | Rise 0:04:05.161251 (4.4 min) 0:08:30.200441 Fall], 
        [Pass Station [<sorts.radar.system.station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object 
        at 0x7f1f903ca340>] | Rise 0:04:05.161251 (4.3 min) 0:08:23.574462 Fall], [Pass Station [<sorts.radar.system.
        station.TX object at 0x7f1f903e1ac0>, <sorts.radar.system.station.RX object at 0x7f1f903eebc0>] | Rise 0:04:05.
        161251 (4.1 min) 0:08:10.322502 Fall]]]

        To compute and plot the zenith angle rate over all the :class:`RX<sorts.radar.system.station.RX>` stations, run:

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> fmt = ["-r", "-g", "-b"]
        >>> for rxi in range(len(radar.rx)):
        ...     ax.plot(t_states[radar_passes[0][rxi][0].inds], radar_passes[0][rxi][0].zenith_angle()[1], fmt[rxi], label=f"Rx {rxi}")
        >>> ax.set_xlabel("$t$ [$s$]")
        >>> ax.set_ylabel("$\\theta_{rx}$ [$deg$]")
        >>> ax.grid()
        >>> ax.legend()
        >>> plt.show()

        Yiedlding: 

        .. figure:: ../../../../figures/passes_example_zenith_angle.png
        '''
        if self.cache:
            if self._zenith_angle is None:
                self._zenith_angle = self.get_zenith_angle(radians=radians)
            return self._zenith_angle
        else:
            return self.get_zenith_angle(radians=radians)



def equidistant_sampling(orbit, start_t, end_t, max_dpos=1e3, eccentricity_tol=0.3):
    '''Finds the temporal sampling of an orbit which is sufficient to achieve a maximum spatial separation.
    
    Assuming an elleptic orbit, the ``equidistant_sampling`` function uses Keplerian propagation to find 
    an equidistant sampling time array. In the case where eccentricity is small, periapsis speed 
    and uniform sampling in time are used to generate the time array.

    .. note::
        The current implementation does not take orbital perturbation patterns into account.
    
    Parameters
    ----------
    orbit : :class:`pyorb.Orbit` 
        Orbit which is being sampled.
    start_t : float 
        Start time (in seconds).
    end_t : float 
        End time (in seconds).
    max_dpos : float 
        Maximum separation between evaluation points (in meters).
    eccentricity_tol : float
        Minimum eccentricity below which the orbit is approximated as a circle and temporal samples are uniform in time.
    
    Returns
    -------
    numpy.ndarray : 
        Vector of sample times in seconds.

    Examples
    --------
    The :func:`equidistant_sampling` function is generally used to generate a time array over which 
    the states of a space object are to be propagated. Consider the following space object:

    >>> import sorts
    >>> Prop_cls = sorts.propagator.Kepler
    >>> Prop_opts = dict(
    ...     settings = dict(
    ...         out_frame='ITRS',
    ...         in_frame='TEME',
    ...     ),
    ... )
    >>> space_object = sorts.SpaceObject(
    ...         Prop_cls,
    ...         propagator_options = Prop_opts,
    ...         a = 7000e3, 
    ...         e = 0.0,
    ...         i = 78,
    ...         raan = 86,
    ...         aop = 0, 
    ...         mu0 = 50,
    ...         epoch = 53005.0,
    ...         parameters = dict(
    ...             d = 0.1,
    ...         ),
    ...     )
    >>> print(space_object)
    Space object 1: <Time object: scale='utc' format='mjd' value=53005.0>:
    a    : 7.0000e+06   x : -7.9830e+05
    e    : 0.0000e+00   y : 4.5663e+06
    i    : 7.8000e+01   z : 5.2451e+06
    omega: 0.0000e+00   vx: -1.4093e+03
    Omega: 8.6000e+01   vy: -5.6962e+03
    anom : 5.0000e+01   vz: 4.7445e+03
    Parameters: C_D=2.3, m=1.0, C_R=1.0, d=0.1

    Given the initial state of the space object (which is described by a :class:`pyorb.Orbit` object),
    it is possible to generate an equidistant sampling time array to propagate the states of the 
    space object over 1 day:

    >>> t_states = sorts.equidistant_sampling(
    ...     orbit=space_object.state, 
    ...     start_t=0, 
    ...     end_t=3600.0, 
    ...     max_dpos=10e3)
    >>> t_states
    array([0.00000000e+00, 1.32519595e+00, 2.65039190e+00, ...,
       3.59658181e+03, 3.59790701e+03, 3.59923220e+03])
    '''
    if len(orbit) > 1:
        raise ValueError(f'Cannot use vectorized orbits: len(orbit) = {len(orbit)}')

    if orbit.e <= eccentricity_tol:
        r = pyorb.elliptic_radius(0.0, orbit.a, orbit.e, degrees=False)
        v = pyorb.orbital_speed(r, orbit.a, orbit.G*(orbit.M0 + orbit.m))[0]
        return np.arange(start_t, end_t, max_dpos/v)

    tmp_orb = orbit.copy()
    tmp_orb.auto_update = False

    tmp_orb.propagate(start_t)
    period = tmp_orb.period

    t_curr = start_t
    t = [t_curr]
    t_repeat = None

    while t_curr < end_t:
        if t_curr - start_t > period:
            if t_repeat is None:
                t_repeat = len(t)

            dt = t[-t_repeat+1] - t[-t_repeat]
            t_curr += dt
        else:
            v = tmp_orb.speed[0]
            dt = max_dpos/v
            t_curr += dt

            tmp_orb.propagate(dt)

        t.append(t_curr)

    return np.array(t, dtype=np.float64)


def find_passes(t, states, station, cache_data=True):
    '''Finds all passes within the FOV of a radar :class:`sorts.Station<sorts.radar.system.station.Station>`
    given a set of space object states.

    In this implementation, a state is considered to be inside the field-of-view (FOV) of a station if the
    elevation of the target is **greater than the minimal elevation of the station beam**.  

    Parameters
    ----------    
    t : numpy.ndarray (N,) 
        Vector of times in seconds to use as a base to find passes.
    states : numpy.ndarray (6, N) 
        ECEF states of the object to find passes for.
    station : :class:`sorts.Station<sorts.radar.system.station.Station>`
        Radar station which defines the FOV.

    Returns
    -------
    passes : list of :class:`Pass`
        List of passes over the radar station.

    Examples
    --------
    Consider a space object passing over the EISCAT_3D radar system:

    >>> import sorts
    >>> Prop_cls = sorts.propagator.Kepler
    >>> Prop_opts = dict(
    ...     settings = dict(
    ...         out_frame='ITRS',
    ...         in_frame='TEME',
    ...     ),
    ... )
    >>> space_object = sorts.SpaceObject(
    ...         Prop_cls,
    ...         propagator_options = Prop_opts,
    ...         a = 7000e3, 
    ...         e = 0.0,
    ...         i = 78,
    ...         raan = 86,
    ...         aop = 0, 
    ...         mu0 = 50,
    ...         epoch = 53005.0,
    ...         parameters = dict(
    ...             d = 0.1,
    ...         ),
    ...     )
    >>> print(space_object)
    Space object 1: <Time object: scale='utc' format='mjd' value=53005.0>:
    a    : 7.0000e+06   x : -7.9830e+05
    e    : 0.0000e+00   y : 4.5663e+06
    i    : 7.8000e+01   z : 5.2451e+06
    omega: 0.0000e+00   vx: -1.4093e+03
    Omega: 8.6000e+01   vy: -5.6962e+03
    anom : 5.0000e+01   vz: 4.7445e+03
    Parameters: C_D=2.3, m=1.0, C_R=1.0, d=0.1

    If the states are propagated over a time period of 1 day:

    >>> t_states = sorts.equidistant_sampling(
    ...     orbit=space_object.state, 
    ...     start_t=0, 
    ...     end_t=3600.0, 
    ...     max_dpos=10e3)
    >>> space_object.get_state(t_states)

    we can identify the passes over the :class:`TX<sorts.radar.system.station.TX>`
    station of the EISCAT_3D radar by calling the function :func:`find_passes`:

    >>> sorts.passes.find_passes(t_states, object_states, radar.tx[0], cache_data=True)
    [Pass Rise 0:04:05.161251 (4.4 min) 0:08:30.200441 Fall]
    '''
    passes = []

    # convert states to enu
    enu = station.enu(states[:3,:])
    
    # mask all states outside the FOV
    check = station.field_of_view(states)
    inds = np.where(check)[0]

    if len(inds) == 0:
        return passes

    # get indices which mark the start and end of a :class:`Pass`
    dind = np.diff(inds)
    splits = np.where(dind > 1)[0]
    splits = np.insert(splits, 0, -1)
    splits = np.insert(splits, len(splits), len(inds)-1)

    # split passes
    splits += 1
    for si in range(len(splits)-1):
        ps_inds = inds[splits[si]:splits[si+1]]

        if cache_data:
            ps = Pass(
                t=t[ps_inds], 
                enu=enu[:, ps_inds], 
                inds=ps_inds, 
                cache=True,
            )
        else:
            ps = Pass(
                t=None, 
                enu=None, 
                inds=ps_inds, 
                cache=True,
            )
            ps._start = t[ps_inds].min()
            ps._end = t[ps_inds].max()

        passes.append(ps)

    return passes


def find_simultaneous_passes(t, states, stations, cache_data=True):
    ''' This function finds all passes (set of consecutive states) which lie *simultaneously*
    within the Field of View of a set of Radar stations.

    Parameters
    ----------
    t : numpy.ndarray of floats (N,)
        Vector of times in seconds to use as a base to find passes.
    states : numpy.ndarray of floats (6, N) 
        ECEF states of the object to find passes for.
    stations : list of :class:`sorts.Station<sorts.radar.system.station.Station>` 
        List of radar stations which define the field of view.
    cache_data : bool, default=True
        If ``True``, the states and time arrays will be stored within the passes.

        .. note:: 
            Enabling this option will increase RAM usage.

    Returns
    -------
    passes : list of :class:`Pass`
        list of passes which are associated with the specified field of view (and object).
    
    Examples
    --------
    Consider a space object passing over the EISCAT_3D radar system:

    >>> import sorts
    >>> Prop_cls = sorts.propagator.Kepler
    >>> Prop_opts = dict(
    ...     settings = dict(
    ...         out_frame='ITRS',
    ...         in_frame='TEME',
    ...     ),
    ... )
    >>> space_object = sorts.SpaceObject(
    ...         Prop_cls,
    ...         propagator_options = Prop_opts,
    ...         a = 7000e3, 
    ...         e = 0.0,
    ...         i = 78,
    ...         raan = 86,
    ...         aop = 0, 
    ...         mu0 = 50,
    ...         epoch = 53005.0,
    ...         parameters = dict(
    ...             d = 0.1,
    ...         ),
    ...     )
    >>> print(space_object)
    Space object 1: <Time object: scale='utc' format='mjd' value=53005.0>:
    a    : 7.0000e+06   x : -7.9830e+05
    e    : 0.0000e+00   y : 4.5663e+06
    i    : 7.8000e+01   z : 5.2451e+06
    omega: 0.0000e+00   vx: -1.4093e+03
    Omega: 8.6000e+01   vy: -5.6962e+03
    anom : 5.0000e+01   vz: 4.7445e+03
    Parameters: C_D=2.3, m=1.0, C_R=1.0, d=0.1

    If the states are propagated over a time period of 1 day:

    >>> t_states = sorts.equidistant_sampling(
    ...     orbit=space_object.state, 
    ...     start_t=0, 
    ...     end_t=3600.0, 
    ...     max_dpos=10e3)
    >>> space_object.get_state(t_states)

    we can identify the passes over the all the stations of the EISCAT_3D radar by 
    calling the function :func:`find_simultaneous_passes`:

    >>> sorts.passes.find_simultaneous_passes(t_states, object_states, radar.tx+radar.rx, cache_data=True)
    [Pass Station [<sorts.radar.system.station.TX object at 0x7ff9af6ccac0>, 
    <sorts.radar.system.station.RX object at 0x7ff99523ba40>, 
    <sorts.radar.system.station.RX object at 0x7ff9951db2c0>, 
    <sorts.radar.system.station.RX object at 0x7ff9951db4c0>] | 
    Rise 0:04:05.161251 (4.1 min) 0:08:10.322502 Fall]
    '''
    passes = []
    enus = []
    check = np.full((len(t),), True, dtype=np.bool)
    
    # check if states are within the fov of each station
    for station in stations:
        if cache_data:  # convert states to enu in the reference frame of the station
            enus.append(station.enu(states))

        check_st = station.field_of_view(states)
        check = np.logical_and(check, check_st)

    # find all time indices which are within the fov of all stations
    inds = np.where(check)[0]
    if len(inds) == 0:
        return passes

    # compute splitting indices 
    split_ids = np.where(np.diff(inds) > 1)[0]+1
    split_ids = np.insert(split_ids, 0, 0)
    split_ids = np.insert(split_ids, len(split_ids), len(inds))

    # create pass objects for each identifed pass
    for pi in range(len(split_ids)-1):
        if split_ids[pi] == split_ids[pi+1]:
            continue

        # pass data
        if cache_data:
            ps = Pass(
                t=t[inds[split_ids[pi]:split_ids[pi+1]]], 
                enu=[enu[:, inds[split_ids[pi]:split_ids[pi+1]]] for enu in enus], 
                inds=inds[split_ids[pi]:split_ids[pi+1]], 
                cache=True,
                stations=stations,
            )
        else:
            ps = Pass(
                t=None, 
                enu=None, 
                inds=inds[split_ids[pi]:split_ids[pi+1]], 
                cache=True,
                stations=stations,
            )
            ps._start = t[inds[split_ids[pi]]]
            ps._end = t[inds[split_ids[pi+1]-1]]

        passes.append(ps)

    return passes


def group_passes(passes):
    '''
    Takes a list of passes structured as ``passes[tx][rx][pass]`` and find all simultaneous passes and groups them 
    according to ``[tx]``, resulting in a ``passes[tx][pass][rx]`` structure.

    Parameters
    ----------
    passes : list of :class:`Pass`
        List of passes structured as ``passes[tx][rx][pass]``.

    Returns
    -------
    passes : list of :class:`Pass`
        List of passes structured as ``passes[tx][pass][rx]``.
    '''

    def overlap(ps1, ps2):
        return ps1.start() <= ps2.end() and ps2.start() <= ps1.end()

    grouped_passes = []
    for tx_passes in passes:
        grouped_passes.append([])

        #first flatten
        flat_passes = [x for rx_passes in tx_passes for x in rx_passes]

        if len(flat_passes) > 0:
            grouped_passes[-1].append([flat_passes[0]])
        else:
            continue

        for x in range(1,len(flat_passes)):
            for y in range(len(grouped_passes[-1])):
                member = False
                for gps in grouped_passes[-1][y]:
                    if overlap(gps, flat_passes[x]):
                        member = True
                        break

                if member:
                    member_id = y
                    break

            if member:
                grouped_passes[-1][member_id].append(flat_passes[x])
            else:
                grouped_passes[-1].append([flat_passes[x]])

    return grouped_passes

