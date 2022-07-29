#!/usr/bin/env python

''' 

'''
import copy
import ctypes
import numpy as np

from .. import passes
from .. import measurements
from ...common import interpolation
from ...transformations import frames
from . import station
from sorts import clibsorts

class Radar(object):
    '''Encapsulates a Radar system.
        
    The Radar class is used to define a Radar system (i.e. a network of :class:`TX<sorts.radar.system.station.TX>`
    /:class:`RX<sorts.radar.system.station.RX>` stations). It provides the used with a set of functionalities to easily 
    access and control the properties of the stations.

    Parameters
    ----------
    tx : :class:`TX<sorts.radar.system.station.TX>`/numpy.ndarray of :class:`TX<sorts.radar.system.station.TX>` radar stations
        Set of Transmitting radar stations.

    rx : :class:`RX<sorts.radar.system.station.RX>`/numpy.ndarray of :class:`RX<sorts.radar.system.station.RX>` radar stations
        Set of Receiving radar stations.

    min_SNRdb : float, optional 
        Minimal SNR detection level (in :math:`dB`). 
        If SNR computations are over this threshold, the measurement will be considered as a positive object detection.
        ``min_SNRdb`` must be a positive float.
        Default : 10.0 dB

    logger : logging.Logger instance, optional
        Logger instance used to log the computation status within class methods
        Default : None

    profiler : :class:`Profiler<sorts.common.profiling.Profiler>` instance
        :class:`Profiler<sorts.common.profiling.Profiler>` instance used to get computation time and performances wint class methods
        Default : None

    joint_stations : list of tuple
        list of stations indices (:class:`TX<sorts.radar.system.station.TX>`/:class:`RX<sorts.radar.system.station.RX>`) which 
        phisically correspond to the same antenna. This is used when one antenna is use both as a receiver and transmitter.
        Default : None

    measurement : measurements.Measurement instance
        Measurement class instance used to simulate SNR/range/range rate measurements. See :class:`Measurements<sorts.radar.measurements.base.Measurement>` for
        more information.
        Default : measurements.Measurement
    
    See Also
    --------
        sorts.radar.controllers : defines a set of standard radar controllers to generate custom radar control sequences.
        sorts.radar.radar_controls.RadarControls : encapsulates a radar control sequence. 

    '''

    TIME_VARIABLES = [
        "t",
        "t_slice",
    ]
    ''' Parametrization of a radar control time slice.

    A radar control time slice is characterized by its start time ``t`` and its duration ``t_slice``. The array of starting points ``t`` of time slices is used 
    throughout the library to date each control, but one has to keep in mind that the control is active from ``t`` to ``t + t_slice`` seconds.
    
    .. note::
        Multiple controls can coexist within a single control time slice, this is for example the case of the scanning controls which can scan simultaneously multiple 
        points. 

    The time slice parameters are set by the controllers, which associate a set of controls for each radar station (such as pulse length, pointing direction, ...)to each time slice.
    
    See Also
    --------
        sorts.radar.controllers : defines a set of standard radar controllers to generate custom radar control sequences.
        sorts.radar.radar_controls.RadarControls : encapsulates a radar control sequence. 
    '''

    def __init__(
        self, 
        tx, 
        rx, 
        min_SNRdb=10.0, 
        logger=None, 
        profiler=None, 
        joint_stations=None, 
        measurement=measurements.Measurement_new
    ): 
        '''
        Default class constructor.
        '''

        self.tx = tx
        ''' List of transmitting stations in the network. '''

        self.rx = rx
        ''' List of receiving stations in the network. '''

        self.logger = logger
        ''' logging.Logger instance used keep track of the computation status. '''

        self.profiler = profiler
        ''' :class:`Profiling<sorts.common.profiling.Profiler> instance used to evaluate the computing performances of the library. '''

        self.min_SNRdb = min_SNRdb
        ''' .. _minsnrdb

        logging.Logger instance used keep track of the computation status '''

        self.measurement_class = measurement(logger=logger, profiler=profiler)
        ''' :class:`Measurement<sorts.radar.measurements.base.Measurement>` instance used simulate radar measurements (SNR, Range, ...). '''

        self.states = None
        ''' dict containing the radar states if cached. '''

        self.joint_stations = []
        ''' list of stations indices (:class:`TX<sorts.radar.system.station.TX>`/:class:`RX<sorts.radar.system.station.RX>`) which 
            phisically correspond to the same antenna. '''
        
        if joint_stations is not None:
            self.joint_stations = joint_stations


    def copy(self):
        '''Creates a deep copy of the radar system.

        This method is used to easily perform a deepcopy (a deepcopy performes a copy of the object itself instead of returning a reference
        to the object to copy) of all the attributes of a radar instance.

        Examples
        --------
        Creating a radar object by using the ``=`` symbole will copy the `reference` of the initial object (i.e. its memory adress)

        >>> import sorts
        >>> radar = sorts.radars.eiscat3d
        >>> radar
        <sorts.radar.system.radar.Radar object at 0x7f593a4b6df0>
        >>> radar_ref_copy = radar
        >>> radar_ref_copy
        <sorts.radar.system.radar.Radar object at 0x7f593a4b6df0>
        
        This means that the new 'copy' will be linked to the object of reference (which in our case is ``radar``).
        If we want to copy every attribute of ``radar`` in a new object with a different memory adress, we instead
        need to call :

        >>> radar_copy = radar.copy()
        >>> radar_copy
        <sorts.radar.system.radar.Radar object at 0x7f591ae978e0>
        
        Note that the newly created radar instance does not share the memory with the first object.

        See Also
        --------
            copy.copy : helper module to define custom copy() methods for complex objects.
        '''
        ret = Radar(
            tx = [],
            rx = [],
            min_SNRdb = copy.deepcopy(self.min_SNRdb),
            joint_stations = copy.deepcopy(self.joint_stations),
        )
        for tx in self.tx:
            ret.tx.append(tx.copy())
        for rx in self.rx:
            ret.rx.append(rx.copy())
        return ret


    def get_station_id(self, station):
        ''' Gets the station index inside the network.

        The ``get_station_id`` method gets the index of a given radar station inside the ``tx``/``rx`` lists containing
        all the stations of the network.

        Parameters
        ----------
        station : sorts.radar.system.station.Station 
            Station instance within the radar network which index we want to know.

        Returns
        -------
        int 
            Station id inside the network.

        Raises
        ------
        Exception :
            if the station instance is not part of the radar network.

        .. note::
            The index of the station corresponds to the index of the station inside the ``tx`` or ``rx`` arrays, which 
            means that we can't know the type of station from the value returned by the function.

            Therefore, one needs to keep track of the station type when calling the ``get_station_id`` method.
    
        Examples
        --------
        For the sake of the example, let's first create a station instance which will correspond to the station of type ``rx``
        of index 1:

        >>> import sorts
        >>> radar = sorts.radars.eiscat3d
        >>> station = radar.rx[1]
        >>> station
        <sorts.radar.system.station.RX object at 0x7f591ae998c0>

        If we want to know the index of the station, one simply needs to call :

        >>> radar.get_station_id(station)
        1
        >>> radar.rx[radar.get_station_id(station)]
        <sorts.radar.system.station.RX object at 0x7f591ae998c0>
        '''
        station_id = None

        # get type and id of station
        for station_type in ("rx", "tx"):
            stations = getattr(self, station_type)
            for sid, station_ in enumerate(stations):
                if station_ == station:
                    station_id = sid
                    break
        
        if station_id is None: raise Exception(f"could not find station {station} in radar {self}")

        return station_id


    def set_beam(self, beam):
        '''Sets the radiation pattern for transmitters and receivers.
        
        This method sets the radiation pattern of the antennas associated to all :class:`TX<sorts.radar.system.station.TX>` and 
        :class:`RX<sorts.radar.system.station.RX>` within the radar network.
    
        Parameters
        ----------
        beam : pyant.Beam
            Radiation pattern to set for all the stations within the radar system.

        See Also
        --------
            pyant.beam : defines the radiation pattern of radar antennas.
        '''
        self.set_tx_beam(beam)
        self.set_rx_beam(beam)


    def set_tx_beam(self, beam):
        '''Sets the radiation pattern for transmitters.
        
        This method sets the radiation pattern of the antennas associated to all :class:`TX<sorts.radar.system.station.TX>` 
        within the radar network.
    
        Parameters
        ----------
        beam : pyant.Beam
            Radiation pattern to set for all the `Transmitting` stations within the radar system.

        See Also
        --------
            pyant.beam : defines the radiation pattern of radar antennas.
        '''
        for tx in self.tx:
            tx.beam = beam.copy()


    def set_rx_beam(self, beam):
        '''Sets the radiation pattern for receivers.
        
        This method sets the radiation pattern of the antennas associated to all :class:`TX<sorts.radar.system.station.TX>` 
        within the radar network.
    
        Parameters
        ----------
        beam : pyant.Beam
            Radiation pattern to set for all the `Receiving` stations within the radar system.

        See Also
        --------
            pyant.beam : defines the radiation pattern of radar antennas.
        '''
        for rx in self.rx:
            rx.beam = beam.copy()


    def field_of_view(self, ecef):
        ''' Returns wether the given states are within the radar FOV.

        This method determines if the ecef states are within the FOV of all stations 
        of the radar system.

        Parameters
        ----------
        ecef : numpy.ndarray (6, N)
            States :math:`\vec{x} = [x, y, z, v_x, v_y, v_z]^T` in the ecef frame.            

        Returns
        -------
        numpy.ndarray of bool (N,)
            For each ecef state, the value will be True if the state is inside the FOV of all stations, False if not. 

        Examples
        --------
        To simplify this example, we will assume that we have created an array of space object states ``states`` in the ecef
        frame of 4 elements. Since all the stations within the radar have a FOV of 120 degrees (min elevation of 30 degrees),
        then we can create the directions :

        >>> dir_1 = np.array([0.33087577, 0.12248111, 0.93569204, 0, 0, 0]) # 90 deg of elevation        
        >>> dir_2 = np.array([0.18286604, 0.28205454, 0.94180956, 0, 0, 0]) # 77.5 deg of elevation     
        >>> dir_3 = np.array([-0.1004115, 0.53090257, 0.84146301, 0, 0, 0]) # 55 deg of elevation       
        >>> dir_4 = np.array([-0.6476016, 0.75067923, 0.13073926, 0, 0, 0]) # 0 deg of elevation
        >>> dirs = np.asfarray([dir_1, dir_2, dir_3, dir_4]).T
        >>> dirs
        array([[ 0.33087577,  0.18286604, -0.1004115 , -0.6476016 ],
               [ 0.12248111,  0.28205454,  0.53090257,  0.75067923],
               [ 0.93569204,  0.94180956,  0.84146301,  0.13073926],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ]])

        If we assume that the directions point from the transmitting station to the space object state, then we get :

        >>> ecef = dirs * np.array([[1550.0, 896.2, 1434.0, 4575.1]])*1e3
        >>> ecef[0:3] = ecef[0:3] + self.radar.tx[0].ecef[:, None]
        >>> ecef 
        array([[2629440.29182936, 2280467.39337736, 1972592.75732936,
                -846259.23183064],
               [ 973346.45896828, 1036278.01721628, 1544815.02384828,
                4217933.28364128],
               [7395791.81355332, 6789518.87922532, 7152127.10789332,
                6543614.33997932],
               [      0.        ,       0.        ,       0.        ,
                      0.        ],
               [      0.        ,       0.        ,       0.        ,
                      0.        ],
               [      0.        ,       0.        ,       0.        ,


        We know that only the first three states will be within the FOV, so we will get : 

        >>> radar.field_of_view(ecef)
        array([ True,  True, True, False])

                
        See Also
        --------
            sorts.radar.system.station.field_of_view : This method determines if the ecef states are within the FOV of a station 
        '''
        ecef = np.atleast_1d(ecef).reshape((6, -1))
        in_fov = np.full((len(ecef[0]),), True, dtype=bool)
    
        for station in self.tx + self.rx:
            in_fov = np.logical_and(in_fov, station.field_of_view(ecef))
           
        return in_fov


    def find_passes(self, t, states, cache_data=True):
        '''Finds all passes that are simultaneously inside a transmitter 
        station FOV and a receiver station FOV. 
        
        Parameters
        ----------
        t : numpy.ndarray (N,)
            Vector of times in seconds to use as a base to find passes.
        states : numpy.ndarray (6, N)
            ECEF states of the object to find passes for.
        cache_data : bool
            Wether the states will be stored.
            If True, the states will be stored in the ENU reference frame of each station.
            default value is True.
        
        Returns
        -------
        list of list of sorts.radar.passes.Pass :
            list of passes indexed by first tx-station and then rx-station.

        Examples
        --------
        To simplify this example, we will assume that we have created an array of space object states ``states`` in the ecef
        frame of 4 elements. Since all the stations within the radar have a FOV of 120 degrees (min elevation of 30 degrees),
        then we can create the directions :

        >>> dir_1 = np.array([0.33087577, 0.12248111, 0.93569204, 0, 0, 0]) # 90 deg of elevation        
        >>> dir_2 = np.array([0.18286604, 0.28205454, 0.94180956, 0, 0, 0]) # 77.5 deg of elevation     
        >>> dir_3 = np.array([-0.1004115, 0.53090257, 0.84146301, 0, 0, 0]) # 55 deg of elevation       
        >>> dir_4 = np.array([-0.6476016, 0.75067923, 0.13073926, 0, 0, 0]) # 0 deg of elevation
        >>> dirs = np.asfarray([dir_1, dir_2, dir_3, dir_4]).T
        >>> dirs
        array([[ 0.33087577,  0.18286604, -0.1004115 , -0.6476016 ],
               [ 0.12248111,  0.28205454,  0.53090257,  0.75067923],
               [ 0.93569204,  0.94180956,  0.84146301,  0.13073926],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ]])

        If we assume that the directions point from the transmitting station to the space object state, then we get :
        
        >>> t = numpy.array([0., 100., 200., 300.])
        >>> ecef = dirs * np.array([[1550.0, 896.2, 1434.0, 4575.1]])*1e3
        >>> ecef[0:3] = ecef[0:3] + self.radar.tx[0].ecef[:, None]
        >>> ecef 
        array([[2629440.29182936, 2280467.39337736, 1972592.75732936,
                -846259.23183064],
               [ 973346.45896828, 1036278.01721628, 1544815.02384828,
                4217933.28364128],
               [7395791.81355332, 6789518.87922532, 7152127.10789332,
                6543614.33997932],
               [      0.        ,       0.        ,       0.        ,
                      0.        ],
               [      0.        ,       0.        ,       0.        ,
                      0.        ],
               [      0.        ,       0.        ,       0.        ,


        We know that only the first three states will be within the FOV, so we will get : 

        >>> radar.find_passes(t, ecef)
        [[[Pass Station [<sorts.radar.system.station.TX object at 0x7f4721b066c0>, <sorts.radar.system.station.RX object at 0x7f470c1cb640>] 
        | Rise 0:00:00 (3.3 min) 0:03:20 Fall], 
        [Pass Station [<sorts.radar.system.station.TX object at 0x7f4721b066c0>, <sorts.radar.system.station.RX object at 0x7f470c1cb540>] 
        | Rise 0:00:00 (3.3 min) 0:03:20 Fall], 
        [Pass Station [<sorts.radar.system.station.TX object at 0x7f4721b066c0>, <sorts.radar.system.station.RX object at 0x7f470c1cb840>] 
        | Rise 0:00:00 (3.3 min) 0:03:20 Fall]]]

        After a quick computation, one notices that 200 seconds is equal to 3 minutes and 20 seconds, which means that the object entered the FOV
        at :math:`t=0s` and left at :math:`t=200s`, which is coherent with our previous observation that only the first 3 states are within the FOV
        of the radar.
        '''
        rd_ps = []
        for txi, tx in enumerate(self.tx):
            rd_ps.append([])
            for rxi, rx in enumerate(self.rx):
                txrx = passes.find_simultaneous_passes(
                    t, states, 
                    [tx, rx], 
                    cache_data=cache_data
                )
                for ps in txrx:
                    ps.station_id = [txi, rxi]
                rd_ps[-1].append(txrx)
        return rd_ps


    def check_control_feasibility(self, control_sequence):
        '''
        This function verifies if any of the controls are in conflict with the physical limitations of the RADAR (for example 
        its speed, elevation, power, ...).

        This function can be freely overrided to check the feasibility of the controls wby a custom Radar system.
        
        Parameters
        ----------
        control_sequence : sorts.radar.radar_controls.RadarControls
            RadarControls instance containing the controls to be sent to the radar. Such controls give the commanded states of the radar
            (such as power, pointing direction, ...) for each control time slice.

        Returns
        -------
        bool :
            True if the controls are compatible with the radar system, False otherwise.
        '''
        return True


    def control(self, control_sequence, cache_pdirs=True, cache_states=True):
        ''' Returns a control structure containing all the radar states for the given controls.

        This function computes the radar states resulting from the given control sequence. First, the algorithm
        will check if the specified control sequence is compatible with the radar physical constraints, and if yes,
        it will return the states of each station for each time slice.

        .. note::
            The default implementation of the Radar class does not include physical constraints. Therefore, if one needs to 
            include such constraints, one needs to overload the :ref:`check_control_feasibility` function.

        Parameters
        ----------
        control_sequence : sorts.radar.radar_controls.RadarControls
            Control sequence instance used to control the radar. This control sequence must be compatible with the radar constraints.
        cache_pdirs : bool, default=True
            If true, the radar states structure will contain the numerical values of the pointing directions

            .. note::
                Enabling the ``cache_pdir`` option will increase the RAM usage, but also reduce computational time.
        cache_states : bool, default=True
            If true, the radar states structure will contain the numerical values of the space object's states.

            .. note::
                Enabling the ``cache_states`` option will increase the RAM usage, but also reduce computational time.

        Examples
        --------
            TODO
        '''
        if self.check_control_feasibility(control_sequence) is False:
            self.logger.error("radar:system:control: control sequence is not compatible with the radar system !")
            return None

        # list of radar properties to be controlled
        radar_states = control_sequence.copy()

        if cache_pdirs is True and radar_states.pdirs is None and radar_states.has_pdirs is True:
            radar_states.set_pdirs(radar_states.pdir_args, cache_pdirs=True)

        # first, the algorithm fills in all the values of the radar states which aren't being contolled 
        # in the specific time slice
        controlled = False
        for period_id in range(radar_states.n_periods): 
            for station in self.rx + self.tx:
                station_id = radar_states.radar.get_station_id(station)
                station_type = station.type

                # for each controlled property of the station
                for property_name in radar_states.controlled_properties[station_type][station_id]:
                    # check if the property is controlled during the specific period id
                    if property_name in radar_states.property_controls[period_id][station_type].keys():
                        if radar_states.property_controls[period_id][station_type][property_name][station_id] is not None:
                            controlled = True
                    else: # if not, create new set of controls
                        n_stations = len(getattr(radar_states.radar, station_type))
                        radar_states.property_controls[period_id][station_type][property_name] = np.ndarray((n_stations,), dtype=object)

                    # if there is no controls for the current property at the specified control period, set default value
                    if controlled is False:
                        data = getattr(station, property_name)
                        radar_states.property_controls[period_id][station_type][property_name][station_id] = data*np.ones(len(radar_states.t[period_id]))        

        # create control fields for each stations properties which aren't controlled
        for station in self.rx + self.tx:
            station_id = radar_states.radar.get_station_id(station)
            station_type = station.type

            for property_name in station.get_properties():
                if property_name not in radar_states.controlled_properties[station_type][station_id]:
                    data = getattr(station, property_name)
                    radar_states.add_property_control(property_name, station, data*np.ones(radar_states.n_control_points))

                    if self.logger is not None:
                        self.logger.info(f"added control {property_name} for station {station_type} id {station_id}")
                    #print(f"added control {property_name} for station {station_type} id {station_id}")
                    radar_states.controlled_properties[station_type][station_id].pop(-1)

        self.states = radar_states
        return radar_states

    def compute_measurements(
        self, 
        radar_states, 
        space_object,
        rx_indices=None,
        tx_indices=None,
        epoch=None, 
        calculate_snr=True, 
        doppler_spread_integrated_snr=False,
        interpolator=interpolation.Linear, 
        max_dpos=50e3,
        snr_limit=True, 
        save_states=False, 
        logger=None,
        profiler=None,
        parallelization=True,
        n_processes=16,
    ):
        ''' Simulates radar measurements associated to an object and to a set of radar states.

        This function wraps the function :func:`measure<sorts.radar.measurement.base.Measurement.measure>` and provides an easy to use interface to simulate
        radar observations of space objects (SNR, range, Range rates, ...).

        Parameters
        ----------
        radar_states : sorts.radar.radar_controls.RadarControls
            Radar states during the measurement. 
            Radar states can be generated by calling the function :func:`control<Radar.control>` providing that the control sequence is feasable by the radar 
            system.        

        space_object : sorts.target.space_object.SpaceObject
            :class:`SpaceObject<sorts.targets.space_object.SpaceObject>` instance which is subject to the measurement.

        rx_indices : list of int/None, default=None
            List of indices of :class:`RX<sorts.radar.system.station.RX>` stations performing the measurement simultaneously.

        tx_indices : list of int/None, default=None
            List of indices of :class:`TX<sorts.radar.system.station.TX>` stations performing the measurement simultaneously.

        epoch : float/None, default=None, 
            Time epoch at the start of the simulation in Modified Julian Date format. See :class:`astropy.time.TimeMJD` for more information.

        calculate_snr=True, 
            If true, the simulation will compute the Signal-To-Noise ratio predictions given the radar and object states.

        doppler_spread_integrated_snr=False,
            If true, the simulation will compute the Doppler Spread Integrated Signal-To-Noise ratio predictions given the radar and object states.

        interpolator : sorts.common.interpolation.Interpolator, default=interpolation.Linear, 
            Interpolation class used to interpolate radar states from low time-resolution computations.

        max_dpos=50e3,
            Maximum distance between two consecutive low time-resolition propagated states.
            see :func:`measure<sorts.radar.measurement.base.Measurement.measure>` for more information.

        snr_limit : bool, default=``True``, 
            If True, SNR measurements under the minimum value provided by :ref:`min_SNRdb<minsnrdb>`_ will be discarded.

        save_states : bool, default=``False``, 
            If True, propagated and interpolated space object states will be saved and returned within the :ref:`output data structure<compute_measurements_ret>`_.

        logger : sorts.common.logging.Logger instance, default=``None``,
            Logger used to keep track of the measurement simulation process.

        profiler : sorts.common.profiling.Profiler instance, default=None
            Profiler used to keep track of the performances of the measurement simulation process.

        parallelization : bool, default=``True``
            If True, the simulation will be parallelized using process-based parallelisation.
            The current implementation is based on the python :ref:`multiprocessing` library.

        n_processes : int, default=16
            Maximum number of simultaneous processes used when :attr:`parallelization` is ``True``.


        .. _compute_measurements_ret::

        Returns
        -------
        dict
            The output data structure is a dictionnary which contains all the results of the simulation computations. 
            The dict contains the following fields : 

            * measurements : dict containing results of the measurements simulation
                - t : numpy.ndarray (n_periods, N,)
                    Time points at which the observations are performed (in seconds).

                - snr : numpy.ndarray (n_periods, Ntx, Nrx, N,)
                    If :attr:`calculate_snr` is True, then ``snr`` will contain the SNR values for each pair of (tx, rx) pais performing the measurements.

                - range : numpy.ndarray (n_periods, Ntx, Nrx, N,)
                    Distance travelled by the radar signal from the :class:`TX<sorts.radar.system.station.TX>` station to the :class:`RX<sorts.radar.system.station.RX>`
                    station (in meters) responsible for the phase shift of the radar signal.

                - range_rate : numpy.ndarray (n_periods, Ntx, Nrx, N,)
                    Total radial velocity of the object (corresponding to the sum of the radial velocity of the object with respect to the :class:`TX<sorts.radar.system.station.TX>` 
                    sation and with respect to the :class:`RX<sorts.radar.system.station.RX>`) responsible for the doppler shift of the radar signal.

                - pointing_direction : numpy.ndarray (n_periods, Ntx, Nrx, N,)
                    pointing direction of each radar station at each measurement point

                - rcs : numpy.ndarray (n_periods, N,)
                    Radar cross section of the object at each measurement point.

                - tx_indices : numpy.ndarray (Ntx,)
                    Indices of the :class:`TX<sorts.radar.system.station.TX>` stations involved in the measurement.

                - rx_indices : numpy.ndarray (Nrx,)
                    Indices of the :class:`RX<sorts.radar.system.station.RX>` stations involved in the measurement.

                - detection : numpy.ndarray (n_periods, N,)
                    At each time point, ``detection`` will be ``True`` if the SNR is greater than the detection value given by :ref:`min_SNRdb<minsnrdb>`_.

            * states : array of space object states (n_periods, 6, N)

            * pass_mask : array of bool (n_periods, 6, N)
                if the value at time t is False, then the space object state is not inside the station fov.

            .. note::
                All the returned arrays are splitted according to the control periods of the control sequence used to generate the radar states. 


        Examples 
        --------

        Before simulating a measurement, we need to define the radar system as well as the radar controls which will be executed during the measurement.
        To simplify this example, we will track a space object passing right over the station.
        '''
        return self.measurement_class.compute_space_object_measurements(
            radar_states, 
            space_object, 
            self, 
            tx_indices=tx_indices,
            rx_indices=rx_indices,
            epoch=epoch, 
            calculate_snr=calculate_snr, 
            doppler_spread_integrated_snr=doppler_spread_integrated_snr,
            interpolator=interpolation.Legendre8, 
            max_dpos=max_dpos,
            snr_limit=snr_limit, 
            save_states=save_states, 
            logger=logger,
            profiler=profiler,
            parallelization=parallelization,
            n_processes=n_processes,
        )

    def observe_passes(
        self, 
        passes,
        radar_states, 
        space_object, 
        epoch=None, 
        calculate_snr=True, 
        doppler_spread_integrated_snr=False,
        interpolator=interpolation.Legendre8, 
        max_dpos=100e3,
        snr_limit=True, 
        save_states=False, 
        logger=None,
        profiler=None,
        parallelization=True,
        n_processes=16,
    ):
        ''' Simulates radar measurements associated to an object and to a set of radar states.

        This function wraps the function :func:`measure<sorts.radar.measurement.base.Measurement.measure>` and provides an easy to use interface to simulate
        radar observations of space objects (SNR, range, Range rates, ...).

        Parameters
        ----------
        txrx_pass : sorts.Pass 
            Object pass over the radar system which we want to observe.

        radar_states : sorts.radar.radar_controls.RadarControls
            Radar states during the measurement. 
            Radar states can be generated by calling the function :func:`control<Radar.control>` providing that the control sequence is feasable by the radar 
            system.        

        space_object : sorts.target.space_object.SpaceObject
            :class:`SpaceObject<sorts.targets.space_object.SpaceObject>` instance which is subject to the measurement.

        rx_indices : list of int/None, default=None
            List of indices of :class:`RX<sorts.radar.system.station.RX>` stations performing the measurement simultaneously.

        tx_indices : list of int/None, default=None
            List of indices of :class:`TX<sorts.radar.system.station.TX>` stations performing the measurement simultaneously.

        epoch : float/None, default=None, 
            Time epoch at the start of the simulation in Modified Julian Date format. See :class:`astropy.time.TimeMJD` for more information.

        calculate_snr=True, 
            If true, the simulation will compute the Signal-To-Noise ratio predictions given the radar and object states.

        doppler_spread_integrated_snr=False,
            If true, the simulation will compute the Doppler Spread Integrated Signal-To-Noise ratio predictions given the radar and object states.

        interpolator : sorts.common.interpolation.Interpolator, default=interpolation.Linear, 
            Interpolation class used to interpolate radar states from low time-resolution computations.

        max_dpos=50e3,
            Maximum distance between two consecutive low time-resolition propagated states.
            see :func:`measure<sorts.radar.measurement.base.Measurement.measure>` for more information.

        snr_limit : bool, default=``True``, 
            If True, SNR measurements under the minimum value provided by :ref:`min_SNRdb<minsnrdb>`_ will be discarded.

        save_states : bool, default=``False``, 
            If True, propagated and interpolated space object states will be saved and returned within the :ref:`output data structure<compute_measurements_ret>`_.

        logger : sorts.common.logging.Logger instance, default=``None``,
            Logger used to keep track of the measurement simulation process.

        profiler : sorts.common.profiling.Profiler instance, default=None
            Profiler used to keep track of the performances of the measurement simulation process.

        parallelization : bool, default=``True``
            If True, the simulation will be parallelized using process-based parallelisation.
            The current implementation is based on the python :ref:`multiprocessing` library.

        n_processes : int, default=16
            Maximum number of simultaneous processes used when :attr:`parallelization` is ``True``.


        .. _compute_measurements_ret::

        Returns
        -------
        dict
            The output data structure is a dictionnary which contains all the results of the simulation computations. 
            The dict contains the following fields : 

            * measurements : dict containing results of the measurements simulation
                - t : numpy.ndarray (n_periods, N,)
                    Time points at which the observations are performed (in seconds).

                - snr : numpy.ndarray (n_periods, Ntx, Nrx, N,)
                    If :attr:`calculate_snr` is True, then ``snr`` will contain the SNR values for each pair of (tx, rx) pais performing the measurements.

                - range : numpy.ndarray (n_periods, Ntx, Nrx, N,)
                    Distance travelled by the radar signal from the :class:`TX<sorts.radar.system.station.TX>` station to the :class:`RX<sorts.radar.system.station.RX>`
                    station (in meters) responsible for the phase shift of the radar signal.

                - range_rate : numpy.ndarray (n_periods, Ntx, Nrx, N,)
                    Total radial velocity of the object (corresponding to the sum of the radial velocity of the object with respect to the :class:`TX<sorts.radar.system.station.TX>` 
                    sation and with respect to the :class:`RX<sorts.radar.system.station.RX>`) responsible for the doppler shift of the radar signal.

                - pointing_direction : numpy.ndarray (n_periods, Ntx, Nrx, N,)
                    pointing direction of each radar station at each measurement point

                - rcs : numpy.ndarray (n_periods, N,)
                    Radar cross section of the object at each measurement point.

                - tx_indices : numpy.ndarray (Ntx,)
                    Indices of the :class:`TX<sorts.radar.system.station.TX>` stations involved in the measurement.

                - rx_indices : numpy.ndarray (Nrx,)
                    Indices of the :class:`RX<sorts.radar.system.station.RX>` stations involved in the measurement.

                - detection : numpy.ndarray (n_periods, N,)
                    At each time point, ``detection`` will be ``True`` if the SNR is greater than the detection value given by :ref:`min_SNRdb<minsnrdb>`_.

            * states : array of space object states (n_periods, 6, N)

            * pass_mask : array of bool (n_periods, 6, N)
                if the value at time t is False, then the space object state is not inside the station fov.

            .. note::
                All the returned arrays are splitted according to the control periods of the control sequence used to generate the radar states. 


        Examples 
        --------

        Before simulating a measurement, we need to define the radar system as well as the radar controls which will be executed during the measurement.
        To simplify this example, we will track a space object passing right over the station.
        '''
        t_start = radar_states.pdirs[0]["t"][0]
        t_end   = radar_states.pdirs[-1]["t"][-1]

        data = []
        pass_iter = 0

        # count all passes
        if logger is not None:
            n_passes = 0
            for txi in range(len(passes)):
                for rxi in range(len(passes[txi])):
                    for ps in passes[txi][rxi]:
                        n_passes += 1

        # observe pass
        for txi in range(len(passes)):
            data.append([])
            for rxi in range(len(passes[txi])):
                data[-1].append([])
                for ps in passes[txi][rxi]:
                    if ps.start() > t_end or ps.end() < t_start: # if the pass is outside of the control range
                        pass_data = None
                    else:
                        if ps.start() <= t_start or ps.end() >= t_end: # if the pass is partially outside of the control range, remove parts which are outside
                            if ps.t is not None:
                                pass_mask = np.logical_and(ps.t >= t_start, ps.t <= t_end)
                                pass_inds = ps.inds[pass_mask]

                                enu_new = [xv[:, pass_inds] for xv in ps.enu]

                                new_ps = passes.Pass(
                                    t=ps.t[pass_mask], 
                                    enu=enu_new, 
                                    inds=pass_inds, 
                                    cache=True,
                                )
                            else:
                                new_ps = passes.Pass(
                                    t=np.array([t_start, t_end]), 
                                    enu=None, 
                                    inds=None, 
                                    cache=True,
                                )
                            ps = new_ps
                            ps.station_id = [txi, rxi]

                        pass_data = self.measurement_class.compute_pass_measurements(
                            ps,
                            radar_states, 
                            space_object, 
                            self, 
                            epoch=epoch, 
                            calculate_snr=calculate_snr, 
                            doppler_spread_integrated_snr=doppler_spread_integrated_snr,
                            interpolator=interpolator, 
                            max_dpos=max_dpos,
                            snr_limit=snr_limit, 
                            save_states=save_states, 
                            logger=logger,
                            profiler=profiler,
                            parallelization=parallelization,
                            n_processes=n_processes,
                        )

                    pass_iter += 1
                    if logger is not None:
                        logger.info(f"observe_passes: iteration {pass_iter}/{n_passes}")

                    data[-1][-1].append(pass_data)
        return data


    def compute_intersection_points(
        self,
        tx_directions,
        rx_directions,
        rtol=0.05,
        ):
        ''' Computes the ECEF points (if they exist) targetted by pointing direction controls.

        Given a set of pointing directions :math:`(\hat{k_{tx}}, \hat{k_{rx}})` and the positions of the stations (from which we can extract the vector 
        :math:`\hat{k_{tx, rx}`), one can compute the theoretical point which was targetted (given that the 3 prevous vectors lie in the same plane) 
        using the formula :
        
        .. math::
               \vec{r^{ij}} = \vec{r_{tx}^i} + \lambda_{ij} \hat{k_{tx}^i}

        with :

        .. math::
            \lambda_{ij} = \frac{(\vec{r_{tx}^i} - \vec{r_{rx}^j}) ((\hat{k_{tx}^i} \dot \hat{k_{rx}^j}) \hat{k_{rx}^j} - \hat{k_{tx}^i})}{1 - (\hat{k_{tx}^i} \dot \hat{k_{rx}^j})^2}
        
        The algorithm computes the intersection points :math:`\vec{r^{ij}}` for each (tx[i], rx[j]) tuple of stations inside the radar system. Then, we compute the
        barycenter for each tx[i] station :

        ..  math::
            \vec{r^{i}} = \frac{1}{N_{rx}} \sum_{k=0}^{N_{rx}-1}\vec{r^{ik}}

        And then we compute the relative distance of each point :math:`\vec{r^{ij}}` to the barycenter :math:`\vec{r^{i}}` :
        
        ..  math::
            \alpha_{ij} = \frac{\| \vec{r^{ij}} - \vec{r^{i}} \|}{\| \vec{r^{ij}} \|}

        If :math:`\alpha_{ij} > \alpha_{tol}`, we discard the intersection point, and if not, we keep the barycenter :math:`\vec{r^{i}}` as the intersection point.

        Parameters
        ----------
        tx_directions : numpy.array (Ntx, 1, 3, N)
            Transmitting stations pointing directions.
        rx_directions : numpy.array (Nrx, Ntx, 3, N)
            Receiving stations pointing directions.
        rtol : float, default=0.05
            Relative tolerance threshold over which we discard the intersection point at time t.

        Returns
        ------- 
        numpy.ndarray (3, N) :
            Array of ecef points targetted by the pointing direction controls.

        ..notes::
            Since this function was designed only for plotting, it only returns the intersection points which were found during the routine and not those which were discarded.
        
        Examples
        --------
        Pointing direction controls are usually generated from a set of ECEF points. For this example, we will consider a point situated directly over the Tx station of 
        the EISCAT_3D radar.
        
        >>> import sorts
        >>> radar = sorts.radars.eiscat3d

        We can compute the theoretical ecef position of the point located at a range of 1000km:

        >>> dir_enu = np.array([0., 0., 1.])
        >>> r_ecef = sorts.frames.enu_to_ecef(radar.tx[0].lat, radar.tx[0].lon, 0., dir_enu)*1e6 + radar.tx[0].ecef
        >>> r_ecef
        array([2447458.62326276,  905981.8472074 , 6881161.19467662])

        Then we can compute the pointing directions of each station :

        tx_ecef = np.array([tx.ecef for tx in radar.tx])[:, None, :, None]
        rx_ecef = np.array([rx.ecef for rx in radar.rx])[:, None, :, None]

        Tx pointing direction :

        >>> tx_dirs = r_ecef[None, None, :, None] - tx_ecef
        >>> tx_dirs = tx_dirs/np.linalg.norm(tx_dirs, axis=2)[:, :, None, :]
        array([[[[0.33087577],
                 [0.12248111],
                 [0.93569204]]]])

        Rx pointing directions :

        >>> rx_dirs = r_ecef[None, None, :, None] - rx_ecef
        >>> rx_dirs = rx_dirs/np.linalg.norm(rx_dirs, axis=2)[:, :, None, :]
        >>> rx_dirs
        array([[[[0.33087577],
                 [0.12248111],
                 [0.93569204]]],


               [[[0.27472276],
                 [0.00885592],
                 [0.9614827 ]]],


               [[[0.20478789],
                 [0.09962972],
                 [0.97372267]]]])
        
        Notice that we have don't have direct access to the range information of the point we started with using only ``rx_dirs`` and ``tx_dirs``, which 
        complicates the plotting of the pointing directions. Therefore, we can call ``compute_intersection_points`` to get the original point ``r_ecef`` :

        >>> radar.compute_intersection_points(tx_dirs, rx_dirs)
        array([[2447458.62326275],
               [ 905981.8472074 ],
               [6881161.19467661]])

        '''
        if np.shape(tx_directions[0]) != np.shape(rx_directions[0]):
            raise Exception(f"tx and rx pdirs directions are not the same shape : {np.shape(tx_directions)} != {np.shape(rx_directions)}")

        # radar pointing directions
        tx_directions = np.ascontiguousarray(np.asfarray(tx_directions), dtype=np.float64)
        rx_directions = np.ascontiguousarray(np.asfarray(rx_directions), dtype=np.float64)

        # radar station positions in ECEF coordinates
        tx_ecef = np.array([tx.ecef for tx in self.tx], dtype=np.float64)
        rx_ecef = np.array([rx.ecef for rx in self.rx], dtype=np.float64)
        
        # initialization of the computation results
        n_points = len(rx_directions[0, 0, 0]) # number of time points

        intersection_points = np.empty((3, n_points,), dtype=np.float64)
        keep = np.zeros((n_points,), dtype=np.int32)

        # Calling c library
        clibsorts.compute_intersection_points.argtypes = [
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=tx_directions.ndim, shape=tx_directions.shape, flags="C"),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=rx_directions.ndim, shape=rx_directions.shape, flags="C"),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=tx_ecef.ndim, shape=tx_ecef.shape, flags="C"),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=rx_ecef.ndim, shape=rx_ecef.shape, flags="C"),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=intersection_points.ndim, shape=intersection_points.shape, flags="C"),
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=keep.ndim, shape=keep.shape),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        clibsorts.compute_intersection_points(
            tx_directions,
            rx_directions,
            tx_ecef,
            rx_ecef,
            intersection_points,
            ctypes.c_double(rtol),
            keep,
            ctypes.c_int(len(self.tx)),
            ctypes.c_int(len(self.rx)),
            ctypes.c_int(n_points),
        ) 

        return intersection_points[:, np.where(keep==1)[0]]