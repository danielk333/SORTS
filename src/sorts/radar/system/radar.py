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

# Define C interface
clibsorts.compute_intersection_points.argtypes = [
    np.ctypeslib.ndpointer(dtype=ctypes.c_double, flags="C"),
    np.ctypeslib.ndpointer(dtype=ctypes.c_double, flags="C"),
    np.ctypeslib.ndpointer(dtype=ctypes.c_double, flags="C"),
    np.ctypeslib.ndpointer(dtype=ctypes.c_double, flags="C"),
    np.ctypeslib.ndpointer(dtype=ctypes.c_double, flags="C"),
    ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


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
        measurement=measurements.Measurement
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
    
        for st in self.tx + self.rx:
            in_fov = np.logical_and(in_fov, st.field_of_view(ecef))
           
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


    def control(
        self, 
        control_sequence, 
        cache_pdirs=True, 
        cache_states=True
    ):
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
            if radar_states.property_controls[period_id] is None:
                continue

            for st in self.rx + self.tx:
                station_id = radar_states.radar.get_station_id(st)
                station_type = st.type

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
                        data = getattr(st, property_name)
                        radar_states.property_controls[period_id][station_type][property_name][station_id] = data*np.ones(len(radar_states.t[period_id]))        

        # create control fields for each stations properties which aren't controlled
        for st in self.rx + self.tx:
            station_id = radar_states.radar.get_station_id(st)
            station_type = st.type

            for property_name in st.get_properties():
                if property_name not in radar_states.controlled_properties[station_type][station_id]:
                    data = getattr(st, property_name)
                    radar_states.add_property_control(property_name, st, data*np.ones(radar_states.n_control_points))

                    if self.logger is not None:
                        self.logger.info(f"added control {property_name} for station {station_type} id {station_id}")

                    # remove from controlled properties (to hide from controlled properties)
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
        exact=False,
        save_states=False, 
        logger=None,
        profiler=None,
        interrupt=False,
        parallelization=True,
        n_processes=16,
    ):
        ''' Simulates radar measurements associated to an object and to a set of radar states.

        This function wraps the function :func:`measure<sorts.radar.measurement.base.Measurement.measure>` and provides an easy to 
        use interface to simulateradar observations of space objects (SNR, range, Range rates, ...).

        .. seealso::
            See :attr:`sorts.Measurement.compute_space_object_measurements<sorts.radar.system.measurements.
            Measurement.compute_space_object_measurements>` to obtain more information about the way measurements
            are simulated.

        Parameters
        ----------
        radar_states : sorts.radar.radar_controls.RadarControls
            Radar states during the measurement. 
            Radar states can be generated by calling the function :func:`control<Radar.control>` providing that the control 
            sequence is feasable by the radar system.        
        space_object : sorts.target.space_object.SpaceObject
            :class:`SpaceObject<sorts.targets.space_object.SpaceObject>` instance which is subject to the measurement.
        rx_indices : list of int/None, default=None
            List of indices of :class:`RX<sorts.radar.system.station.RX>` stations performing the measurement simultaneously.
        tx_indices : list of int/None, default=None
            List of indices of :class:`TX<sorts.radar.system.station.TX>` stations performing the measurement simultaneously.
        epoch : float/None, default=None, 
            Time epoch at the start of the simulation in Modified Julian Date format. See :class:`astropy.time.TimeMJD` for more 
            information.
        calculate_snr=True, 
            If true, the simulation will compute the Signal-To-Noise ratio predictions given the radar and object states.
        doppler_spread_integrated_snr=False,
            If true, the simulation will compute the Doppler Spread Integrated Signal-To-Noise ratio predictions given the radar 
            and object states.
        interpolator : sorts.common.interpolation.Interpolator, default=interpolation.Linear, 
            Interpolation class used to interpolate radar states from low time-resolution computations.
        max_dpos=50e3,
            Maximum distance between two consecutive low time-resolition propagated states.
            see :func:`measure<sorts.radar.measurement.base.Measurement.measure>` for more information.
        snr_limit : bool, default=``True``, 
            If True, SNR measurements under the minimum value provided by :ref:`min_SNRdb<minsnrdb>`_ will be discarded.
        exact : bool, default=False
            If True, the states will be propagated at each time point ``t``, if not, they will be propagated to 
            meet the condition set by ``max_dpos``.
        save_states : bool, default=``False``, 
            If True, propagated and interpolated space object states will be saved and returned within the :ref:`output data 
            structure<compute_measurements_ret>`_.
        logger : sorts.common.logging.Logger instance, default=``None``,
            Logger used to keep track of the measurement simulation process.
        profiler : sorts.common.profiling.Profiler instance, default=None
            Profiler used to keep track of the performances of the measurement simulation process.
        interrupt : bool, default=False
            If ``True``, the measurement simulator will evalate the stop condition (defined within 
            the :attr:`sorts.Measurement.stop_condition<sorts.radar.measurements.measurement.Measurement.stop_condition>`)
            The simulation will stop at the first time step where the condition is satisfied and return the results of the
            previous steps.

            .. note::
                The default implementation of sorts does not provide any implementation for the ``stop_condition`` method.
                Therefore, to use the stop_condition feature, it is necessary to create a new :class:`sorts.Measurement
                <sorts.radar.measurements.measurement.Measurement>` class inherited from the first measurement
                class and provide a custom implementation satisfying the requirements of the project.
                
        parallelization : bool, default=``True``
            If True, the simulation will be parallelized using process-based parallelisation.
            The current implementation is based on the python :ref:`multiprocessing` library.
        n_processes : int, default=16
            Maximum number of simultaneous processes used when :attr:`parallelization` is ``True``.

        .. _compute_measurements_ret::

        Returns
        -------
        data : list of dict
            List of measurement data structure, see :attr:`Measurement.measure_states` for more information about the 
            output data structure of the measurement simulation.
            The list of data structures is arranged such that ``data[txi][rxi][pi]`` corresponds to the measurement data
            of the ``pi``^th pass over the Tx station of index ``txi`` and the Rx station of index ``rxi``.
        '''
        return self.measurement_class.compute_space_object_measurements(
            radar_states, 
            space_object, 
            tx_indices=tx_indices,
            rx_indices=rx_indices,
            epoch=epoch, 
            calculate_snr=calculate_snr, 
            doppler_spread_integrated_snr=doppler_spread_integrated_snr,
            interpolator=interpolation.Legendre8, 
            max_dpos=max_dpos,
            exact=exact,
            snr_limit=snr_limit, 
            save_states=save_states, 
            logger=logger,
            profiler=profiler,
            parallelization=parallelization,
            n_processes=n_processes,
            interrupt=interrupt,
        )

    def observe_passes(
        self, 
        pass_list,
        radar_states, 
        space_object, 
        epoch=None, 
        calculate_snr=True, 
        doppler_spread_integrated_snr=False,
        interpolator=interpolation.Legendre8, 
        max_dpos=100e3,
        snr_limit=True, 
        exact=False,
        use_cached_states=True,
        save_states=False, 
        logger=None,
        profiler=None,
        interrupt=False,
        parallelization=True,
        n_processes=16,
    ):
        ''' Simulates radar measurements associated to an object and to a set of radar states.

        This function wraps the function :func:`measure<sorts.radar.measurement.base.Measurement.measure>` and provides an easy to use interface to simulate
        radar observations of space objects (SNR, range, Range rates, ...).

        Parameters
        ----------
        pass_list : sorts.Pass 
            List of object passes over the radar system which we want to observe.
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
        exact : bool, default=False
            If True, the states will be propagated at each time point ``t``, if not, they will be propagated to 
            meet the condition set by ``max_dpos``.
        use_cached_states : bool, default=True
            If True, the function will use the space object states cached within the radar pass if available.
        snr_limit : bool, default=``True``, 
            If True, SNR measurements under the minimum value provided by :ref:`min_SNRdb<minsnrdb>`_ will be discarded.
        save_states : bool, default=``False``, 
            If True, propagated and interpolated space object states will be saved and returned within the :ref:`output data structure<compute_measurements_ret>`_.
        logger : sorts.common.logging.Logger instance, default=``None``,
            Logger used to keep track of the measurement simulation process.
        profiler : sorts.common.profiling.Profiler instance, default=None
            Profiler used to keep track of the performances of the measurement simulation process.
        interrupt : bool, default=False
            If ``True``, the measurement simulator will evalate the stop condition (defined within 
            the :attr:`sorts.Measurement.stop_condition<sorts.radar.measurements.measurement.Measurement.stop_condition>`)
            The simulation will stop at the first time step where the condition is satisfied and return the results of the
            previous steps.

            .. note::
                The default implementation of sorts does not provide any implementation for the ``stop_condition`` method.
                Therefore, to use the stop_condition feature, it is necessary to create a new :class:`sorts.Measurement
                <sorts.radar.measurements.measurement.Measurement>` class inherited from the first measurement
                class and provide a custom implementation satisfying the requirements of the project.
                
        parallelization : bool, default=``True``
            If True, the simulation will be parallelized using process-based parallelisation.
            The current implementation is based on the python :ref:`multiprocessing` library.
        n_processes : int, default=16
            Maximum number of simultaneous processes used when :attr:`parallelization` is ``True``.


        .. _compute_measurements_ret::

        Returns
        -------
        data : list of dict
            List of measurement data structure, see :attr:`Measurement.measure_states` for more information about the 
            output data structure of the measurement simulation.
            The list of data structures is arranged such that ``data[txi][rxi][pi]`` corresponds to the measurement data
            of the ``pi``^th pass over the Tx station of index ``txi`` and the Rx station of index ``rxi``.

        Examples 
        --------
        Before simulating a measurement, we need to define the radar system as well as the radar controls which will be 
        executed during the measurement. 

        >>> import sorts
        >>> import numpy as np
        >>> radar = sorts.radars.eiscat3d

        Consider now a space object (Kepler propagator) passing over the **EISCAT_3D** radar system:

        >>> Prop_cls = sorts.propagator.Kepler
        >>> Prop_opts = dict(
        ...         settings = dict(
        ...                 out_frame='ITRS',
        ...                 in_frame='TEME',
        ...         ),
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
        ...                 d = 0.1,
        ...         ),
        ... )
        
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
        ...         orbit=space_object.state, 
        ...         start_t=0, 
        ...         end_t=3600.0, 
        ...         max_dpos=10e3)
        >>> object_states = space_object.get_state(t_states)
        
        We can recover the passes over the radar station by running:

        >>> passes = sorts.passes.find_simultaneous_passes(t_states, object_states, radar.tx+radar.rx, cache_data=True)
        [Pass Station [<sorts.radar.system.station.TX object at 0x7ff9af6ccac0>, 
        <sorts.radar.system.station.RX object at 0x7ff99523ba40>, 
        <sorts.radar.system.station.RX object at 0x7ff9951db2c0>, 
        <sorts.radar.system.station.RX object at 0x7ff9951db4c0>] | 
        Rise 0:04:05.161251 (4.1 min) 0:08:10.322502 Fall]

        For the sake of this example, suppose that the radar is controlled by a Tracking controller:

        >>> controller = sorts.Tracker()

        We can generate the controls for the first pass by running:

        >>> tracking_states = object_states[:, passes[0].inds]
        >>> t_states_i = t_states[passes[0].inds]
        >>> t_controller = np.arange(t_states_i[0], t_states_i[-1], 10)
        >>> controls = controller.generate_controls(
        ...         t_controller, 
        ...         radar, 
        ...         t_states_i, 
        ...         tracking_states, 
        ...         t_slice=0.1, 
        ...         scheduler=None, 
        ...         states_per_slice=1, 
        ...         interpolator=sorts.interpolation.Legendre8)

        From which we can generate the radar states during the Tracking time interval:

        >>> radar_states = radar.control(controls)

        Finally, we can run the measurement simulation over the pass by finding the corresponding pass
        for each tuple of station (tx/rx) and running the ``observe_passes`` function:

        >>> pass_list = radar.find_passes(t_states_i, tracking_states, cache_data=True)
        >>> data = radar.observe_passes(pass_list, radar_states, space_object)

        We can then print the results by executing the following instructions:

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure()
        >>> ax1 = fig.add_subplot(311)
        >>> ax2 = fig.add_subplot(312, sharex=ax1)
        >>> ax3 = fig.add_subplot(313, sharex=ax1)
        >>> fmt = ["-r", "-g", "-b"]
        >>> for station_id in range(len(radar.rx)):
        ...         measurements = data[0][station_id][0]["measurements"] # extract measurements for each rx station for the first pass
        ...         ax1.plot(measurements['t_measurements'], measurements['range']*1e-3, fmt[station_id], label=f"rx{station_id}")
        ...         ax2.plot(measurements['t_measurements'], measurements['range_rate']*1e-3, fmt[station_id], label=f"rx{station_id}")
        ...         ax3.plot(measurements['t_measurements'], 10*np.log10(measurements['snr']), fmt[station_id], label=f"rx{station_id}")
        >>> ax1.set_ylabel("$R$ [$km$]")
        >>> ax2.set_ylabel("$v_r$ [$km/s$]")
        >>> ax3.set_ylabel("$\\rho$ [$dB$]")
        >>> ax3.set_xlabel("$t$ [$s$]")
        >>> ax1.tick_params(labeltop=False, labelbottom=False)
        >>> ax2.tick_params(labeltop=False, labelbottom=False)
        >>> ax3.tick_params(labeltop=False)
        >>> ax1.grid()
        >>> ax2.grid()
        >>> ax3.grid()
        >>> fig.subplots_adjust(hspace=0)
        >>> plt.legend()
        >>> plt.show()

        Which gives us the following results:

        .. figure:: ../../../../figures/radar_example_observe_passes.png

        '''
        t_start = radar_states.pdirs[0]["t"][0]
        t_end   = radar_states.pdirs[-1]["t"][-1]

        data = []
        pass_iter = 0

        # count all passes
        if logger is not None:
            n_passes = 0
            for txi in range(len(pass_list)):
                for rxi in range(len(pass_list[txi])):
                    for ps in pass_list[txi][rxi]:
                        n_passes += 1

        # observe pass
        for txi in range(len(pass_list)):
            data.append([])
            for rxi in range(len(pass_list[txi])):
                data[-1].append([])
                for ps in pass_list[txi][rxi]:
                    print(ps.start())
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
                                    t=np.array([ps.start(), ps.end()]), 
                                    enu=None, 
                                    inds=None, 
                                    cache=True,
                                )
                                print("start, ", new_ps.start())
                            ps = new_ps
                            ps.station_id = [txi, rxi]

                        pass_data = self.measurement_class.compute_pass_measurements(
                            ps,
                            radar_states, 
                            space_object, 
                            epoch=epoch, 
                            calculate_snr=calculate_snr, 
                            doppler_spread_integrated_snr=doppler_spread_integrated_snr,
                            interpolator=interpolator, 
                            max_dpos=max_dpos,
                            snr_limit=snr_limit, 
                            exact=exact,
                            use_cached_states=use_cached_states,
                            save_states=save_states, 
                            logger=logger,
                            profiler=profiler,
                            parallelization=parallelization,
                            n_processes=n_processes,
                            interrupt=interrupt,
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