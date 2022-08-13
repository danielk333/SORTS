import numpy as np
from tabulate import tabulate
import ctypes

from .base import RadarSchedulerBase
from sorts import clibsorts

from ..controllers import radar_controller
from .. import radar_controls


class StaticPriorityScheduler(RadarSchedulerBase):
    ''' Defines the static priority scheduler.
    
    The :class:`static priority scheduler <StaticPriorityScheduler>` schedules the control
    time slices of a list of controllers according to their priority. There are two different 
    ways to define the priority of time slices:

     *  By setting a constant priority common to each time slice when calling the :attr:`generate_controls<sorts.radar.controllers.radar_controller.RadarController.generate_controls>` method.
     *  By setting variable priorities for each control time slice by calling:

            >>> controls.priority = priority

            with ``priority`` a positive integer or an array of positive integers.

    The scheduler effectively compares the priorities of competing time slices and 
    picks the ones with highest priority amongst them (highest priority being 
    lower ``priority`` values).

    .. seealso::
        - :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>` : Encapsulates a radar control structure.
        - :class:`sorts.Radar<sorts.radar.system.radar.Radar>` : Encapsulates a radar system.
        - :ref:`controllers` : Module defining radar controllers.
        - :ref:`schedulers` : Module defining radar schedulers.

    Parameters
    ----------
    radar : :class:`sorts.Radar<sorts.radar.system.radar.Radar>`
        Radar instance being controlled.
    t0 : float
        Start time of the scheduler with respect to the reference time of the
        simulation (in seconds).
    scheduler_period : float
        Duration of a scheduler period (in seconds). 

        The input :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>` 
        control periods must coincide with the scheduler periods (see :ref:`radar_controls` 
        to obtain more information about control period synchronization). 
    logger : :class:`logging.Logger`, default=None
        Logger instance used to log the compuation status of the class methods.
    profiler : :class:`sorts.profiling.Profiler<sorts.common.profiling.Profiler>`, default=None
        Profiler instance used to monitor the computation performances of the class methods.

    Examples
    --------
    Let us take the example of a RADAR measurement campain where multiple scientist want to use the same RADAR system to conduct different observations.
    Each scientist will require a different set of controls, which aren't necessarily compatible with each other.

    The :ref:`schedulers` module can be used to solve this problem as follow:
      -  Each scientist can generate the controls he wants to perform with the radar (i.e. tracking of a meteoroid, random scan each 10s, ...).
      - A unique priority can be affected to each control.
      - The different controls are stacked into a single array.
      - Finally, one can call :attr:`scheduler.run<base.RadarSchedulerBase.run>` method to get a final control sequence which (hopefully) will satisfy the requirements of each scientist.
    

    The following example showcases the generation of a schedule over a sub time interval of 4 control sequences
    (29 to 70 seconds).

    .. code-block:: Python

        import numpy as np
        import matplotlib.pyplot as plt

        import sorts


        # scheduler and Computation properties
        t0                  = 0
        scheduler_period    = 50 # [s]
        end_t               = 100

        # RADAR definition
        eiscat3d = sorts.radars.eiscat3d

        # ======================================= Scheduler ========================================

        scheduler = sorts.StaticPriorityScheduler(eiscat3d, t0, scheduler_period)

        # ======================================== Controls =========================================
        # ---------------------------------------- Scanner -----------------------------------------


        # controls parameters 
        controls_period     = np.array([1, 2, 2])
        controls_t_slice    = np.array([0.5, 1, 1.5])
        controls_priority   = np.array([3, 2, 1], dtype=int) # control priorities -> 1, 2, 3
        controls_start      = np.array([0, 4, 49.5], dtype=float)
        controls_end        = np.array([end_t, end_t, end_t], dtype=float)
        controls_az         = np.array([90, 45, 80])
        controls_el         = np.array([10, 20, 30])

        scans = []
        for i in range(len(controls_period)):
            scans.append(sorts.scans.Fence(azimuth=controls_az[i], min_elevation=controls_el[i], dwell=controls_t_slice[i], num=int(controls_end[i]/controls_t_slice[i])))

        # Generate scanning controls
        controls        = []
        scanner_ctrl    = sorts.Scanner()
        for i in range(len(controls_period)):
            t = np.arange(controls_start[i], controls_end[i], controls_period[i])
            controls.append(scanner_ctrl.generate_controls(t, eiscat3d, scans[i], scheduler=scheduler, priority=controls_priority[i]))

        # TRACKER
        # Object definition
        # Propagator
        Prop_cls = sorts.propagator.Kepler
        Prop_opts = dict(
            settings = dict(
                out_frame='ITRS',
                in_frame='TEME',
            ),
        )

        # ---------------------------------------- Tracker -----------------------------------------

        # Creating space object
        space_object = sorts.SpaceObject(
                Prop_cls,
                propagator_options = Prop_opts,
                a = 7200e3, 
                e = 0.1,
                i = 80.0,
                raan = 86.0,
                aop = 0.0,
                mu0 = 60.0,
                
                epoch = 53005.0,
                parameters = dict(
                    d = 0.1,
                ),
            )

        # create state time array
        t_states = sorts.equidistant_sampling(
            orbit=space_object.state, 
            start_t=0, 
            end_t=end_t, 
            max_dpos=50e3,
        )

        # get object states in ECEF frame and passes
        object_states   = space_object.get_state(t_states)
        eiscat_passes   = sorts.find_simultaneous_passes(t_states, object_states, [*eiscat3d.tx, *eiscat3d.rx])

        # Tracker controller parameters
        tracking_period = 15
        t_slice         = 10

        # create Tracker controller with highest priority (prio=0)
        tracker_controller = sorts.Tracker()

        for pass_id in range(np.shape(eiscat_passes)[0]):
            tracking_states = object_states[:, eiscat_passes[pass_id].inds]
            t_states_i      = t_states[eiscat_passes[pass_id].inds]
            t_controller    = np.arange(t_states_i[0], t_states_i[-1]+tracking_period, tracking_period)
            controls.append(tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, scheduler=scheduler, priority=0, states_per_slice=10, interpolator=sorts.interpolation.Linear))
            
        # ==================================== Run scheduler ======================================


        final_control_sequence = scheduler.run(controls, t_start=29, t_end=70)


        # ======================================= Plotting ========================================

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plotting station ECEF positions and grid earth
        sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)
        for tx in eiscat3d.tx:
            ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
        for rx in eiscat3d.rx:
            ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

        # plot passes
        for pass_id in range(np.shape(eiscat_passes)[0]):
            tracking_states = object_states[:, eiscat_passes[pass_id].inds]
            ax.plot(tracking_states[0], tracking_states[1], tracking_states[2], "-b")

        # plot scheduler schedule
        figs = sorts.plotting.plot_scheduler_control_sequence(controls, final_control_sequence, scheduler)

        # plot control sequences pointing directions
        fmts = ["b-", "m-", "k-", "-c"]
        for period_id in range(final_control_sequence.n_periods):
            for ctrl_i in range(len(controls)):
                ax = sorts.plotting.plot_beam_directions(controls[ctrl_i].get_pdirs(period_id), eiscat3d, ax=ax, fmt=fmts[ctrl_i], linewidth_rx=0.08, linewidth_tx=0.08, alpha=0.001)

        # plot scheduler pointing directions
        for period_id in range(final_control_sequence.n_periods):
            ax = sorts.plotting.plot_beam_directions(final_control_sequence.get_pdirs(period_id), eiscat3d, ax=ax)

        plt.show()
    

    This example produces the following outputs:

    .. figure:: ../../../../figures/example_scheduler_static_prio.png

        Pointing directions of the scheduler control sequence (red/green) and of the raw control sequences used
        to generate the schedule (blue, magenta, cyan and black).


    .. figure:: ../../../../figures/example_scheduler_static_prio_p1.png

        Scheduler and raw control sequences uptime (1st scheduler period).


    .. figure:: ../../../../figures/example_scheduler_static_prio_p2.png

        Scheduler and raw control sequences uptime (2nd scheduler period).
    '''
    def __init__(self, radar, t0, scheduler_period, logger=None, profiler=None):
        ''' Default class constructor. '''
        super().__init__(radar, t0, scheduler_period, logger=logger, profiler=profiler)

    def run(self, controls, t_start=None, t_end=None, log_performances=True):
        ''' Used to create a single radar control set from a set of multiple controls which 
        is compatible with the control priorities, time points and time slices. 

        As an example, if one wants to execute multiple control types at the same time (scanning, tracking, ...)
        then one needs to generate all the wanted controls independantly, and then call the scheduler over the 
        list of generated controls to get a new control set which corresponds to the fusion (weighted with 
        respect to the priority of each control subsets) of all the previous controls. 

        Parameters
        ----------
        controls : list / numpy.ndarray of :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            List or array of controls to be managed. Those controls can have been generated from different kinds of controllers.
            The scheduler will extract the final controls and time points depending on the priority of each control. 
            To do so, the scheduler will discard any overlapping time slices and will only keep the time slice of 
            highest priority.
        t_start : int, default=None
            Start time of the scheduler (in seconds). If not set, the start time will correspond to the first time point of the 
            control time arrays.
        t_end : int, default=None
            End time of the scheduler (in seconds). If not set, the end time will correspond to the lastest time point of the 
            control time arrays.
        log_performances : bool, default=True
            If true, the scheduler will print the performances (success rate, number of slices per controls, ...) of the scheduler 
            to debug the schedule quality. 

        Returns
        -------
        final_control_sequence : :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            Final scheduled radar control sequence.
        '''
        # Check input values   
        controls=np.asarray(controls, dtype=object) # convert control list to an np.array of objects  

        # check if the priority of each control is valid, and if not, use FIFO.
        self.check_priority(controls)

        # get the max end time of the controls if t_end is not given 
        if t_end is None:
            t_end = 0

            for ctrl_id, ctrl in enumerate(controls):
                t_last_i = controls[ctrl_id].t[-1][-1]
                if t_last_i > t_end: t_end = t_last_i
        
        # get the max end time of the controls if t_end is not given 
        if t_start is None: 
            t_start = 0

            for ctrl_id, ctrl in enumerate(controls):
                t_last_i = controls[ctrl_id].t[0][0]
                if t_last_i < t_start: t_start = t_last_i

        new_time_array = []
        new_ctrl_id_array = []

        # compute start and number of scheduler periods when start and end times are set
        scheduler_period_start = int((t_start - self.t0)//self._scheduler_period)
        scheduler_period_count = int(np.ceil((t_end - (t_start//self._scheduler_period)*self._scheduler_period)/self._scheduler_period))

        # Initialization of the scheduling C functions
        c_fnc_pointers, C_FNC_TYPES = self.__setup_c_callback_functions(controls, scheduler_period_count, new_time_array, new_ctrl_id_array) # intialize C callback functions 

        # save functions to a C library struct for easy access during schedule computations
        clibsorts.init_static_priority_scheduler.argtypes = C_FNC_TYPES
        clibsorts.init_static_priority_scheduler(*c_fnc_pointers)

        t_start_i = self.t0 + t_start

        # initializes results
        final_t = [None]*scheduler_period_count
        final_t_slice = [None]*scheduler_period_count
        final_active_control = [None]*scheduler_period_count

        # manages the controls for each scheduler period 
        for scheduler_period_id in range(scheduler_period_count):
            abs_scheduler_period_id = scheduler_period_id + scheduler_period_start
            t_end_i = t_start_i + self._scheduler_period # compute the end time of the scheduler period

            if t_end_i > t_end:
                t_end_i = t_end

            # initialize library call function
            clibsorts.run_static_priority_scheduler.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double]
            clibsorts.run_static_priority_scheduler.restype = ctypes.c_double

            # run clib function -> returns the position of the time cursor t_start_i
            t_start_i = clibsorts.run_static_priority_scheduler(
                ctypes.c_int(abs_scheduler_period_id),
                ctypes.c_int(len(controls)),
                ctypes.c_double(t_start_i), 
                ctypes.c_double(t_end_i))

            # get scheduler arrays
            time_array, time_slices = self.__get_control_sequence(controls, abs_scheduler_period_id, new_time_array, new_ctrl_id_array)
            
            # save scheduler arrays
            final_t[scheduler_period_id] = time_array
            final_t_slice[scheduler_period_id] = time_slices
            final_active_control[scheduler_period_id] = np.asarray(new_ctrl_id_array).astype(np.int32)

        # convert each output to a numpy array
        final_control_sequence = radar_controls.RadarControls(self.radar, None, scheduler=self, priority=None)
        final_control_sequence.active_control = np.asarray(final_active_control, dtype=object)
        final_control_sequence.set_time_slices(
            np.asarray(final_t, dtype=object), 
            np.asarray(final_t_slice, dtype=object),
            )
        final_control_sequence.meta["scheduled_controls"] = controls

        if log_performances is True:
            self.log_scheduler_performances(final_control_sequence)

        return self.extract_control_sequence(controls, final_control_sequence)

    def __setup_c_callback_functions(self, controls, n_periods, new_time_array, new_ctrl_id_array):
        ''' Sets up all the callback functions needed to interact properly with the scheduler C library.

        Parameters
        ----------
        controls : list / numpy.ndarray of :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            List or array of controls to be managed.
        n_periods : int
            number of scheduling periods to be scheduled.
        new_time_array : numpy.ndarray of int
            rray o arrays of indices corresponding to the indices of the time slices which will be extracted from the 
            list of controls in order to create the final control sequence.   
        new_ctrl_id_array : numpy.ndarray of int
            Array of indices corresponding to the indices the control sequence which time slice which will 
            be extracted in order to create the final control sequence.

        Returns
        -------
        c_fnc_pointers : list
            List of ctypes function pointers used for communication between the C library and Python.
        C_FNC_TYPES : list
            List of ctypes fonction types (contating information about the arguments and type of the 
            return value of the functions).
        '''
        ptr_time_array_c = [[None]*len(controls)]*n_periods
        ptr_t_slice_c = [[None]*len(controls)]*n_periods
        ptr_priority_c = [[None]*len(controls)]*n_periods

        # C CALLBACK FUNCTIONS
        def get_control_period_id_callback(control_id, scheduler_period_id):
            ''' Gets the active time control subarray associated with a given scheduler period ID. 
            
            Parameters
            ----------
            control_id : int
                Index of the current control sequence within the list of control sequences being scheduled.
            scheduler_period_id : int 
                Index of the scheduler period considered.

            Returns
            -------
            control_period_id : int
                Index of the current control at the specified scheduler period index.

            '''
            nonlocal controls
            return controls[control_id].get_control_period_id(scheduler_period_id)

        GET_CONTROL_PERIOD_ID_FNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
        get_control_period_id_c = GET_CONTROL_PERIOD_ID_FNC(get_control_period_id_callback)

        def get_control_parameters_callback(control_id, control_period_id, index, t, t_slice, priority):
            ''' Copies the priority, time and t_slice of a given control, time index and control 
            period (which corresponds to a control sliced time subarray) from Python to C pointers.
            This function is used to prevent sending the whole time data to the library at once.

            Parameters
            ----------
            control_id : int
                Index of the current control sequence within the list of control sequences being scheduled.
            control_period_id : int 
                Index of the control period considered.
            index : index
                index of the control time slice which properties will be extracted. 
            t : pointer
                Pointer to the current time slice start time (in seconds).
            t_slice : pointer
                Pointer to the current time slice duration (in seconds).
            priority : pointer
                Pointer to the current time slice priority.
            '''
            nonlocal controls

            if control_period_id >= len(controls[control_id].t) or control_period_id == -1: # no controls available in given control period
                return -1
            else:
                # assigns the control parameters to the C pointers
                t[0] = controls[control_id].t[control_period_id][index]
                t_slice[0] = controls[control_id].t_slice[control_period_id][index]
                priority[0] = controls[control_id].priority

                return 1

        GET_CONTROL_PARAMETERS_FNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int))
        get_control_parameters_c = GET_CONTROL_PARAMETERS_FNC(get_control_parameters_callback)

        def get_control_arrays_callback(period_id, time_array_c, t_slice_c, priority_c, size_c):
            ''' Copies the priority, time and t_slice arrays of a given control and control period 
            (which corresponds to a control sliced time subarray) from Python to C allocated array.
            This function is used to prevent sending the whole time data to the library at once.

            Parameters
            ----------
            period_id : int
                Index of the scheduler period which arrays will be copied.
            time_array_c : ctypes pointer
                Pointer to the time slice start time array during the current period index.
            t_slice_c : ctypes pointer
                Pointer to the time slice duration array during the current period index.
            priority_c : ctypes pointer
                Pointer to the time slice priority array during the current period index.
            size_c : ctypes pointer
                Number of time slices during the current scheduler period.
            '''
            nonlocal controls

            c_double_p = ctypes.POINTER(ctypes.c_double)
            c_int_p = ctypes.POINTER(ctypes.c_int)

            for ctrl_id in range(len(controls)):
                ctrl_period_id = controls[ctrl_id].get_control_period_id(period_id)

                if ctrl_period_id != -1:                    
                    # get array pointers
                    ptr_time_array_c[period_id][ctrl_id]   = np.asarray(controls[ctrl_id].t[ctrl_period_id]).astype(np.float64).ctypes.data_as(c_double_p)
                    ptr_t_slice_c[period_id][ctrl_id]      = np.asarray(controls[ctrl_id].t_slice[ctrl_period_id]).astype(np.float64).ctypes.data_as(c_double_p)                    
                    ptr_priority_c[period_id][ctrl_id]     = np.asarray(controls[ctrl_id].priority[ctrl_period_id]).astype(np.int32).ctypes.data_as(c_int_p)
                    
                    time_array_c[ctrl_id]       = ptr_time_array_c[period_id][ctrl_id]
                    t_slice_c[ctrl_id]          = ptr_t_slice_c[period_id][ctrl_id]
                    priority_c[ctrl_id]         = ptr_priority_c[period_id][ctrl_id]

                    # get new size
                    size_c[ctrl_id]             = len(controls[ctrl_id].t[ctrl_period_id])
                else:
                    size_c[ctrl_id]             = -1

        GET_CONTROL_ARRAYS_FNC = ctypes.CFUNCTYPE(
            None, 
            ctypes.c_int, 
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
            ctypes.POINTER(ctypes.c_int))

        get_control_arrays_c = GET_CONTROL_ARRAYS_FNC(get_control_arrays_callback)

        def save_new_control_arrays_callback(new_time_array_c, new_ctrl_id_array_c, size):
            nonlocal new_time_array, new_ctrl_id_array
            ''' Allows Python to copy the control arrays :
                - time_array : list of time slice indices extracted from the list of controls by the scheduler.
                - ctrl_id_array : list of control sequence indices corresponding to the extracted time slices.

            from the C arrays, created after running the scheduler over a given scheduler period, to the 
            corresponding python arrays. This function allows the use of temporary dynamically allocated 
            arrays in C.

            Parameters
            ----------
            new_time_array_c : pointer
                pointer to the C ``time_array``.
            new_ctrl_id_array_c : pointer
                pointer to the C ``ctrl_id_array``.
            size : int
                Number of time slices extracted.
            '''
            self.__save_new_control_arrays(new_time_array_c, new_ctrl_id_array_c, size, new_time_array, new_ctrl_id_array)

        SAVE_NEW_CTRL_ARRAY_FNC = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int)
        save_new_control_arrays_c = SAVE_NEW_CTRL_ARRAY_FNC(save_new_control_arrays_callback)

        C_FNC_TYPES = (GET_CONTROL_PERIOD_ID_FNC, GET_CONTROL_PARAMETERS_FNC, GET_CONTROL_ARRAYS_FNC, SAVE_NEW_CTRL_ARRAY_FNC)
        c_fnc_pointers = (get_control_period_id_c, get_control_parameters_c, get_control_arrays_c, save_new_control_arrays_c)

        return c_fnc_pointers, C_FNC_TYPES


    def __get_control_sequence(self, controls, period_id, new_time_array, new_ctrl_id_array):
        ''' Extract the control time and time_slice arrays from the glogal controls array.

        This function is called after having conputed the active control and time point indices (arrays of int) 
        over a given scheduler period to retreive the actual time points and time slices.

        Parameters
        ----------
        controls : list / numpy.ndarray of :class:`sorts.radar.radar_controls.RadarControls`
            List of radar control sequences being scheduled.
        period_id : int
            Index of the period index considered.
        new_time_array : numpy.ndarray of int
            rray o arrays of indices corresponding to the indices of the time slices which will be extracted from the 
            list of controls in order to create the final control sequence.   
        new_ctrl_id_array : numpy.ndarray of int
            Array of indices corresponding to the indices the control sequence which time slice which will 
            be extracted in order to create the final control sequence.

        Returns
        -------
        time_array : numpy.ndarray of float
            Time slice start time of the final control sequence (in seconds).
        time_slices : numpy.ndarray of float
            Time slice duration of the final control sequence (in seconds).
        '''

        # initializes the output arrays
        time_array = np.empty(len(new_ctrl_id_array), dtype=float)
        time_slices = np.empty(len(new_ctrl_id_array), dtype=float)
        
        for i in range(len(new_ctrl_id_array)):
            # compute the sliced time array id for the given controller
            ctrl_period = period_id - int(controls[new_ctrl_id_array[i]].t[0][0]/self._scheduler_period)

            # copies the values contained inside the controls array to the new arrays
            time_array[i] = controls[new_ctrl_id_array[i]].t[ctrl_period][new_time_array[i]]
            time_slices[i] = controls[new_ctrl_id_array[i]].t_slice[ctrl_period][new_time_array[i]]
                                    
        return time_array, time_slices


    def __get_control_array_size(self, controls, control_id, control_period_id):
        ''' Gets the number of time points in a given control time subarray.

        Parameters
        ----------
        controls : list / numpy.ndarray of :class:`sorts.radar.radar_controls.RadarControls`
            List of radar control sequences being scheduled.
        control_id : int
            Index of the current control sequence within the list of control sequences being scheduled.
        control_period_id : int 
            Index of the control period considered.

        Returns
        -------
        time_array_size : int
            Number of time slices in the current control for the given control period.
        '''
        if control_period_id == -1:
            time_array_size = 0;
        else:
            time_array_size = np.size(controls[control_id].t[control_period_id])

        return time_array_size 
        

    def __save_new_control_arrays(self, new_time_array_c, new_ctrl_id_array_c, size, new_time_array, new_ctrl_id_array):
        ''' Allows Python to copy the control arrays :
            - time_array : list of time slice indices extracted from the list of controls by the scheduler.
            - ctrl_id_array : list of control sequence indices corresponding to the extracted time slices.

        from the C arrays, created after running the scheduler over a given scheduler period, to the 
        corresponding python arrays. This function allows the use of temporary dynamically allocated 
        arrays in C.

        Parameters
        ----------
        new_time_array_c : pointer
            pointer to the C ``time_array``.
        new_ctrl_id_array_c : pointer
            pointer to the C ``ctrl_id_array``.
        size : int
            Number of time slices extracted.
        new_time_array : numpy.ndarray of int
            rray o arrays of indices corresponding to the indices of the time slices which will be extracted from the 
            list of controls in order to create the final control sequence.   
        new_ctrl_id_array : numpy.ndarray of int
            Array of indices corresponding to the indices the control sequence which time slice which will 
            be extracted in order to create the final control sequence.

        Returns
        -------
        None
        '''
        if size == 0:
            new_time_array = np.array([], dtype=int)
            new_ctrl_id_array = np.array([], dtype=int)
        else:
            # copy new_time_array
            buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
            buffer_from_memory.restype = ctypes.py_object
            buffer = buffer_from_memory(new_time_array_c, np.dtype(np.int32).itemsize*size)

            new_time_array[:] = np.frombuffer(buffer, np.int32).astype(int)

            # copy new_ctrl_id_array
            buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
            buffer_from_memory.restype = ctypes.py_object
            buffer = buffer_from_memory(new_ctrl_id_array_c, np.dtype(np.int32).itemsize*size)

            new_ctrl_id_array[:] = np.frombuffer(buffer, np.int32).astype(int)


    def log_scheduler_performances(self, final_control_sequence):   
        ''' Logs the performances of the given scheduled control sequence.

        This function prints the performances of the scheduling process :
         -  The theoretical uptime of each control sequence [s]
         -  The real (or effective) uptime of each control sequence [s]
         -  The schedule success rate [%]

        Parameters
        ----------
        final_control_sequence : :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            Scheduled control sequence which quality we want to evaluate.

        Returns
        -------
        None

        Raises
        ------
        ValueError :
            If the control sequence hasn't been scheduled by a :class:`StaticPriorityScheduler` instance.
        '''     
        if not isinstance(final_control_sequence.meta['scheduler'], self.__class__):
            raise ValueError('the final control sequence provided has not been scheduled using the StaticPriorityScheduler.') 

        data = []

        print("")

        t_start     = final_control_sequence.t[0][0]
        t_end       = final_control_sequence.t[-1][-1]
        controls    = final_control_sequence.meta["scheduled_controls"]

        for ctrl_id in range(len(controls)):
            control_uptime_th       = 0.0
            control_time_point_ids  = 0.0
            control_uptime_real     = 0.0

            controller_type = controls[ctrl_id].meta["controller_type"]

            for scheduler_period_id in range(len(final_control_sequence.t)):
                control_period_id = controls[ctrl_id].get_control_period_id(scheduler_period_id)

                if control_period_id != -1:
                    msk = np.logical_and(controls[ctrl_id].t[control_period_id] >= t_start, controls[ctrl_id].t[control_period_id] <= t_end)
                    control_uptime_th += np.sum(controls[ctrl_id].t_slice[control_period_id][msk])

                    control_time_point_ids = np.where(final_control_sequence.active_control[scheduler_period_id] == ctrl_id)[0]
                    control_uptime_real += np.sum(final_control_sequence.t_slice[scheduler_period_id][control_time_point_ids])

            succes_rate = control_uptime_real/control_uptime_th*100
            data.append([ctrl_id, controller_type, controls[ctrl_id].priority[0][0], control_uptime_th, control_uptime_real, succes_rate])
            
        header = ['Control index\n[-]', 'Controller type\n[-]', 'Priority\n[-]', 'Theoretical uptime\n[s]', 'Real uptime\n[s]', 'Success rate\n[%]']
        tab = str(tabulate(data, headers=header, tablefmt="presto", numalign="center", floatfmt="5.2f"))

        width = tab.find('\n', 0)

        str_ = f'{" scheduler performance analysis ".center(width, "-")}\n'
        str_ += tab + '\n'
        str_ += '-'*width + '\n'

        print(str_)


    def check_priority(self, controls):
        ''' Checks and corrects (if feasable) the control priorities to be scheduled.

        The static priority scheduler requires for the priority of each control sequence to be
        arrays or scalar values of positive ordered integers. If some priorities are None, 
        they will automatically set by using a FIFO algorithm. 
        
        Parameters
        ----------
        controls : list / numpy.ndarray of :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            list of control sequences which are to be scheduled.
        
        Returns
        -------
        None

        Raises
        ------
        ValueError :
            If the values are not integers or arrays of the same shape as the time array of the corresponding
            control sequence.
        ''' 
        # convert priorities to arrays if not already done
        for ctrl_id, control in enumerate(controls):
            if control.priority is None:
                control.priority = ctrl_id + len(controls) # put the control at the end in FIFO mode

            if not isinstance(control.priority, int) and not isinstance(control.priority, np.ndarray):
                raise ValueError("Control priorities must be arrays of shape (n_periods, n_points_per_periods) or integers.")

            # if the priority of a control is given as an array
            if isinstance(control.priority, np.ndarray):
                error_flag = False
                # check if shape corresponds to the control periods and number of time slices
                if len(control.priority) != control.n_periods and len(control.priority) != control.n_control_points:
                    error_flag = True
                elif len(control.priority) == control.n_control_points:
                    control.priority = control.split_array(control.priority)
                else:
                    for period_id in range(control.n_periods):
                        if np.size(control.priority[period_id]) != np.size(control.t[period_id]):
                            error_flag = True
                            break
                if error_flag is True:
                    raise ValueError("The control priority must be of the same shape as the time array.")
            else:
                # if not given as an array, priorities are set to be the same shape as the control / time arrays
                control.priority = control.split_array(np.repeat(control.priority, control.n_control_points))