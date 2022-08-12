from abc import ABC, abstractmethod
import numpy as np

import ctypes

from sorts import clibsorts

class RadarSchedulerBase(ABC):
    ''' Defines the fundamental structure of a radar Scheduler.

    The role of a scheduler is to schedule the time slices which will 
    be run by the radar system by satisfying a set of constraints. The
    generated control sequence can then be used to control a radar system.

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
    '''
    def __init__(self, radar, t0, scheduler_period, logger=None, profiler=None):
        self.profiler = profiler
        ''' Profiler instance used to monitor the computation performances of the class methods. '''
        self.logger = logger
        ''' Logger instance used to log the compuation status of the class methods. '''
        self.radar = radar
        ''' Radar instance being controlled. '''
        self._t0 = t0
        ''' Start time of the scheduler with respect to the reference time of the
        simulation (in seconds). 
        '''
        self._scheduler_period = scheduler_period
        ''' Duration of a scheduler period (in seconds). 

        The input :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>` 
        control periods must coincide with the scheduler periods (see :ref:`radar_controls` 
        to obtain more information about control period synchronization).
        '''

        if self.logger is not None:
            self.logger.info(f"RadarSchedulerBase:init -> setting scheduling start time t0={t0}")

            if self._scheduler_period is not None:
                self.logger.info(f"RadarSchedulerBase:init -> setting scheduling period : scheduler_period={scheduler_period}")        
            else:
                self.logger.info("RadarSchedulerBase:init -> ignoring scheduling period...")   


    @abstractmethod
    def run(self, controls):
        ''' Runs the control scheduler algorithm to obtain the final RADAR control 
        sequence sent to the RADAR.

        This method must generate a new control sequence containing non-overlapping 
        time slices. Those time slices must be extracted from the list of radar controls
        which are to be scheduled. The scheduler shall also implement the constraints 
        relative to the choice of time slices to run.

        Parameters
        ----------
        controls : numpy.ndarray / list
            Array of RADAR controls to be managed. The algorithm will arrange the time 
            slices from those controls to create a control sequence compatible with the 
            RADAR system.

        Returns
        -------
        final_control_sequence : :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            Final RADAR control sequence compatible with the RADAR system.

        Examples
        --------
        This simple example implementation returns a combination of all the control sequences
        (in order of arrival).

        .. code-block:: Python

                # TODO Test 

                def run(self, controls):
                    time_slice_start_time   = np.ndarray(0, dtype=float)
                    time_slice_duration     = np.ndarray(0, dtype=float)
                    controls_id             = np.ndarray(0, dtype=int)

                    for ctrl_id, ctrl in controls:
                        controls_id         = np.append(controls_id, np.full(ctrl.n_control_points, ctrl_id, int))

                        for period_id in range(ctrl.n_periods):
                            time_slice_start_time   = np.append(time_slice_start_time, ctrl.t[period_id])
                            time_slice_duration     = np.append(time_slice_duration, ctrl.t_slice[period_id])
                    
                    final_control_sequence = sorts.RadarControls(self.radar, None, scheduler=self, priority=None)
                    final_control_sequence.set_time_slices(time_slice_start_time, time_slice_duration)

                    final_control_sequence.active_control = final_control_sequence.split_array(controls_id)
                    final_control_sequence.meta["scheduled_controls"] = controls

                    return self.extract_control_sequence(controls, final_control_sequence)         
        '''
        pass


    def generate_schedule(self, control_sequence):
        ''' Extracts the results of the scheduler from the generated control sequence. '''
        pass

    @property
    def scheduler_period(self):
        ''' Scheduler period time (in seconds). '''
        return self._scheduler_period


    @scheduler_period.setter
    def scheduler_period(self, val):
        ''' Scheduler period time (in seconds). '''
        try:
            val = float(val)
        except:
            raise ValueError("The scheduler period has to be a number (int/float)")

        self._scheduler_period = val

        if self.logger is not None:
            self.logger.info(f"RadarSchedulerBase:scheduler_period:setter -> setting scheduling period : scheduler_period={val}")        


    @property
    def t0(self):
        ''' Scheduler start time (in seconds). '''
        return self._t0
    

    @t0.setter
    def t0(self, val):
        ''' Scheduler start time (in seconds). '''
        try:
            val = float(val)
        except:
            raise ValueError("The scheduler start time has to be a number (int/float)")
        
        self._t0 = val
    
        if self.logger is not None:
            self.logger.info(f"RadarSchedulerBase:t0:setter -> setting scheduling start time : t0={val}")     


    def extract_control_sequence(self, controls, final_control_sequence):
        ''' Extracts the scheduled control at specific time points and control indices after the time slices 
        have been scheduled.

        This function extracts the pointing directions and property controls corresponding to the time slices 
        scheduled by the radar scheduler from the list of radar control sequences. This step is essential after
        completing the time slice scheduling to ensure that the generated control sequence possesses the right 
        property and pointing direction controls.

        The algorithm goes through each scheduled time slice and gets the all the pointing and property controls 
        from corresponding control structure (i.e. the one which has generated time slice chosen by the scheduler).

        Parameters
        ----------
        controls : list / numpy.ndarray of :class:`sorts.RadarControllers<sorts.radar.radar_controls.RadarControls>`
            List of radar control sequences being scheduled.
        final_control_sequence : :class:`sorts.RadarControllers<sorts.radar.radar_controls.RadarControls>`
            Final radar control sequence generated by the scheduler.

        Returns
        -------
        final_control_sequence : :class:`sorts.RadarControllers<sorts.radar.radar_controls.RadarControls>`
            Updated 
        '''
        # extract pointing directions (as splitted arrays)
        final_control_sequence.pdirs = self.get_pointing_direction_sequence(controls, final_control_sequence)
        final_control_sequence.has_pdirs = True

        # extract all other control properties
        controlled_properties = dict()
        controlled_properties["tx"] = [[] for rxi in range(len(controls[0].radar.tx))]
        controlled_properties["rx"] = [[] for txi in range(len(controls[0].radar.rx))]


        # get all radar properties being controlled by the different controls
        for control in controls:
            for station_type in controlled_properties.keys():
                stations = getattr(control.radar, station_type)

                for station_id, station in enumerate(stations):
                    station_controls = control.get_property_control_list(station)

                    for property_name in station.PROPERTIES:
                        if property_name not in controlled_properties[station_type][station_id] and property_name in station_controls:
                            controlled_properties[station_type][station_id].append(property_name)


        # copy all controls to new control sequence
        for station_type in controlled_properties.keys():
            stations = getattr(control.radar, station_type) # get all the stations of type tx/rx in self.radar

            # copy the controls for each station
            for sid, station in enumerate(stations):
                # for each property being controlled by at least one control
                for property_name in controlled_properties[station_type][sid]: 
                    tmp_final_control_data = np.ndarray((final_control_sequence.n_periods,), dtype=object)

                    for period_id in range(control.n_periods):
                        if final_control_sequence.t[period_id] is None:
                            tmp_final_control_data[period_id] = None
                            continue

                        tmp_final_control_data[period_id] = np.ndarray((len(final_control_sequence.t[period_id]),), dtype=control.property_controls[period_id][station_type][property_name][sid].dtype)

                        # get control values for each controls
                        for ctrl_id, control in enumerate(controls):   
                            mask = (final_control_sequence.active_control[period_id] == ctrl_id)
                            if property_name in control.get_property_control_list(station):
                                controls_period_id = control.get_control_period_id(period_id)
                                inds = np.intersect1d(control.t[controls_period_id], final_control_sequence.t[period_id][mask], return_indices=True)
                                control_point_ids = inds[1]
                                del inds
                            
                                tmp_final_control_data[period_id][mask] = control.property_controls[controls_period_id][station_type][property_name][sid][control_point_ids]
                            else:
                                # get default value
                                tmp_final_control_data[period_id][mask] = getattr(station, property_name)
                
                    final_control_sequence.add_property_control(property_name, station, tmp_final_control_data)

        return final_control_sequence


    def get_pointing_direction_sequence(self, controls, final_control_sequence):
        ''' Extracts the pointing directions at specific time points and control indices.

        This function extracts the pointing directions corresponding to the time slices scheduled
        by the radar scheduler from the list of radar control sequences.

        Parameters
        ----------
        controls : list / numpy.ndarray of :class:`sorts.RadarControllers<sorts.radar.radar_controls.RadarControls>`
            List of radar control sequences being scheduled.
        final_control_sequence : :class:`sorts.RadarControllers<sorts.radar.radar_controls.RadarControls>`
            Final radar control sequence generated by the scheduler.

        Returns
        -------
        pointing_direction : dict
            Data structure containing the array of scheduled pointing directions (see :attr:`RadarController.compute_pointing_directions
            <sorts.radar.radar_controls.RadarControls.compute_pointing_directions>` for more information about the pointing direction 
            data structure).
        '''
        # get number of Rx/Tx stations
        n_tx = len(controls[0].radar.tx)
        n_rx = len(controls[0].radar.rx)

        pointing_direction = np.ndarray((final_control_sequence.n_periods,), dtype=object)

        # gather all pointing directions
        for period_id in range(len(final_control_sequence.t)):
            if final_control_sequence.t[period_id] is None:
                 pointing_direction[period_id] = None
                 continue

            # intialization of the pointing direction sequence

            # allocate memory
            pointing_direction[period_id]       = dict()
            pointing_direction[period_id]["t"]  = np.array([], dtype=np.float64)
            control_t_ids   = np.array([], dtype=np.int32)
            control_ids     = np.array([], dtype=np.int32)


            # callback function to save arrays
            def save_arrays(array_pdir_time_points_c, arrays_control_ids_c, arrays_control_t_ids_c, size):
                ''' Save pointing direction arrays (time, control index, time index) from the C library. '''
                nonlocal pointing_direction, control_ids, control_t_ids, period_id

                if size > 0:
                    pointing_direction[period_id]["t"] = np.ndarray((size,), dtype=np.float64)
                    control_ids = np.ndarray((size,), dtype=np.int32)
                    control_t_ids = np.ndarray((size,), dtype=np.int32)

                    # arrays control ids
                    buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
                    buffer_from_memory.restype = ctypes.py_object
                    buffer = buffer_from_memory(arrays_control_ids_c, np.dtype(np.int32).itemsize*size)
                    control_ids[:] = np.frombuffer(buffer, np.int32)

                    # arrays time ids
                    buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
                    buffer_from_memory.restype = ctypes.py_object
                    buffer = buffer_from_memory(arrays_control_t_ids_c, np.dtype(np.int32).itemsize*size)
                    control_t_ids[:] = np.frombuffer(buffer, np.int32)

                    # array time points
                    buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
                    buffer_from_memory.restype = ctypes.py_object
                    buffer = buffer_from_memory(array_pdir_time_points_c, np.dtype(np.float64).itemsize*size)
                    pointing_direction[period_id]["t"][:] = np.frombuffer(buffer, np.float64)

            SAVE_ARRAYS = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int)
            save_arrays_c = SAVE_ARRAYS(save_arrays)

            first_iteration = True
            controls_pdirs = []
            
            # initializes the pointing direction arrays
            for ctrl_id, control in enumerate(controls):
                if control.has_pdirs is True:
                    # compute the corresponding control period
                    ctrl_period_id = control.get_control_period_id(period_id)

                    # if there is indeed a sliced control time array of index "ctrl_period_id"
                    if ctrl_period_id != -1:
                        # get the pointing directions of the current control
                        controls_pdirs.append(control.get_pdirs(ctrl_period_id))

                        clibsorts.get_control_sequence_time_indices.argtypes = [
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=final_control_sequence.t[period_id].ndim, shape=final_control_sequence.t[period_id].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=final_control_sequence.t_slice[period_id].ndim, shape=final_control_sequence.t_slice[period_id].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=pointing_direction[period_id]["t"].ndim, shape=pointing_direction[period_id]["t"].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=control.t[ctrl_period_id].ndim, shape=control.t[ctrl_period_id].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=controls_pdirs[-1]["t"].ndim, shape=controls_pdirs[-1]["t"].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=control_ids.ndim, shape=control_ids.shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=control_t_ids.ndim, shape=control_t_ids.shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=final_control_sequence.active_control[period_id].ndim, shape=final_control_sequence.active_control[period_id].shape),
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int,
                            SAVE_ARRAYS,
                        ]

                        print("t = ", final_control_sequence.t[period_id])
                        print("t_slice = ", final_control_sequence.t_slice[period_id])
                        print("active_control = ", final_control_sequence.active_control[period_id])
                        print("t_dirs = ", controls_pdirs[-1]["t"])
                        print("pointing_direction[period_id][t] = " , pointing_direction[period_id]["t"])

                        clibsorts.get_control_sequence_time_indices(
                            final_control_sequence.t[period_id],
                            final_control_sequence.t_slice[period_id],
                            pointing_direction[period_id]["t"],
                            control.t[ctrl_period_id].astype(float),
                            controls_pdirs[-1]["t"].astype(float),
                            control_ids,
                            control_t_ids,
                            final_control_sequence.active_control[period_id].astype(np.int32),
                            ctypes.c_int(ctrl_id),
                            ctypes.c_int(len(control.t[ctrl_period_id])),
                            ctypes.c_int(len(controls_pdirs[-1]["t"])),
                            ctypes.c_int(len(final_control_sequence.t[period_id])),
                            ctypes.c_int(len(control_ids)),
                            ctypes.c_int(first_iteration),
                            save_arrays_c,
                        )

                        first_iteration = False

            # create final pointing direction array
            n_directions = len(pointing_direction[period_id]["t"])
            pointing_direction[period_id]['tx'] = np.ndarray((n_tx, 1, 3, n_directions), dtype=float)
            pointing_direction[period_id]['rx'] = np.ndarray((n_rx, n_tx, 3, n_directions), dtype=float)

            # get directions from individual control arrays
            for ctrl_id in range(len(controls)):
                if controls[ctrl_id].has_pdirs is True:
                    msk = control_ids == ctrl_id
                    print(msk)
                    
                    if len(np.where(msk)[0]) > 0:
                        control_pdir = controls_pdirs.pop(0)
                        print(control_pdir)
                        
                        for txi in range(n_tx):
                            pointing_direction[period_id]['tx'][txi, 0, :, msk] = control_pdir["tx"][txi, 0, :, control_t_ids[msk]]

                            for rxi in range(n_rx):
                                pointing_direction[period_id]['rx'][rxi, txi, :, msk] = control_pdir["rx"][rxi, txi, :, control_t_ids[msk]]

        return pointing_direction