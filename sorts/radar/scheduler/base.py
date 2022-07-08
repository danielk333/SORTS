from abc import ABC, abstractmethod
import numpy as np

import ctypes

from sorts import clibsorts

class RadarSchedulerBase(ABC):
    def __init__(self, radar, t0, scheduler_period, logger=None, profiler=None):
        self.profiler = profiler
        self.logger = logger

        self.radar = radar
        self._t0 = t0
        self._scheduler_period = scheduler_period

        if self.logger is not None:
            self.logger.info(f"RadarSchedulerBase:init -> setting scheduling start time t0={t0}")

            if self._scheduler_period is not None:
                self.logger.info(f"RadarSchedulerBase:init -> setting scheduling period : scheduler_period={scheduler_period}")        
            else:
                self.logger.info("RadarSchedulerBase:init -> ignoring scheduling period...")   


    @abstractmethod
    def run(self, controls):
        '''Runs the control scheduler algorithm to obtain the final RADAR control sequence sent to the RADAR.

        Parameters
        ----------

        controls : np.ndarray/list
        Array of RADAR controls to be managed. The algorithm will arrange the time slices from those controls to create a control sequence compatible with the RADAR system.

        Return value
        ------------

        final_control_sequence : dict
        Final RADAR control sequence compatible with the RADAR system

        '''
        pass


    @property
    def scheduler_period(self):
        return self._scheduler_period


    @scheduler_period.setter
    def scheduler_period(self, val):
        try:
            val = float(val)
        except:
            raise ValueError("The scheduler period has to be a number (int/float)")

        self._scheduler_period = val

        if self.logger is not None:
            self.logger.info(f"RadarSchedulerBase:scheduler_period:setter -> setting scheduling period : scheduler_period={val}")        


    @property
    def t0(self):
        return self._t0
    

    @t0.setter
    def t0(self, val):
        try:
            val = float(val)
        except:
            raise ValueError("The scheduler start time has to be a number (int/float)")
        
        self._t0 = val
    
        if self.logger is not None:
            self.logger.info(f"RadarSchedulerBase:t0:setter -> setting scheduling start time : t0={val}")     


    def extract_control_sequence(self, controls, final_control_sequence):
        ''' 
        Extracts the control data at the given time points and control ids
        '''
        # extract pointing directions (as splitted arrays)
        final_control_sequence.pdirs = self.get_pointing_direction_sequence(controls, final_control_sequence)


        # extract all other control properties
        controlled_properties = dict()
        controlled_properties["tx"] = []
        controlled_properties["rx"] = []

        # get all radar properties being controlled by the different controls
        for control in controls:
            for station_type in controlled_properties.keys():
                stations = getattr(control.radar, station_type)

                for station in stations:
                    station_controls = control.get_property_control_list(station)

                    for property_name in station.PROPERTIES:
                        if property_name not in controlled_properties[station_type] and property_name in station_controls:
                            controlled_properties[station_type].append(property_name)

        
        # copy all controls to new control sequence
        for station_type in controlled_properties.keys():
            stations = getattr(control.radar, station_type) # get all the stations of type tx/rx in self.radar

            # for each property being controlled by at least one control
            for property_name in controlled_properties[station_type]:
                final_control_sequence.property_controls[station_type][property_name] = np.ndarray((len(stations),), dtype=object)
                
                # copy the controls for each station
                for sid, station in enumerate(stations):
                    tmp_final_control_data = np.ndarray((final_control_sequence.n_control_points,), dtype=np.float64)
                    tmp_final_control_data = final_control_sequence.split_array(tmp_final_control_data)

                    # get control values for each controls
                    for ctrl_id, control in enumerate(controls):   
                        station_controls = control.get_property_control_list(station)

                        for period_id in range(controls.n_periods):
                            mask = (final_control_sequence.active_control[period_id] == ctrl_id)

                            if property_name in station_controls:
                                inds = np.intersect1d(control.t[period_id], final_control_sequence.t[period_id][mask], return_indices=True)
                                control_point_ids = inds[1]
                                del inds
                            
                                tmp_final_control_data[period_id][mask] = control.property_controls[station_type][property_name][sid][period_id][control_point_ids]
                            else:
                                # get default value
                                tmp_final_control_data[period_id][mask] = getattr(station, property_name)
                
                    final_control_sequence.property_controls[station_type][property_name][sid] = tmp_final_control_data
                    del tmp_final_control_data


        return final_control_sequence


    def get_pointing_direction_sequence(self, controls, final_control_sequence):
        ''' 
        Get the pointing directions at specific time points and control ids
        '''
        # get number of Rx/Tx stations
        n_tx = len(controls[0].radar.tx)
        n_rx = len(controls[0].radar.rx)

        pointing_direction = np.ndarray((final_control_sequence.n_periods,), dtype=object)

        # gather all pointing directions
        for period_id in range(len(final_control_sequence.t)):
            # intialization of the pointing direction sequence

            # allocate memory
            pointing_direction[period_id] = dict()
            pointing_direction[period_id]["t"] = np.array([], dtype=np.float64)

            control_t_ids = np.array([], dtype=np.int32)
            control_ids = np.array([], dtype=np.int32)


            # callback function to save arrays
            def save_arrays(array_pdir_time_points_c, arrays_control_ids_c, arrays_control_t_ids_c, size):
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
                # compute the corresponding control period
                ctrl_period_id = control.get_control_period_id(period_id)

                # if there is indeed a sliced control time array of index "ctrl_period_id"
                if ctrl_period_id != -1:
                    # get the pointing directions of the current control
                    controls_pdirs.append(control.get_pdirs(ctrl_period_id))
                    print("keys, ", controls_pdirs[-1].keys())

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

            n_directions = len(pointing_direction[period_id]["t"])
            pointing_direction[period_id]['tx'] = np.ndarray((n_tx, 1, 3, n_directions), dtype=float)
            pointing_direction[period_id]['rx'] = np.ndarray((n_rx, n_tx, 3, n_directions), dtype=float)

            # get directions from individual control arrays
            ctrl_id = 0
            while(len(controls_pdirs) > 0):
                control_pdir = controls_pdirs.pop(0)
                msk = control_ids == ctrl_id
                ctrl_id += 1

                for txi in range(n_tx):
                    pointing_direction[period_id]['tx'][txi, 0, :, msk] = control_pdir["tx"][txi, 0, :, control_t_ids[msk]]

                    for rxi in range(n_rx):
                        pointing_direction[period_id]['rx'][rxi, txi, :, msk] = control_pdir["rx"][rxi, txi, :, control_t_ids[msk]]

        return pointing_direction