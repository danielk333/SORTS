import numpy as np
from tabulate import tabulate
import ctypes

from .base import RadarControlManagerBase
from sorts import clibsorts

from ..controllers import radar_controller




class SimpleRadarControl(RadarControlManagerBase):
    def __init__(self, radar, t0, manager_period, logger=None, profiler=None):
        super().__init__(radar, t0, manager_period, logger=logger, profiler=profiler)

    def run(self, controls, t_end=None, log_performances=True):
        ''' 
        This function is used to create a single radar control set from a set of multiple controls which is compatible with the control priorities, time points and time slices.
        The final control set will be free of overlapping time slices. 

        As an example, if one wants to execute multiple control types at the same time (scanning, tracking, ...)
        then one needs to generate all the wanted controls independantly, and then call the manager over the list of generated controls to get a new control set 
        which corresponds to the fusion (weighted with respect to the priority of each control subsets) of all the previous controls. 

        Parameters :
        ------------
            controls : list/numpy.ndarray
                List or array of controls to be managed. Those controls can have been generated from different kinds of controllers.
                The manager will extract the final controls and time points depending on the priority of each control. To do so, the manager will 
                discard any overlapping time slices and will only keep the time slice of highest priority.
            t_end (optional) : int
                end time of the manager. If not set, the end time will correspond to the lastest time point of the control time arrays 

        Return value :
        --------------
            As stated before, this function returns a single set of controls which can be executed by the radar. 

        Example :
        ---------
            Let us take the example of a RADAR measurement campain where multiple scientist want to use the same RADAR system to conduct different observations.
            Each scientist will require a different set of controls, which aren't necessarily compatible with each other.

            The sorts.manager moduler can be used to solve this problem :
             - Each scientist can generate the controls he wants to perform with the radar (i.e. tracking of a meteoroid, a random scan each 10s, ...)
             - A unique priority can be affected to each control
             - The different controls are stacked into a single array
             - Finally, one can call Manager.run function to get a final control sequence which (hopefully) will satisfy the requirements of each scientist

              

        '''

        # Check input values   
        controls=np.asarray(controls, dtype=object) # convert control list to an np.array of objects  
        if t_end is None: # get the max end time of the controls if t_end is not given 
            t_end = 0

            for ctrl_id, ctrl in enumerate(controls):
                t_last_i = controls[ctrl_id]["t"][-1][-1]
                if t_last_i > t_end: t_end = t_last_i

                if ctrl["priority"] is None:
                    # TODO Modify FIFO priority allocation
                    ctrl["priority"] = ctrl_id*np.ones(len(ctrl["t"])) # FIFO used if no priority set

            del t_last_i

        # Initialization
        self.__setup_c_callback_functions(controls) # intialize C callback functions 

        # save functions to a C library struct for easy access during schedule computations
        clibsorts.init_manager.argtypes = self.C_FNC_TYPES
        clibsorts.init_manager(*self.c_fnc_pointers)

        # compute number of manager periods 
        manager_period_count = int((t_end - self.t0)//self._manager_period) + 1
        t_start_i = self.t0

        # initializes results
        final_control_sequence = dict()
        final_control_sequence['t'] = [None]*manager_period_count
        final_control_sequence['t_slice'] = [None]*manager_period_count
        final_control_sequence['active_control'] = [None]*manager_period_count

        # manages the controls for each manager period 
        for manager_period_id in range(manager_period_count):
            t_end_i = t_start_i + self._manager_period # compute the end time of the manager period

            self.new_time_array = []
            self.new_ctrl_id_array = []

            # initialize library call funtion
            clibsorts.run_manager.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double]
            clibsorts.run_manager.restype = ctypes.c_double

            # run clib function -> returns the position of the time cursor t_start_i
            t_start_i = clibsorts.run_manager(
                ctypes.c_int(manager_period_id),
                ctypes.c_int(len(controls)),
                ctypes.c_double(t_start_i), 
                ctypes.c_double(t_end_i))

            # get manager arrays
            time_array, time_slices = self.__get_control_sequence(controls, manager_period_id)
            
            # save manager arrays
            final_control_sequence['t'][manager_period_id] = time_array
            final_control_sequence['t_slice'][manager_period_id] = time_slices
            final_control_sequence['active_control'][manager_period_id] = self.new_ctrl_id_array

        # convert each output to a numpy array
        final_control_sequence['t'] = np.asarray(final_control_sequence['t'])
        final_control_sequence['t_slice'] = np.asarray(final_control_sequence['t_slice'])
        final_control_sequence['active_control'] = np.asarray(final_control_sequence['active_control'], dtype=object)

        if log_performances is True and self.logger is not None:
            self.__log_manager_performances(controls, final_control_sequence)

        return self.__extract_control_sequence(controls, final_control_sequence)
    
    def __setup_c_callback_functions(self, controls):
        ''' 
        Sets up all the callback functions needed to interact properly with the manager C library.
        '''

        # C CALLBACK FUNCTIONS
        def get_control_period_id_callback(control_id, manager_period_id):
            ''' 
            Gets the active time control subarray associated with a given manager period ID  
            '''
            nonlocal controls
            return self.__get_control_period_id(controls, control_id, manager_period_id)

        GET_CONTROL_PERIOD_ID_FNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
        get_control_period_id_c = GET_CONTROL_PERIOD_ID_FNC(get_control_period_id_callback)

        def get_control_array_size_callback(control_id, control_period_id):
            ''' 
            Gets the number of time points in a given control time subarray
            '''
            nonlocal controls
            return self.__get_control_array_size(controls, control_id, control_period_id)

        GET_CONTROL_ARRAY_SIZE_FNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
        get_control_array_size_c = GET_CONTROL_ARRAY_SIZE_FNC(get_control_array_size_callback)

        def get_control_parameters_callback(control_id, control_period_id, index, t, t_slice, priority):
            ''' 
            Copies the priority, time and t_slice of a given control, time index and control period (which corresponds to a control sliced time subarray) from Python to C pointers.
            This function is used to prevent sending the whole time data to the library at once.
            '''
            nonlocal controls

            if control_period_id >= len(controls[control_id]) or control_period_id == -1: # no controls available in given control period
                return -1
            else:
                # assigns the control parameters to the C pointers
                t[0] = controls[control_id]["t"][control_period_id][index]
                t_slice[0] = controls[control_id]["t_slice"][control_period_id][index]
                priority[0] = controls[control_id]["priority"][control_period_id][index]

                return 1

        GET_CONTROL_PARAMETERS_FNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int))
        get_control_parameters_c = GET_CONTROL_PARAMETERS_FNC(get_control_parameters_callback)

        def get_control_arrays_callback(control_id, control_period_id, time_array_c, t_slice_c, priority_c):
            ''' 
            Copies the priority, time and t_slice arrays of a given control and control period (which corresponds to a control sliced time subarray) from Python to C allocated array.
            This function is used to prevent sending the whole time data to the library at once.
            '''
            nonlocal controls

            doublep = ctypes.POINTER(ctypes.c_double)
            intp = ctypes.POINTER(ctypes.c_int)

            # get new time points 
            time_array = controls[control_id]["t"][control_period_id]
            time_array_c_arr = np.ctypeslib.as_array(time_array_c, (len(time_array),))
            time_array_c_arr[:] = time_array
            del time_array
            
            # get new time slices 
            t_slice = controls[control_id]["t_slice"][control_period_id]
            t_slice_c_arr = np.ctypeslib.as_array(t_slice_c, (len(t_slice),))
            t_slice_c_arr[:] = t_slice
            del t_slice
            
            # get new priority
            priority = controls[control_id]["priority"][control_period_id]
            priority_c_arr = np.ctypeslib.as_array(priority_c, (len(priority),))
            priority_c_arr[:] = priority
            del priority

        GET_CONTROL_ARRAYS_FNC = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int))
        get_control_arrays_c = GET_CONTROL_ARRAYS_FNC(get_control_arrays_callback)

        def save_new_control_arrays_callback(new_time_array_c, new_ctrl_id_array_c, size):
            ''' 
            Allows Python to copy the control arrays :
                - time_array
                - ctrl_id_array

            from the C arrays, created after running the manager over a given manager period, to the corresponding python arrays.
            This function allows the use of temporary dynamically allocated arrays in C.
            '''
            self.__save_new_control_arrays(new_time_array_c, new_ctrl_id_array_c, size)

        SAVE_NEW_CTRL_ARRAY_FNC = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int)
        save_new_control_arrays_c = SAVE_NEW_CTRL_ARRAY_FNC(save_new_control_arrays_callback)

        self.C_FNC_TYPES = (GET_CONTROL_PERIOD_ID_FNC, GET_CONTROL_ARRAY_SIZE_FNC, GET_CONTROL_PARAMETERS_FNC, GET_CONTROL_ARRAYS_FNC, SAVE_NEW_CTRL_ARRAY_FNC)
        self.c_fnc_pointers = (get_control_period_id_c, get_control_array_size_c, get_control_parameters_c, get_control_arrays_c, save_new_control_arrays_c)


    def __get_control_sequence(self, controls, period_id):
        ''' 
        Extract the control time and time_slice arrays from the glogal controls array.

        This function is called after having conputed the active control and time point indices (arrays of int) over a given manager period to retreive
        the actual time points and time slices.
        '''

        # initializes the output arrays
        time_array = np.empty(len(self.new_ctrl_id_array), dtype=float)
        time_slices = np.empty(len(self.new_ctrl_id_array), dtype=float)
        
        for i in range(len(self.new_ctrl_id_array)):
            # compute the sliced time array id for the given controller
            ctrl_period = period_id - int(controls[self.new_ctrl_id_array[i]]['t'][0][0]/self._manager_period)

            # copies the values contained inside the controls array to the new arrays
            time_array[i] = controls[self.new_ctrl_id_array[i]]["t"][ctrl_period][self.new_time_array[i]]
            time_slices[i] = controls[self.new_ctrl_id_array[i]]["t_slice"][ctrl_period][self.new_time_array[i]]
                                    
        return time_array, time_slices

    def __get_control_period_id(self, controls, control_id, manager_period_id):
        ''' 
        Computes the control period associated with a given manager period.
        If there is no control if index 'control_id' at the given control period, then the function will return -1
        '''
        if control_id >= len(controls): # control_id is bigger than the number of controls in the given control structure 
            ctrl_period = -1
        else:
            ctrl_period = manager_period_id - int(controls[control_id]['t'][0][0]/self.manager_period) # computes the time subarray id

            if ctrl_period < 0 or ctrl_period >= len(controls[control_id]['t']): # the time subarray id is bigger than the number of time subarrays in the given control structure
                ctrl_period = -1

        return ctrl_period

    def __get_control_array_size(self, controls, control_id, control_period_id):
        ''' 
        Gets the number of time points in a given control time subarray
        '''
        if control_period_id == -1:
            time_array_size = 0;
        else:
            time_array_size = np.size(controls[control_id]["t"][control_period_id])

        return time_array_size 

    def __save_new_control_arrays(self, new_time_array_c, new_ctrl_id_array_c, size):
        ''' 
        Allows Python to copy the control arrays :
            - time_array
            - ctrl_id_array

        from the C arrays, created after running the manager over a given manager period, to the corresponding python arrays.
        This function allows the use of temporary dynamically allocated arrays in C.
        '''
        if size == 0:
            self.new_time_array = np.array([], dtype=int)
            self.new_ctrl_id_array = np.array([], dtype=int)
        else:
            # copy new_time_array
            buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
            buffer_from_memory.restype = ctypes.py_object
            buffer = buffer_from_memory(new_time_array_c, np.dtype(np.int32).itemsize*size)

            self.new_time_array = np.frombuffer(buffer, np.int32).astype(int)

            # copy new_ctrl_id_array
            buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
            buffer_from_memory.restype = ctypes.py_object
            buffer = buffer_from_memory(new_ctrl_id_array_c, np.dtype(np.int32).itemsize*size)

            self.new_ctrl_id_array = np.frombuffer(buffer, np.int32).astype(int)

    def __extract_control_sequence(self, controls, final_control_sequence):
        ''' 
        Extracts the control data at the given time points and control ids
        '''
        control_list = []

        # get all control variables
        for control in controls:
            for control_variable in control.keys():
                if control_variable not in radar_controller.NON_CONTROL_FIELDS and control_variable not in control_list:
                    control_list.append(control_variable)


        final_control_sequence["controls"] = dict()
        for control_variable in control_list:

            if control_variable == "pointing_direction":
                final_control_sequence["controls"][control_variable] = self.__get_pointing_direction_sequence(controls, final_control_sequence)

        return final_control_sequence

    def __get_pointing_direction_sequence(self, controls, final_control_sequence):
        ''' 
        Get the pointing directions at specific time points and control ids
        '''
        for period_id in range(len(final_control_sequence["t"])):
            pointing_direction = dict()

            # get number of Rx/Tx stations
            n_tx = len(controls[0]["meta"]["radar"].tx)
            n_rx = len(controls[0]["meta"]["radar"].rx)

            # initializes the pointing direction arrays
            pointing_direction["tx"] = np.ndarray((n_tx, 1, len(final_control_sequence["t"][period_id])), dtype=object)
            pointing_direction["rx"] = np.ndarray((n_rx, n_tx, len(final_control_sequence["t"][period_id])), dtype=object)

            for ctrl_id, control in enumerate(controls):
                if "pointing_direction" in control.keys():
                    # compute the corresponding control period
                    ctrl_period_id = self.__get_control_period_id(controls, ctrl_id, period_id)

                    # if there is indeed a sliced control time array of index "ctrl_period_id"
                    if ctrl_period_id != -1:
                        # get ids of time points which active control corresponds to the current control id
                        ids = np.where(final_control_sequence["active_control"][period_id] == ctrl_id)[0]
                        ids_t = np.zeros(len(ids), dtype=int)

                        # get ids of time points which exist both in the final_control_sequence array and the current control array 
                        for i, id_ in enumerate(ids):
                            for j in range(len(control["t"][ctrl_period_id])):
                                if final_control_sequence["t"][period_id][id_] == control["t"][ctrl_period_id][j]:
                                    ids_t[i] = j
                                    break
                        
                        # get the pointing directions of the current control
                        pdirs = next(control["pointing_direction"])

                        # copy the directions for each tx and rx station
                        for txi in range(n_tx):
                            for i in range(len(ids_t)):
                                pointing_direction["tx"][txi][0][ids[i]] = pdirs["tx"][txi][0][ids_t[i]]

                        for rxi in range(n_rx):
                            for txi in range(n_tx):
                                for i in range(len(ids_t)):
                                    pointing_direction["rx"][rxi][txi][ids[i]] = pdirs["rx"][rxi][txi][ids_t[i]]

            yield pointing_direction

    def __log_manager_performances(self, controls, final_control_sequence):        
        data = []

        print("")
        self.logger.info("Logging manager performance analysis :")

        for ctrl_id in range(len(controls)):
            control_uptime_th = 0.0
            control_time_point_ids = 0.0
            control_uptime_real = 0.0

            for manager_period_id in range(len(final_control_sequence["t"])):
                control_period_id = self.__get_control_period_id(controls, ctrl_id, manager_period_id)

                if control_period_id != -1:
                    control_uptime_th += np.sum(controls[ctrl_id]["t_slice"][control_period_id])

                    control_time_point_ids = np.where(final_control_sequence["active_control"][manager_period_id] == ctrl_id)[0]
                    control_uptime_real += np.sum(final_control_sequence["t_slice"][manager_period_id][control_time_point_ids])

            control_duty_cycle = control_uptime_real/control_uptime_th*100
            data.append([ctrl_id, controls[ctrl_id]["priority"][0][0], control_uptime_th, control_uptime_real, control_duty_cycle])
            
        header = ['Control index\n[-]', 'Priority\n[-]', 'Theoretical uptime\n[s]', 'Real uptime\n[s]', 'Duty cycle\n[%]']
        tab = str(tabulate(data, headers=header, tablefmt="presto", numalign="center", floatfmt="5.2f"))

        width = tab.find('\n', 0)

        str_ = f'{" Manager performance analysis ".center(width, "-")}\n'
        str_ += tab + '\n'
        str_ += '-'*width + '\n'

        print(str_)

