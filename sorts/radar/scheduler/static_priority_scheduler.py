import numpy as np
from tabulate import tabulate
import ctypes

from .base import RadarSchedulerBase
from sorts import clibsorts

from ..controllers import radar_controller
from .. import radar_controls


class StaticPriorityScheduler(RadarSchedulerBase):
    def __init__(self, radar, t0, scheduler_period, logger=None, profiler=None):
        super().__init__(radar, t0, scheduler_period, logger=logger, profiler=profiler)

    def run(self, controls, t_end=None, t_start=None, log_performances=True):
        ''' 
        This function is used to create a single radar control set from a set of multiple controls which is compatible with the control priorities, time points and time slices.
        The final control set will be free of overlapping time slices. 

        As an example, if one wants to execute multiple control types at the same time (scanning, tracking, ...)
        then one needs to generate all the wanted controls independantly, and then call the scheduler over the list of generated controls to get a new control set 
        which corresponds to the fusion (weighted with respect to the priority of each control subsets) of all the previous controls. 

        Parameters :
        ------------
            controls : list/numpy.ndarray
                List or array of controls to be managed. Those controls can have been generated from different kinds of controllers.
                The scheduler will extract the final controls and time points depending on the priority of each control. To do so, the scheduler will 
                discard any overlapping time slices and will only keep the time slice of highest priority.
            t_end (optional) : int
                end time of the scheduler. If not set, the end time will correspond to the lastest time point of the control time arrays 

        Return value :
        --------------
            As stated before, this function returns a single set of controls which can be executed by the radar. 

        Example :
        ---------
            Let us take the example of a RADAR measurement campain where multiple scientist want to use the same RADAR system to conduct different observations.
            Each scientist will require a different set of controls, which aren't necessarily compatible with each other.

            The sorts.scheduler moduler can be used to solve this problem :
             - Each scientist can generate the controls he wants to perform with the radar (i.e. tracking of a meteoroid, a random scan each 10s, ...)
             - A unique priority can be affected to each control
             - The different controls are stacked into a single array
             - Finally, one can call scheduler.run function to get a final control sequence which (hopefully) will satisfy the requirements of each scientist
        '''
        # Check input values   
        controls=np.asarray(controls, dtype=object) # convert control list to an np.array of objects  

        self.check_priority(controls)

        if t_end is None: # get the max end time of the controls if t_end is not given 
            t_end = 0

            for ctrl_id, ctrl in enumerate(controls):
                t_last_i = controls[ctrl_id].t[-1][-1]
                if t_last_i > t_end: t_end = t_last_i

        if t_start is None: # get the max end time of the controls if t_end is not given 
            t_start = 0

            for ctrl_id, ctrl in enumerate(controls):
                t_last_i = controls[ctrl_id].t[0][0]
                if t_last_i < t_start: t_start = t_last_i

        new_time_array = []
        new_ctrl_id_array = []

        # compute number of scheduler periods 
        scheduler_period_start = int((t_start - self.t0)//self._scheduler_period)
        scheduler_period_count = int(np.ceil((t_end - (t_start//self._scheduler_period)*self._scheduler_period)/self._scheduler_period))

        # Initialization
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

            # initialize library call funtion
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

        if log_performances is True:
            self.log_scheduler_performances(controls, final_control_sequence)

        return self.extract_control_sequence(controls, final_control_sequence)

    def __setup_c_callback_functions(self, controls, n_periods, new_time_array, new_ctrl_id_array):
        ''' 
        Sets up all the callback functions needed to interact properly with the scheduler C library.
        '''
        ptr_time_array_c = [[None]*len(controls)]*n_periods
        ptr_t_slice_c = [[None]*len(controls)]*n_periods
        ptr_priority_c = [[None]*len(controls)]*n_periods

        # C CALLBACK FUNCTIONS
        def get_control_period_id_callback(control_id, scheduler_period_id):
            ''' 
            Gets the active time control subarray associated with a given scheduler period ID  
            '''
            nonlocal controls
            return controls[control_id].get_control_period_id(scheduler_period_id)

        GET_CONTROL_PERIOD_ID_FNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
        get_control_period_id_c = GET_CONTROL_PERIOD_ID_FNC(get_control_period_id_callback)

        def get_control_parameters_callback(control_id, control_period_id, index, t, t_slice, priority):
            ''' 
            Copies the priority, time and t_slice of a given control, time index and control period (which corresponds to a control sliced time subarray) from Python to C pointers.
            This function is used to prevent sending the whole time data to the library at once.
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
            ''' 
            Copies the priority, time and t_slice arrays of a given control and control period (which corresponds to a control sliced time subarray) from Python to C allocated array.
            This function is used to prevent sending the whole time data to the library at once.
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
            ''' 
            Allows Python to copy the control arrays :
                - time_array
                - ctrl_id_array

            from the C arrays, created after running the scheduler over a given scheduler period, to the corresponding python arrays.
            This function allows the use of temporary dynamically allocated arrays in C.
            '''
            self.__save_new_control_arrays(new_time_array_c, new_ctrl_id_array_c, size, new_time_array, new_ctrl_id_array)

        SAVE_NEW_CTRL_ARRAY_FNC = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int)
        save_new_control_arrays_c = SAVE_NEW_CTRL_ARRAY_FNC(save_new_control_arrays_callback)

        C_FNC_TYPES = (GET_CONTROL_PERIOD_ID_FNC, GET_CONTROL_PARAMETERS_FNC, GET_CONTROL_ARRAYS_FNC, SAVE_NEW_CTRL_ARRAY_FNC)
        c_fnc_pointers = (get_control_period_id_c, get_control_parameters_c, get_control_arrays_c, save_new_control_arrays_c)

        return c_fnc_pointers, C_FNC_TYPES


    def __get_control_sequence(self, controls, period_id, new_time_array, new_ctrl_id_array):
        ''' 
        Extract the control time and time_slice arrays from the glogal controls array.

        This function is called after having conputed the active control and time point indices (arrays of int) over a given scheduler period to retreive
        the actual time points and time slices.
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
        ''' 
        Gets the number of time points in a given control time subarray
        '''
        if control_period_id == -1:
            time_array_size = 0;
        else:
            time_array_size = np.size(controls[control_id].t[control_period_id])

        return time_array_size 
        

    def __save_new_control_arrays(self, new_time_array_c, new_ctrl_id_array_c, size, new_time_array, new_ctrl_id_array):
        ''' 
        Allows Python to copy the control arrays :
            - time_array
            - ctrl_id_array

        from the C arrays, created after running the scheduler over a given scheduler period, to the corresponding python arrays.
        This function allows the use of temporary dynamically allocated arrays in C.
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



    def log_scheduler_performances(self, controls, final_control_sequence):        
        data = []

        print("")

        t_start = final_control_sequence.t[0][0]
        t_end = final_control_sequence.t[-1][-1]

        for ctrl_id in range(len(controls)):
            control_uptime_th = 0.0
            control_time_point_ids = 0.0
            control_uptime_real = 0.0

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
        # convert priorities to arrays if not already done
        for ctrl_id, control in enumerate(controls):
            if control.priority is None:
                control.priority = ctrl_id + len(controls) # put the control at the end in FIFO mode

            if not isinstance(control.priority, int) and not isinstance(control.priority, np.ndarray):
                raise ValueError("Control priorities must be arrays of shape (n_periods, n_points_per_periods) or integers.")

            if isinstance(control.priority, np.ndarray):
                error_flag = False
                if len(control.priority) != control.n_periods and len(control.priority) != control.n_control_points: # not the 
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
                control.priority = control.split_array(np.repeat(control.priority, control.n_control_points))