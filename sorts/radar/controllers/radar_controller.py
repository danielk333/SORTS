#!/usr/bin/env python

'''This module is used to define the radar controller

'''
#Python standard import
from abc import ABC, abstractmethod
import copy
import ctypes

#Third party import
import numpy as np

#Local import
from ..controls_manager import RadarControlManagerBase
from sorts import clibsorts

NON_CONTROL_FIELDS = [
        "meta",
        "t",
        "t_slice",
        "priority",
        "beam_enabled"
    ]

class RadarController(ABC):
    '''
        Implements the basic structure of a radar controller. The objective of the Radar controller is to generate the instructions for the Radar system (Tx/Rx) to follow (i.e. follow an object, scan a given area, ...).
        The controller shall be able to output the set of controls in a format which is understandable for the RADAR system (refer to the radar class for more information about Radar controls)
        
    '''
    
    # TODO : do I need to add a static meta field for t_slice ?

    META_FIELDS = [
        'controller_type',
    ]

    def __init__(self, profiler=None, logger=None):
        self.logger = logger
        self.profiler = profiler
    
        # set controller metadata        
        self.meta = dict()
        self.meta['controller_type'] = self.__class__

    def _split_time_array(self, t, t_slice, priority, manager, max_points):
        '''
        Usage
        -----
        
        Split the controls time array according to the manager period and start time (if the manager is provided).       
        If the manager is None, then the time array will be splitted to ensure that the number of time points in a given controls subarray does not exceed max_points
        This function returns the splitted control time array as well as the number of time subarrays
        '''
        if manager is not None:
            if self.logger is not None:
                self.logger.info("radar_controller:_split_time_array -> using manager master clock")
                self.logger.info("radar_controller:_split_time_array -> skipping max_points (max time points limit)")
                
            if not issubclass(manager.__class__, RadarControlManagerBase):
                raise ValueError(f"manager has to be an instance of {RadarControlManagerBase}, not {manager}")
            
            # compute manager period incices
            sch_period = manager.manager_period
            t0 = manager.t0
            
            period_idx = (t - t0)//sch_period
            t_start_subarray_indices = np.array(np.where((period_idx[1:] - period_idx[:-1]) == 1)[0]) + 1
            del period_idx
            
            t = np.array(np.split(t, t_start_subarray_indices), dtype=object)
            t_slice = np.array(np.split(t_slice, t_start_subarray_indices), dtype=object)
            priority = np.array(np.split(priority, t_start_subarray_indices), dtype=object)

            del t_start_subarray_indices

        else:
            if self.logger is not None:
                self.logger.info("radar_controller:_split_time_array -> No manager provided, skipping master clock splitting...")
                self.logger.info(f"radar_controller:_split_time_array -> using max_points={max_points} (max time points limit)")
                
            # compute the number of iterations needed
            if(np.size(t) > max_points):                
                t_start_subarray_indices = np.arange(max_points, np.size(t), max_points, dtype=int)

                t = np.array(np.split(t, t_start_subarray_indices), dtype=object)
                t_slice = np.array(np.split(t_slice, t_start_subarray_indices), dtype=object)
                priority = np.array(np.split(priority, t_start_subarray_indices), dtype=object)

                del t_start_subarray_indices
                
            else:
                t = t[None, :]
                t_slice = t_slice[None, :]
                priority = priority[None, :]

        return t, t_slice, priority, np.shape(t)[0]

    @abstractmethod
    def generate_controls(self, t, radar, **kwargs):
        '''Abstract method used to create a method which generates RADAR controls for a given radar and sampling time. 
        This method can be called multiple times to generate different controls for different radar systems.
        
        Parameters
        ----------
        
        t : numpy.ndarray 
            Time points at which the controls are to be generated [s]
        radar : radar.system.Radar 
            Radar instance to be controlled
       
        Returns
        -------
        Python dictionary 
            Controls to be applied to the radar to perform the required scanning scheme. 
        
        In the case of the Scanner controller, the controls are the following :
    
        - "t"
            1D array of time points at which the controls need to be executed by the radar
        
        - "t_slice"
            1D array containing the duration of a time slice. A time slice represent the elementary quanta which corresponds to a single measurement. Therefore, it also corresponds to the maximal time resolution achievable by our system.
        
        - "priority"
            Priority of the generated controls, only used by the manager to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority management algorithms.
            
        - "enabled"
           State of the radar (enabled/disabled). If this value is a single boolean, then radar measurements will be enabled at each time step. If the value is an array, then the state of the radar
           at time t[k] will be enabled[k]=True/False
       
        - "beam_direction"
            Python generator list of control subarrays (each comprised of max_points time_steps). The data from each subarray is stored in a python dictionary of keys :
            
            - "tx"
                Array of unit vectors representing the target direction of the beam for a given Tx station at time t. 
                To ensure that this controls can be understood by the radar and the manager, the orientation data is given in the following standard:
                
                - Dimension 0 : 
                    Tx station index. In the case where there is only one Tx station considered, the value of this index will be 0.
               
                - Dimension 1 : 
                    Rx station ID associated with Tx. When no stations is associated to Tx, the index is 0 (which is generally the case)   
                    
                - Dimension 2 : 
                    Starting date of each control time slice.
                    
                - Dimension 3 : 
                    Control points per time slice (each time slice can contain multiple orientation controls per time slice)
                       
                - Dimension 4 : 
                    position index (0, 1, 2) = (x, y, z)

            - "rx"
                Array of unit vectors representing the target direction of the beam for a given Rx station at time t. 
                To ensure that this controls can be understood by the radar and the manager, the orientation data is given in the following standard:
            
                - Dimension 0 : 
                    Rx station index. In the case where there is only one Rx station considered, the value of this index will be 0.
               
                - Dimension 1 : 
                    Tx station ID associated with Rx. When there is one Tx stations is associated to Rx, 
                    the controls for the (Rx, Tx) tuple can be accessed by setting this index to 0.
                    
                - Dimension 2 : 
                    Starting date of each control time slice.
                    
                - Dimension 3 : 
                    Control points per time slice (each time slice can contain multiple orientation controls per time slice)
                       
                - Dimension 4 : 
                    position index (0, 1, 2) = (x, y, z)
        
        - "meta":
            This field contains the controls metadata :
                
            - "scan" : 
                sorts.radar.Scan instance used to generate the scan.
                
            - "radar" :
                sorts.radar.system.Radar instance being controlled.
                
            - "max_points" :
                Number of time points per control subarray (this value is not used if the manager is not none)
                
            - "manager" :
                sorts.radar.controls_manager.RadarControlsManagerBase instance associated with the control array. This instance is used to 
                divide the controls into subarrays of controls according to the manager scheduling period to 
                reduce memory/computational overhead.
                If the manager is not none, the value of the scheduling period will take over the max points parameter (see above)
                
        - "priority"
            Priority of the generated controls, only used by the manager to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority management algorithms.
      '''
        pass



def normalize_direction_controls(directions, logger=None):
    '''
    Compute the normalized beam direction vectors.
    
    The input format of the beam direction vectors are :
        - Tx/Rx stations :
            - dim 0: station id
            - dim 1: associated station id (if it exists, else the index will be 0)
            - dim 2: array of time slices
            - dim 3: points per time slice
            - dim 4: x, y, z coordinate axis
            
    TODO -> implementation in C/C++
    TODO -> implement callback python functions to support non-homogeneous arrays of controls
    '''
    directions = np.asfarray(directions)                        
    return directions/np.linalg.norm(directions, axis=4)[:, :, :, :, None]


def get_time_slice(t_controller, time_slice, t, flatten=True, logger=None, profiler=None):
    '''
    get the indices of the active time slices of a given controller array at time t.
    
    Parameters:
    -----------
    t_controller : numpy.ndarray
        starting point of each controller time slice
    time_slice : float/numpy.ndarray
        time slice duration
    t : float/numpy.ndarray
        points at which to get the active time slices
    falten (optional) : boolean
        if true and if the time array of the controller is sliced, then the output array will be 1D and each index will correspond to the non-sliced time array
    logger (optional) : logging.Logger
        profiler instance used to monitor execution performances
    profiler (optional) : profiling.Profiler
        profiler instance used to monitor execution performances

    Returns:
    --------
    t_in_slice_inds : numpy.ndarray
        indices of the time slices for each time point given by t

    '''
    # convert input variables to float arrays for vectorized operations
    t = np.asfarray(t)
    time_slice = np.asfarray(time_slice)

    # Logging execution status
    if logger is not None:
        logger.info(f"getting time slices at t={t}s")

    if profiler is not None:
        profiler.start(f"radar_controller:get_time_slice")

    # if time_slice 
    if(np.size(time_slice) == 1):
        time_slice = np.ones(np.size(t_controller))*time_slice

    # check if time array is multidimensional -> True : the time array has been splitted
    if isinstance(t_controller[0], np.ndarray):
        t_in_slice_inds_1 = []
        t_in_slice_inds_2 = []
        
        time_index = 0

        for ti, t_sub in enumerate(t_controller):
            t_in_subarray_msk = np.logical_and(t >= t_sub[0], t <= t_sub[-1])
            
            dt = t_sub[:, None] - t[t_in_subarray_msk][None, :]
            inds = np.asfarray(np.where(np.logical_and(dt <= time_slice[time_index:time_index+np.size(t_sub), None], dt >= 0))) #[t_ctrl, t]
            
            if flatten:
                inds[0] = inds[0] + time_index

            t_in_slice_inds_1.append(inds[0])
            t_in_slice_inds_2.append(inds[1])
            
            time_index += np.size(t_sub)
              
        if flatten:
            t_in_slice_inds_1 = np.array([item for sub in t_in_slice_inds_1 for item in sub], dtype=float)
            t_in_slice_inds_2 = np.array([item for sub in t_in_slice_inds_2 for item in sub], dtype=float)
            
        t_in_slice_inds = np.array([t_in_slice_inds_1, t_in_slice_inds_2])
    else:
        t_in_slice_inds = np.where((t_controller[:, None] - t[None, :]) <= time_slice)[0]
    if profiler is not None:
        profiler.stop(f"radar_controller:get_time_slice")

    return t_in_slice_inds

def check_time_slice_overlap(t_controller, time_slice, logger=None, profiler=None):
    '''
    Checks wether or not time slices overlap within a given control array.
    If two different time slices overlap, the function will return the indices of the time points which overlap.
    
    Parameters:
    -----------
    t_controller : numpy.ndarray
        starting point of each controller time slice
    time_slice : float/numpy.ndarray
        time slice duration
    logger (optional) : logging.Logger
        profiler instance used to monitor execution performances
    profiler (optional) : profiling.Profiler
        profiler instance used to monitor execution performances

    Returns:
    --------
    indices_1 : numpy.ndarray
        first element of the tuple of overlaping indices
    indices_2 : numpy.ndarray
        second element of the tuple of overlaping indices

    if the time slices 3 and 4 are overlapping, check_time_slice_overlap(...) will return :
        - indices_1 = numpy.array([3])
        - indices_2 = numpy.array([4])

    '''
    # convert input variables to float arrays for vectorized operations
    t_controller = np.asarray(t_controller, dtype=np.float64)
    time_slice = np.asarray(time_slice, dtype=np.float64)

    # Logging execution status
    if logger is not None:
        logger.info(f"checking time slice overlap")

    if profiler is not None:
        profiler.start(f"radar_controller:get_time_slice")

    indices = np.array([0], dtype=np.int32)

    SAVE_ARRAY_FNC = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int), ctypes.c_int)

    def callback_save_array(array, size):
        nonlocal indices

        if size == 0:
            indices = np.array([])
        else:
            buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
            buffer_from_memory.restype = ctypes.py_object
            buffer = buffer_from_memory(array, np.dtype(np.int32).itemsize*size)

            indices = np.frombuffer(buffer, np.int32)

    callback_save_array_c = SAVE_ARRAY_FNC(callback_save_array)
    
    clibsorts.check_time_slice_overlap.argtypes = [
                                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t_controller.ndim, shape=t_controller.shape),
                                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=time_slice.ndim, shape=time_slice.shape),
                                            ctypes.c_int,
                                            np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=indices.ndim, shape=indices.shape) ,
                                            SAVE_ARRAY_FNC]

    clibsorts.check_time_slice_overlap(t_controller, time_slice, ctypes.c_int(len(t_controller)), indices, callback_save_array_c)
    
    if profiler is not None:
        profiler.stop(f"radar_controller:get_time_slice")

    return indices

def check_controls_overlap(t_controller_1, time_slice_1, t_controller_2, time_slice_2, logger=None, profiler=None):
    '''Checks wether or not time slices overlap between two controls (with the same time reference).
    If two different time slices overlap, the function will return the indices of the time points which overlap.
    
    Parameters:
    -----------
    t_controller_1 : numpy.ndarray
        starting point of each of the time slices of the first control array
    time_slice_1 : float/numpy.ndarray
        time slice duration of the first control array
    t_controller_2 : numpy.ndarray
        starting point of each of the time slices of the second control array
    time_slice_2 : float/numpy.ndarray
        time slice duration of the second control array
    logger (optional) : logging.Logger
        profiler instance used to monitor execution performances
    profiler (optional) : profiling.Profiler
        profiler instance used to monitor execution performances

    Returns:
    --------
    indices_1 : numpy.ndarray
        overlaping indices of the first control array
    indices_2 : numpy.ndarray
        overlaping indices of the second control array

    if the time slices 15 of the first array and 4, 3 of the second array are overlapping, check_time_slice_overlap(...) will return :
        - indices_1 = numpy.array([15, 15])
        - indices_2 = numpy.array([3, 4])
    '''

    # convert input variables to float arrays for vectorized operations
    # check time_slice_1 input data
    if not isinstance(time_slice_1, np.ndarray):
        time_slice_1 = np.asarray(time_slice_1)
    if isinstance(time_slice_1[0], np.ndarray):
        raise(ValueError("time_slice_1 array is sliced (more than one dimension). Please provide 1D time arrays"))
        
    # check time_slice_2 input data
    if not isinstance(time_slice_2, np.ndarray): 
        time_slice_2 = np.asarray(time_slice_2)
    if isinstance(time_slice_2[0], np.ndarray): 
        raise(ValueError("time_slice_2 array is sliced (more than one dimension). Please provide 1D time arrays"))
    
    # check t_controller_1 & t_controller_2 input data
    if isinstance(t_controller_1[0], np.ndarray) or isinstance(t_controller_2[0], np.ndarray):
        raise(ValueError("controller time arrays t_controller_1/t_controller_2 are sliced (more than one dimension). Please provide 1D time arrays"))

    # Logging execution status
    if logger is not None:
        logger.info(f"radar_controller:check_controls_overlap: checking time slice overlap")

    if profiler is not None:
        profiler.start(f"radar_controller:check_controls_overlap")

    # if time_slice are floats, convert to array
    if(np.size(time_slice_1) == 1): 
        time_slice_1 = np.ones(np.size(t_controller_1))*time_slice_1
    if(np.size(time_slice_2) == 1): 
        time_slice_2 = np.ones(np.size(t_controller_2))*time_slice_2

    # find the common time points between the two controller time arrays
    array_overlap_msk_1 = np.logical_and(t_controller_1 >= t_controller_2[0], t_controller_1 <= t_controller_2[-1] + time_slice_2[-1])
    array_overlap_msk_2 = np.logical_and(t_controller_2 >= t_controller_1[0], t_controller_2 <= t_controller_1[-1] + time_slice_1[-1])

    if np.size(np.where(array_overlap_msk_1)[0]) == 0 and np.size(np.where(array_overlap_msk_2)[0]) == 0:
        if logger is not None:
            logger.info(f"radar_controller:check_controls_overlap: controller time arrays are not on the same interval -> no overlap")
        
        indices_1 = np.array([])
        indices_2 = np.array([])
    else:
        # find time points inside the other controller time slices 
        tslice_overlap_msk_1 = t_controller_1[:, None] - t_controller_2[None, :] <= time_slice_1
        tslice_overlap_msk_2 = t_controller_2[:, None] - t_controller_1[None, :] <= time_slice_2

        indices_1 = np.where(np.logical_and(tslice_overlap_msk_1, array_overlap_msk_1))[0]
        indices_2 = np.where(np.logical_and(tslice_overlap_msk_2, array_overlap_msk_2))[0]

    if profiler is not None:
        profiler.stop(f"radar_controller:check_controls_overlap")

    return indices_1, indices_2