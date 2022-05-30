#!/usr/bin/env python

'''This module is used to define the radar controller

'''
#Python standard import
from abc import ABC, abstractmethod
import copy

#Third party import
import numpy as np

#Local import


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

    def _split_time_array(self, t, scheduler, max_points):
        '''
        Usage
        -----
        
        Split the controls time array according to the scheduler period and start time (if the scheduler is provided).       
        If the scheduler is None, then the time array will be splitted to ensure that the number of time points in a given controls subarray does not exceed max_points
        This function returns the splitted control time array as well as the number of time subarrays
        '''
        if scheduler is not None:
            if self.logger is not None:
                self.logger.info("radar_controller:_split_time_array -> using scheduler master clock")
                self.logger.info("radar_controller:_split_time_array -> skipping max_points (max time points limit)")
                
            if not isinstance(scheduler, Scheduler):
                raise ValueError(f"scheduler has to be an instance of {Scheduler}, not {scheduler}")
            
            # compute scheduler period incices
            sch_period = scheduler.scheduler_period
            t0 = scheduler.t0
            
            period_idx = (t - t0)//sch_period
            t_start_subarray_indices = np.array(np.where((period_idx[1:] - period_idx[:-1]) == 1)[0]) + 1
            del period_idx
            
            t = np.array(np.split(t, t_start_subarray_indices), dtype=object)
            del t_start_subarray_indices

        else:
            if self.logger is not None:
                self.logger.info("radar_controller:_split_time_array -> No scheduler provided, skipping master clock splitting...")
                self.logger.info(f"radar_controller:_split_time_array -> using max_points={max_points} (max time points limit)")
                
            # compute the number of iterations needed
            if(np.size(t) > max_points):                
                t_start_subarray_indices = np.arange(max_points, np.size(t), max_points, dtype=int)
                t = np.array(np.split(t, t_start_subarray_indices), dtype=object)
                del t_start_subarray_indices
                
            else:
                t = t[None, :]

        return t, np.shape(t)[0]

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
            Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority scheduler algorithms.
            
        - "enabled"
           State of the radar (enabled/disabled). If this value is a single boolean, then radar measurements will be enabled at each time step. If the value is an array, then the state of the radar
           at time t[k] will be enabled[k]=True/False
       
        - "beam_direction"
            Python generator list of control subarrays (each comprised of max_points time_steps). The data from each subarray is stored in a python dictionary of keys :
            
            - "tx"
                Array of unit vectors representing the target direction of the beam for a given Tx station at time t. 
                To ensure that this controls can be understood by the radar and the scheduler, the orientation data is given in the following standard:
                
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
                To ensure that this controls can be understood by the radar and the scheduler, the orientation data is given in the following standard:
            
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
                Number of time points per control subarray (this value is not used if the scheduler is not none)
                
            - "scheduler" :
                sorts.scheduler.Scheduler instance associated with the control array. This instance is used to 
                divide the controls into subarrays of controls according to the scheduler scheduling period to 
                reduce memory/computational overhead.
                If the scheduler is not none, the value of the scheduling period will take over the max points parameter (see above)
                
        - "priority"
            Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority scheduler algorithms.
      '''
        pass



def normalize_direction_controls(directions):
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

    n_stations = np.shape(directions)[0]

    for station_id in range(n_stations):
        n_associated_stations = np.shape(directions[station_id])[0]
        
        for associated_station_id in range(n_associated_stations):
            n_time_points = np.shape(directions[station_id, associated_station_id])[0]
            
            for ti in range(n_time_points):
                n_points_per_slice = np.shape(directions[station_id, associated_station_id, ti])[0]
                
                for t_slice_id in range(n_points_per_slice):
                    direction = directions[station_id, associated_station_id, ti, t_slice_id]
                    direction = direction/np.linalg.norm(direction)
                    
                    directions[station_id, associated_station_id, ti, t_slice_id] = direction

    return directions
