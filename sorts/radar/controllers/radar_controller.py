#!/usr/bin/env python

'''This module is used to define the radar controller

'''
#Python standard import
from abc import ABC, abstractmethod
import copy

#Third party import
import numpy as np

#Local import
from ..scheduler import Scheduler

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
            sch_period = scheduler.schedule_period
            t0 = scheduler.t0
            period_idx = (t - t0)//sch_period
            
            t_start_period_indices = np.array(np.where((period_idx[1:] - period_idx[:-1]) == 1)[0]) + 1
            
            del period_idx
        else:
            if self.logger is not None:
                self.logger.info("radar_controller:_split_time_array -> No scheduler provided, skipping master clock splitting...")
                self.logger.info("radar_controller:_split_time_array -> using max_points={max_points} (max time points limit)")
                
            # compute the number of iterations needed
            if(np.size(t) > max_points):
                sub_controls_count = int((len(t[0])-1)/max_points) + 1    
                
                t_start_subarray_indices = np.arange(0, np.size(t), max_points, dtype=int)
                t = np.split(t, t_start_subarray_indices)
            else:
                t = t[None, :]

        return t, np.shape(t)[0] 
    
    @abstractmethod
    def generate_controls(self, t, radar, **kwargs):
        '''
            Purpose
            -------
            This method is used to generate the controls (generated as a dictionnary containing the instructions to be executed for each time step) for a given radar instance. 
            
            Independant controls can be generated using the same controller instance as follow :
                
                >>> controls_radar_1 = controller.generate_controls(t, radar_1)
                >>> controls_radar_2 = controller.generate_controls(t, radar_2)

            The controls can then be used by the scheduler (or directly by the radar) to perform a given observation scheme. 
            Beware that the controls are generated independantly from the ability of the radar to perform them, therefore the scheduler 
            (or user, in case no scheduler is used) has to make sure that the controls remain feasible given th physical constraints of the Radar system (if any)
            
            Parameters
            ----------
            
            :np.ndarray/float t: array of time points at which the radar controls shall be generated. the minimal interval between each time step has to be greater than the dwel time of each step.
            :radar.system.Radar radar: radar instance which is controlled
            
            Return value
            ------------
            
            This function returns the controls (python dictionnary) containing each instruction to be executed by the Radar as array of values
            
            Individual controls can be accessed as follow :
                
                >>> controls = controller.generate_controls(t, radar)

                >>> controls_t = controls["t"]                           # get time array
                >>> controls_dir_tx = controls["beam_direction_tx"]      # get controls for tx direction
        '''
        pass