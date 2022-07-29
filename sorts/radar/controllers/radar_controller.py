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
from ..scheduler import RadarSchedulerBase
from sorts import clibsorts

class RadarController(ABC):
    '''
        Implements the basic structure of a radar controller. The objective of the Radar controller is to generate the instructions for the Radar system (Tx/Rx) to follow (i.e. follow an object, scan a given area, ...).
        The controller shall be able to output the set of controls in a format which is understandable for the RADAR system (refer to the radar class for more information about Radar controls)
        
    '''
    
    META_FIELDS = [
        'controller_type',
    ]

    def __init__(self, profiler=None, logger=None):
        self.logger = logger
        self.profiler = profiler
    
        # set controller metadata        
        self.meta = dict()
        self.meta['controller_type'] = self.__class__

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
            Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority management algorithms.
            
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
                sorts.radar.controls_scheduler.RadarControlsschedulerBase instance associated with the control array. This instance is used to 
                divide the controls into subarrays of controls according to the scheduler scheduling period to 
                reduce memory/computational overhead.
                If the scheduler is not none, the value of the scheduling period will take over the max points parameter (see above)
                
        - "priority"
            Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority management algorithms.
      '''
        pass


    def compute_pointing_direction(
        self, 
        controls,
        period_id, 
        args):
        '''
        Computes the pointing directions of the radar given the controller type and the array of time slices
        '''
        raise Exception("No pointing directions set")


    @staticmethod
    def coh_integration(controls, radar, dwell):
        '''Set the coherent integration settings based on the dwell time.
        '''
        dwell = np.atleast_1d(dwell)

        for tx in radar.tx:
            controls.add_property_control("n_ipp", tx, (dwell/tx.ipp).astype(int))
            controls.add_property_control("coh_int_bandwidth", tx, 1.0/(tx.pulse_length*tx.n_ipp))