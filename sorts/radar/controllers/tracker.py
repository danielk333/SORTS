#!/usr/bin/env python

'''#TODO

'''

import numpy as np

from .radar_controller import RadarController


class Tracker(RadarController):
    '''Takes in ECEF points and a time vector and creates a tracking control.
    '''
    
    META_FIELDS = RadarController.META_FIELDS

    def __init__(self, profiler=None, logger=None, **kwargs):
        super().__init__(profiler=profiler, logger=logger, **kwargs)

        if self.logger is not None:
            self.logger.info(f'Tracker:init')
            
    def compute_sub_controls(
                            self, 
                            t, 
                            radar, 
                            target_states,
                            priority=None, 
                            max_points=100,
                            ):
        
        beam_controls = dict()
        
        if self.profiler is not None:
            self.profiler.start('Tracker:generate_controls:compute_controls')
        
        for ti in range(len(t)):
            if self.profiler is not None:
                self.profiler.start('Tracker:generate_controls:step')

            if self.return_copy:
                radar = self.radar.copy()
            else:
                radar = self.radar

            dt = t[ti] - self.t
            check = np.logical_and(dt >= 0, dt < self.dwell)
            ind = np.argmax(check)
            meta = self.default_meta()

            # TODO : add this to radar class : RadarController.coh_integration(radar, self.dwell)
            
            self.toggle_stations(t[ti], radar)
            self.point_radar(radar, ind)
            
            if self.profiler is not None:
                self.profiler.stop('Tracker:generate_controls:step')
            
            yield radar, meta

        if self.profiler is not None:
            self.profiler.stop('Tracker:generate_controls')
        if self.logger is not None:
            self.logger.debug(f'Tracker:generate_controls:completed')
        
        yield beam_controls

    def generate_controls(
                            self, 
                            t, 
                            t_slice,
                            radar, 
                            target_states,
                            priority=None, 
                            max_points=100,
                            ):
        '''
        Usage
        -----
        
        Generate Radar tracking for a given radar, target and sampling time. This method can be called multiple times to generate different controls for different radar systems.
        
        Parameters
        ----------
        
        :numpy.ndarray t: Time points at which the controls are to be generated [s]
        :radar.system.Radar radar: Radar instance to be controlled
        :numpy.ndarray target_states: target states used to generate the tracking controls
        :numpy.ndarray r (optional): Array of ranges from the transmitter where the receivers need to target simultaneously at a given time t [m]
        :int priority (optional): Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority scheduler algorithms.
        :int max_points (optional): 
    
        - "t"
            1D array of time points at which the controls need to be executed by the radar
        
        - "t_slice"
            1D array containing the duration of a time slice. A time slice represent the elementary quanta which corresponds to a single measurement. Therefore, it also corresponds to the maximal time resolution achievable by our system.
       - "priority"
            Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority scheduler algorithms.
            
       - "enabled"
           State of the radar (enabled/disabled). If this value is a single boolean, then radar measurements will be perforenabled at each time step. If the value is an array, then the state of the radar
           at time t[k] will be enabled[k]=True/False
       
        - "beam_direction"
            Python generator list of control subarrays (each comprised of max_points time_steps). The data from each subarray is stored in a python dictionary of keys :
            
            - "tx"
                Array of unit vectors representing the target direction of the beam for a given Tx station at time t. 
                To ensure that this controls can be understood by the radar and the scheduler, the orientation data is given in the following standard:
                    
                - Dimension 0 : 
                    Tx station index. In the case where there is only one Tx station considered, the value of this index will be 0.
               
                - Dimension 1 : 
                    position index (0, 1, 2) = (x, y, z)
                
                - Dimension 2 : 
                    time index
            
            - "rx"
                Array of unit vectors representing the target direction of the beam for a given Rx station at time t. 
                To ensure that this controls can be understood by the radar and the scheduler, the orientation data is given in the following standard:
                
                - Dimension 0 : 
                    Rx station index. in the case where there is only one Rx station considered, the value of this index will be 0.
                
                - Dimension 1 : 
                    Associated Tx station index. Since the Rx stations must target the same points as the Tx stations, each Rx station gets at least as many beam orientation controls as there are Tx stations. In the case where there is only one Tx station considered, the value of the index will be 0.
               
                - Dimension 2 : 
                    Range index : one Rx station can simultaneously target the same Tx beam at multiple ranges (from the Tx station), those ranges are given when calling the generate_controls method by setting the argument r.
               
                - Dimension 3 : 
                    position index (0, 1, 2) = (x, y, z)
                
                - Dimension 4 : 
                    time index 
        
        - "meta":
            This field contains the controls metadata :
                
            - "scan" : 
                sorts.radar.Scan instance used to generate the scan.
                
            - "radar" :
                sorts.radar.system.Radar instance being controlled.
                
            - "max_points" :
                Number of time points per control subarray. 
                
        - "priority"
            Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority scheduler algorithms.
            
         
        Examples 
        --------
        
        Suppose that there is a sinle Tx station performing a scan at 100 different time points. The shape of the output "beam_direction_tx" array will be (1, 3, 100). 
        
        - To get the z component of the beam direction of the Tx station at the second time step, one must call : 
            
            >>> ctrl = controls["beam_direction_tx"][0, 2, 2]
            
        - To get the x component of the beam direction of the Tx station at the 35th time step, one must call :
        
            >>> ctrl = controls["beam_direction_tx"][0, 0, 34]
        

        Suppose now that there is a sinle Tx station performing a scan at 100 different time points, that there are 2 Rx stations performing simultaneous scans at 10 different ranges. The shape of the output "beam_direction_rx" array will be (2, 1, 10, 3, 100). 
        
        - To get the z component of the beam direction of the first Rx station with respect to the second simultaneous scan at range r of the only Tx beam at the second time step, one must call : 
             
             >>> ctrl = controls["beam_direction_rx"][0, 0, 1, 2, 2]
         
        - To get the y component of the beam direction of the second Rx station with respect to the 5th simultaneous scan at range r of the only Tx beam at the 80th time step, one must call 
             
            >>> ctrl = controls["beam_direction_rx"][1, 0, 4, 1, 79]
            
          '''
        if self.profiler is not None:
            self.profiler.start('Tracker:generate_controls')
        
        # checks input values to make sure they are compatible with the implementation of the function
        if priority is not None:
            if not isinstance(priority, int): raise TypeError("the priority must be an integer.")
            else: 
                if priority < 0: raise ValueError("the priority must be positive [0; +inf] or equal to -1.")
            
        if not isinstance(radar, Radar): raise TypeError(f"the radar must be an instance of {Radar}.")

        # add support for both arrays and floats
        t = np.asarray(t)
        if len(np.shape(t)) > 1: raise TypeError("t must be a 1-dimensional array or a float")
            
        controls = dict()
        controls["t"] = t
        controls["t_slice"] = t_slice
        controls["priority"] = priority # set the controls priority
        controls["enabled"] = True # set the radar state (on/off). TODO : check if necessary
        
        controls["meta"] = dict()
        controls["meta"]["radar"] = radar
        controls["meta"]["controller_type"] = "tracker"
        controls["meta"]["max_points"] = max_points

        controls["beam_orientation"] = []
        
        # compute the number of iterations needed
        sub_controls_count = int((len(t)-1)/max_points) + 1
        
        # compute sub control arrays
        for i in range(sub_controls_count):
            if self.profiler is not None:
                self.profiler.start('Scanner:generate_controls:compute_controls_subarray:')
            
            if i == sub_controls_count-1:
                t_sub_array = t[i*max_points:]
            else:
                t_sub_array = t[i*max_points:max_points*(i+1)]
                
            controls["beam_orientation"].append(self.__compute_controls_subarray( 
                                                                                t, 
                                                                                t_slice,
                                                                                radar, 
                                                                                target_states,
                                                                                ))
                        
            if self.profiler is not None:
                self.profiler.start('Scanner:generate_controls:compute_controls_subarray')
         
        if self.profiler is not None:
            self.profiler.stop('Scanner:generate_controls')
        
        return controls
