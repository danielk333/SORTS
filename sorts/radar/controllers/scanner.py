#!/usr/bin/env python

'''
Usage
-----

This module defines the scanner controller class that can be used to create the Radar controls necessary to perform a scanning observation scheme. 

'''

import numpy as np

from .radar_controller import RadarController
from ..system.radar import Radar

class Scanner(RadarController):
    '''
    Usage
    -----
    
    This class is used to create scanning controls for a given radar system. To create a control, one instanciates the scanner class by giving a profiler and logger depending on the application. 
   
    
    Examples
    --------
    
    calling the function generate_controls() with the radar to be controlled and the scan algorithm to be executed will return the corresponding array of controls. This can be achieved in the following manner :
        
        >>> scanner_controller = controllers.Scanner()
        
        >>> controls_radar_1 = scanner_controller.generate_controls(t, radar_1) # radar_1 is an instance of radar.system.Radar
        >>> controls_radar_2 = scanner_controller.generate_controls(t, radar_2) # radar_2 is an instance of radar.system.Radar
            
    Only one Scanner controller is needed to create multiple controls for multiple scan methods and radars.
    '''

    META_FIELDS = RadarController.META_FIELDS + [
        'scan_type',
    ]

    def __init__(
            self, 
            profiler=None, 
            logger=None, 
            **kwargs
            ):
        '''
        Usage
        -----
        Creates a Scanner controller to generate radar scanning controls. Only one Scanner controller is needed to create multiple controls for differnet scan methods and radars.
        
        Parameters
        ----------
        :Profiler profiler: Profiler instance used to check the generate_controls method performance.
        :logging.Logger logger: Logger instance used to log the execttion of the generate_controls method.
        
        Return value
        ------------
        
        The constructor returns an instance of controllers.Scanner that can be used to generate scanning controls.
        
        Example
        -------
        
        An instance of the Scanner controller can be created in the following manner :
            
            >>> scanner_controller = controllers.Scanner()
        
        '''
        super().__init__(profiler=profiler, logger=logger, **kwargs) # call the constructor from RadaController (parent class)
        
         # defines the controller type in the meta data
        self.meta['scan_type'] = self.__class__

        if self.logger is not None:
            self.logger.info('Scanner:init')   
            
    def __compute_beam_orientation(
            self, 
            t,
            radar, 
            scan,
            r, 
            ):
        '''
        Usage
        -----
        
        Compute the beam orientation controls assiciated with the controller. 
        This function returns a generator allowing the computation of the said subarray (containing a maximum of max_points time points)
        '''
        if self.profiler is not None:
            self.profiler.start('Scanner:generate_controls:compute_controls_subarray:tx') 
        
        # initializing results
        beam_controls = dict()

        # get the position of the Tx/Rx stations
        tx_ecef = np.array([tx.ecef for tx in radar.tx], dtype=float).reshape((len(radar.tx), 3, 1)) # get the position of each Tx station (ECEF frame)
        rx_ecef = np.array([rx.ecef for rx in radar.rx], dtype=float) # get the position of each Rx station (ECEF frame)
        
        # compute directions for stations where tx and rx < 200 meters apart => same location for pointing
        rx_close_to_tx = np.linalg.norm(tx_ecef - rx_ecef.transpose(), axis=1) < 200.0
        inds_rx_close_to_tx = np.array(np.where(rx_close_to_tx)) # [txinds, rxinds]
        del rx_close_to_tx
        
        # get Tx pointing directions
        points = scan.ecef_pointing(t, radar.tx) # [ind_tx][x, y, z][t]
        # get Tx pointing directions and target points for Rx
        point_tx = points + tx_ecef
 
        # Compute Tx pointing directions
        tx_dirs = point_tx - tx_ecef
        beam_controls['tx'] = tx_dirs/np.linalg.norm(tx_dirs, axis=1) # the beam directions are given as unit vectors in the ecef frame of reference
        del tx_dirs
        
        if self.profiler is not None:
            self.profiler.stop('Scanner:generate_controls:compute_controls_subarray:tx') 
            self.profiler.start('Scanner:generate_controls:compute_controls_subarray:rx') 
        
        # get Rx target points on the Tx beam
        point_rx_to_tx = points[:, None, :, :]*r[None, :, None, None] + tx_ecef # compute the target points for the Rx stations
        del tx_ecef, points
        point_rx = np.repeat(point_rx_to_tx[None, :], len(radar.rx), axis=0) 
        del point_rx_to_tx
        
        # correct pointing directions for stations too close to each other
        point_rx[inds_rx_close_to_tx[1], :, :, :] = point_tx[None, None, inds_rx_close_to_tx[0], :, :]
        del inds_rx_close_to_tx, point_tx
        
        # compute actual pointing direction
        rx_dirs = point_rx - rx_ecef[:, None, None, :, None]
        del point_rx
        
        # save computation results
        beam_controls['rx'] = rx_dirs/np.linalg.norm(rx_dirs, axis=3)[:, :, :, None] # the beam directions are given as unit vectors in the ecef frame of reference
        del rx_ecef
        
        if self.profiler is not None:
            self.profiler.stop('Scanner:generate_controls:compute_controls_subarray:rx')
            
        yield beam_controls

    def generate_controls(
            self, 
            t, 
            radar, 
            scan,
            r=np.linspace(300e3, 1000e3, num=10), 
            priority=None, 
            scheduler=None,
            max_points=100,
            ):
        '''
        Usage
        -----
        
        Generate RADAR scanning controls for a given radar, scanning algorithm and sampling time. This method can be called multiple times to generate different controls for different radar systems.
        
        Parameters
        ----------
        
        :numpy.ndarray t: Time points at which the controls are to be generated [s]
        :radar.system.Radar radar: Radar instance to be controlled
        :scans.Scan scan: Scan instance used to generate the scanning controls
        :numpy.ndarray r (optional): Array of ranges from the transmitter where the receivers need to target simultaneously at a given time t [m]
        :int priority (optional): Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority scheduler algorithms.
        :int max_points (optional): 

        Return value
        ----------
        
        Dictionnary containing the controls to be applied to the radar to perform the required scanning scheme. In the case of the Scanner controller, the controls are the following.
    
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
            self.profiler.start('Scanner:generate_controls')
        
        # controls computation initialization
        # checks input values to make sure they are compatible with the implementation of the function
        if priority is not None:
            if not isinstance(priority, int): raise TypeError("the priority must be an integer.")
            else: 
                if priority < 0: raise ValueError("the priority must be positive [0; +inf] or equal to -1.")
            
        if not isinstance(radar, Radar): raise TypeError(f"the radar must be an instance of {Radar}.")

        # add support for both arrays and floats
        t = np.asarray(t)
        if len(np.shape(t)) > 1: raise TypeError("t must be a 1-dimensional array or a float")
        
        # split time array into scheduler periods if a scheduler is attached to the controls
        t, sub_controls_count = super()._split_time_array(t, scheduler, max_points)
        
        # output data initialization
        controls = dict()  # the controls structure is defined as a dictionnary of subcontrols
        controls['t'] = t  # save the time points of the controls
        controls['t_slice'] = scan.dwell(t) # save the dwell time of each time point
        controls["priority"] = priority # set the controls priority
        controls["enabled"] = True # set the radar state (on/off)
        
        # setting metadata
        controls["meta"] = dict()
        controls["meta"]["scan"] = scan
        controls["meta"]["radar"] = radar
        controls["meta"]["controller_type"] = "scanner"
        controls["meta"]["scheduler"] = scheduler # set the radar state (on/off)
        controls["meta"]["sub_controls_count"] = sub_controls_count
        
        # creating orientation controls generator array
        controls["beam_orientation"] = []
        
        # Computations
        # compute controls for each time sub array
        for subcontrol_index in range(sub_controls_count):
            if self.profiler is not None:
                self.profiler.start('Scanner:generate_controls:compute_controls_subarray')

            controls["beam_orientation"].append(self.compute_beam_orientation(t[subcontrol_index], radar, scan, r))
            
            if self.profiler is not None:
                self.profiler.start('Scanner:generate_controls:compute_controls_subarray')
         
        if self.profiler is not None:
            self.profiler.stop('Scanner:generate_controls')
        
        return controls
