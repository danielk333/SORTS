#!/usr/bin/env python

'''
Usage
-----

This module defines the scanner controller class that can be used to create the Radar controls necessary to perform a scanning observation scheme. 

'''

import numpy as np

from . import radar_controller
from sorts.radar.system.radar import Radar

class Scanner(radar_controller.RadarController):
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

    META_FIELDS = radar_controller.RadarController.META_FIELDS + [
        'scan_type',
    ]

    def __init__(
            self, 
            profiler=None, 
            logger=None, 
            **kwargs
            ):
        ''' Creates a Scanner controller to generate radar scanning controls. Only one Scanner controller is needed to create multiple controls for differnet scan methods and radars.
        
        Parameters
        ----------
        profiler : Profiler
            Profiler instance used to check the generate_controls method performance.
        logger : logging.Logger 
            Logger instance used to log the execttion of the generate_controls method.
        
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
        Computes the beam orientation controls assiciated with the controller. 
        This function returns a generator allowing the computation of the said subarray (containing a maximum of max_points time points)
        '''
        if self.profiler is not None:
            self.profiler.start('Scanner:generate_controls:compute_controls_subarray:tx') 
        
        # initializing results
        beam_controls = dict()

        # get the position of the Tx/Rx stations
        tx_ecef = np.array([tx.ecef for tx in radar.tx], dtype=float) # get the position of each Tx station (ECEF frame)
        rx_ecef = np.array([rx.ecef for rx in radar.rx], dtype=float) # get the position of each Rx station (ECEF frame)
        
        # compute directions for stations where tx and rx < 200 meters apart => same location for pointing
        
        rx_close_to_tx = np.linalg.norm(tx_ecef[:, None, :] - rx_ecef[None, :, :], axis=2) < 200.0
        inds_rx_close_to_tx = np.array(np.where(rx_close_to_tx)) # [txinds, rxinds]
        del rx_close_to_tx
        
        # get Tx pointing directions
        # [ind_tx][x, y, z][t] ->[ind_tx][t][x, y, z]
        points = scan.ecef_pointing(t, radar.tx).transpose(0, 2, 1)

        # get Tx pointing directions and target points for Rx
        # [txi, rxi, t, t_slice, (xyz)]
        point_tx = points[:, None, :, None, :] + tx_ecef[:, None, None, None, :]
 
        # Compute Tx pointing directions
        beam_controls['tx'] = points[:, None, :, None, :]#super()._normalize(tx_dirs) # the beam directions are given as unit vectors in the ecef frame of reference
        
        if self.profiler is not None:
            self.profiler.stop('Scanner:generate_controls:compute_controls_subarray:tx') 
            self.profiler.start('Scanner:generate_controls:compute_controls_subarray:rx') 
        
        # get Rx target points on the Tx beam
        point_rx_to_tx = points[:, :, None, :]*r[None, :, None] + tx_ecef[None, :, None, None, :] # compute the target points for the Rx stations
        del tx_ecef, points
        point_rx = np.repeat(point_rx_to_tx, len(radar.rx), axis=0) 
        del point_rx_to_tx
        
        # correct pointing directions for stations too close to each other
        # point_rx[inds_rx_close_to_tx[1], :, :, :, :] = point_tx[inds_rx_close_to_tx[0], :, :, :, :]
        # del inds_rx_close_to_tx, point_tx
        
        # compute actual pointing direction
        rx_dirs = point_rx - rx_ecef[:, None, None, None, :]
        del point_rx, rx_ecef
        
        # save computation results

        beam_controls['rx'] = radar_controller.normalize_direction_controls(rx_dirs) # the beam directions are given as unit vectors in the ecef frame of reference
        
        if self.profiler is not None:
            self.profiler.stop('Scanner:generate_controls:compute_controls_subarray:rx')
            
        yield beam_controls

    def generate_controls(
            self, 
            t, 
            radar, 
            scan,
            r=np.linspace(300e3, 1000e3, num=10), 
            as_altitude=False, 
            scheduler=None,
            priority=None,
            max_points=1000,
            ):
        '''Generates RADAR scanning controls in a given direction. 
        This method can be called multiple times to generate different controls for different radar systems.
        
        Usage
        -----
        
        One can generate scanning controls for a given target as follows :

        >>> logger = profiling.get_logger('static')
        >>> p = profiling.Profiler() # Profiler

        >>> # Computation / test setup
        >>> end_t = 24*3600
        >>> nbplots = 1
        >>> t_slice = 0.1
        >>> max_points = 1000
        >>> log_array_sizes = True

        >>> eiscat3d = instances.eiscat3d # RADAR definition

        >>> # create scan and controller
        >>> scan = Fence(azimuth=90, min_elevation=30, dwell=t_slice, num=50)
        >>> scanner_controller = controllers.Scanner(profiler=p, logger=logger)

        >>> t = np.arange(0, end_t, t_slice)
        >>> controls = scanner_controller.generate_controls(t, eiscat3d, scan, t_slice=t_slice, max_points=max_points)

            
        Parameters
        ----------
        
        t : numpy.ndarray 
            Time points at which the controls are to be generated [s]
        radar : radar.system.Radar 
            Radar instance to be controlled
        scan : scans.Scan 
            Instance of the scanning algorithm used to perform the RADAR scanning scheme
        t_slice : float/numpy.ndarray 
            Array of time slice durations. The duration of the time slice for a given control must be less than or equal to the time step
        r : float/numpy.ndarray 
            Array of ranges from the transmitter beam at which the receiver will perform scans during a given time slice 
        scheduler : int priority (optional)
            Scheduler instance used for scheduling time sunchromization between controls for tims slicing. 
            Time slicing refers to the slicing of the time array into multiple subcontrol arrays (given as generator objects) to reduce memory (RAM) usage.
            This parameter is only useful when multiple controls are sent to a given scheduler.
            If the scheduler is not provided, the controller will slice the controls using the max_points parameter.
        priority : int priority (optional)
            Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority scheduler algorithms.
        max_points : int (optional)
            Max number of points for a given control array computed simultaneously. This number is used to limit the impact of computations over RAM
            Note that lowering this number might increase computation time, while increasing this number might cause problems depending on the available RAM on your machine
        
       
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
        
        Examples
        --------

        Suppose that there is a sinle Tx station performing a scan at 100 different time points. We alse suppose that 
        the echos are gathered by two receiver stations Rx that will perform a scan at 10 different altitudes at each time slice.
        Note that there will be only one control per time slice for the Tx controls since the role of Tx is to send a pulse 
        in the direction of the target. In theory there would be multiple pulses per time slice, but since the direction 
        would remain the same, we chose to only keep one direction control per time slice. Since the orientation for Tx 
        remains the same for a given time slice, the discrepency between the number of controls between Rx/Tx can simply 
        be solved by comparing the number of controls per time slice for Tx and Rx.
        
        The shape of the output "beam_direction_tx" array will be (1, 1, 100, 1, 3). 
        The shape of the output "beam_direction_rx" array will be (3, 1, 100, 10, 3). 
        
        - To get the z component of the beam direction of the Tx station at the second time step. 
            
            We assume that there is no Rx station associated with the Tx station and that there is only one control per time slice. 
            
            Therefore, we have :
            
            - Dimension 0: Tx id -> 0 (only one station)
            - Dimension 1: Rx id -> 0 (no Rx station associated with Tx)
            - Dimension 2: Time slice -> 2 (second time step)
            - Dimension 3: Control point -> 0 (only one control per time slice)
            - Dimension 4: (x, y, z) -> 2 (we want to get the z coordinate)
                
            Yielding :
            
            >>> ctrl = controls["beam_direction"]["tx"][0, 0, 2, 0, 2]
        
        - To get the x component of the beam direction of the Tx station at the 35th time step. 
        
            We assume that there is only one Tx station and that we want to get every orientation control per time slice. 
            There are no Rx stations associated with Tx.
            
            Therefore, we have :
                
            - Dimension 0: Tx id -> 0 (only one Tx station)
            - Dimension 1: Rx id -> 0 (no Rx station associated with Tx)
            - Dimension 2: Time slice -> 34 (second time step)
            - Dimension 3: Control point -> : (we want to get every orientation control for the given time slice)
            - Dimension 4: (x, y, z) -> 0 (we want to get the x coordinate)
            
            Yielding :
        
            >>> ctrl = controls["beam_direction"]["tx"][0, 0, 34, :, 0]
            
        - To get the z component of the beam direction of the first Rx station with respect to the 2nd scan at range r of the Tx beam at during the third time slice.
            
            Therefore, we have :
                
            - Dimension 0: Rx id -> 0 (first Rx station)
            - Dimension 1: Tx id -> 0 (only one Tx station)
            - Dimension 2: Time slice -> 2 (third time step)
            - Dimension 3: Control point -> 1 : (we want to get the second orientation control for the given time slice)
            - Dimension 4: (x, y, z) -> 2 (we want to get the z coordinate)
            
            >>> ctrl = controls["beam_direction"]["rx"][0, 0, 2, 1, 2]
         
        - To get the y component of the beam direction of the second Rx station with respect to the 5th scan at range r of the Tx beam at the 80th time slice.
            
            Therefore, we have :
                
            - Dimension 0: Rx id -> 1 (second Rx station)
            - Dimension 1: Tx id -> 0 (only one Tx station)
            - Dimension 2: Time slice -> 79 (80th time step)
            - Dimension 3: Control point -> 4 : (5th orientation control for the given time slice)
            - Dimension 4: (x, y, z) -> 1 (we want to get the y coordinate)
        
            >>> ctrl = controls["beam_direction"]["rx"][1, 0, 79, 4, 1]
      '''
        # add new profiler entry
        if self.profiler is not None:
            self.profiler.start('Scanner:generate_controls')
            
        # controls computation initialization
        # checks input values to make sure they are compatible with the implementation of the function
        if priority is not None:
            if not isinstance(priority, int): raise TypeError("priority must be an integer.")
            else: 
                if priority < 0: raise ValueError("priority must be positive [0; +inf] or equal to -1.")
             
        if not isinstance(radar, Radar): 
            raise TypeError(f"radar must be an instance of {Radar}.")

        # add support for both arrays and floats
        t = np.asarray(t)
        if len(np.shape(t)) > 1: raise TypeError("t must be a 1-dimensional array or a float")

        # Check if time slices are overlapping
        check_overlap_indices = radar_controller.check_time_slice_overlap(t, np.ones(np.size(t))*scan.dwell())
        if np.size(check_overlap_indices) > 0:
            if self.logger is not None:
                self.logger.warning(f"Tracker:generate_controls -> control time slices are overlapping at indices {check_overlap_indices}")
        del check_overlap_indices
        
        # split time array into scheduler periods if a scheduler is attached to the controls
        t, sub_controls_count = super()._split_time_array(t, scheduler, max_points)
        
        # output data initialization
        controls = dict()  # the controls structure is defined as a dictionnary of subcontrols
        controls["t"] = t  # save the time points of the controls
        controls["t_slice"] = np.ones(np.size(t))*scan.dwell() # save the dwell time of each time point
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
            controls["beam_orientation"].append(self.__compute_beam_orientation(t[subcontrol_index], radar, scan, r))

        if self.profiler is not None:
            self.profiler.stop('Scanner:generate_controls')
        
        return controls
