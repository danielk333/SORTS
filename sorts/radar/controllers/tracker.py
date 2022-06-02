#!/usr/bin/env python

'''#TODO

'''

import numpy as np

from . import radar_controller
from ..system import Radar
from ..scheduler import Scheduler


class Tracker(radar_controller.RadarController):
    '''
    Takes in ECEF points and a time vector and creates a tracking control.
    '''


    META_FIELDS = radar_controller.RadarController.META_FIELDS + [
        'scan_type',
    ]


    def __init__(self, profiler=None, logger=None, **kwargs):
        super().__init__(profiler=profiler, logger=logger, **kwargs)
        
        self.meta['scan_type'] = self.__class__
        
        if self.logger is not None:
            self.logger.info(f'Tracker:init')

    

    def __get_states_split_indices(self, t, t_slice, t_states, target_states):
        dt = t[:, None] - t_states[None, :]
        check = np.logical_and(dt >= 0, dt < t_slice)
        del dt
        
        ind_states = np.where(check)[1]
        ind_ctrl = np.where(check)[0]
        del check
        
        target_group_transition_mask = (ind_ctrl[1:] - ind_ctrl[:-1] > 0)
        target_states = target_states[:, ind_states]
        del ind_states
        
        t = t[np.delete(ind_ctrl, np.array(np.where(target_group_transition_mask==False)[0]))]
        del ind_ctrl

        return t, target_states, target_group_transition_mask
    

    def __split_time_array(
            self, 
            t, 
            split_mask, 
            target_states, 
            scheduler, 
            max_points
            ):
        # get splitted time array (using scheduler and max_points)
        t, sub_controls_count = super()._split_time_array(t, scheduler, max_points)
        
        # intitialize new states array
        target_states_splitted = np.ndarray(np.size(t), dtype=object)
        
        t_split_index = 0
        
        # split states array according to the scheduler period and to the controller time slices
        for ti, t_sub in enumerate(t):            
            t_size = np.size(t_sub)
            
            split_indices = np.array(np.where(split_mask[t_split_index:t_split_index+t_size] == True)[0]) + 1

            if ti < len(t) - 1:
                split_indices = split_indices[0:-1]
            
            target_states_splitted[ti] = np.array(np.split(target_states[0:3, t_split_index:t_split_index+t_size], split_indices, axis=1), dtype=object).transpose(0, 2, 1)
            del split_indices
            
            t_split_index += t_size
        
        del split_mask        
        
        return t, target_states_splitted, sub_controls_count


    def _compute_beam_orientation(
            self, 
            t,
            radar, 
            target_ecef,
            ):
        '''
        Compute the beam orientation for sub-arrays of radar controls. This function returns a genereator which can be used to compute the sub controls.
        '''
        
        if self.profiler is not None:
            self.profiler.start('Static:generate_controls:compute_beam_orientation')
        
        # initializing results
        beam_controls = dict()
    
        # get the position of the Tx/Rx stations
        tx_ecef = np.array([tx.ecef for tx in radar.tx], dtype=float).reshape((len(radar.tx), 3)) # get the position of each Tx station (ECEF frame)
        rx_ecef = np.array([rx.ecef for rx in radar.rx], dtype=float) # get the position of each Rx station (ECEF frame)
        
        if self.profiler is not None:
            self.profiler.start('Static:generate_controls:compute_beam_orientation:tx')
        
        # Compute Tx pointing directions
        tx_dirs = target_ecef[None, None, :, :, :] - tx_ecef[:, None, None, None, :]
        del tx_ecef
        
        beam_controls['tx'] = radar_controller.normalize_direction_controls(tx_dirs, logger=self.logger) # the beam directions are given as unit vectors in the ecef frame of reference
        del tx_dirs
        
        if self.profiler is not None:
            self.profiler.stop('Static:generate_controls:compute_beam_orientation:tx') 
            self.profiler.start('Static:generate_controls:compute_beam_orientation:rx') 
        
        # compute Rx pointing direction
        rx_dirs = target_ecef[None, None, :, :, :]  - rx_ecef[:, None, None, None, :]     
        del rx_ecef
        
        # save computation results
        beam_controls['rx'] = radar_controller.normalize_direction_controls(rx_dirs, logger=self.logger) # the beam directions are given as unit vectors in the ecef frame of reference
        del rx_dirs
        
        if self.profiler is not None:
            self.profiler.stop('Static:generate_controls:compute_beam_orientation:rx')
            self.profiler.stop('Static:generate_controls:compute_beam_orientation')
        
        # TODO : include this in radar -> RadarController.coh_integration(self.radar, self.meta['dwell'])
        
        yield beam_controls
    

    def generate_controls(
            self, 
            t, 
            radar, 
            t_states,
            target_states, 
            t_slice=0.1, 
            scheduler=None,
            priority=None, 
            max_points=100,
            ):
        '''Generates RADAR tracking controls for a given radar and sampling time and target. 
        This method can be called multiple times to generate different controls for different radar systems.
        
        Usage
        -----
        
        One can generate tracking controls for a given target as follows :

            >>> # Target properties
            >>> orbit_a = 7200     # semi-major axis - km
            >>> orbit_i = 80       # inclination - deg
            >>> orbit_raan = 86    # longitude of the ascending node - deg
            >>> orbit_aop = 0      # argument of perigee - deg
            >>> orbit_mu0 = 50     # mean anomaly - deg

            >>> # Target instanciation
            >>> space_object = space_object.SpaceObject(
            >>>         Prop_cls,
            >>>         propagator_options = Prop_opts,
            >>>         a = orbit_a, 
            >>>         e = 0.1,
            >>>         i = orbits_i,
            >>>         raan = orbit_raan,
            >>>         aop = orbit_aop,
            >>>         mu0 = orbit_mu0,
            >>>         
            >>>         epoch = 53005.0,
            >>>         parameters = dict(
            >>>             d = 0.1,
            >>>         ),
            >>>     )
            
            >>> Compute target states
            >>> t_states = equidistant_sampling(orbit = space_object.state, start_t = 0, end_t = end_t, max_dpos=50e3) # create state time array
            >>> object_states = space_object.get_state(t_states) # computes object states in ECEF frame
            >>> eiscat_passes = find_simultaneous_passes(t_states, object_states, [*eiscat3d.tx, *eiscat3d.rx]) # reduce state array
            
            >>> # create controller
            >>> t_slice = 0.2
            >>> t_controller = np.arange(0, end_t, t_slice)
            >>> tracker_controller = controllers.tracker.Tracker(logger=logger, profiler=p) # instantiate controller
            
            >>> # compute and plot controls for each pass
            >>> for pass_id in range(np.shape(eiscat_passes)[0]):
            >>>     tracking_states = object_states[:, eiscat_passes[pass_id].inds]
            >>>     t_states_i = t_states[eiscat_passes[pass_id].inds]
            >>>    # generate controls 
            >>>    controls = tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, max_points=10)

            
        Parameters
        ----------
        
        t : numpy.ndarray 
            Time points at which the controls are to be generated [s]
        radar : radar.system.Radar 
            Radar instance to be controlled
            
        TODO
        target_states : float 
            Azimuth of the target beam
        azimuth : float 
            Elevation of the target beam
        scan : scans.Scan
            Scan instance used to generate the scanning controls
            
        t_slice : float/numpy.ndarray 
            Array of time slice durations. The duration of the time slice for a given control must be less than or equal to the time step
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
            
            >>> ctrl = controls["beam_direction_tx"][0, 0, 2, 0, 2]
        
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
        
            >>> ctrl = controls["beam_direction_tx"][0, 0, 34, :, 0]
            
        - To get the z component of the beam direction of the first Rx station with respect to the 2nd scan at range r of the Tx beam at during the third time slice.
            
            Therefore, we have :
                
            - Dimension 0: Rx id -> 0 (first Rx station)
            - Dimension 1: Tx id -> 0 (only one Tx station)
            - Dimension 2: Time slice -> 2 (third time step)
            - Dimension 3: Control point -> 1 : (we want to get the second orientation control for the given time slice)
            - Dimension 4: (x, y, z) -> 2 (we want to get the z coordinate)
            
            >>> ctrl = controls["beam_direction_rx"][0, 0, 2, 1, 2]
         
        - To get the y component of the beam direction of the second Rx station with respect to the 5th scan at range r of the Tx beam at the 80th time slice.
            
            Therefore, we have :
                
            - Dimension 0: Rx id -> 1 (second Rx station)
            - Dimension 1: Tx id -> 0 (only one Tx station)
            - Dimension 2: Time slice -> 79 (80th time step)
            - Dimension 3: Control point -> 4 : (5th orientation control for the given time slice)
            - Dimension 4: (x, y, z) -> 1 (we want to get the y coordinate)
        
            >>> ctrl = controls["beam_direction_rx"][1, 0, 79, 4, 1]
      '''
        # add new profiler entry
        if self.profiler is not None:
            self.profiler.start('Tracker:generate_controls')
            
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
        
        # split time array into scheduler periods and target states if a scheduler is attached to the controls
        t, target_states, split_mask = self.__get_states_split_indices(t, t_slice, t_states, target_states)
        t, target_states, sub_controls_count = self.__split_time_array(t, split_mask, target_states, scheduler, max_points)
                
        # output data initialization
        controls = dict()  # the controls structure is defined as a dictionnary of subcontrols
        controls["t"] = t  # save the time points of the controls
        controls["t_slice"] = np.ones(np.size(t))*t_slice # save the dwell time of each time point
        controls["priority"] = priority # set the controls priority
        controls["enabled"] = True # set the radar state (on/off)
        
        # setting metadata
        controls["meta"] = dict()
        controls["meta"]["radar"] = radar
        controls["meta"]["controller_type"] = "tracker"
        controls["meta"]["scheduler"] = scheduler # set the radar state (on/off)
        controls["meta"]["sub_controls_count"] = sub_controls_count
        
        # creating orientation controls generator array
        controls["beam_orientation"] = []
        
        # Computations
        # compute controls for each time sub array
        for subcontrol_index in range(sub_controls_count):
            t_sub = t[subcontrol_index]
            target_ecef = target_states[subcontrol_index]
            
            controls["beam_orientation"].append(self._compute_beam_orientation(t, radar, target_ecef))
    
        if self.profiler is not None:
            self.profiler.stop('Tracker:generate_controls')
        
        return controls