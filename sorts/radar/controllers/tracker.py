#!/usr/bin/env python

'''#TODO

'''

import numpy as np

from . import radar_controller
from ..system import Radar
from ..scheduler import RadarSchedulerBase
from .. import radar_controls

from sorts.common import interpolation

class Tracker(radar_controller.RadarController):
    '''
    Takes in ECEF points and a time vector and creates a tracking control.
    '''

    META_FIELDS = radar_controller.RadarController.META_FIELDS

    def __init__(self, profiler=None, logger=None, **kwargs):
        super().__init__(profiler=profiler, logger=logger, **kwargs)
        
        self.meta['controller_type'] = self.__class__
        
        if self.logger is not None:
            self.logger.info(f'Tracker:init')


    # def __get_states_split_indices(self, t, t_slice, t_states, target_states):
    #     dt = t[:, None] - t_states[None, :]
    #     check = np.logical_and(dt >= 0, dt < t_slice)
    #     del dt
        
    #     ind_states = np.where(check)[1]
    #     ind_ctrl = np.where(check)[0]
    #     del check
        
    #     target_group_transition_mask = (ind_ctrl[1:] - ind_ctrl[:-1] > 0)
    #     target_states = target_states[:, ind_states]
    #     del ind_states
        
    #     t = t[np.delete(ind_ctrl, np.array(np.where(target_group_transition_mask==False)[0]))]
    #     del ind_ctrl

    #     return t, target_states, target_group_transition_mask
    

    def __retreive_target_states(
            self, 
            controls, 
            t_states_sub,
            state_interpolator,
            states_per_slice,
            ):
        '''Retreives target's states given the time array and the state interpolator.

        Parameters:
        -----------
            t : numpy.ndarray
                splitted array of time points used to generate the controls. Each time point corresponds to the start of a time slice.
            t_slice : float/numpy.ndarray
                duration of a time slice.
            state_interpolator : sorts.interpolation.Interpolator
                class used to interpolate the states of the object (i.e. Legendre8, ...)
            states_per_slice : int
                number of target states per slice. This number can be used when multiple measurements are performed during a given time slice (i.e. for coherent integration)
        '''
        n_control_periods = controls.n_periods
        target_states   = np.empty((n_control_periods, ), dtype=object)

        t_states        = np.empty((n_control_periods, ), dtype=object)
        
        # find states within the control interval
        states_msk = np.logical_and(t_states_sub >= controls.t[0][0], t_states_sub <= controls.t[-1][-1] + controls.t_slice[-1][-1])
        
        if np.size(np.where(states_msk)[0]) > 0:
            t_shape = np.shape(controls.t) #[scheduling slice/control subarray][time points]
            
            # get start and end points of the object pass
            t_start = t_states_sub[states_msk][0]
            t_end = t_states_sub[states_msk][-1]
            del states_msk
            
            flag_found_pass = False
            keep = np.full((n_control_periods,), False, dtype=bool)
            
            # get the states for each time sub-array
            for period_id in range(len(controls.t)):   
                pass_msk = np.logical_and(controls.t[period_id] >= t_start, controls.t[period_id] + controls.t_slice[period_id] <= t_end) # get all time slices in the pass

                if np.size(np.where(pass_msk)) < len(pass_msk): 
                    if self.logger is not None:
                        self.logger.warning(f"tracker:__retreive_target_states: some incomplete tracking control slices have been discarded between t={controls.t[period_id][0]} and t={controls.t[period_id][-1]} seconds (control sub array {period_id})")
                
                # if there are some time slices inside the pass
                if np.size(np.where(pass_msk)[0]) > 0:
                    flag_found_pass = True      
                    keep[period_id] = True

                    dt_states = controls.t_slice[period_id][pass_msk]/float(states_per_slice) # time interval between states
                    t_states_sub = np.repeat(controls.t[period_id][pass_msk], states_per_slice).astype(np.float64) # initializes space object state sampling time array
                    
                    # add intermediate time points to sample space object states
                    for ix in range(states_per_slice):
                        t_states_sub[ix::states_per_slice] = t_states_sub[ix::states_per_slice] + ix*dt_states

                    t_states[period_id]           = t_states_sub
                    del t_states_sub

                    target_states[period_id]      = state_interpolator.get_state(t_states[period_id])[0:3, :].astype(np.float64) #[scheduling slice/control subarray][time points][xyz]
                else:                    
                    if flag_found_pass is True:
                        break

        # remove control periods where no space object states are present
        controls.remove_periods(keep)

        return target_states[keep], t_states[keep]


    def compute_pointing_direction(
            self, 
            controls,
            period_id, 
            args, 
            ):
        '''
        Compute the beam orientation for sub-arrays of radar controls. This function returns a genereator which can be used to compute the sub controls.
        '''
        interpolated_states, t_dirs = args

        # compute pointing directions for each control sub time array      
        if interpolated_states[period_id] is not None:
            if self.profiler is not None:
                self.profiler.start('Static:generate_controls:compute_beam_orientation')
            
            # initializing results
            pointing_direction = dict()
        
            # get the position of the Tx/Rx stations
            tx_ecef = np.array([tx.ecef for tx in controls.radar.tx], dtype=np.float64) # get the position of each Tx station (ECEF frame)
            rx_ecef = np.array([rx.ecef for rx in controls.radar.rx], dtype=np.float64) # get the position of each Rx station (ECEF frame)
            
            if self.profiler is not None:
                self.profiler.start('Static:generate_controls:compute_beam_orientation:tx')

            # Compute Tx pointing directions
            tx_dirs = interpolated_states[period_id][None, None, :, :] - tx_ecef[:, None, :, None]
            del tx_ecef
            
            pointing_direction['tx'] = tx_dirs/np.linalg.norm(tx_dirs, axis=2)[:, :, None, :] # the beam directions are given as unit vectors in the ecef frame of reference
            del tx_dirs
            
            if self.profiler is not None:
                self.profiler.stop('Static:generate_controls:compute_beam_orientation:tx') 
                self.profiler.start('Static:generate_controls:compute_beam_orientation:rx') 
            
            # compute Rx pointing direction
            rx_dirs = interpolated_states[period_id][None, None, :, :]  - rx_ecef[:, None, :, None]     
            del rx_ecef

            rx_dirs = np.repeat(rx_dirs, len(controls.radar.tx), axis=1)
            
            # save computation results
            pointing_direction['rx'] = rx_dirs/np.linalg.norm(rx_dirs, axis=2)[:, :, None, :] # the beam directions are given as unit vectors in the ecef frame of reference
            del rx_dirs

            pointing_direction['t'] = t_dirs[period_id]

            if self.profiler is not None:
                self.profiler.stop('Static:generate_controls:compute_beam_orientation:rx')
                self.profiler.stop('Static:generate_controls:compute_beam_orientation')
            
            # TODO : include this in radar -> RadarController.coh_integration(self.radar, self.meta['dwell'])
        else:
            pointing_direction = dict()

            pointing_direction['rx'] = []
            pointing_direction['tx'] = []
            pointing_direction['t'] = []

        return pointing_direction


    def generate_controls(
            self, 
            t, 
            radar, 
            t_states,
            target_states, 
            t_slice=0.1, 
            states_per_slice=1,
            interpolator=interpolation.Linear,
            scheduler=None,
            priority=None, 
            max_points=100,
            beam_enabled=True,
            cache_pdirs=False,
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
        t_states : numpy.ndarray 
            time point of each target state vector 
        target_states : numpy.ndarray 
            State vectors [1x6] of the target to be tracked by the radar system (given in the ECEF frame)
        t_slice : float/numpy.ndarray 
            Array of time slice durations. The duration of the time slice for a given control must be less than or equal to the time step
        state_per_slice (optional) : int
            number of target points per time slice
        interpolator (optional) : interpolation.Interpolator
            interpolator algorithm instance used to reconstruct the states of the target at each needed time point
        scheduler : sorts.Scheduler (optional)
            scheduler instance used for scheduling time sunchromization between controls for tims slicing. 
            Time slicing refers to the slicing of the time array into multiple subcontrol arrays (given as generator objects) to reduce memory (RAM) usage.
            This parameter is only useful when multiple controls are sent to a given scheduler.
            If the scheduler is not provided, the controller will slice the controls using the max_points parameter.
        priority : int (optional)
            Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority management algorithms.
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
                sorts.radar.controls_scheduler.RadarControlschedulerBase instance associated with the control array. This instance is used to 
                divide the controls into subarrays of controls according to the scheduler period to 
                reduce memory/computational overhead.
                If the scheduler is not none, the value of the period will take over the max points parameter (see above)
                
        - "priority"
            Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority management algorithms.
        
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

        if not issubclass(interpolator, interpolation.Interpolator):
            raise TypeError(f"interpolator must be an instance of {interpolation.Interpolator}.")
        else:
            state_interpolator = interpolator(target_states, t_states)
            
            if self.logger is not None:
                self.logger.info(f"Tracker:generate_controls -> creating state interpolator {state_interpolator}")
        
        # output data initialization
        controls = radar_controls.RadarControls(radar, self, scheduler=scheduler, priority=priority, logger=self.logger, profiler=self.profiler)  # the controls structure is defined as a dictionnary of subcontrols
        controls.interpolator = interpolator
        
        controls.set_time_slices(t, t_slice, max_points=max_points)

        # split time array into scheduler periods and target states if a scheduler is attached to the controls
        # TODO move split time array to radar controls
        target_states_interp, t_states_interp = self.__retreive_target_states(controls, t_states, state_interpolator, states_per_slice)

        # Compute controls
        pdir_args = (target_states_interp, t_states_interp)
        controls.set_pdirs(pdir_args, cache_pdirs=cache_pdirs)

        if self.profiler is not None:
            self.profiler.stop('Tracker:generate_controls')
        
        return controls