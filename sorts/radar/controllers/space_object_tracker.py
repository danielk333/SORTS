import numpy as np
import multiprocessing as mp
import ctypes

import pyorb
from astropy.time 		import Time, TimeDelta

from ..system.radar		import Radar
from .					import Tracker
from .					import radar_controller

from ...common 			import interpolation
from ..passes 			import find_simultaneous_passes, equidistant_sampling
from ...common 			import multiprocessing_tools as mptools

class SpaceObjectTracker(radar_controller.RadarController):
	'''
    Generates pointing direction controls allowing for the tracking of a list of space objects 
    '''

	META_FIELDS = radar_controller.RadarController.META_FIELDS

	def __init__(self, logger=None, profiler=None, **kwargs):
		''' This class can be used to generate a set of states (ECEF frame) which allow for the tracking of multiple space objects
		while satisfying a set of observational requirements. The generated states can then be sent to a Tracking controller to get the corresponding pointing directions.

		Parameters :
			logger (optional) : logging.logger
				logger instance used to log comptutation status on the terminal
			profiler (optional) : sorts.profiling.profiler
				profiler instance used to monitor the computational performances of the class' functions

		'''
		super().__init__(profiler=profiler, logger=logger)

		self.meta['space_objects'] = self.__class__

		if self.logger is not None:
		    self.logger.info(f'SpaceObjectTracker:Init')

	def generate_controls(
		self, 
		t, 
		radar, 
		space_objects, 
		epoch, 
		t_slice, 
		states_per_slice=1, 
		priority=None, 
		interpolator=None, 
		max_dpos=50e3, 
		max_samp=1000, 
		max_processes=16,
		scheduler=None, 
		controls_priority=None, 
		max_points=100,
		beam_enabled=True,
		save_states=False,
	):
		'''Generates RADAR tracking controls for a given radar and list of space objects to track. 
        This method can be called multiple times to generate different controls for different radar systems.
            
        Parameters
        ----------
		t : np.ndarray
			Time points at which the space objects are to be observed. Each time point correspond to the starting instant of a Tracking time slice.
		radar : sorts.radar.system.Radar
			Radar instance observing the space objects
		space_objects : list/np.ndarray of sorts.targets.SpaceObject
			Space objects to be observed during the specified time interval.
		epoch : int/float
			Time epoch (at the start of the simulation) given in MJD format (Modified Julian Date)
		t_slice : float/numpy.ndarray 
            Array of time slice durations. The duration of the time slice for a given control must be less than or equal to the time step
        state_per_slice (optional) : int
            number of target points per time slice
		priority (optional) : np.ndarray (int)
			priority of each space object (given in the same order as space_objects). if None, the priorities will be set automatically following the FIFO algorithm.
			If the priorities of two space are identical, the 
		interpolator (optional) : sorts.interpolation.Interpolation
			Interpolation algorithm used to reconstruct interpolate target states between propagated states.
			If no interpolator is profided, the default interpolation class used will be sorts.interpolation.Linear 
		max_dpos (optional) : int
			maximum distance in meters between two consecutive propagated states.
			The default value is 50km.
		max_samp (optional) : int
			maximum number of time samples used for target states propagation.
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
        
        See controllers.radar_controller for more information about how to use the controls data structure.
		
		Examples :
		----------

		To allow the tracking of multiple objects, one first needs to define the array of objects to be tracked.
		For example, an array of five objects can be defined as follows :

		>>> # Target propagator options 
		>>> Prop_cls = Kepler
		>>> Prop_opts = dict(
		>>>     settings = dict(
		>>>         out_frame='ITRS',
		>>>         in_frame='TEME',
		>>>     ),
		>>> )

		>>> # define objects' orbital parameters
		>>> orbits_a = np.array([7200, 7200, 8500, 12000, 10000])*1e3 # semi major axis - m
		>>> orbits_i = np.array([80, 80, 105, 105, 80]) # inclination - deg
		>>> orbits_raan = np.array([86, 86, 160, 180, 90]) # right ascension of ascending node - deg
		>>> orbits_aop = np.array([0, 0, 50, 40, 55]) # argument of perigee - deg
		>>> orbits_mu0 = np.array([60, 50, 5, 30, 8]) # mean anomaly - deg

		>>> # Initialization of each space object
		>>> space_objects = []
		>>> for so_id in range(len(orbits_a)):
		>>>     space_objects.append(space_object.SpaceObject(
		>>>             Prop_cls,
		>>>             propagator_options = Prop_opts,
		>>>             a = orbits_a[so_id], 
		>>>             e = 0.1,
		>>>             i = orbits_i[so_id],
		>>>             raan = orbits_raan[so_id],
		>>>             aop = orbits_aop[so_id],
		>>>             mu0 = orbits_mu0[so_id],
		>>>             
		>>>             epoch = epoch,
		>>>             parameters = dict(
		>>>                 d = 0.1,
		>>>            ),
		>>>         ))

		When space objects are created, one has to define a tracking priority for each one of them. The lower the scalar for the tracking priority is for a given object,
		the more importance will be given by the scheduler to the observation of this object.

		>>> priority = np.array([3, 2, 0, 1, 4]) # observational priority associated to each object
		
		Then, we can initialize the scheduler :

		>>> tracking_scheduler = TrackingScheduler(logger=logger, profiler=p)
		
		Finally, after defining the radar and the tracking time points, one can generate the tracking schedule :

		>>> eiscat3d = instances.eiscat3d # definition of the radar instance used to track the objects

		>>> # definition of tracking time points
		>>> t_start = 0
		>>> t_end = 100000
		>>> t_slice = 0.1
		>>> states_per_slice = 1
		>>> epoch = 53005.0
		>>> t_tracking = np.arange(t_start, t_end, tracking_period)

		>>> # generation of the tracking points in the ECEF frame
		>>> controls = tracking_scheduler.generate_controls(t_tracking, eiscat3d, space_objects, epoch, t_slice, states_per_slice, priority=priority)

		'''
		# check the validity of the input values
		space_objects = np.asarray(space_objects) # if list, convert space_objects to a numpy array

		# check if radar is an instance of sorts.Radar
		if not isinstance(radar, Radar): raise ValueError(f"radar must be an instance of {Radar}, not {radar.__class__}")

		# check priority
		if priority is None: # if none, the priority is generated automatically following the FIFO algorithm
			if self.logger is not None: self.logger.warning("TrackingScheduler:generate_schedule -> no priority information provided, using FIFO algorithm")
			priority = np.arange(0, len(space_objects), 1)
		else: # if not, checks if priority is a positive array of integers of the same size as space_objects
			priority = np.asarray(priority).astype(int) 

			if np.size(priority) < np.size(space_objects):
				raise ValueError(f"priority ({np.size(priority)}) must be the same size as space_objects ({np.size(space_objects)})")

			neg_priority_indices = np.where(priority < 0)[0]
			if np.size(neg_priority_indices) > 0:
				raise ValueError(f"priority must be positive. check indices {neg_priority_indices}")
			del neg_priority_indices

		# check/initialize interpolator
		if interpolator is None:
			if self.logger is not None: self.logger.warning("TrackingScheduler:generate_schedule -> no interpolator information provided, using interpolator.Linear")
			interpolator = interpolation.Linear

		# check/initialize epoch
		if isinstance(epoch, float):
			epoch = Time(epoch, format='mjd')

		# check time array 
		t = np.asarray(t).astype(float)
		if len(t) == 0:
			raise ValueError("t must be an array")

		if self.profiler is not None: self.profiler.start("TrackingScheduler:generate_schedule")

		# create the output arrays as shared arrays
		# Those arrays are shared between all processes
		# each process will updated the array values to get the final tracking states 
		mp_shared_final_states = mptools.convert_to_shared_array(np.empty((6, len(t)), dtype=float), ctypes.c_double)
		mp_shared_state_priorities = mptools.convert_to_shared_array(np.repeat([-1], len(t)).astype(np.int32), ctypes.c_int)
		mp_shared_object_indices = mptools.convert_to_shared_array(np.repeat([-1], len(t)).astype(np.int32), ctypes.c_int)

		# create final output arrays from the memory slots of the shared arrays
		final_states = mptools.convert_to_numpy_array(mp_shared_final_states, (6, len(t)))
		state_priorities = mptools.convert_to_numpy_array(mp_shared_state_priorities, (len(t)))
		object_indices = mptools.convert_to_numpy_array(mp_shared_object_indices, (len(t)))

		if self.logger is not None: self.logger.info(f"TrackingScheduler:generate_schedule -> Starting multiprocessing computations") 

		# function used to create each subprocess 
		def schedule_tracking(space_object_index, mutex):
			self.__compute_tracking_states(t, space_objects[space_object_index], space_object_index, radar, epoch, priority[space_object_index], interpolator, max_dpos, max_samp, mp_shared_final_states, mp_shared_state_priorities, mp_shared_object_indices, mutex)

		# subgroup of processes (size max_processes) will be created to propagate and find the states to track for each space object
		for process_subgroup_id in range(int(len(space_objects)/max_processes) + 1):
			if int(len(space_objects) - process_subgroup_id*max_processes) >= max_processes:
				n_process_in_subgroup = max_processes
			else:
				n_process_in_subgroup = int(len(space_objects) - process_subgroup_id*max_processes)

			mutex = mp.Lock() # create the mp.Lock mutex to ensure critical ressources sync between processes
			process_subgroup = []

			# initializes each process and associate them to an object in the list of targets to follow
			for i in range(n_process_in_subgroup):
				so_id = process_subgroup_id * max_processes + i # get the object's id
				process = mp.Process(target=schedule_tracking, args=(so_id, mutex,)) # create new process

				if self.logger is not None: self.logger.info(f"TrackingScheduler:generate_schedule -> (process pid {mptools.get_process_id()}) creating subprocess id {so_id}") 
				
				process_subgroup.append(process)
				process.start()

			# wait for each process to be finished
			for process in process_subgroup:
				process.join()

		# check if there are some empty spots in the target states array and removes them
		data_msk_ids = np.where(state_priorities != -1)[0]	

		if self.profiler is not None: self.profiler.stop("TrackingScheduler:generate_schedule")
		
		controller = Tracker(profiler=self.profiler, logger=self.logger)
		
		controls = controller.generate_controls(t[data_msk_ids], radar, t[data_msk_ids],final_states[:,data_msk_ids], t_slice=t_slice, states_per_slice=states_per_slice,interpolator=interpolator,scheduler=scheduler,priority=controls_priority, max_points=max_points, beam_enabled=beam_enabled)
		del controller

		if save_states is True:
			controls.object_indices = object_indices[data_msk_ids]
			controls.state_priorities = state_priorities[data_msk_ids]
			controls.space_objects_states = final_states[:,data_msk_ids]

		return controls

	def __compute_tracking_states(
		self, 
		t, 
		space_object, 
		space_object_index, 
		radar, 
		epoch, 
		priority, 
		interpolator, 
		max_dpos, 
		max_samp, 
		final_states_shared, 
		state_priorities_shared, 
		object_indices_shared, 
		mutex
	):
		''' Process subroutine used to gather the states of one of the object to follow.
			After computing the states, this function will add the states which satify the set of constraints to the schedule 
		'''
		if self.logger is not None: self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) object {space_object_index} starting")
		dt = (space_object.epoch - epoch).to_value("sec")

		# compute target states 
		t_states, states = self.__get_states(t, space_object, epoch, max_dpos) # propagates the target states
		interp_states = interpolator(states, t_states) # intitializes the interpolation object associated with those states

		# find simultaneous passes for all stations 
		passes = self.__get_passes(t_states, states, radar)
		del states

		# check if there are conflicts with existing final tracking states and modifies the schedule
		for pass_id in range(len(passes)):
			if self.profiler is not None: self.profiler.start("TrackingScheduler:generate_schedule:extract_states")
			if self.logger is not None: self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) Extracting states for radar pass {pass_id}")
			if self.logger is not None: self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) Found {len(passes[pass_id].inds)} time points")

			t_pass_start = t_states[passes[pass_id].inds][0]
			t_pass_end = t_states[passes[pass_id].inds][-1]

			# get shared memory arrays
			final_states = mptools.convert_to_numpy_array(final_states_shared, (6, len(t)))
			state_priorities = mptools.convert_to_numpy_array(state_priorities_shared, (len(t)))
			object_indices = mptools.convert_to_numpy_array(object_indices_shared, (len(t)))

			# get states in the pass which satisfy the constraints
			state_mask = np.logical_and(np.logical_or(state_priorities == -1, state_priorities > priority), np.logical_and(t >= t_pass_start, t <= t_pass_end))

			if self.logger is not None: self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) saving {len(np.where(state_mask)[0])} states")
			
			mutex.acquire()
			
			# modify the tracking schedule where constraints are statisfied 
			final_states[:, np.where(state_mask)[0]] = interp_states.get_state(t[state_mask] - dt)
			state_priorities[state_mask] = priority
			object_indices[state_mask] = space_object_index

			mutex.release()

			if self.profiler is not None: self.profiler.stop("TrackingScheduler:generate_schedule:extract_states")



	def __get_states(self, t, space_object, epoch, max_dpos):
		''' Process subroutine used to compute the ecef states of one of the object to follow.
		'''
		if self.logger is not None: self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) generating state time array")
		if self.profiler is not None: self.profiler.start("TrackingScheduler:generate_schedule:create_sampling_time_array")

		# generate states propagation time points  
		if isinstance(space_object.state, pyorb.Orbit): # by finding spatially equidistant time points
			t_states = equidistant_sampling(
			orbit = space_object.state, 
			start_t = t[0], 
			end_t = t[-1], 
			max_dpos = max_dpos,
		)
		else: # or by finding temporally equidistant time points
			t_states = np.arange(start_time, end_time, max_samp)		

		if self.profiler is not None: self.profiler.stop("TrackingScheduler:generate_schedule:create_sampling_time_array")	
		if self.profiler is not None: self.profiler.start("TrackingScheduler:generate_schedule:get_states")

		# propagates the object's states 
		states = space_object.get_state(t_states)
		
		if self.profiler is not None: self.profiler.stop("TrackingScheduler:generate_schedule:get_states")

		return t_states, states



	def __get_passes(self, t_states, states, radar):
		''' Process subroutine used to find the passes which are simultaneously in the FOV of all the radar stations
		'''
		if self.logger is not None: self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) searching for simultaneous passes")
		if self.profiler is not None: self.profiler.start("TrackingScheduler:generate_schedule:find_simultaneous_passes")

		passes = find_simultaneous_passes(t_states, states, [*radar.tx, *radar.rx])

		if self.profiler is not None: self.profiler.stop("TrackingScheduler:generate_schedule:find_simultaneous_passes")
		if self.logger is not None: self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) found {len(passes)} passes")

		return passes