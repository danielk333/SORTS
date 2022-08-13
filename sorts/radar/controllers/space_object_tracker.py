import numpy as np
import multiprocessing as mp
import ctypes

import pyorb
from astropy.time 		import Time, TimeDelta

from ...common 			import interpolation
from ...common 			import multiprocessing_tools as mptools
from ..system.radar		import Radar
from ..passes 			import find_simultaneous_passes, equidistant_sampling
from ..					import radar_controls
from .					import Tracker, radar_controller


class SpaceObjectTracker(radar_controller.RadarController):
	''' Generates a control sequence which tracks multiple objects based on their priority.

	The :class:`SpaceObjectTracker` controller generates a set of tracking controls allowing for 
    the tracking multiple space objects in time based on their priority.

    .. note::
        The tracking of space objects assumes that our knowledge of the objects' orbit at time
        :math:`t_0` is sufficient to propagate the orbits in time to determine the consecutive 
        ECEF points to be targetted by the radar system.

    When coherent/incoherent integration is used, the implementation of the controller allows for
    multiple pointing directions per time slice (but this number is set to be constant).

    .. seealso::

        * :class:`sorts.Radar<sorts.radar.system.radar.Radar>` : class encapsulating the radar system.
        * :class:`sorts.RadarController<sorts.radar.controllers.radar_congirtroller.RadarController>` : radar controller base class.
        * :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>` : class encapsulating radar control sequences.
        * :class:`sorts.SpaceObject<sorts.targets.space_object.SpaceObject>` : class encapsulating a space object.
    
    Parameters
    ----------
    profiler : :class:`sorts.Profiler<sorts.common.profing.Profiler>`, default=None
        Profiler instance used to monitor the computation performances of the class methods. 
    logger : :class:`logging.Logger`, default=None
        Logger instance used to log the computation status of the class methods.
    
    Examples
    --------
    .. _tracker_controller_example:

    This example showcases the generation of multiple controls sequences allowing to track a 
    :class:`Space object<sorts.targets.space_object.SpaceObject>` passing over the EISCAT_3D radar system:

    .. code-block:: Python

        import numpy as np
        import matplotlib.pyplot as plt

        import sorts

        # RADAR system definition
        eiscat3d = sorts.radars.eiscat3d

        # controller and simulation parameters
        max_points = 100
        end_t = 3600*12
        t_slice = 10
        tracking_period = 20
        states_per_slice = 10

        # Propagator
        Prop_cls = sorts.propagator.Kepler
        Prop_opts = dict(
            settings = dict(
                out_frame='ITRS',
                in_frame='TEME',
            ),
        )
        # Object definition
        space_object = sorts.SpaceObject(
                Prop_cls,
                propagator_options = Prop_opts,
                a = 7200.0e3, 
                e = 0.1,
                i = 80.0,
                raan = 86.0,
                aop = 0.0,
                mu0 = 50.0,
                epoch = 53005.0,
                parameters = dict(
                    d = 0.1,
                ),
            )
        # create state time array
        t_states = sorts.equidistant_sampling(
            orbit = space_object.state, 
            start_t = 0, 
            end_t = end_t, 
            max_dpos=50e3,
        )

        # create tracking controller
        tracker_controller = sorts.controllers.Tracker()

        # get object states/passes in ECEF frame
        object_states = space_object.get_state(t_states)
        eiscat_passes = sorts.find_simultaneous_passes(t_states, object_states, eiscat3d.tx + eiscat3d.rx)

        # plot results
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plotting station ECEF positions and earth grid
        for tx in eiscat3d.tx:
            ax.plot([tx.ecef[0]], [tx.ecef[1]], [tx.ecef[2]],'or')

        for rx in eiscat3d.rx:
            ax.plot([rx.ecef[0]], [rx.ecef[1]], [rx.ecef[2]],'og')

        sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

        # plotting object states
        ax.plot(object_states[0], object_states[1], object_states[2], "--b", alpha=0.2)

        # compute and plot controls for each pass
        for pass_id in range(np.shape(eiscat_passes)[0]):
            # get states within pass to generate tracking controls
            tracking_states = object_states[:, eiscat_passes[pass_id].inds]
            t_states_i = t_states[eiscat_passes[pass_id].inds]
            
            # generate controls
            t_controller = np.arange(t_states_i[0], t_states_i[-1] + tracking_period, tracking_period)
            controls = tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, max_points=max_points, states_per_slice=states_per_slice)
            
            # plot states being tracked
            ax.plot(tracking_states[0], tracking_states[1], tracking_states[2], "-", color="blue")
            
            # plot beam directions over each control period
            for period_id in range(controls.n_periods):
                ctrl = controls.get_pdirs(period_id)
                sorts.plotting.plot_beam_directions(ctrl, eiscat3d, ax=ax, tx_beam=True, rx_beam=True, zoom_level=0.9, azimuth=10, elevation=10)

        plt.show()

    .. figure:: ../../../../figures/example_tracker_controller.png

    '''
	
	META_FIELDS = radar_controller.RadarController.META_FIELDS

	def __init__(self, logger=None, profiler=None, **kwargs):
		''' 
		This class can be used to generate a set of states (ECEF frame) which allow for the tracking of multiple space objects while satisfying a set of observational 
		requirements.

		Parameters
		----------
			logger (optional) : logging.logger
				logger instance used to log comptutation status on the terminal
			profiler (optional) : sorts.profiling.profiler
				profiler instance used to monitor the computational performances of the class' functions

		'''
		super().__init__(profiler=profiler, logger=logger)

		self.tracker_controller = Tracker(logger=logger, profiler=profiler)

		if self.logger is not None:
		    self.logger.info(f'SpaceObjectTracker:Init')

	def compute_pointing_directions(
		self, 
		controls,
		period_id, 
		args,
	):
		''' 
		Computes the pointing directions over the specified period id. This method is used automatically by the RadarControls class to generate the 
		tracking pointing directions. 

		As such, the user can indirectly call this method by calling tracking_controls.get_pdirs(period_id) where tracking_controls is an instance
		of RadarControls which has been created by the SpaceObjectTracker controller.

		Parameters
		----------
			controls : sorts.RadarControls
				radar controls instance used to store the Radar controls generated by the SpaceObjectTracker controller.

			period_id : int
				index of the control period which corresponding pointing directions are to be computed. see RadarControls documentation for more information.

			args : list
				set of aditional arguments (relative to the compute_pointing_direction implementation of the controller) used to compute the Radar pointing
				directions. In the case of the SpaceObjectTracker controller args = (final_states, final_t_tracking), which correspond to the 
				array of interpolated states and time points where pointing directions are to be computed
		'''
		return self.tracker_controller.compute_pointing_directions(
			controls,
			period_id, 
			args,
		)

	def generate_controls(
		self, 
		t, 
		radar, 
		space_objects, 
		epoch, 
		t_slice, 
		states_per_slice=1, 
		space_object_priorities=None, 
		priority=None, 
		interpolator=None, 
		max_dpos=50e3, 
		max_samp=1000, 
		max_processes=16,
		scheduler=None, 
		max_points=100,
		save_states=False,
		cache_pdirs=False,
	):
		''' This method can be called to generate a sequence of radar controls which allow for the tracking of multiple space 
		objects. The generated controls will contain the pointing directions for each radar station. 

		The current implementation of this algorithm relies on the concept of static priority to choose which space object will 
		be tracked at a given time t. 
		It is important to note that when a specific space object is chosen for tracking at the start of the time slice, 
		this object will remain tracked until then end of the time slice, even if a higher priority object enters the field 
		of view.

		The generate_controls method consists of two distinct parts :
			- 	First the states of each space object are propagated over the whole control time interval (given by t). 
				When all the states are computed, data reduction will be achieved by removing all the states lying outside 
				of the radar system field of view. 
				Finally, the tracking schedule (i.e. the choice of which space object to follow) will be done according 
				to the priority of each space object.
			- 	When the tracking schedule is generated, the algorithm will then perform an interpolation to the 
				:math:`i^{th}` object for all tracking instants :math:`t_{ik}` associated with this space object. 

		As an example, consider that we want to track 3 space objects (0, 1, 2) of priority (0, 1, 2) at times 
		:math:`t = [0, 10, 20, 30]`

		The first part of the algorithm has propagated the states and has found the following passes (sequence of 
		points inside the FOV of all stations) :

			- Object 0 (p = 0) :        :math:`t_0^p = [10, 15]`
			- Object 1 (p = 1) :        :math:`t_0^p = [30]`
			- Object 2 (p = 2) :        :math:`t_0^p = [10, 20, 30]`

		Since all objects are outside the radar's FOV at time 0, we can remove the control time slice at :math:`t=0`, 
		yielding :math:`t = [10, 20, 30]`. We can then follow an iterative procedure : first we create an array 
		containing the index of the space object beaing tracked during the time slice :math:`t_k` :

		.. math:: C = [-1 \hspace{1.5mm} -1 \hspace{1.5mm} -1]

		Looking at the first object, we can see that at time :math:`t = 10s`, the object will be inside the radar FOV. 
		Therefore, we allocate the first time slice to the tracking of this space object :

		.. math:: C = [0 \hspace{1.5mm} -1 \hspace{1.5mm} -1]

		Then, we see that the second space object is inside the stations' FOV at time :math:`t = 30s`. Since there are 
		no conflicts with other space objects at this time, we allocate the last time slice to the tracking of this 
		space object :

		.. math:: C = [0 \hspace{1.5mm} -1 \hspace{1.5mm} 1]

		Finally, we can see that the last space object is visible during the whole control interval, but the first and 
		last time slices are already allocated for the tracking of higher priority objects. Therefore, the middle time 
		slice will be allocated for the tracking of the last space object :

		.. math:: C = [0 \hspace{1.5mm} 2 \hspace{1.5mm} 1]

		The last part of this algorithm consists in computing the states of the objects at each tracking point, and then 
		computing the pointing directions for each station.

        Parameters
        ----------
		t : numpy.ndarray (N,)
			Start time of Tracking time slices (in seconds). 
			This array limits the controls time interval between t[0] and t[-1].
			
			.. note::
				If no space objects are found at a given time slice, the latter will be discarded such that the final control
				array only contains time slices allocated for the observation of a space object.
		radar : :class:`sorts.Radar<sorts.radar.system.radar.Radar>`
			Radar instance being controlled to track the space objects. 
		space_objects : list / numpy.ndarray of :class:`sorts.SpaceObject<sorts.targets.space_object.SpaceObject>`
			Array of space objects being targeted during the specified time interval (t[0], t[-1]). 
		epoch : int / float (in ``MJD`` format)
			Time epoch (at the start of the simulation) given in MJD format (Modified Julian Date). This epoch is used 
			to synchronize the states of the space objects with the simulation time reference.
		t_slice : float / numpy.ndarray (N,)
            Time slice durations (in seconds). The duration of each time slice must be less or equal to the 
            time step between two consecutive time slices.
        states_per_slice : int, default=1
            Number of pointing directions per time slice. A space object can be tracked at multiple time points within 
            a given time slice. If ``state_per_slice`` is greater than one, then additional tracking points will be added 
            within time slices separated by a duration

            .. math:: \\delta t^k = \\frac{\\Delta t_{slice}^k}{n}.

            where :math:`\\Delta t_slice` is the duration of the :math:`k^{k}` time slice and :math:`n` is the number of 
            pointing directions per time slice.
		space_object_priorities : numpy.ndarray of int, default=None
			Priority of each space object (given in the same order as space_objects array). if None, the priorities 
			will be set automatically following the FIFO algorithm.
			Low numbers indicate a high tracking prioriy.
		priority : int, default=None
            Priority of the generated controls, only used by the scheduler to choose between overlapping controls. 
            Low numbers indicate a high control prioriy. ``None`` is used for dynamic priority management algorithms.
		interpolator : :class:`sorts.Interpolator<sorts.common.interpolation.Interpolator>, default=None
			Interpolation algorithm used to reconstruct interpolate target states between propagated states.
			If no interpolator is provided, the default interpolation class used will be 
			:class:`sorts.interpolation.Linear<sorts.common.interpolation.Linear>`. 
		max_dpos : float, default=50e3
			Maximum distance in meters between two consecutive propagated states.
		max_samp : int, default=1000
			Maximum number of time samples used for target states propagation.
		max_processes : int, default=16
			Maximum number of simultaneous processes used during the generation of the tracking schedule. 
		scheduler : :class:`sorts.RadarSchedulerBase<sorts.radar.scheduler.base.RadarSchedulerBase>`, default=None
            RadarSchedulerBase instance used for time synchronization between control periods.
            This parameter is useful when multiple controls are sent to a given scheduler.
            If the scheduler is not provided, the control periods will be generated using the ``max_points``
            parameter.
        max_points : default=100
            Max number of points for a given control array computed simultaneously. This number is used to limit the 
            impact of computations over RAM.

            .. note::
            	Lowering this number might increase computation time, while increasing this number might cause 
            	problems depending on the available RAM on your machine.

		save_states : bool, default=False
			If True, the states of the space objects will be saved within the ``meta`` field of the generated controls. 
			Setting ``save_states`` to True might drasticly increase your RAM usage.
		cache_pdirs : bool, default=False
			If True, the pointing directions computation results will be cached within the control structure. 
			Setting ``cache_pdirs`` to True might drasticly increase your RAM usage.

		Returns
        -------
        controls : :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            Radar control sequence to be applied to the radar to perform tracking of the the given array of space objects 
            during the control interval t. 
		
		Examples
		--------
		This example showcases the simultaneous tracking of 5 space objects by the ``EISCAT_3D`` radar system. 
		
		.. code-block:: Python

			import numpy as np
			import matplotlib.pyplot as plt

			import sorts

			# RADAR definition
			eiscat3d = sorts.radars.eiscat3d

			# Object definition
			# Propagator
			Prop_cls = sorts.propagator.Kepler
			Prop_opts = dict(
			    settings = dict(
			        out_frame='ITRS',
			        in_frame='TEME',
			    ),
			)

			# Object properties
			orbits_a = np.array([7200, 7200, 8500, 12000, 10000])*1e3 # m
			orbits_i = np.array([80, 80, 105, 105, 80]) # deg
			orbits_raan = np.array([86, 86, 160, 180, 90]) # deg
			orbits_aop = np.array([0, 0, 50, 40, 55]) # deg
			orbits_mu0 = np.array([60, 50, 5, 30, 8]) # deg
			priorities = np.array([4, 3, 1, 2, 5])
			epoch = 53005.0

			# Creating space objects
			space_objects = []
			for so_id in range(len(orbits_a)):
			    space_objects.append(sorts.SpaceObject(
			            Prop_cls,
			            propagator_options = Prop_opts,
			            a=orbits_a[so_id], 
			            e=0.1,
			            i=orbits_i[so_id],
			            raan=orbits_raan[so_id],
			            aop=orbits_aop[so_id],
			            mu0=orbits_mu0[so_id],
			            epoch=epoch,
			            parameters = dict(
			                d=0.1,
			            ),
			        ))



			# Radar controller parameters
			tracking_period = 50.0
			t_slice = 2.0
			t_start = 0.0
			t_end = 3600*5

			# intialization of the space object tracker controller
			so_tracking_controller = sorts.SpaceObjectTracker()

			# generate controls
			t_tracking = np.arange(t_start, t_end, tracking_period)
			controls = so_tracking_controller.generate_controls(t_tracking, eiscat3d, space_objects, epoch, t_slice, space_object_priorities=priorities, save_states=True)



			# plotting results
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')

			# Plotting station ECEF positions and earth grid
			sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res=100, color='black', hide_ax=True)
			for tx in eiscat3d.tx:
			    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')

			for rx in eiscat3d.rx:
			    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')       

			# plot all space object states
			for space_object_index in range(len(space_objects)):
			    states = controls.meta["objects_states"][space_object_index]
			    ax.plot(states[0], states[1], states[2], '--', label=f"so-{space_object_index} (p={priorities[space_object_index]})", alpha=0.35)


			# plot states being tracked and 
			ecef_tracking = controls.meta["tracking_states"]
			object_ids = controls.meta["state_priorities"]

			for period_id in range(controls.n_periods):
			    # compute transitions between the tracking of multiple objects (used to plot segments)
			    mask = np.logical_or(np.abs(controls.t[period_id][1:] - controls.t[period_id][:-1]) > tracking_period, object_ids[period_id][1:] - object_ids[period_id][:-1] != 0)
			    transition_ids = np.where(mask)[0]+1

			    # plot control sequence beam directions
			    ax = sorts.plotting.plot_beam_directions(controls.get_pdirs(period_id), eiscat3d, ax=ax, zoom_level=0.6, azimuth=10, elevation=20)

			    # plot states being tracked as segments
			    for i in range(len(transition_ids)+1):
			        if i == 0:
			            i_start = 0
			        else:
			            i_start = transition_ids[i-1]

			        if i == len(transition_ids):
			            i_end = len(t_tracking)+1
			        else:
			            i_end = transition_ids[i]
			             
			        ax.plot(ecef_tracking[period_id][0, i_start:i_end], ecef_tracking[period_id][1, i_start:i_end], ecef_tracking[period_id][2, i_start:i_end], '-b')
			             
			ax.legend()
			plt.show()

		.. figure:: ../../../../figures/example_spaceobject_tracker_controller.png
		
		'''
		# check the validity of the input values
		space_objects = np.asarray(space_objects) # if list, convert space_objects to a numpy array

		n_space_objects = len(space_objects)
		space_object_priorities = self.check_space_object_priorities(space_object_priorities, n_space_objects)

		# check/initialize interpolator
		if interpolator is None:
			if self.logger is not None: self.logger.warning("SpaceObjectTracker:generate_controls -> no interpolator information provided, using interpolator.Linear")
			interpolator = interpolation.Linear

		# check/initialize epoch
		if isinstance(epoch, float):
			epoch = Time(epoch, format='mjd')

		# check time array 
		t = np.asarray(t).astype(float)
		if len(t) == 0:
			raise ValueError("t must be an array")

		if self.profiler is not None: 
			self.profiler.start("SpaceObjectTracker:generate_controls")

		n_control_points = len(t)

		# create the output arrays as shared arrays
		# Those arrays are shared between all processes
		# each process will updated the array values to get the final tracking states 
		mp_shared_final_states 		= mptools.convert_to_shared_array(np.ndarray((6, n_control_points*states_per_slice), dtype=float), ctypes.c_double)
		mp_shared_state_priorities 	= mptools.convert_to_shared_array(np.full((n_control_points,), -1, np.int32), ctypes.c_int)
		mp_shared_object_indices 	= mptools.convert_to_shared_array(np.full((n_control_points,), -1, np.int32), ctypes.c_int)

		# create final output arrays from the memory slots of the shared arrays
		final_states 				= mptools.convert_to_numpy_array(mp_shared_final_states, 		(6, n_control_points*states_per_slice))
		state_priorities 			= mptools.convert_to_numpy_array(mp_shared_state_priorities, 	(n_control_points,))
		object_indices 				= mptools.convert_to_numpy_array(mp_shared_object_indices, 		(n_control_points,))

		# if save states is true, create arrays to save propagated states.
		if save_states is True:
			mp_shared_obj_states 	= mptools.convert_to_shared_array(np.ndarray((n_space_objects, 6, n_control_points), dtype=float), ctypes.c_double)
			obj_states 				= mptools.convert_to_numpy_array(mp_shared_obj_states, 		(n_space_objects, 6, n_control_points))
		else:
			mp_shared_obj_states 	= None
			obj_states 				= None

		# generate tracking time array (adding itermediate time points to reach states_per_slice)
		dt_tracking = t_slice/float(states_per_slice)

		final_t_tracking = np.repeat(t, states_per_slice)
		for ti in range(states_per_slice):
			final_t_tracking[ti::states_per_slice] = final_t_tracking[ti::states_per_slice] + dt_tracking*ti


		if self.logger is not None: 
			self.logger.info(f"SpaceObjectTracker:generate_controls -> Starting multiprocessing computations") 

		# function used to create each subprocess 
		def schedule_tracking(space_object_index, mutex):
			self.compute_tracking_states(
				t, 
				t_slice, 
				states_per_slice,
				final_t_tracking, 
				space_objects[space_object_index], 
				space_object_index, 
				n_space_objects,
				radar, 
				epoch, 
				space_object_priorities[space_object_index], 
				interpolator, 
				max_dpos,
				max_samp, 
				mp_shared_obj_states,
				mp_shared_final_states, 
				mp_shared_state_priorities, 
				mp_shared_object_indices, 
				mutex,
				save_states,
			)

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

				if self.logger is not None: 
					self.logger.info(f"SpaceObjectTracker:generate_controls -> (process pid {mptools.get_process_id()}) creating subprocess id {so_id}") 

				process_subgroup.append(process)
				process.start()

			# wait for each process to be finished
			for process in process_subgroup:
				process.join()

		# check if there are some empty spots in the target states array and removes them
		ctrl_msk = state_priorities != -1
		state_msk = np.repeat(ctrl_msk, states_per_slice)

		# creating output tracking controls
		controls = radar_controls.RadarControls(radar, self, scheduler=scheduler, priority=priority, logger=self.logger, profiler=self.profiler)  # the controls structure is defined as a dictionnary of subcontrols
		controls.meta["interpolator"] = interpolator

		controls.set_time_slices(t[ctrl_msk], t_slice, max_points=max_points)

		# remove points where no space objects were found and split arrays according to control time slices
		if controls.splitting_indices is not None:
			print(controls.splitting_indices)
			final_states 		= np.hsplit(final_states[:, state_msk][0:3], states_per_slice*controls.splitting_indices)
			final_t_tracking 	= np.hsplit(final_t_tracking[state_msk], states_per_slice*controls.splitting_indices)
			state_priorities 	= np.hsplit(state_priorities[ctrl_msk], controls.splitting_indices)
			object_indices 		= np.hsplit(object_indices[ctrl_msk], controls.splitting_indices)
		else:
			final_states 		= final_states[:, state_msk][0:3][None,...]
			final_t_tracking 	= final_t_tracking[state_msk][None,...]
			state_priorities 	= state_priorities[ctrl_msk][None,...]
			object_indices 		= object_indices[ctrl_msk][None,...]

		if self.profiler is not None: 
			self.profiler.stop("SpaceObjectTracker:generate_controls")

		# Compute controls
		pdir_args = (final_states, final_t_tracking)
		controls.set_pdirs(pdir_args, cache_pdirs=cache_pdirs)

		radar_controller.RadarController.coh_integration(controls, radar, t_slice)

		if save_states is True:
			controls.meta["object_indices"] = object_indices
			controls.meta["state_priorities"] = state_priorities
			controls.meta["tracking_states"] = final_states
			controls.meta["objects_states"] = obj_states

		return controls

	def compute_tracking_states(
		self, 
		t_controller, 
		t_slice,
		states_per_slice,
		final_t_tracking, 
		space_object, 
		space_object_index, 
		n_space_objects, 
		radar, 
		epoch, 
		space_object_priority, 
		interpolator, 
		max_dpos, 
		max_samp, 
		object_states_shared, 
		final_states_shared, 
		state_priorities_shared, 
		object_indices_shared, 
		mutex,
		save_states,
	):
		''' Process subroutine used to gather the states of one of the object to follow. 

		After computing the states, this function will add the states which satify the set of constraints to the schedule.

		First, the algorithm computes the states of the given space object over the entirety of the control time interval. Then, 
		it performs a data reduction procedure by only keeping the states which are within the field of view of all the stations. 
		Then, the algorithm will compare the priorities of the previous pointing direction controls and will overwrite the ones 
		with a lower priority compared to the current space object. Finally, the algorithm will performs an interpolation to 
		compute the states at each time point where a tracking pointing direction control will be applied (states_per_slice 
		pointing directions will be performed during each time slice).
		
		Parameters
		----------
		t_controller : numpy.ndarray (N,)
			Start time of Tracking time slices (in seconds). 
			This array limits the controls time interval between t[0] and t[-1].
			
			.. note::
				If no space objects are found at a given time slice, the latter will be discarded such that the final control
				array only contains time slices allocated for the observation of a space object.

		t_slice : float / numpy.ndarray (N,)
			Time slice durations (in seconds). The duration of each time slice must be less or equal to the 
            time step between two consecutive time slices.
		states_per_slice : int
			Number of pointing directions per time slice. A space object can be tracked at multiple time points within 
            a given time slice. If ``state_per_slice`` is greater than one, then additional tracking points will be added 
            within time slices separated by a duration

            .. math:: \\delta t^k = \\frac{\\Delta t_slice^k}{n}.

            where :math:`\\Delta t_slice` is the duration of the :math:`k^{k}` time slice and :math:`n` is the number of 
            pointing directions per time slice.
		final_t_tracking : numpy.ndarray (M,)
			Array containing the instants at which a pointing direction control will be performed to track a set of space objects.
		space_object : :class:`sorts.SpaceObject<sorts.targets.space_object.SpaceObject>`
			Space object to be tracked by the controller.
		space_object_index : numpy.ndarray (M,)
			Index of the space object to be tracked by the controller inside the population of objects tracked by 
			the controller.
		radar : :class:`sorts.Radar<sorts.radar.system.radar.Radar>`
			Radar instance being controlled to track the space objects. 
		epoch : float (in ``MJD`` format)
			Time epoch (at the start of the simulation) given in MJD format (Modified Julian Date). This epoch is used 
			to synchronize the states of the space objects with the simulation time reference.
		space_object_priority : int
			Priority of the current space object. The priority is used to choose which space object to track when multiple 
			space objects are within the field of view of the radar. In this case, the object of highest priority (i.e. 
			space_object_priority is the lowest number of all) at a given instant will be the one being tracked. 
		interpolator : :class:`sorts.Interpolator<sorts.common.interpolation.Interpolator>
			Interpolation algorithm used to reconstruct interpolate target states between propagated states.
			If no interpolator is provided, the default interpolation class used will be 
			:class:`sorts.interpolation.Linear<sorts.common.interpolation.Linear>`. 
		max_dpos : float
			maximum distance between two consecutive states when the equidistant sampling method is used when computing the space 
			object states. This sampling method is used when the state of the object (retreivable by calling space_object.state) 
			is given as a pyorb.Orbit instance.
		max_samp : int
			Maximum number of time points in the array when the constant time sampling method is used when computing the space 
			object states. This sampling method is used when the state of the object (retreivable by calling space_object.state) 
			is NOT given as a pyorb.Orbit instance, but for example as (6,1) vector.
		object_states_shared : multiprocessing.Array (n_objects, 6, N)
			Array (shared between processes) containing the states of each space objects evaluated at the start of each control 
			time slice. If ``save_states`` is False, this array will be None.
		final_states_shared, : multiprocessing.Array (6, M)
			Array (shared between processes) containing the states of the different space objects (ECEF coordinate frame) for each 
			pointing direction control points. 
		state_priorities_shared : multiprocessing.Array (N,)
			Array (shared between processes) containing the priorities of each pointing direction control points. The priority of 
			a given pointing direction control points correspond to the priority of the space object being tracked at this instant. 
		object_indices_shared : multiprocessing.Array (N,)
			Array (shared between processes) containing the space object id being tracked at each pointing direction control points. 
		mutex : processing.Lock
			Mutex instance used for synchronization between processes.

		Returns
		-------
		None
		'''
		if self.logger is not None: 
			self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) object {space_object_index} starting")

		# get epoch correction time 
		dt = (space_object.epoch - epoch).to_value("sec")

		# compute and interpolate target states 
		t_states, states = self.get_states(t_controller - dt, t_slice, space_object, epoch, max_dpos, max_samp) # propagates the target states
		state_interpolator = interpolator(states, t_states) # intitializes the interpolation object associated with those states

		# find simultaneous passes for all stations 
		passes = self.get_passes(t_states, states, radar)
		del states

		# get shared memory arrays
		state_priorities 	= mptools.convert_to_numpy_array(state_priorities_shared, (len(t_controller),))
		final_states 		= mptools.convert_to_numpy_array(final_states_shared, (6, len(t_controller)*states_per_slice))
		object_indices 		= mptools.convert_to_numpy_array(object_indices_shared, (len(t_controller),))

		# if save_states is true, save states for each time slice 
		if save_states is True:
			obj_states = mptools.convert_to_numpy_array(object_states_shared, (n_space_objects, 6, len(t_controller))) 
			obj_states[space_object_index] = state_interpolator.get_state(t_controller)

		# check if there are conflicts with existing final tracking states and modifies the schedule
		for pass_id in range(len(passes)):
			if self.profiler is not None: 
				self.profiler.start("TrackingScheduler:generate_schedule:extract_states")

			if self.logger is not None: 
				self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) Extracting states for radar pass {pass_id}")
				self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) Found {len(passes[pass_id].inds)} time points")

			# get states in the pass which satisfy the constraints (priority is greater than already scheduled observations (or free obs. slot available) and the controller time slice is inside the pass time interval) 
			pass_mask 		= np.logical_and(t_controller >= t_states[passes[pass_id].inds][0], t_controller <= t_states[passes[pass_id].inds][-1] - t_slice)	# remove control points outside of the pass
			priority_mask 	= np.logical_or(state_priorities == -1, state_priorities > space_object_priority)	# remove points where there is already a higher priority observation
			
			ctrl_mask 		= np.logical_and(pass_mask, priority_mask) # compute intersection of priority constraint and observational constraints
			state_mask 		= np.repeat(ctrl_mask, states_per_slice)
			del priority_mask, pass_mask
			
			if self.logger is not None: 
				self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) saving {len(np.where(state_mask)[0])} states")
			
			# modify the tracking schedule where constraints are statisfied 
			mutex.acquire()
			final_states[:, state_mask] = state_interpolator.get_state(final_t_tracking[state_mask] - dt)
			state_priorities[ctrl_mask] = space_object_priority
			object_indices[ctrl_mask] 	= space_object_index
			mutex.release()

			if self.profiler is not None: 
				self.profiler.stop("TrackingScheduler:generate_schedule:extract_states")


	def get_states(
		self, 
		t, 
		t_slice, 
		space_object, 
		epoch, 
		max_dpos, 
		max_samp,
	):
		''' Computes the ecef states of a given space object over the control time interval.

		This function computes the space object states 

		.. math:: \\mathbf{x^k} = [x^k, y^k, z^k, v_x^k, v_y^k, v_z^k]^T

		over the control interval given by ``t`` and ``t_slice``.

		Parameters
		----------
		t : numpy.ndarray (N,)
			Starting point of each controller time slice (in seconds). 
		t_slice : float / numpy.ndarray (N,)
			Duration of each controller time slice (in seconds). 
		space_object : :class:`sorts.SpaceObject<sorts.targets.space_object.SpaceObject>`
			Space object instance which states we wish to propagate in time.
		max_dpos : float
			Maximum distance (in meters) between two consecutive states when the equidistant sampling method is used (1st method).
		max_samp : int 
			Maximum number of time points in the array when the constant time sampling method is used (2nd method).

		Returns
		-------
		t_states : numpy.ndarray (M,)
			State sampling time array of the given space object (in seconds).
		states : numpy.ndarray (6, M)
			Propagated states of the space object over the given control time interval.
		'''
		t_states = self.get_sampling_time(t, t_slice, space_object, max_dpos, max_samp)

		# propagates the object's states 
		if self.profiler is not None: self.profiler.start("TrackingScheduler:generate_schedule:get_states")	
		states = space_object.get_state(t_states)		
		if self.profiler is not None: self.profiler.stop("TrackingScheduler:generate_schedule:get_states")

		return t_states, states



	def get_sampling_time(
		self, 
		t, 
		t_slice, 
		space_object, 
		max_dpos, 
		max_samp,
	):
		""" This function returns the array of sampling time points needed to propagate the states of a given space object. 

		There are two possible ways to generate sampling time arrays :
			1. When the states of the space objects are given as a pyorb.Orbit instance, the time points are generated to ensure that the distance between two consecutive states on the orbit stays constant. 
			2. If the states of the space object are not given as a pyorb.Orbit instance, the time points are instead generated to ensure that the time interval between two consecutive states stays constant.

		Parameters
		----------
		t : numpy.ndarray (N,)
			Starting point of each controller time slice (in seconds). 
		t_slice : float / numpy.ndarray (N,)
			Duration of each controller time slice (in seconds). 
		space_object : sorts.SpaceObject
			Space object instance which states we wish to propagate in time.
		max_dpos : float
			Maximum distance (in meters) between two consecutive states when the equidistant sampling method is used (1st method).
		max_samp : int 
			Maximum number of time points in the array when the constant time sampling method is used (2nd method).

		Returns
		-------
		t_states : numpy.ndarray (M,)
			State sampling time array of the given space object (in seconds).
		"""
		if self.logger is not None: 
			self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) generating state time array")

		if self.profiler is not None:
		 	self.profiler.start("TrackingScheduler:generate_schedule:create_sampling_time_array")

		# generate states propagation time points  
		if isinstance(space_object.state, pyorb.Orbit): # by finding spatially equidistant time points
			t_states = equidistant_sampling(
				orbit = space_object.state, 
				start_t = t[0], 
				end_t = t[-1] + t_slice*10, 
				max_dpos = max_dpos,
			)
		else: # or by finding temporally equidistant time points
			t_states = np.arange(t[0], t[-1] + t_slice, max_samp)		

		if self.profiler is not None: 
			self.profiler.stop("TrackingScheduler:generate_schedule:create_sampling_time_array")	

		return t_states



	def get_passes(
		self, 
		t_states, 
		states, 
		radar
	):
		''' Finds the passes which are simultaneously in the FOV of all the radar stations. 

		Parameters
		----------
		t_states : numpy.ndarray (N,)
			this array contains the time points (in seconds) at which the states were propagated. 
			It is assumed that the time values are given with respect to the space object's epoch.
		states : numpy.ndarray (6, N) 
			This array contains the space object's states 

			.. math:: \\mathbf{x^k} = [x^k, y^k, z^k, v_x^k, v_y^k, v_z^k]^T

			associated to each time point of ``t_states``.
		radar : sorts.Radar
			radar instance over which we want to compute the space object passes.

		Returns
		-------
		passes : numpy.ndarray
			Array of :class:`sorts.Pass<sorts.radar.passes.Pass>` objects which correspond to all 
			the radar passes found for the given state array and radar instance.
		'''
		if self.logger is not None: 
			self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) searching for simultaneous passes")

		if self.profiler is not None: 
			self.profiler.start("TrackingScheduler:generate_schedule:find_simultaneous_passes")

		passes = find_simultaneous_passes(t_states, states, [*radar.tx, *radar.rx])

		if self.profiler is not None: 
			self.profiler.stop("TrackingScheduler:generate_schedule:find_simultaneous_passes")

		if self.logger is not None: 
			self.logger.info(f"TrackingScheduler:generate_schedule -> (subprocess pid {mptools.get_process_id()}) found {len(passes)} passes")

		return passes


	def check_space_object_priorities(
		self, 
		space_object_priorities, 
		n_space_objects,
	):
		""" Checks the validity of the space_object_priorities array. 

		If space_object_priorities is None, the space objects priorities will be created following a FIFO algorithm.
		The space object priority is used to choose which space object will be tracked when multiple objects are within 
		range at the same time.

		Parameters
		----------
		space_object_priorities : numpy.ndarray
			Priority of each space object. If None, the priority of each space object will be set using the FIFO algorithm.
		n_space_objects : int
			Number of space objects.

		Returns
		-------
		space_object_priorities : numpy.ndarray
			returns the validated/generated space object priority array.
		"""
		if space_object_priorities is None: # if none, the priority is generated automatically following the FIFO algorithm
			if self.logger is not None: 
				self.logger.warning("TrackingScheduler:generate_schedule -> no priority information provided, using FIFO algorithm")
			
			space_object_priorities = np.arange(0, n_space_objects, 1)
		else: 
			# if not, checks if space_object_priorities is a positive array of integers of the same size as space_objects
			space_object_priorities = np.asarray(space_object_priorities).astype(int) 

			if np.size(space_object_priorities) < n_space_objects:
				raise ValueError(f"space_object_priorities ({np.size(space_object_priorities)}) must be the same size as space_objects ({n_space_objects})")

			neg_priority_indices = np.where(space_object_priorities < 0)[0]

			if np.size(neg_priority_indices) > 0:
				raise ValueError(f"space_object_priorities must be positive. check indices {neg_priority_indices}")

			del neg_priority_indices

		return space_object_priorities