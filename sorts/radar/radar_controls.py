import numpy as np
import copy

from . import scheduler
from .system.radar import Radar

class TimeSliceOverlapError(Exception):
	''' Raised when two control time slices are overlapping.

	This Error is raised when two time slices are overlapping, i.e. when the end point 
	of a time slice is greater that the starting point of the following time slice within
	the same control sequence. 
	'''
	pass

class NegativePriorityError(Exception):
	''' Raised when a priority is negative. '''
	pass

class ControlFieldError(Exception):
	''' Errors related to property controls. '''
	pass


class RadarControls(object):
	''' Encapsulates a radar control sequence.

	Radar controls are used to modify the **pointing directions** and **properties** of a station 
	(such as wavelength, power, ...) in time. This class provide a standard data structure which
	allows for the easy creation, storage and manipulation of radar control sequences.

	.. seealso ::
		:ref:`radar_controls` : radar controls module definitions

	Parameters
	----------
	radar : :class:`sorts.Radar<sorts.radar.system.radar.Radar>`
		:class:`sorts.Radar<sorts.radar.system.radar.Radar>` instance being controlled.
	controller : :class:`sorts.RadarController<sorts.radar.controllers.radar_controller.RadarController>`
		:class:`sorts.RadarController<sorts.radar.controllers.radar_controller.RadarController>` instance 
		handling the generation of the control instance.
	scheduler : :class:`sorts.RadarSchedulerBase<sorts.radar.scheduler.base.RadarSchedulerBase>`, default=None
		:class:`sorts.RadarSchedulerBase<sorts.radar.scheduler.base.RadarSchedulerBase>` instance handling the 
		generation of the control schedule for the current radar system. 
		In this class, the scheduler is used for time synchronization between multiple radar controls.
	priority : int, default=None
		Radar control priority. Used by the static priority scheduler to determine which control time slice will 
		be executed at a given time (see :class:`sorts.StaticPriorityScheduler<sorts.radar.scheduler.static_priority_scheduler.StaticPriorityScheduler>` for additional information). 
		Priority must be a positive integer.
	logger : :class:`logging.Logger`, default=None
		logging.Logger instance used to log the main execution steps of class methods
	profiler : :class:`logging.Profiler<sorts.common.profiling.Profiler>`, default=None
		Profiler instance used to measure the execution performances of class methods.		

	Raises
	------
	TypeError
		If ``radar`` is not an instance of :class:`sorts.Radar<sorts.radar.system.radar.Radar>`.
	:class:`NegativePriorityError`
		If ``priority`` is negative.
	'''
	def __init__(self, radar, controller, scheduler=None, priority=None, logger=None, profiler=None):
		''' Default class constructor. '''
		self.logger = logger
		''' Logger instance used to log the computation status within the class. '''
		self.profiler = profiler
		''' Profiler instance used to monitor the computation performances within the class. '''

		# check radar
		if not isinstance(radar, Radar): 
			raise TypeError(f"radar must be an instance of {Radar}.")

		self.radar = radar
		''' :class:`sorts.Radar<sorts.radar.system.radar.Radar>` instance being controlled by the control sequence. '''
		self.controller = controller 
		''' :class:`sorts.RadarController<sorts.radar.controllers.radar_controller.RadarController>` instance 
		which generated the control sequence. 
		'''
		self.scheduler = scheduler
		''' :class:`sorts.RadarSchedulerBase<sorts.radar.scheduler.base.RadarSchedulerBase>` instance which 
		generated the control sequence. 
		'''

		self._priority = priority
		''' Control sequence priority. 

		.. note:: priority must be a positive integer.
		''' 
		if self._priority is not None:
			self._priority = int(self._priority)

			if self._priority < 0:
				raise NegativePriorityError("priority must be positive [0; +inf]")

		self._t = None
		self._t_slice = None

		self.pdirs = None
		''' Control pointing directions. 

		The station **pointing direction** corresponds to the *normalized Poynting vector* of the transmitted signal 
		(i.e. its direction of propagation / direction of the beam). It is possible to have multiple pointing 
		direction vectors for a sinle station per time slice. This feature can therefore be used to model digital 
		beam steering of phased array receiver antennas (see radar_eiscat3d).
		''' 
		self.pdir_args = None
		''' :attr:`compute_pointing_directions<sorts.radar.controllers.radar_controller.RadarController.compute_pointing_directions>`
		method arguments used to generate the pointing direction sequence.

		.. seealso::
			:attr:`compute_pointing_directions<sorts.radar.controllers.radar_controller.RadarController.compute_pointing_directions>`
		''' 
		self.has_pdirs = False
		''' If ``True``, the current :class:`RadarControls` instance has cached pointing direction values. 

		Therefore, :attr:`RadarControls.get_pdirs` will return the cached pointing directions corresponding to the 
		given period index.
		''' 

		self.n_control_points = None
		''' Total number of control time slices. '''
		self.n_periods = None
		''' Total number of control periods. '''
		self.max_points = None
		''' Maximum number of control time slices per control period. '''
		self.splitting_indices = None
		''' List of indices corresponding to the transition between two consecuticve control periods. 

		This array is used to split property control arrays (1D arrays of length ``n_control_points``)
		into arrays of the same shape as :attr:`RadarControls.t` : (number of periods, number of time slices
		per period).
	
		Examples 
		--------
		>>> controls.t
		array([[0., 1.], [2., 3., 4.], [5.]])
		>>> control.splitting_indices
		array([[2, 5])
		'''

		# keep track of the parameters being contolled
		self.property_controls = None
		''' Radar properties controls. 
		
		This structure contains all the property controls for each radar
		station of the network. All the controls are splitted according to 
		the control periods and can be accessed as follow:

		>>> controls.property_controls[period_id][station_type][name][station_id]
		>>> # types :                numpy.ndarray    dict      dict numpy.ndarray

		Where:
		 * period_id : int
		 	index of the control period considered.
		 * station_type : str, "tx" / "rx" 
		 	Type of the station being controlled.
		 * name : str
		 	Name of the property being controlled (i.e. "wavelength", "n_ipp", ...).

		 	.. seealso::
		 		Refer to :attr:`TX.PROPERTIES<sorts.radar.system.station.TX.PROPERTIES>` 
		 		or :attr:`RX.PROPERTIES<sorts.radar.system.station.RX.PROPERTIES>` for more information
		 		about the radar station properties which can be controlled.

		 * station_id : int
		 	Index of the station considered. The index will correspond to the index of the
		 	TX station in the list :attr:`radar.tx<sorts.radar.system.radar.Radar.tx>` if  ``station_type=='tx'`` 
		 	and to the index of the RX station in the list :attr:`radar.rx<sorts.radar.system.radar.Radar.rx>` 
		 	if  ``station_type=='rx'``. 
		'''
		self.controlled_properties = dict()
		''' List of radar properties being controlled. 

		Contains a list of the controlled properties for each stations, which can be accessed 
		as follows :

		>>> controls.property_controls[station_type][station_id]
		>>> # types :                      dict     numpy.ndarray

		Where:
		 * station_type : str, "tx" / "rx" 
		 	Type of the station being controlled.
		 * station_id : int
		 	Index of the station considered. The index will correspond to the index of the
		 	TX station in the list :attr:`radar.tx<sorts.radar.system.radar.Radar.tx>` if  ``station_type=='tx'`` 
		 	and to the index of the RX station in the list :attr:`radar.rx<sorts.radar.system.radar.Radar.rx>` 
		 	if  ``station_type=='rx'``. 
		'''
		self.controlled_properties["tx"] = np.ndarray((len(self.radar.tx,)), dtype=object)
		self.controlled_properties["rx"] = np.ndarray((len(self.radar.rx,)), dtype=object)

		for station_type in ("tx", "rx"):
			for station_id in range(len(getattr(self.radar, station_type))):
				self.controlled_properties[station_type][station_id] = []

		self.meta = dict()
		''' Control structure metadata. '''
		self.meta['controller_type'] = self.controller.__class__
		self.meta['scheduler'] = self.scheduler


	def copy(self):
		''' 
		Performs a deepcopy of the radar control structure.

		Examples
		--------
		A deepcopy of a radar control structure ``controls`` can be achieved by 
		calling:

		>>> controls_copy = controls.copy()
		'''
		ret = RadarControls(self.radar, self.controller, scheduler=self.scheduler, priority=None, logger=self.logger, profiler=self.profiler)
		ret._t 				= copy.copy(self._t)
		ret._t_slice 		= copy.copy(self._t_slice)
		ret.pdir_args 		= copy.copy(self.pdir_args)
		ret.has_pdirs 		= copy.copy(self.has_pdirs)
		ret._priority 		= copy.copy(self.priority)
		ret.pdirs 			= copy.copy(self.pdirs)

		ret.n_control_points 		= self.n_control_points
		ret.n_periods 				= self.n_periods
		ret.max_points 				= self.max_points
		ret.splitting_indices 		= self.splitting_indices

		# setup controls
		ret.controlled_properties 	= copy.copy(self.controlled_properties)
		ret.property_controls 		= copy.copy(self.property_controls)

		ret.meta = copy.copy(self.meta)
		return ret


	def set_time_slices(self, t, duration, max_points=100):
		''' 
		Sets control time slice properties.

		This function is used to set the properties of a control time slice. Time slices are the
		most basic building bloc used for the definition of radar controls : it is an indivisible
		time interval during which a controller controls a given Radar system. In SORTS, time is
		continuous and the start time and duration of a time slice can take any values possible.

		A time slice can be defined a follow :
		
		.. math::	t_{i}^k = t_{start}^k \\\\
		.. math::	t_{f}^k = t_{start}^k + \\Delta t^k
		
		with :math:`\\Delta t^k` the duration of the time slice.

		The :attr:`set_time_slices<RadarControls.set_time_slices>` function will verify the validity 
		of the control time slices and separate the duration and starting points arrays into multiple 
		sub-arrays according to the scheduler period if provided, or according to the max_points criterion 
		(see :ref:`radar_controls` for more information).

		The control period can be defined in two different ways :

      - max_points :
	      By setting the maximum number of points max_points within each control period, 
	      the controls will be generated such that each control period contains max_points 
	      time points for period_id<n_periods-1 and less than max_points time points for 
	      period_id == n_periods-1.

      - scheduler :
      	The scheduler period (defined by the scheduler start time ``t0`` and the duration 
      	scheduler_period), all the will be generated such that all control periods coincide 
      	with the scheduler periods. Beware that while the controller and scheduler periods 
      	will correspond to the sime time intervals, the index of the controller period in 
      	the control arrays will be different from the corresponding scheduler period if 
      	the start time of the scheduler t0 is different from the start time of the controller

      .. note::
      	When using multiple radar controls, the control periods must be generated using the second 
      	method (scheduler) to ensure time synchronization between each the periods of different 
      	controls.

		Parameters
		----------
		t : 1D ndarray of floats
			start time of each control time slice. 
		duration : 1D ndarray of floats / float
			Duration of each time slice. 
			If duration is a float, each time slice will have the same duration. 
			For varying time slice durations, a 1D np.ndarray of the same size and shape as
			t must be provided
			Must positive a positive float.
		max_points : int, default=100
			Maximum number of control points (time slices) per time slice sub-array (used for the
			splitting of time arrays into multiple control periods).

			.. note::
				High values of ``max_points`` increase RAM usage.

		Returns
		-------
		None

		Raises
		------
		:class:`TimeSliceOverlap`
			If two time slices are overlapping.
		ValueError
			- If the duration of a time slice is negative.
			- If the time slice duration and start point arrays are not of the same size.
		TypeError
			If t is not a one dimensional array of floats.
		'''
		if max_points <= 0 or not isinstance(max_points, int):
			raise ValueError("max_points must be a positive integer")

		if np.size(t[0]) == 1 and not isinstance(t[0], np.ndarray) and t[0] is not None:
			# both the time slice and the time arrays are protected to ensure that they are the same size and add verification
			t 			= np.atleast_1d(t).astype(np.float64)
			duration 	= np.atleast_1d(duration).astype(np.float64)

			# check time array  
			if len(np.shape(t)) > 1: 
				raise TypeError("t must be a 1-dimensional array or a float")

			self.n_control_points = len(t)

			# check time slice array
			if np.where(duration < 0)[0] > 0: 
				raise ValueError("duration must be positive [0; +inf]")
			else:
				if np.size(duration) == 1:
					duration = np.repeat(duration, np.size(t)) 

			if len(duration) != self.n_control_points:
				raise ValueError(f"t and duration must be the same size (size {self.n_control_points} and {len(duration)}).")

			# check if time slices are overlapping
			if self.check_time_slice_overlap(t, duration):
				raise TimeSliceOverlapError("Time slices are overlapping")

			self._t = t
			self._t_slice = duration
			self._priority = np.repeat(self._priority, self.n_control_points)
			
			# split time slices according to scheduler periods or max points for performance
			self.max_points = max_points
			self.get_splitting_indices()

			# slice arrays according to scheduler period  or max_points requirements
			self._t 		= self.split_array(self._t)
			self._t_slice 	= self.split_array(self._t_slice)
			self._priority 	= self.split_array(self._priority)
		else:
			if len(t) != len(duration):
				raise Exception("t must be the same length as the duration array")

			self.splitting_indices = np.ndarray((len(t),), dtype=int)
			print(t)
			start_time = t[0][0]//self.scheduler.scheduler_period*self.scheduler.scheduler_period

			index = 0
			for period_id in range(len(t)):
				if t[period_id] is not None:
					index += len(t[period_id])
					self.splitting_indices[period_id] = index

					if self.scheduler is not None:
						print(t[period_id][-1])
						if t[period_id][-1] > start_time + (period_id + 1)*self.scheduler.scheduler_period or t[period_id][0] < start_time + period_id*self.scheduler.scheduler_period:
							raise Exception("t is not synchronized with the scheduler period. Please provide a valid splitted time array t or use the automatic splitting feature by calling set_time_slices with flat time arrays")
					else:
						if len(t[period_id]) > max_points:
							raise Exception(f"time subarrays have more elements than max_points (max_points={max_points}). Please provide a valid splitted time array t or use the automatic splitting feature by calling set_time_slices with flat time arrays ")

					if len(t[period_id]) != len(duration[period_id]):
						raise Exception("t must be the same length as the duration array for each period index")
				else:
					self.splitting_indices[period_id] = -1

					if duration[period_id] is not None:
						raise Exception("t must be the same length as the duration array for each period index")

			self._t = t
			self._t_slice = duration
			self.n_control_points = 0
			self.n_periods 	= len(self._t)

			tmp_priority = self._priority
			self._priority = np.ndarray((len(t),), dtype=object)

			for period_id in range(len(t)):
				if self._t[period_id] is not None:
					self.n_control_points += len(self._t[period_id])
					self._priority[period_id] = np.repeat(np.atleast_1d(tmp_priority), len(self._t[period_id]))
				else:
					self._priority[period_id] = None

		# set array of property controls		
		self.property_controls = np.ndarray((self.n_periods,), dtype=object)
		for period_id in range(self.n_periods):
			self.property_controls[period_id] = dict() # station types

			for station_type in ("tx", "rx"):
				self.property_controls[period_id][station_type] = dict() # property names


	def remove_periods(self, mask):
		''' Removes specific control periods.

		This function deletes specific contol periods and updates the control
		structure after removal. 

		Parameters
		----------
		mask : numpy.array of bool (``n_periods``)
			Mask used to remove specific control periods. If ``mask[i]`` is ``True``, then the 
			control period of index ``i`` will be removed. 
		Returns
		-------
		None

		Raises
		------
		Exception :
			If time slices are not set (i.e., the method :attr:`RadarControls.set_time_slices` hasn't been called).
		ValueError :
			If the length of the removal mask is not equal to the number of control periods. 

		Examples
		--------
		Consider a radar control sequence ``controls`` containing 3 control periods such that :

		>>> controls.t
		array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

		To remove all but the second period:

		>>> controls.remove_periods([True, False, True])
		>>> controls.t
		array([[0.3, 0.4]])
		'''
		if self._t is None:
			raise Exception("no time slice parameters set, please call set_time_slices() instead")

		if len(mask) != self.n_periods:
			raise ValueError("the length of the removal mask must be the same length as the number of control periods")

		# the number of points is the same for all periods but the last, so treat the last separatly to get total number of points remaining
		self.n_control_points = 0
		for period_id in range(self.n_periods):
			if self._t[period_id] is not None:
				self.n_control_points += len(self._t[period_id])*mask[period_id]

		# update time slices and priority
		self._t 				= self._t[mask]
		self._t_slice 			= self._t_slice[mask]
		self._priority 			= self._priority[mask]

		# update property controls
		self.property_controls 	= self.property_controls[mask]
		self.n_periods 			= len(self._t)

		# updates pointing directions
		if self.pdirs is not None:
			self.pdirs = self.pdirs[mask]


	@property
	def t(self):
		''' Time slice start time. '''
		return self._t

	@property
	def t_slice(self):
		''' Time slice duration. '''
		return self._t_slice

	@property
	def priority(self):
		''' Time slice priority. '''
		return self._priority

	@priority.setter
	def priority(self, value):
		''' Set time slice priority. '''
		if self._t is None:
			raise AttributeError("Control sequence time slices are not set, please set the control time slice properties before setting the priority.")

		# check validity of value
		error = False
		split = False
		if isinstance(value, int) or isinstance(value, float): # if integer, convert to array
			if value <= 0:
				error = True
			else:
				value = np.full((self.n_control_points,), int(value), int)
				split = True
		else:
			if not isinstance(value, np.ndarray): # if not array, error
				raise ValueError("priority must be a positive integer or an array")
			else:
				shape_error = False
				if len(value) != self.n_control_points:
					if len(value) != self.n_periods:
						shape_error = True
					else:
						for period_id in range(self.n_periods):
							if np.size(value[period_id]) != len(controls.t[period_id]):
								shape_error = True	
							else:
								value[period_id] = value[period_id].astype(int)
				else:
					split = True

		if split is True:
			value = self.split_array(value)
		if error is True:
			raise ValueError("priority must have the same shape as control.t or must be of length controls.n_control_points.")
	


	def set_pdirs(self, pdir_args, cache_pdirs=False):
		''' Sets the needed arguments to generate radar pointing directions.

		This function performs an initialization of the pointing direction 
		computations. If the ``cache_pdirs`` option is True, then the 
		pointing direction values are stored within the RAM and are returned
		when calling 

		>>> controls.get_pdirs(period_id)

		Or,

		>>> controls.pdirs

		Parameters
		----------
		pdir_args : list
			List of arguments required by the function :attr:`sorts.RadarController.compute_pointing_directions
			<sorts.radar.radar_controller.RadarController.compute_pointing_directions>`.

			.. seealso::
				Refer to List of arguments required by the function :attr:`sorts.RadarController.compute_pointing_directions
				<sorts.radar.controllers.radar_controller.RadarController.compute_pointing_directions>` to get more information about
				the way pointing directions must be generated.

		cache_pdirs : bool, default=False
			Wether the pointing directions are stored within RAM or calculated at each call of 
			:attr:`sorts.RadarControls.get_pdirs<sorts.radar.radar_controls.RadarControls.get_pdirs>`

		Returns 
		-------
		None
		'''
		self.pdir_args = pdir_args
		self.has_pdirs = True

		# if pdir arrays are cached as numerical values
		if cache_pdirs is True:
			self.pdirs = np.ndarray((self.n_periods,), dtype=object)

			for period_id in range(self.n_periods):
				# the function compute_pointing_directions is called by passing the arguments specific to the current controller and a reference to the 
				# controls instance
				self.pdirs[period_id] = self.controller.compute_pointing_directions(self, period_id, self.pdir_args)


	def get_pdirs(self, period_id):
		''' Computes the pointing direction over a single period.

		This function computes (if :attr:`RadarControls.has_pdirs<sorts.radar.radar_controls.RadarControls.has_pdirs>` 
		is ``False``) returns the radar pointing directions over a single control 
		period.

		.. note::
			Before calling this function, it is necessary to make sure that the function :attr:`RadarControls.set_pdirs
			<sorts.radar.radar_controls.RadarControls.set_pdirs>` has already been called and that the controller (if 
			:attr:`RadarControls.has_pdirs<sorts.radar.radar_controls.RadarControls.has_pdirs>` is ``False``) associated with the
			current radar controls structure is correctly defined.  
		
		The computation of the pointing directions is achieved by calling the function 
		:attr:`compute_pointing_directions<sorts.radar.controllers.radar_controller.RadarController.compute_pointing_directions>` 
		of the controller which has generated the current :class:`RadarControls` structure. 

		.. seealso::
			:class:`RadarController<sorts.radar.controllers.radar_controller.RadarController>` : class encapsulating the radar controller which is responsible for generating radar control sequences.

		Parameters
		----------
		period_id : int
			Index of the period over which the pointing directions are to be computed.

		Returns
		-------
		pdirs : dict
			Pointing direction computation results. The data is organized as a dictionnary with 3 keys :

			- "tx":
				Contains the pointing directions of all radar :class:`sorts.TX<sorts.radar.system.station.TX>` stations.
				The data within ``pdirs["tx"]`` is organized as follows :

				>>> pdirs['tx'][txi, 0, i, j]

				With :
				- txi :
					Index of the :class:`sorts.TX<sorts.radar.system.station.TX>` station within the :attr:`Radar.tx<sorts.radar.system.radar.Radar.tx>` list.
				- i :
					:math:`i^{th}` component of the pointing direction. :math:`i \\in [\\![ 0, 3 [\\![`
				- j : 
					:math:`j^{th}` time point.
					Beware that since there can be multiple pointing directions per time slice, the number of pointing directions
					for a single station is greater or equal to the number of time slices of the control sequence.

			- "rx":
				Contains the pointing directions of all radar :class:`sorts.RX<sorts.radar.system.station.RX>` stations.
				The data within ``pdirs["rx"]`` is organized as follows :

				>>> pdirs['rx'][rxi, txi, i, j]

				With :
				- txi :
					Index of the :class:`sorts.RX<sorts.radar.system.station.RX>` station within the :attr:`Radar.rx<sorts.radar.system.radar.Radar.rx>` list.
				- i :
					:math:`i^{th}` component of the pointing direction. :math:`i \\in [\\![ 0, 3 [\\![`
				- j : 
					:math:`j^{th}` time point.
					Beware that since there can be multiple pointing directions per time slice, the number of pointing directions
					for a single station is greater or equal to the number of time slices of the control sequence.
			
			- "t": 
				Contains the pointing direction time array. When there are more than one pointing direction per time slice, the
				number of time points within the pointing direction time array will be greater than the number of time slices within the 
				control sequence.

		Examples
		--------
		Refer to the examples within the documentation of the :attr:`compute_pointing_directions<sorts.radar.controllers.radar_controller.RadarController.compute_pointing_directions>` 
		method of each one of the controller types (scanner, tracker, ...).
		'''
		if self.has_pdirs is True:
			if self.pdirs is not None:
				pointing_direction = self.pdirs[period_id]
			else:
				# the function compute_pointing_directions is called by passing the arguments specific to the current controller and a reference to the 
				# controls instance
				pointing_direction = self.controller.compute_pointing_directions(self, period_id, self.pdir_args)
		else:
			pointing_direction = None

		return pointing_direction


	def add_property_control(
		self, 
		name, 
		station, 
		data):
		'''
		Sets the control data corresponding to the specified property and station.

		Parameters
		----------
		name : str
			Name of the station property to be controlled. 

			.. note:: 
				The property must be a controlled property of the station to be controlled. Refer to 
				:attr:`TX.PROPERTIES<sorts.radar.system.station.TX.PROPERTIES>` 
		 		or :attr:`RX.PROPERTIES<sorts.radar.system.station.RX.PROPERTIES>` for more information
		 		about the radar station properties which can be controlled.

		station : :class:`sorts.Station<sorts.radar.system.station.Station>`
			Station instance to be controlled.
		data : float / numpy.ndarray
			Property values for each time slice. The array must have the same shape as :attr:`RadarControls.t`
			or the shape (:attr:`RadarControls.n_control_points`).

		Returns
		-------
		None

		Raises
		------
		ValueError :
		 	If the name of the property is not a string.
		Exception :
			If the size of ``data`` is not the same as the one described in the **Parameters** section.
		ControlFieldError :
			if the station has no controllable property named ``name``.

		See Also
		--------
		:attr:`RadarControls.create_new_property_control_field`
		
		:attr:`RadarControls.split_array`

		Examples
		--------
		Consider a control structure with the following time slice start time array :

		>>> controls.t
		array([[0.1, 0.2], [0.3, 0.4], [0.5]])

		If we want to add a new entry to control the wavelength (0.6) of the first TX radar station, we need to run :

		>>> add_property_control("wavelength", controls.radar.tx[0], 0.6)
		>>> controls.get_property_control("wavelength", controls.radar.tx[0], period_id=0)
		array([0.6, 0.6])
		>>> controls.get_property_control("wavelength", controls.radar.tx[0], period_id=1)
		array([0.6, 0.6])
		>>> controls.get_property_control("wavelength", controls.radar.tx[0], period_id=2)
		array([0.6])
		'''
		if not isinstance(name, str):
			raise ValueError("name must be a string")

		station = np.atleast_1d(station)
		
		# check array size (if not already splitted : split data array)
		if np.size(data) == 1:
			data = np.repeat(np.atleast_1d(data), self.n_control_points)

		if np.size(data) != self.n_periods and not isinstance(data[0], np.ndarray):
			if np.size(data) != self.n_control_points:
				raise Exception("the control data shall either be of the shape (n_control_points,), where n_control_points is the total number of time points inside the time slice array, or (n_periods,...) where n_periods is the number of control periods")
			else:
				data = self.split_array(data)

		# create new control field if doesn't already exist
		self.create_new_property_control_field(name, station)	
		print("created new field : ", name)	
		print("in : ", self.controlled_properties)	

		# add control for each station
		for station_ in station:
			# get type and id of station
			station_id = self.radar.get_station_id(station_)
			station_type = station_.type
			
			for period_id in range(self.n_periods):
				self.property_controls[period_id][station_type][name][station_id] = data[period_id]


	def create_new_property_control_field(self, name, stations):
		''' Adds a new empty property control field to the :class:`radar controls<RadarControls>` for 
		all stations in ``stations``.

		.. warning::
			One must set the time slices before creating new radar control entries (since they are created 
			over each control period).
		
		Parameters
		----------
		name : str
			Name of the property field to be created. 
			The function will create a new property control for each control period.

			.. note::
				The property must be a controllable property of the station (see Refer to 
				:attr:`TX.PROPERTIES<sorts.radar.system.station.TX.PROPERTIES>` or :attr:
				`RX.PROPERTIES<sorts.radar.system.station.RX.PROPERTIES>` for more information
		 		about the radar station properties which can be controlled.)

		stations : list of :class:`sorts.Station<sorts.radar.system.station.Station>`
			List of stations which property will be controlled. The new control field will be created over all stations 
			present in ``stations``.

		Returns 
		-------
		None

		Raises
		------
		ControlFieldError :
			if the station has no controllable property named ``name``.
		'''
		for station_ in stations:
			if not name in station_.PROPERTIES:
				raise ControlFieldError(f"station {station_} has no control variable named {name}. Available controls are : {station_.get_properties()}")
			else:
				station_type = station_.type
				station_id = self.radar.get_station_id(station_)

				# if there is no control field of name ``name``, add new field and create new control arrays at each period id
				if name not in self.controlled_properties[station_type][station_id]:
					self.controlled_properties[station_type][station_id].append(name)

					# create new field for each period
					for period_id in range(self.n_periods):
						if not name in self.property_controls[period_id][station_type].keys():
							self.property_controls[period_id][station_type][name] = np.ndarray((len(getattr(self.radar, station_type)),), dtype=object)
				
	def get_property_control(self, name, station, period_id):
		''' Gets the control data corresponding to the specified control period id.

		Parameters
		----------
		name : str
			Name of the station property to be controlled. 

			.. note:: 
				The property must be a controlled property of the station to be controlled. Refer to 
				:attr:`TX.PROPERTIES<sorts.radar.system.station.TX.PROPERTIES>` 
		 		or :attr:`RX.PROPERTIES<sorts.radar.system.station.RX.PROPERTIES>` for more information
		 		about the radar station properties which can be controlled.

		station : :class:`sorts.Station<sorts.radar.system.station.Station>`
			Station instance to be controlled.
		period_id : int
			Control period index which controls we want to extract.

		Returns
		-------
		property_controls : ndarray (n_periods, ...)
			Array of controls for the control_variable. if ``period_id`` is provided then ``n_periods=1``, else n_periods will be the 
			total number of periods :attr:`RadarControls.n_periods`.

		Raises
		------
		ControlFieldError :
			If the station has no controllable property named ``name``.
		ValueError :
			If ``name`` is not a string.

		Examples
		--------
		Consider a control structure with the following time slice start time array :

		>>> controls.t
		array([[0.1, 0.2], [0.3, 0.4], [0.5]])

		If we want to add a new entry to control the wavelength (0.6) of the first TX radar station, we need to run :

		>>> add_property_control("wavelength", controls.radar.tx[0], 0.6)

		To get the corresponding controls, we need to run:
		
		>>> controls.get_property_control("wavelength", controls.radar.tx[0], period_id=0)
		array([0.6, 0.6])
		>>> controls.get_property_control("wavelength", controls.radar.tx[0], period_id=1)
		array([0.6, 0.6])
		>>> controls.get_property_control("wavelength", controls.radar.tx[0], period_id=2)
		array([0.6])
		'''
		if not isinstance(name, str):
			raise ValueError("name must be a string")

		# get type and id of station
		station_id = self.radar.get_station_id(station)
		station_type = station.type
		
		if not name in station.PROPERTIES:
			raise ControlFieldError(f"station {station} has no control variable named {name}. Available controls are : {station.get_properties()}")

		return self.property_controls[period_id][station_type][name][station_id]


	def get_property_control_list(self, station):
		''' Returns the list of controlled properties of a station.

		This function returns the list of properties being controlled in the current control structure  
		for a given station.

		Parameters
		----------
		station : :class:`sorts.Station<sorts.radar.system.station.Station>`
			Station instance which list of controlled properties we want to get.

		Returns
		-------
		properties : list of str
			List of ``station`` properties being controlled by the controls structure.

		Examples
		--------
		Consider a control structure named ``controls`` controlling the properties "wavelength" and "n_ipp" of a Station 
		instance named ``station``. To get the list of controlled properties, one can run:

		>>> controls.get_property_control_list(station)
		["wavelength", "n_ipp"]
		'''
		# get type and id of station
		station_id = self.radar.get_station_id(station)
		station_type = station.type

		properties = []

		for period_id in range(self.n_periods):
			for property_name in station.PROPERTIES:
				if property_name in self.property_controls[period_id][station_type].keys() and property_name not in properties:
					if self.property_controls[period_id][station_type][property_name][station_id] is not None:
						properties.append(property_name)

		return properties


	def get_control_period_id(self, scheduler_period_id):
		''' Computes the control period associated with a given scheduler period.

		This function computes the control period corresponding to a given scheduler period index.
		If there is no control if index 'control_id' at the given control period, then the function 
		will return -1, and if there is no scheduler associated with the control structure, the function
		will return ``scheduler_period_id``.

		Parameters
		----------
		scheduler_period_id : int
			Index of the scheduler period.

		Returns 
		-------
		ctrl_period_id : int
			Index of the control period corresponding to the given scheduler period.

		Examples
		--------
		Consider the scheduler and control sequence given by the following figure :

		.. figure:: ../../../figures/example_control_period.png

		The control period corresponding to ``scheduler_period_id=1`` is ``ctrl_period_id=0``, and the one corresponding to the	
		``scheduler_period_id=4`` is ``ctrl_period_id=3``.
		'''
		if self.scheduler is None: 
			ctrl_period_id = scheduler_period_id
		else:
			ctrl_period_id = scheduler_period_id - int((self._t[0][0] - self.scheduler.t0)/self.scheduler.scheduler_period)  # computes the time subarray id

			# the time subarray id is bigger than the number of time subarrays in the given control structure
			if ctrl_period_id < 0 or ctrl_period_id > len(self._t)-1: 
				ctrl_period_id = -1

		return ctrl_period_id


	def check_time_slice_overlap(self, t, duration):
		''' Checks wether or not time slices overlap within a given control array.

		If two different time slices overlap, the function will return the indices 
		of the time points which overlap.

		Parameters
		----------
		t : numpy.ndarray (N,)
			Time slice start time (in seconds).
		duration : numpy.ndarray (N,)
			Time slice duration (in seconds).
		'''
		# Logging execution status
		if self.logger is not None: 
			self.logger.info("checking time slice overlap")

		overlap = False
		
		# compute time interval between start time and previous time slice end time
		dt = t[1:] - (t[:-1] + duration[:-1])
		# get indices where the difference is negative (i.e. a time slice has started 
		# before the previous one ended).
		superposition_mask_ids = np.where(dt < -1e-10)[0]

		if np.size(superposition_mask_ids) > 0:
			if self.logger is not None: 
				self.logger.info(f"Time slices overlapping at transition indices {superposition_mask_ids}. Stopping")
			
			overlap = True

		return overlap


	def split_array(self, array):
		''' Split an array according to the scheduler period and start time (if the scheduler is provided).       
		
		If the scheduler is None, then the time array will be splitted to ensure that the number of time 
		points in a given controls subarray does not exceed max_points.

		Parameters
		----------
		array : numpy.ndarray (N,)
			Array to be splitted.

		Returns
		-------
		splitted_array : numpy.ndarray (:attr:`RadarControls.n_periods`, ``n_points_per_period[i]``)
			Array splitted according to the control periods. ``n_points_per_period[i]`` corresponds to 
			the number of time slices inside the :math:`i^{th}` control period. 
		'''
		# if the splitting indices have not been computed, run computation first
		if self.n_periods is None and self.splitting_indices is None:
			self.splitting_indices 	= self.get_splitting_indices()

		# Split arrays according to transition indices
		if self.splitting_indices is not None:
			splitted_array = np.ndarray((self.n_periods,), dtype=object)
			print(self.splitting_indices)
			print(self.n_periods)

			id_start = 0
			for period_id in range(self.n_periods):
				if self.splitting_indices[period_id] == -1: # if the periods does not contain any controls
						splitted_array[period_id] = None
				else:
					# get end index of the control period in the linear array
					id_end = self.splitting_indices[period_id]

					# copy values from the array in the corresponding control period
					splitted_array[period_id] = array[id_start:id_end]
					id_start = id_end
		else:
			splitted_array = array[None, :]

		return splitted_array


	def get_splitting_indices(self):
		''' Computes the indices at which the time and control arrays must be 
		sliced to meet the scheduler/max_points requirements.
		
		.. note::
			Control time slices must have been set before calling this function. 
			To set the control time slices, call the function :attr:`RadarControls.set_time_slices`.
		'''
		self.splitting_indices = None

		if self._t is None or self._t_slice is None:
			raise Exception("time slices must be set before computing splitting_indices. please set time slices by calling : set_time_slices(t, duration)")
		
		# get period transition indices
		if self.scheduler is not None:
			if self.logger is not None:
				self.logger.info("radar_controls:get_splitting_indices -> using scheduler master clock")
				self.logger.info("radar_controls:get_splitting_indices -> skipping max_points (max time points limit)")

			if not issubclass(self.scheduler.__class__, scheduler.RadarSchedulerBase):
				raise ValueError(f"scheduler has to be an instance of {RadarSchedulerBase}, not {scheduler}")
			
			start_time = self._t[0]//self.scheduler.scheduler_period*self.scheduler.scheduler_period

			self.n_periods = int((self._t[-1] - start_time)//self.scheduler.scheduler_period)+1
			self.splitting_indices = np.ndarray((self.n_periods,), dtype=int)

			i_start = 0
			for period_id in range(self.n_periods):
				# compute next time index (last point inside the current period)
				if period_id < self.n_periods-1:
					i_end = int(np.argmax(self._t[i_start:] >= start_time + (period_id+1)*self.scheduler.scheduler_period)) + i_start - 1
				else:
					i_end = len(self._t) - 1

				# if the point has been found within the control array
				add_index = False
				if i_end > -1:
					if i_start == i_end:
						if self._t[i_end] < start_time + (period_id+1)*self.scheduler.scheduler_period: # if equal and t_end inside control period, add a single point
							add_index = True
					else:
						if self._t[i_end] - self._t[i_start] > self.scheduler.scheduler_period or i_start > i_end:
							add_index = False
						else:
							add_index = True

				if add_index is True:
					self.splitting_indices[period_id] = i_end+1
					i_start = i_end+1
				else:
					self.splitting_indices[period_id] = -1
		else:
			print("ok")
			if self.logger is not None:
				self.logger.info("radar_controls:get_splitting_indices -> No scheduler provided, skipping master clock splitting...")
				self.logger.info(f"radar_controls:get_splitting_indices -> using max_points={self.max_points} (max time points limit)")

			if(np.size(self._t) > self.max_points):                
				self.splitting_indices = np.arange(self.max_points, np.size(self._t), self.max_points, dtype=int)
				self.splitting_indices = np.append(self.splitting_indices, np.size(self._t))
				self.n_periods = len(self.splitting_indices)
			else:
				self.n_periods = 1 


	def __str__(self):
		''' Overloaded __str__ method. '''
		print("RADAR Controls structure : ")
		print("")

		if self.n_periods is not None:
			print("Pointing direction controls (RadarControls.get_pdirs(period_id)) :")
			for period_id in range(self.n_periods):
				print("Period ", period_id, " : ", self.get_pdirs(period_id))

			print("\n-------------- Optional/Property controls --------------")

			for station in self.radar.tx + self.radar.rx:
				station_id = self.radar.get_station_id(station)
				print(f"Station (type {station.type}, id {station_id} property controls : ")

				for name in self.controlled_properties[station.type][station_id]:
					periods = []
					for period_id in range(self.n_periods):
						if self.t[period_id] is not None:
							if self.property_controls[period_id][station.type][name][station_id] is not None:
								periods.append(period_id)

					print(f"    - {name} -> (periods {periods})")
								
			print("\n-------------------------------------------------------")
		else:
			print("time and time slice arrays not set.")

		return ""
