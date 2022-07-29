"""
====================================================
Chebyshev Series (:mod:`numpy.polynomial.chebyshev`)
====================================================

This module provides a the classes and tools useful for the definition of 
Radar controls

Classes
-------
.. autosummary::
   :toctree: generated/

    TimeSliceOverlapError
	NegativePriorityError
	RadarControls

Notes
-----
Radar controls are used to modify the properties of a station (such as 
wavelength, pointing direction, power, ...) in time.

The most basic component of a radar control is called a `Time slice`. A
time slice is an indivisible time interval during which a set of controls 
are applied to the stations of a given Radar system. A time slice is defined
by its start time and its duration

For performance reasons, SORTS divides control sequences (i.e. array of 
station parameter values at each time slice) into sub arrays called `control 
periods`. Within a given control period, operations are done on homogeneous 
arrays to increase performances. But to reduce the impact on RAM usage, the
user can also choose to do only perform computations over a single control
period at a time. 




"""
import numpy as np
import copy

from . import scheduler
from .system.radar import Radar

class TimeSliceOverlapError(Exception):
	""" 
	Raised when two control time slices are overlapping

	This Error is raised when two time slices are overlapping, i.e. when the end point 
	of a time slice is greater that the starting point of the following time slice within
	the same control sequence. 
	"""
	pass

class NegativePriorityError(Exception):
	""" 
	Raised when a priority is negative.
	"""
	pass

class ControlFieldError(Exception):
	""" 
	Errors related to property controls.
	"""
	pass

class RadarControls(object):
	""" 
	Encapsulates a set of Radar controls.

	This class encapsulates a set of radar controls which are used to temporarily modify 
	the properties of the stations within a specified Radar system.
	"""
	def __init__(self, radar, controller, scheduler=None, priority=None, logger=None, profiler=None):
		"""
		Default class constructor.

		Parameters
		----------
		radar : sorts.Radar instance
			Radar instance being controlled.
		controller : sorts.RadarController instance
			RadarController instance handling the generation of the control instance.
		scheduler : sorts.Scheduler instance, optional
			Scheduler instance handling the generation of the control schedule for the current radar system. 
			In this class, the scheduler is used for time synchronization between multiple radar controls.
		priority : sorts.Scheduler instance, optional
			Priority of the controls.
			Used by the scheduler to determine which control time slice will be executed at a given time (see 
			..mod::`sorts.radar.scheduler` for additional information)
			Must be a positive integer, or None (Default)
		logger : sorts.Scheduler instance, optional
			logging.Logger instance used to log the main execution steps of class methods
		profiler : sorts.Scheduler instance, optional
			Profiler instance used to measure the execution performances of class methods.		

		Raises
		------
		TypeError
			If radar is not in instance of sorts.Radar
		NegativePriorityError
			If the control priority is negative
		"""
		self.logger = logger
		self.profiler = profiler

		# check radar
		if not isinstance(radar, Radar): 
			raise TypeError(f"radar must be an instance of {Radar}.")

		self.radar = radar
		self.controller = controller 
		self.scheduler = scheduler

		self._priority = priority
		if self._priority is not None:
			self._priority = int(self._priority)

			if self._priority < 0:
				raise NegativePriorityError("priority must be positive [0; +inf]")

		self._t = None
		self._t_slice = None

		self.pdirs = None
		self.pdir_args = None
		self.has_pdirs = False

		self.n_control_points = None
		self.n_periods = None
		self.max_points = None
		self.splitting_indices = None

		# keep track of the parameters being contolled
		self.property_controls = None
		self.controlled_properties = dict()
		self.controlled_properties["tx"] = np.ndarray((len(self.radar.tx,)), dtype=object)
		self.controlled_properties["rx"] = np.ndarray((len(self.radar.rx,)), dtype=object)

		for station_type in ("tx", "rx"):
			for station_id in range(len(getattr(self.radar, station_type))):
				self.controlled_properties[station_type][station_id] = []

		self.meta = dict()
		self.meta['controller_type'] = self.controller.__class__


	def copy(self):
		''' 
		Performs a deepcopy of the radar control structure.
		'''
		ret = RadarControls(self.radar, self.controller, scheduler=self.scheduler, priority=None, logger=self.logger, profiler=self.profiler)
		ret._t 				= copy.copy(self._t)
		ret._t_slice 		= copy.copy(self._t_slice)
		ret.pdir_args 		= copy.copy(self.pdir_args)
		ret.has_pdirs 		= copy.copy(self.has_pdirs)
		ret._priority 		= copy.copy(self.priority)
		ret.pdirs 			= copy.copy(self.pdirs)

		ret.n_control_points 		= self.n_control_points
		ret.n_periods 					= self.n_periods
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

		.. math::
			t_{i} = t_{start} \\\\
			t_{f} = t_{start} + \\Delta t
		
		with \\Delta t the duration of the time slice.

		The set_time_slices function will verify the validity of the control time slices and separate
		the duration and starting points arrays into multiple sub-arrays according to the scheduler
		period if provided, or according to the max_points criterion (see ..mod::`sorts.radar.radar_controls`
		for more information).

  		Parameters
    	----------
		t : 1D ndarray of floats
			start time of each control time slice. 
		duration : 1D ndarray of floats/ float
			Duration of each time slice. 
			If duration is a float, each time slice will have the same duration. 
			For varying time slice durations, a 1D np.ndarray of the same size and shape as
			t must be provided
			Must positive a positive float.
		max_points : int, optional
			Maximum number of control points (time slices) per time slice sub-array (used for the
			splitting of time arrays into multiple control periods).
			Increasing max_points might result in an increase in RAM usage.
			Must be a positive integer, default value is 100.

		Raises
		------
		TimeSliceOverlap
			If two time slices are overlapping.
		ValueError
			- If the duration of a time slice is negative.
			- If the time slice duration and start point arrays are not of the same size.
		TypeError
			If t is not a one dimensional array of floats.
		'''
		if max_points <= 0 or not isinstance(max_points, int):
			raise ValueError("max_points must be a positive integer")

		if np.size(np.atleast_1d(t[0])) == 1 and not isinstance(t[0], np.ndarray):
			# both the time slice and the time arrays are protected to ensure that they are the same size and add verification
			t 				= np.atleast_1d(t).astype(np.float64)
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
			self.get_splitting_indices(max_points=max_points)

			# slice arrays according to scheduler period  or max_points requirements
			self._t 			= self.split_array(self._t)
			self._t_slice 	= self.split_array(self._t_slice)
			self._priority = self.split_array(self._priority)
		else:
			if len(t) != len(duration):
				raise Exception("t must be the same length as the duration array")

			self.splitting_indices = np.ndarray((len(t)-1,), dtype=int)

			index = 0
			for period_id in range(len(t)):
				if period_id < len(t)-1:
					index += len(t[period_id])
					self.splitting_indices[period_id] = index

				if self.scheduler is not None:
					if t[period_id][-1] > self.scheduler.t0 + (period_id + 1)*self.scheduler.scheduler_period or t[period_id][0] < self.scheduler.t0 + period_id*self.scheduler.scheduler_period:
						raise Exception("t is not synchronized with the scheduler period. Please provide a valid splitted time array t or use the automatic splitting feature by calling set_time_slices with flat time arrays")
				else:
					if len(t[period_id]) > max_points:
						raise Exception(f"time subarrays have more elements than max_points (max_points={max_points}). Please provide a valid splitted time array t or use the automatic splitting feature by calling set_time_slices with flat time arrays ")

				if len(t[period_id]) != len(duration[period_id]):
					raise Exception("t must be the same length as the duration array for each period index")

			self._t = t
			self._t_slice = duration
			self.n_control_points = 0
			self.n_periods 	= len(self._t)

			tmp_priority = self._priority
			self._priority = np.ndarray((len(t),), dtype=object)

			for period_id in range(len(t)):
				self.n_control_points += len(t[period_id])
				self._priority[period_id] = np.repeat(np.atleast_1d(tmp_priority), len(t[period_id]))

		# set array of property controls		
		self.property_controls = np.ndarray((self.n_periods,), dtype=object)
		for period_id in range(self.n_periods):
			self.property_controls[period_id] = dict() # station types

			for station_type in ("tx", "rx"):
				self.property_controls[period_id][station_type] = dict() # property names


	def remove_periods(self, mask):
		'''
		Updates the time slice starting point.
		'''
		if self._t is None:
			raise Exception("no time slice parameters set, please call set_time_slices() instead")

		if len(mask) != self.n_periods:
			raise Exception("the length of the removal mask must be the same length as the number of control periods")

		# the number of points is the same for all periods but the last, so treat the last separatly to get total number of points remaining
		self.n_control_points = 0
		for period_id in range(self.n_periods):
			self.n_control_points += len(self._t[period_id])*mask[period_id]

		self._t 		= self._t[mask]
		self._t_slice 	= self._t_slice[mask]
		self._priority 	= self._priority[mask]

		self.property_controls = self.property_controls[mask]
		self.n_periods 	= len(self._t)


	@property
	def t(self):
		'''
		Gets time slice starting points.
		'''
		return self._t

	@property
	def t_slice(self):
		'''
		Gets time slice durations.
		'''
		return self._t_slice

	@property
	def priority(self):
		'''
		Gets the time slice priority
		'''
		return self._priority


	def set_pdirs(self, pdir_args, cache_pdirs=False):
		'''
		Sets the needed arguments to generate radar pointing directions
		'''
		self.pdir_args = pdir_args
		self.has_pdirs = True

		# if pdir arrays are cached as numerical values
		if cache_pdirs is True:
			self.pdirs = np.ndarray((self.n_periods,), dtype=object)

			for period_id in range(self.n_periods):
				# the function compute_pointing_direction is called by passing the arguments specific to the current controller and a reference to the 
				# controls instance
				self.pdirs[period_id] = self.controller.compute_pointing_direction(self, period_id, self.pdir_args)


	def get_pdirs(self, period_id):
		'''
		Compute the pointing direction 
		'''
		if self.has_pdirs is True:
			if self.pdirs is not None:
				pointing_direction = self.pdirs[period_id]
			else:
				# the function compute_pointing_direction is called by passing the arguments specific to the current controller and a reference to the 
				# controls instance
				pointing_direction = self.controller.compute_pointing_direction(self, period_id, self.pdir_args)
		else:
			pointing_direction = None

		return pointing_direction


	def add_property_control(self, name, station, data):
		'''
		Sets the control data corresponding to the specified property and station.

		Parameters :
		------------
			TODO

			station : instance of station to control

			period_id : int
				index of the control period
		'''
		if not isinstance(name, str):
			raise ValueError("name must be a string")

		station = np.atleast_1d(station)
		
		# check array size (if not alreadysplitted, split data array)
		if np.size(data) == 1:
			data = np.repeat(np.atleast_1d(data), self.n_control_points)

		if np.size(data) != self.n_periods and not isinstance(data[0], np.ndarray):
			if np.size(data) != self.n_control_points:
				raise Exception("the control data shall either be of the shape (n_control_points,), where n_control_points is the total number of time points inside the time slice array, or (n_periods,...) where n_periods is the number of control periods")
			else:
				data = self.split_array(data)

		# create new control field if doesn't already exist
		self.__create_new_property_control_field(name, station)		

		# add control for each station
		for station_ in station:
			# get type and id of station
			station_id = self.radar.get_station_id(station_)
			station_type = station_.type
			
			for period_id in range(self.n_periods):
				self.property_controls[period_id][station_type][name][station_id] = data[period_id]


	def __create_new_property_control_field(self, name, stations):
		"""
		Adds a new empty property control field to the :class:`radar controls<RadarControls>` for 
		all stations in ``stations``.

		.. warning::
			One must set the time slices before creating new radar control entries (since they are created over each control period).
		
		Parameters
		----------
		name : str
			Name of the property field.
			The function will create a new property control for each control period.

		stations : array-like, list of :class:`Station<sorts.radar.system.station.Station>` instances
			List of stations which property will be controlled. The new control field will be created over all stations 
			present in ``stations``.
		"""
		for station_ in stations:
			if not name in station_.PROPERTIES:
				raise ControlFieldError(f"station {station_} has no control variable named {name}. Available controls are : {station_.get_properties()}")
			else:
				station_type = station_.type
				station_id = self.radar.get_station_id(station_)

				# if there is no control field of name ``name``, add new field and create new control arrays at each period id
				if name not in self.controlled_properties[station_type][station_id]:
					self.controlled_properties[station_type][station_id].append(name)

					for period_id in range(self.n_periods):
						if not name in self.property_controls[period_id][station_type].keys():
							self.property_controls[period_id][station_type][name] = np.ndarray((len(getattr(self.radar, station_type)),), dtype=object)
				
	def get_property_control(self, name, station, period_id):
		'''
		Gets the control data corresponding to the specified control period id.

		Parameters
		----------
		control_variable : str
			name of the control variable/field to be controled for a given radar instance (corresponding to the radar instance used during 
			initialization). 

		period_id : int
			control period index which controls we want to exctract.

		Returns
		-------
		(n_periods, ...) ndarray
			Array of controls for the control_variable. if period_id is provided n_periods=1, else n_periods will be the total number of periods 
		'''
		if not isinstance(name, str):
			raise ValueError("name must be a string")

		# get type and id of station
		station_id = self.radar.get_station_id(station)
		station_type = station.type
		
		if not name in station.PROPERTIES:
			raise ValueError(f"station {station} has no control variable named {name}. Available controls are : {station.get_properties()}")

		return self.property_controls[period_id][station_type][name][station_id]


	def get_property_control_list(self, station):
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
		''' 
		Computes the control period associated with a given scheduler period.
		If there is no control if index 'control_id' at the given control period, then the function will return -1
		'''
		if self.scheduler is None: 
			ctrl_period_id = scheduler_period_id
		else:
			ctrl_period_id = scheduler_period_id - int(self._t[0][0]//self.scheduler.scheduler_period)  # computes the time subarray id

			# the time subarray id is bigger than the number of time subarrays in the given control structure
			if ctrl_period_id < 0 or ctrl_period_id > len(self._t)-1: 
				ctrl_period_id = -1

		return ctrl_period_id


	def check_time_slice_overlap(self, t, duration):
		'''
		Checks wether or not time slices overlap within a given control array.
		If two different time slices overlap, the function will return the indices of the time points which overlap.
		'''
		# Logging execution status
		if self.logger is not None: 
			self.logger.info("checking time slice overlap")

		overlap = False
		
		dt = t[1:] - (t[:-1] + duration[:-1])
		superposition_mask_ids = np.where(dt < -1e-10)[0]

		if np.size(superposition_mask_ids) > 0:
			if self.logger is not None: 
				self.logger.info(f"Time slices overlapping at transition indices {superposition_mask_ids}. Stopping")
			
			overlap = True

		return overlap


	def split_array(self, array):
		'''
		Usage :
		-------

		Split an array according to the scheduler period and start time (if the scheduler is provided).       
		If the scheduler is None, then the time array will be splitted to ensure that the number of time points in a given controls subarray does not exceed max_points
		This function returns the splitted array

		Parameters :
		------------
			max_points : int (optional)
				maximum number of time points inside a given control period
		'''
		# if the splitting indices have not been computed, run computation first
		if self.n_periods is None and self.splitting_indices is None:
			self.splitting_indices 	= self.get_splitting_indices()

		# Split arrays according to transition indices
		if self.splitting_indices is not None:
			splitted_array = np.ndarray((self.n_periods,), dtype=object)

			id_start = 0
			for period_id in range(self.n_periods):
				# get end index of the control period in the linear array
				if period_id < self.n_periods-1:
					id_end = self.splitting_indices[period_id]
				else:
					id_end = self.n_control_points

				# copy values from the array in the corresponding control period
				splitted_array[period_id] = array[id_start:id_end]
				id_start = id_end
		else:
			splitted_array = array[None, :]

		return splitted_array


	def get_splitting_indices(self, max_points=100):
		'''
		Computes the indices at which the time and control arrays must be sliced to meet the scheduler/max_points requirements.
		Beware that the time slice must have been set before calling this function
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

			period_idx = (self._t - self.scheduler.t0)//self.scheduler.scheduler_period

			self.splitting_indices = np.array(np.where((period_idx[1:] - period_idx[:-1]) == 1)[0]) + 1
			self.n_periods = len(self.splitting_indices) + 1
			del period_idx
		else:
			if self.logger is not None:
				self.logger.info("radar_controls:get_splitting_indices -> No scheduler provided, skipping master clock splitting...")
				self.logger.info(f"radar_controls:get_splitting_indices -> using max_points={max_points} (max time points limit)")

			if(np.size(self._t) > max_points):                
				self.splitting_indices = np.arange(max_points, np.size(self._t), max_points, dtype=int)
				self.n_periods = len(self.splitting_indices) + 1
			else:
				self.n_periods = 1 


	def __str__(self):
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
						if self.property_controls[period_id][station.type][name][station_id] is not None:
							periods.append(period_id)

					print(f"    - {name} -> (periods {periods})")
								
			print("\n-------------------------------------------------------")
		else:
			print("time and time slice arrays not set.")

		return ""
