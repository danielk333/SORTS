import numpy as np

from . import scheduler
from .system.radar import Radar

class RadarControls(object):
	def __init__(self, radar, controller, scheduler=None, priority=None, logger=None, profiler=None):
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
				raise ValueError("priority must be positive [0; +inf]")

		# keep track of the parameters being contolled
		self.property_controls = dict()
		self.property_controls["tx"] = dict()
		self.property_controls["rx"] = dict()

		self._t = None
		self._t_slice = None


	def set_time_slices(self, t, duration, max_points=100):
		'''
		Paramertrizes the time slices (start time and duration)
		'''
		if np.size(np.atleast_1d(t[0])) == 1:
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
				raise Exception("Time slices are overlapping")

			self._t = t
			self._t_slice = duration
			self._priority = np.repeat(self._priority, self.n_control_points)
			
			# split time slices according to scheduler periods or max points for performance
			self.get_splitting_indices(max_points=max_points)

			# slice arrays according to scheduler period  or max_points requirements
			self._t 		= self.split_array(self._t)
			self._t_slice 	= self.split_array(self._t_slice)
			self._priority 	= self.split_array(self._priority)
		else:
			if len(t) != len(duration):
				raise Exception("t must be the same length as the duration array")

			for period_id in range(len(t)):
				if scheduler is not None:
					if t[period_id][-1] > self.scheduler.t0 + (period_id + 1)*self.scheduler.scheduler_period or t[period_id][0] < self.scheduler.t0 + period_id*self.scheduler.scheduler_period:
						raise Exception("t is not synchronized with the scheduler period. Please provide a valid splitted time array t or use the automatic splitting feature by calling set_time_slices with flat time arrays")
				else:
					if len(t[period_id]) > max_points:
						raise Exception("time subarrays have more elements than max_points (max_points={max_points}). Please provide a valid splitted time array t or use the automatic splitting feature by calling set_time_slices with flat time arrays ")

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

	def remove_periods(self, mask):
		'''
		Updates the time slice starting point.
		'''
		if self._t is None:
			raise Exception("no time slice parameters set, please call set_time_slices() instead")

		if len(mask) != self.n_periods:
			raise Exception("the length of the removal mask must be the same length as the number of control periods")

		self._t 		= self._t[mask]
		self._t_slice 	= self._t_slice[mask]
		self._priority 	= self._priority[mask]

		self.n_periods 	= len(self._t)

	@property
	def t(self):
		'''
		Gets the time slice starting point.
		'''
		return self._t

	@property
	def t_slice(self):
		'''
		Gets the time slice duration.
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

		# if pdir arrays are cached as numerical values
		if cache_pdirs is True:
			self.pdirs = np.ndarray((self.n_periods,), dtype=object)

			for period_id in range(len(self.n_periods)):
				# the function compute_pointing_direction is called by passing the arguments specific to the current controller and a reference to the 
				# controls instance
				self.pdirs[period_id] = self.controller.compute_pointing_direction(self, period_id, self.pdir_args)


	def get_pdirs(self, period_id):
		'''
		Compute the pointing direction 
		'''
		if hasattr(self, "pdirs"):
			pointing_direction = self.pdirs[period_id]
		else:
			# the function compute_pointing_direction is called by passing the arguments specific to the current controller and a reference to the 
			# controls instance
			pointing_direction = self.controller.compute_pointing_direction(self, period_id, self.pdir_args)

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

		# get type and id of station
		station_id, station_type = self.radar.get_station_id_and_type(station)
		
		if not name in station.PROPERTIES:
			raise ValueError(f"station {station} has no control variable named {name}. Available controls are : {station.get_properties()}")

		if len(data) != self.n_control_points:
			raise ValueError(f"data must be an array of size {self.n_control_points}")

		if name not in self.property_controls[station_type].keys():
			# create new array of controls for each station of type station_type (tx/rx)
			self.property_controls[station_type][name] = np.ndarray((len(getattr(self.radar, station_type)),), dtype=object)

		if len(data) != self.n_periods:
			if len(data) != self.n_control_points:
				raise Exception("the control data shall either be of the shape (n_control_points,), where n_control_points is the total number of time points inside the time slice array, or (n_periods,...) where n_periods is the number of control periods")
			else:
				self.property_controls[station_type][name][station_id] = self.split_array(data)
		else:
			self.property_controls[station_type][name][station_id] = self.split_array(data)


	def get_property_control(self, name, station):
		'''
		Gets the control data corresponding to the specified control period id.

		Parameters :
		------------
			control_variable : str
				name of the control variable/field to be controled for a given radar instance (corresponding to the radar instance used during 
				initialization). 
			period_id : int (optional)
				if provided, the function will return the control data of the period given by period_id

		Returns :
		---------
			np.ndarray : shape (n_periods,...)
				Array of controls for the control_variable. if period_id is provided n_periods=1, else n_periods will be the total number of periods 
		'''
		if not isinstance(name, str):
			raise ValueError("name must be a string")

		# get type and id of station
		station_id, station_type = self.radar.get_station_id_and_type(station)
		
		if not name in station.PROPERTIES:
			raise ValueError(f"station {station} has no control variable named {name}. Available controls are : {station.get_properties()}")

		return self.property_controls[station_type][name][station_id]


	def get_property_control_list(self, station):
		# get type and id of station
		station_id, station_type = self.radar.get_station_id_and_type(station)

		properties = []

		for property_name in station.PROPERTIES:
			if property_name in self.property_controls[station_type].keys():
				if self.property_controls[station_type][property_name][station_id] is not None:
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
			ctrl_period_id = scheduler_period_id - int(self._t[0][0]/self.scheduler.scheduler_period) # computes the time subarray id

			# the time subarray id is bigger than the number of time subarrays in the given control structure
			if ctrl_period_id < 0 or ctrl_period_id >= len(self._t): 
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
		if not hasattr(self, "splitting_indices"):
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
		
		print("Pointing direction controls (RadarControls.get_pdirs(period_id)) :")
		print(self.get_pdirs())

		print("-------------- Optional/Property controls --------------")

		for name in self.control_list:
			data = getattr(self, name)

			print("Control " + name)
