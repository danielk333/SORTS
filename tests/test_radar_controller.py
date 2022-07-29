import sys
import os

import unittest
import numpy as np
import numpy.testing as nt

import pyorb

import sorts
from sorts.radar import RadarController, Static, Scanner, SpaceObjectTracker, Tracker
from sorts.radar.scans import Fence, Beampark
from sorts import RadarControls
from sorts import radars


class TestRadarControllers(unittest.TestCase):
	def setUp(self):
		self.radar = radars.eiscat3d

	def test_set_coh_integration(self):
		pdirs_th = np.array([
			np.array([[1, 0, 0], [1, 0, 0]], dtype=float), 
			np.array([[1, 0, 0]], dtype=float)
		], dtype=object)

		# time slice parameters
		t           = np.linspace(0.0, 2.0, 3)
		duration    = np.ones(3)*0.1

		class TestController(RadarController):
			def __init__(self, profiler=None, logger=None):
				super().__init__(profiler=None, logger=None)

			def generate_controls(self, t, t_slice, radar):
				controls = RadarControls(radar, self, scheduler=None)
				controls.set_time_slices(t, t_slice, max_points=2)

				controls.pdirs = controls.set_pdirs(None)
				RadarController.coh_integration(controls, radar, t_slice)

				return controls

			def compute_pointing_direction(self, controls, period_id, args):
				pdirs["tx"] = pdirs_th[period_id]
				pdirs["rx"] = pdirs_th[period_id]
				pdirs["t"] = controls.t[period_id]

				return pdirs


		controller = TestController()
		controls = controller.generate_controls(t, duration, self.radar)

		for tx in self.radar.tx:
			for period_id in range(len(controls.t)):
				n_ipp_ref = np.atleast_1d(controls.t_slice[period_id]/tx.ipp).astype(np.int64)
				coh_int_bandwidth_ref = 1.0/(tx.pulse_length*tx.n_ipp)

				n_ipp = controls.get_property_control("n_ipp", tx, period_id)
				coh_int_bandwidth = controls.get_property_control("coh_int_bandwidth", tx, period_id)

				nt.assert_array_equal(n_ipp, n_ipp_ref)
				nt.assert_array_equal(coh_int_bandwidth, coh_int_bandwidth_ref)


	def test_static_controller(self):
		azimuth=0.0
		elevation=90.0
		t_slice=0.1

		r = np.linspace(100e3, 1000e3)
		t = np.linspace(0.0, 2.0, 3)

		controller = Static()
		controls = controller.generate_controls(t, self.radar, t_slice=t_slice, r=r, azimuth=azimuth, elevation=elevation, max_points=1)

		tx_ecef = np.array([tx.ecef for tx in controls.radar.tx], dtype=np.float64) # get the position of each Tx station (ECEF frame)
		rx_ecef = np.array([rx.ecef for rx in controls.radar.rx], dtype=np.float64) # get the position of each Rx station (ECEF frame)

		scan = Beampark(azimuth=azimuth, elevation=elevation, dwell=t_slice)
		
		for period_id in range(len(controls.t)):
			# compute reference pointing directions
			points = scan.ecef_pointing(controls.t[period_id], controls.radar.tx).reshape((len(tx_ecef), 3, -1))

			point_rx_to_tx = np.repeat(points[:, None, :, :], len(r), axis=3)*np.tile(r, len(points[0, 0]))[None, None, None, :] + tx_ecef[:, None, :, None] # compute the target points for the Rx stations
			point_rx = np.repeat(point_rx_to_tx, len(controls.radar.rx), axis=0) 
			rx_dirs = point_rx - rx_ecef[:, None, :, None]

			pdirs_ref = dict()
			pdirs_ref['tx'] = np.repeat(points[:, None, :, :], len(r), axis=3) 
			pdirs_ref['rx'] = rx_dirs/np.linalg.norm(rx_dirs, axis=2)[:, :, None, :]
			pdirs_ref['t'] = np.repeat(controls.t[period_id], len(r))

			# get poiting directions computed by controller 
			pdirs = controller.compute_pointing_direction(controls, period_id, (r))

			# compare pointing directions
			nt.assert_array_equal(pdirs["tx"], pdirs_ref["tx"])
			nt.assert_array_equal(pdirs["rx"], pdirs_ref["rx"])
			nt.assert_array_equal(pdirs["t"], pdirs_ref["t"])


	def test_scanner_controller(self):
		azimuth = 90.0 
		min_elevation = 30.0
		t_slice = 0.1
		n_scan = 5

		r = np.linspace(100e3, 1000e3)
		t = np.linspace(0.0, 4.0, 5)

		scan = Fence(azimuth=azimuth, min_elevation=min_elevation, dwell=t_slice, num=n_scan)

		controller = Scanner()
		controls = controller.generate_controls(t, self.radar, scan, r=r, max_points=2)

		tx_ecef = np.array([tx.ecef for tx in controls.radar.tx], dtype=np.float64) # get the position of each Tx station (ECEF frame)
		rx_ecef = np.array([rx.ecef for rx in controls.radar.rx], dtype=np.float64) # get the position of each Rx station (ECEF frame)
		
		for period_id in range(len(controls.t)):
			# compute reference pointing directions
			points = scan.ecef_pointing(controls.t[period_id], controls.radar.tx).reshape((len(tx_ecef), 3, -1))

			point_rx_to_tx = np.repeat(points[:, None, :, :], len(r), axis=3)*np.tile(r, len(points[0, 0]))[None, None, None, :] + tx_ecef[:, None, :, None] # compute the target points for the Rx stations
			point_rx = np.repeat(point_rx_to_tx, len(controls.radar.rx), axis=0) 
			rx_dirs = point_rx - rx_ecef[:, None, :, None]

			pdirs_ref = dict()
			pdirs_ref['tx'] = np.repeat(points[:, None, :, :], len(r), axis=3) 
			pdirs_ref['rx'] = rx_dirs/np.linalg.norm(rx_dirs, axis=2)[:, :, None, :]
			pdirs_ref['t'] = np.repeat(controls.t[period_id], len(r))

			# get poiting directions computed by controller 
			pdirs = controller.compute_pointing_direction(controls, period_id, (r))

			# compare pointing directions
			nt.assert_array_equal(pdirs["tx"], pdirs_ref["tx"])
			nt.assert_array_equal(pdirs["rx"], pdirs_ref["rx"])
			nt.assert_array_equal(pdirs["t"], pdirs_ref["t"])


	def test_tracker_controller(self):
		n_states = 4
		n_points_ctrl = 10
		states_per_slice = 2

		delta_theta = 30
		r0 = 10000e3
		r = np.array([r0, 0, 0]) # in the orbit frame

		# get longitude and latitude from reference station
		station_ref = self.radar.tx[0]
		lon0 = station_ref.lon*np.pi/180
		lat0 = station_ref.lat*np.pi/180

		# compute intersecting orbit properties
		mu_earth = pyorb.M_earth*pyorb.G
		omega = np.sqrt(mu_earth/r0**3)
		
		dtheta = delta_theta/n_states # deg
		dt = dtheta/omega

		tf = n_states*dt
		t_slice = dt/10

		# array of true anomaly values
		theta = np.linspace(-delta_theta/2, delta_theta/2, n_states)*np.pi/180.0

		# create time arrays (controller)
		t = np.linspace(dt, tf + dt, n_points_ctrl) # remove first and last point
		pass_mask = np.full((len(t),), True, bool)
		pass_mask[0] = False
		pass_mask[-1] = False

		# target states sampling time points
		t_states = np.linspace(0.0, tf, n_states)
		
		# compute target states
		# we assume that the perigee is located over the station at theta = 0 deg
		# compute DCM for rotation sequence (lambda z (lon), phi y (lat), theta z (anomaly))
		M = np.array([
			[np.cos(lon0)*np.cos(theta)*np.cos(lat0) - np.sin(lon0)*np.sin(theta), 		-np.sin(theta)*np.cos(lat0)*np.cos(lon0) - np.cos(theta)*np.sin(lon0), 	-np.sin(lat0)*np.cos(lon0)*np.ones(len(theta))],
			[np.sin(lon0)*np.cos(theta)*np.cos(lat0) - np.cos(lon0)*np.sin(theta), 		-np.sin(theta)*np.cos(lat0)*np.sin(lon0) - np.cos(theta)*np.cos(lon0), 	-np.cos(lat0)*np.cos(lon0)*np.ones(len(theta))],
			[np.cos(theta)*np.sin(lat0), 												 np.sin(theta)*np.sin(lat0),											 np.cos(lat0)*np.ones(len(theta))]])

		states_ecef = np.einsum("ijk, j->ik", M, r)

		# interpolate target states (ecef)
		interp_states = sorts.interpolation.Linear(states_ecef, t_states)
		t_interp = np.repeat(t, states_per_slice)
		dt_interp = t_slice/states_per_slice

		for i in range(states_per_slice):
			t_interp[i::states_per_slice] = t_interp[i::states_per_slice] + i*dt_interp

		states_ecef_interp = interp_states.get_state(t_interp)

		# split arrays according to max time points in controller
		states_ecef_interp = np.hsplit(states_ecef_interp, [5, 10, 15])
		t_interp = np.hsplit(t_interp, [5, 10, 15])

		# create controller and generate controls
		controller = sorts.Tracker()
		controls = controller.generate_controls(t, self.radar, t_states, np.tile(states_ecef, (2, 1)), t_slice=t_slice, states_per_slice=states_per_slice, max_points=5)

		tx_ecef = np.array([tx.ecef for tx in controls.radar.tx], dtype=np.float64) # get the position of each Tx station (ECEF frame)
		rx_ecef = np.array([rx.ecef for rx in controls.radar.rx], dtype=np.float64) # get the position of each Rx station (ECEF frame)

		for period_id in range(len(controls.t)):
			# compute reference pointing directions
			tx_dirs = states_ecef_interp[period_id][None, None, :, :] - tx_ecef[:, None, :, None]
			rx_dirs = states_ecef_interp[period_id][None, None, :, :] - rx_ecef[:, None, :, None]

			pdirs_ref = dict()
			pdirs_ref['tx'] = tx_dirs/np.linalg.norm(tx_dirs, axis=2)[:, :, None, :]
			pdirs_ref['rx'] = rx_dirs/np.linalg.norm(rx_dirs, axis=2)[:, :, None, :]
			pdirs_ref['t'] = t_interp[period_id]

			# get poiting directions computed by controller 
			pdirs = controller.compute_pointing_direction(controls, period_id, (states_ecef_interp, t_interp))

			# compare pointing directions
			nt.assert_array_equal(pdirs["tx"], pdirs_ref["tx"])
			nt.assert_array_equal(pdirs["rx"], pdirs_ref["rx"])
			nt.assert_array_equal(pdirs["t"], pdirs_ref["t"])

	def test_space_object_tracker(self):
		epoch = 53005.0
		t_slice = 1
		tf = 2000
		max_dpos=10e3

		max_points = 2
		n_tracking_points = 10
		states_per_slice = 2

		# Propagator
		Prop_cls = sorts.propagator.Kepler
		Prop_opts = dict(
		    settings = dict(
		        out_frame='ITRS',
		        in_frame='TEME',
		    ),
		)

		space_object_1 = sorts.SpaceObject(
			Prop_cls,
			propagator_options = Prop_opts,
			a = 1.5*6378e3, 
			e = 0.0,
			i = 72.2,
			raan = 0,
			aop = 66.6,
			mu0 = 0,

			epoch = epoch,
			parameters = dict(
				d = 0.1,
			),
		)

		space_object_2 = sorts.SpaceObject(
			Prop_cls,
			propagator_options = Prop_opts,
			a = 2.5*6378e3, 
			e = 0.0,
			i = 72,
			raan = 20,
			aop = 66.6,
			mu0 = 12,

			epoch = epoch,
			parameters = dict(
				d = 0.1,
			),
		)

		t = np.linspace(0, tf, n_tracking_points)

		space_objects = np.array([space_object_1, space_object_2])
		obj_priorities = np.array([0, 1], dtype=int)
		object_indices_ref 	= np.array([1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)

		# initialize space object tracker controller
		so_tracking_controller = sorts.SpaceObjectTracker()
		controls = so_tracking_controller.generate_controls(t, self.radar, space_objects, epoch, t_slice, states_per_slice=states_per_slice, space_object_priorities=obj_priorities, save_states=True, max_points=max_points, max_dpos=max_dpos)

		# verify that correct objects are being tracked
		nt.assert_array_equal(controls.meta["object_indices"], object_indices_ref)
		object_indices_ref = np.repeat(object_indices_ref, 2)

		# generate tracking time array (adding itermediate time points to reach states_per_slice)
		t_ref = t[0:-1] # last point is discarded by the controller (no space object in FOV)
		dt_tracking = t_slice/float(states_per_slice)
		t_dirs = np.repeat(t_ref, states_per_slice)
		for ti in range(states_per_slice):
			t_dirs[ti::states_per_slice] = t_dirs[ti::states_per_slice] + dt_tracking*ti

		# split arrays according to control periods
		splitting_inds = np.arange(max_points, len(t_ref), max_points)
		t_ref = np.hsplit(t_ref, splitting_inds)
		t_dirs = np.hsplit(t_dirs, splitting_inds*states_per_slice)
		object_indices_ref = np.hsplit(object_indices_ref, splitting_inds*states_per_slice)

		# compute reference ecef states
		ecef_ref = np.ndarray((len(t_dirs),), dtype=object)
		for period_id in range(controls.n_periods):
			ecef_ref[period_id] = np.ndarray((3, len(t_dirs[period_id])), dtype=float)
			for soid, so in enumerate(space_objects):
				obj_msk = object_indices_ref[period_id] == soid
				ecef_ref[period_id][:, obj_msk] = so.get_state(t_dirs[period_id])[0:3][:, obj_msk]

		# station positions
		tx_ecef = np.array([tx.ecef for tx in controls.radar.tx], dtype=np.float64) # get the position of each Tx station (ECEF frame)
		rx_ecef = np.array([rx.ecef for rx in controls.radar.rx], dtype=np.float64) # get the position of each Rx station (ECEF frame)

		i_start = 0
		for period_id in range(controls.n_periods):
			nt.assert_array_equal(controls.t[period_id], t_ref[period_id])

			tx_dirs = ecef_ref[period_id][None, None, :, :] - tx_ecef[:, None, :, None]
			rx_dirs = ecef_ref[period_id][None, None, :, :] - rx_ecef[:, None, :, None]

			pdirs_ref = dict()
			pdirs_ref['tx'] = tx_dirs/np.linalg.norm(tx_dirs, axis=2)[:, :, None, :]
			pdirs_ref['rx'] = rx_dirs/np.linalg.norm(rx_dirs, axis=2)[:, :, None, :]
			
			pdirs = controls.get_pdirs(period_id)

			nt.assert_array_almost_equal(pdirs_ref['tx'], pdirs['tx'], decimal=5)
			nt.assert_array_almost_equal(pdirs_ref['rx'], pdirs['rx'], decimal=5)
			nt.assert_array_almost_equal(t_dirs[period_id], pdirs['t'], decimal=5)