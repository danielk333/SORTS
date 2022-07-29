import unittest
import numpy as np
import numpy.testing as nt

import sorts

class TestRadar(unittest.TestCase):
	def setUp(self):
		self.radar = sorts.radars.eiscat3d


	def test_compute_intersection_points(self):
		tx_ecef = np.array([tx.ecef for tx in self.radar.tx])
		rx_ecef = np.array([rx.ecef for rx in self.radar.rx])

		# compute theoretical intersection points
		intersection_points = np.ndarray((3, 3), dtype=np.float64)

		intersection_points[:, 0] = tx_ecef[0] + np.array([2478e3, 9864e3, 687e3])		
		intersection_points[:, 1] = rx_ecef[1] + np.array([-2478e3, -9864e3, -687e3])
		intersection_points[:, 2] = rx_ecef[0] # invalid value

		pdirs_tx = np.ndarray((len(self.radar.tx), 1, 3, 3), dtype=np.float64)
		pdirs_rx = np.ndarray((len(self.radar.rx), len(self.radar.tx), 3, 3), dtype=np.float64)

		pdirs_tx 		= intersection_points[None, None, :, :] - tx_ecef[:, None, :, None]
		pdirs_rx		= intersection_points[None, None, :, :] - rx_ecef[:, None, :, None]

		pdirs_tx 		= pdirs_tx/np.linalg.norm(pdirs_tx, axis=2)[:, :, None, :]
		pdirs_rx 		= pdirs_rx/np.linalg.norm(pdirs_rx, axis=2)[:, :, None, :]

		estimated_intersection_points = self.radar.compute_intersection_points(pdirs_tx, pdirs_rx)

		nt.assert_almost_equal(estimated_intersection_points, intersection_points[:, 0:-1], decimal=2)


	def test_access_properties(self):
		for txi in range(len(self.radar.tx)):
			for name in self.radar.tx[txi].PROPERTIES:
				x = eval("self.radar.tx[txi]." + name)



	def test_read_write_properties(self):
		data = np.array([0.1, 1.0, -2.1, 3.8]).astype(np.float64)

		# try to set and get attributes in radar TX stations
		for txi in range(len(self.radar.tx)):
			for name in self.radar.tx[txi].PROPERTIES:
				# set data to control
				exec("self.radar.tx[txi]." + name + " = data")

				# compare to theoretical control sent to radar stations
				nt.assert_array_equal(eval("self.radar.tx[txi]." + name), data)

		# try to set and get attributes in radar RX stations
		for rxi in range(len(self.radar.rx)):
			for name in self.radar.rx[rxi].PROPERTIES:
				# set data to control
				exec("self.radar.rx[rxi]." + name + " = data")

				# compare to theoretical control sent to radar stations
				nt.assert_array_equal(eval("self.radar.rx[rxi]." + name), data)

		# try to get a non-existing attribute 
		with self.assertRaises(AttributeError):
			x = self.radar.tx[0].angular_velocity_tx

		with self.assertRaises(AttributeError):
			x = self.radar.rx[0].angular_velocity_tx



	def test_add_controls_during_runtime(self):
		name = "angular_velocity"

		data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

		for txi in range(len(self.radar.tx)):
			self.radar.tx[txi].add_property(name)

			assert hasattr(self.radar.tx[txi], name) 				== True

			self.radar.tx[txi].angular_velocity = data

			nt.assert_array_equal(self.radar.tx[txi].angular_velocity, data)

			# try to get a non-existing attribute 
			with self.assertRaises(AttributeError):
				x = self.radar.angular_velocity_tx

	def test_get_station_id_and_type(self):
		# test retreival of tx stations
		for station_type in ("rx", "tx"):
			stations = getattr(self.radar, station_type)

			for station_id_ref, station in enumerate(stations):
				sid = self.radar.get_station_id(station)
				stype = station.type

				assert sid 		== station_id_ref
				assert stype 	== station_type


	def test_in_fov(self):
		dir_1 = np.array([0.33087577, 0.12248111, 0.93569204, 0, 0, 0]) # 90 deg of elevation        
        dir_2 = np.array([0.18286604, 0.28205454, 0.94180956, 0, 0, 0]) # 77.5 deg of elevation     
        dir_3 = np.array([-0.1004115, 0.53090257, 0.84146301, 0, 0, 0]) # 55 deg of elevation       
        dir_4 = np.array([-0.6476016, 0.75067923, 0.13073926, 0, 0, 0]) # 0 deg of elevation
		dirs = np.asfarray([dir_1, dir_2, dir_3, dir_4]).T

		ecef = dirs * np.array([[1550.0, 896.2, 1434.0, 4575.1]])*1e3
		ecef[0:3] = ecef[0:3] + self.radar.tx[0].ecef[:, None]
		is_in_fov = self.radar.field_of_view(ecef)

		# check output
		assert is_in_fov[0] == True
		assert is_in_fov[1] == True
		assert is_in_fov[2] == True
		assert is_in_fov[3] == False