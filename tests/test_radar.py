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
		tx_stations = self.radar.tx
		rx_stations = self.radar.rx

		# test retreival of tx stations
		for txi, tx in enumerate(tx_stations):
			sid, stype = self.radar.get_station_id_and_type(tx)

			assert sid 		== txi
			assert stype 	== "tx"

		# test retreival of rx stations
		for rxi, rx in enumerate(rx_stations):
			sid, stype = self.radar.get_station_id_and_type(rx)

			assert sid 		== rxi
			assert stype 	== "rx"


