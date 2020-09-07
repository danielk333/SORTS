import unittest
import numpy as np
import numpy.testing as nt

from sorts import frames
from sorts import constants


class TestFrames(unittest.TestCase):

    def test_geodetic_to_ecef(self):
        dec = 3
        x = frames.geodetic_to_ecef(90.0, 0.0, 0.0, radians=False)
        nt.assert_almost_equal(x[2], constants.WGS84.b, decimal = dec)
        
        x = frames.geodetic_to_ecef(-90.0, 0.0, 0.0, radians=False)
        nt.assert_almost_equal(x[2], -constants.WGS84.b, decimal = dec)
        
        x = frames.geodetic_to_ecef(0.0, 0.0, 0.0, radians=False)
        nt.assert_almost_equal(x[0], constants.WGS84.a, decimal = dec)
        
        x = frames.geodetic_to_ecef(0.0, 90.0, 0.0, radians=False)
        nt.assert_almost_equal(x[1], constants.WGS84.a, decimal = dec)
        
        x = frames.geodetic_to_ecef(90.0, 0.0, 100., radians=False)
        nt.assert_almost_equal(x[2], constants.WGS84.b+100., decimal = dec)

    def test_ned_to_ecef(self):
        dec = 3
        
        lat, lon, alt = 0.0, 0.0, 0.0
        x = np.array([0.0, 0.0, 0.0])
        g = frames.ned_to_ecef(lat, lon, alt, x[0], x[1], x[2], radians=False)
        nt.assert_array_almost_equal(g, x, decimal=dec)
        
        
        lat, lon = 0.0, 0.0
        x = np.array([0.0, 0.0, -100.0])
        g = frames.ned_to_ecef(lat, lon, alt, x[0], x[1], x[2])
        g_ref  = np.array([-x[2], 0.0, 0.0])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)
    
        x = np.array([0.0, 0.0, 100.0])
        g = frames.ned_to_ecef(lat, lon, alt, x[0], x[1], x[2])
        g_ref  = np.array([-x[2], 0.0, 0.0])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)

        
        lat, lon = 90.0, 0.0
        x = np.array([0.0, 0.0, 100.0])
        g = frames.ned_to_ecef(lat, lon, alt, x[0], x[1], x[2])
        g_ref  = np.array([0.0, 0.0, -x[2]])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)

        lat, lon = 45.0, 0.0
        x = np.array([0.0, 0.0, -np.sqrt(2.0)])
        g = frames.ned_to_ecef(lat, lon, alt, x[0], x[1], x[2])
        g_ref  = np.array([1.0, 0.0, 1.0])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)

    def test_ecef2local(self):
        dec = 3
        
        lat, lon, alt = 0.0, 0.0, 0.0
        x_ref = np.array([0.0, 0.0, 100.0]) #enu
        ecef = np.array([100.0, 0.0, 0.0])
        x = frames.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        x_ref = np.array([0.0, 100.0, 0.0])
        ecef = np.array([0.0, 0.0, 100.0])
        x = frames.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        x_ref = np.array([100.0, 0.0, 0.0])
        ecef = np.array([0.0, 100.0, 0.0])
        x = frames.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        lat, lon = 0.0, 180.0
        x_ref = np.array([0.0, 0.0, 100.0])
        ecef = np.array([-100.0, 0.0, 0.0])
        x = frames.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        lat, lon = 90.0, 0.0
        x_ref = np.array([0.0, 0.0, 100.0])
        ecef = np.array([0.0, 0.0, 100.0])
        x = frames.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        lat, lon = 45.0, 0.0
        x_ref = np.array([0.0, 0.0, np.sqrt(200.0)])
        ecef = np.array([10.0, 0.0, 10.0])
        x = frames.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)


    def test_ecef2local_inverse(self):
        dec = 3

        lat, lon, alt = 0.0, 0.0, 0.0
        x = np.array([0.0, 0.0, 0.0])
        g = frames.enu2ecef(lat, lon, alt, x[0], x[1], x[2])
        x_ref = frames.ecef2local(lat, lon, alt, g[0], g[1], g[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        
        lat, lon = 0.0, 0.0
        x = np.array([0.0, 0.0, -100.0])
        g = frames.enu2ecef(lat, lon, alt, x[0], x[1], x[2])
        x_ref = frames.ecef2local(lat, lon, alt, g[0], g[1], g[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
    
        x = np.array([0.0, 0.0, 100.0])
        g = frames.enu2ecef(lat, lon, alt, x[0], x[1], x[2])
        x_ref = frames.ecef2local(lat, lon, alt, g[0], g[1], g[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        
        lat, lon = 90.0, 0.0
        x = np.array([0.0, 0.0, 100.0])
        g = frames.enu2ecef(lat, lon, alt, x[0], x[1], x[2])
        x_ref = frames.ecef2local(lat, lon, alt, g[0], g[1], g[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        lat, lon = 45.0, 0.0
        x = np.array([0.0, 0.0, -np.sqrt(2.0)])
        g = frames.enu2ecef(lat, lon, alt, x[0], x[1], x[2])
        x_ref = frames.ecef2local(lat, lon, alt, g[0], g[1], g[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
    
    
    def test_cart_to_azel(self):
        y = np.array([10.0, 0.0, 0.0])
        az, el, r = frames.cart_to_azel(y)
        self.assertAlmostEqual(az, 90.0)
        self.assertAlmostEqual(el, 0.0)
        self.assertAlmostEqual(r, 10.0)

        y = np.array([0.0, 10.0, 0.0])
        az, el, r = frames.cart_to_azel(y)
        self.assertAlmostEqual(az, 0.0)
        self.assertAlmostEqual(el, 0.0)
        self.assertAlmostEqual(r, 10.0)

        y = np.array([0.0, 0.0, 10.0])
        az, el, r = frames.cart_to_azel(y)
        self.assertAlmostEqual(az, 0.0)
        self.assertAlmostEqual(el, 90.0)
        self.assertAlmostEqual(r, 10.0)

        y = np.array([10.0, 0.0, 10.0])
        az, el, r = frames.cart_to_azel(y)
        self.assertAlmostEqual(az, 90.0)
        self.assertAlmostEqual(el, 45.0)
        self.assertAlmostEqual(r, np.sqrt(200.0))
        
    def test_azel_to_cart(self):
        dec = 7
        y = np.array([10.0, 0.0, 0.0])
        y_ref = frames.azel_to_cart(90.0, 0.0, 10.0)
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)

        y = np.array([0.0, 10.0, 0.0])
        y_ref = frames.azel_to_cart(0.0, 0.0, 10.0)
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)

        y = np.array([0.0, 0.0, 10.0])
        y_ref = frames.azel_to_cart(0.0, 90.0, 10.0)
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)

        y = np.array([10.0, 0.0, 10.0])
        y_ref = frames.azel_to_cart(90.0, 45.0, np.sqrt(200.0))
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)
        
    
    def test_ecef_geo_inverse(self):
        dec = 3
        y = np.array((90.0, 0.0, 0.0))
        x = frames.geodetic2ecef(y[0], y[1], y[2])
        y_ref = frames.ecef2geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = np.array((-90.0, 0.0, 0.0))
        x = frames.geodetic2ecef(y[0], y[1], y[2])
        y_ref = frames.ecef2geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = np.array((0.0, 0.0, 0.0))
        x = frames.geodetic2ecef(y[0], y[1], y[2])
        y_ref = frames.ecef2geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = np.array((0.0, 90.0, 0.0))
        x = frames.geodetic2ecef(y[0], y[1], y[2])
        y_ref = frames.ecef2geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = np.array((90.0, 0.0, 100.0))
        x = frames.geodetic2ecef(y[0], y[1], y[2])
        y_ref = frames.ecef2geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)
        
    def test_ecef2geodetic(self):
        dec = 3
        x = frames.ecef2geodetic(0.0, 0.0, frames.b)
        y = np.array((90.0, 0.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = frames.ecef2geodetic(0.0, 0.0, -frames.b)
        y = np.array((-90.0, 0.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = frames.ecef2geodetic(frames.a, 0.0, 0.0)
        y = np.array((0.0, 0.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = frames.ecef2geodetic(0.0, frames.a, 0.0)
        y = np.array((0.0, 90.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = frames.ecef2geodetic(0.0, 0.0, frames.b+100.)
        y = np.array((90.0, 0.0, 100.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)

    
    def test_geodetic_to_az_el_r(self):
        lat, lon, alt = 0.0, 0.0, 0.0
        x = frames.geodetic_to_az_el_r(lat, lon, alt, lat, lon, alt + 100.0)
        #self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(x[1], 90.0)
        self.assertAlmostEqual(x[2], 100.0)

        lat, lon, alt = 90.0, 0.0, 0.0
        x = frames.geodetic_to_az_el_r(lat, lon, alt, lat, lon, alt + 100.0)
        #self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(x[1], 90.0)
        self.assertAlmostEqual(x[2], 100.0)

        lat, lon, alt = 90.0, 45.0, 100.0
        x = frames.geodetic_to_az_el_r(lat, lon, alt, lat, lon, alt + 100.0)
        #self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(x[1], 90.0)
        self.assertAlmostEqual(x[2], 100.0)

        lat, lon, alt = 42.61950, 288.50827, 146.0
        delta = 1e-3
        x = frames.geodetic_to_az_el_r(lat, lon, alt, lat+delta, lon, alt)
        self.assertAlmostEqual(x[0], 0.0)
        nt.assert_almost_equal(x[1], 0.0, decimal = 3)
        x = frames.geodetic_to_az_el_r(lat, lon, alt, lat-delta, lon, alt)
        self.assertAlmostEqual((x[0] + 360.0) % 360.0, 180.0)
        nt.assert_almost_equal(x[1], 0.0, decimal = 3)
        x = frames.geodetic_to_az_el_r(lat, lon, alt, lat, lon+delta, alt)
        nt.assert_almost_equal(x[0], 90.0, decimal = 3)
        nt.assert_almost_equal(x[1], 0.0, decimal = 3)
        x = frames.geodetic_to_az_el_r(lat, lon, alt, lat, lon-delta, alt)
        nt.assert_almost_equal((x[0]+360.0) % 360.0, 270.0, decimal = 3)
        nt.assert_almost_equal(x[1], 0.0, decimal = 3)

    def test_az_el_r2geodetic(self):
        raise Exception('Test not written')
    
    def test_angle_deg(self):
        x = np.array([1,0,0])
        y = np.array([1,1,0])
        theta = frames.angle_deg(x, y)
        self.assertAlmostEqual(theta, 45.0)
        
        y = np.array([0,1,0])
        theta = frames.angle_deg(x, y)
        self.assertAlmostEqual(theta, 90.0)
        
        theta = frames.angle_deg(x, x)
        self.assertAlmostEqual(theta, 0.0)
        
        theta = frames.angle_deg(x, -x)
        self.assertAlmostEqual(theta, 180.0)


        X = np.array([0.11300039,-0.85537661,0.50553118])
        theta = frames.angle_deg(X, X)
        self.assertAlmostEqual(theta, 0.0)
