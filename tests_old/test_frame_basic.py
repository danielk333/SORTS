import unittest
import numpy as np
import numpy.testing as nt

from sorts import frames
from sorts import constants


class TestFrames(unittest.TestCase):

    def test_geodetic_to_ITRS(self):
        dec = 3
        x = frames.geodetic_to_ITRS(90.0, 0.0, 0.0, radians=False)
        nt.assert_almost_equal(x[2], constants.WGS84.b, decimal = dec)
        
        x = frames.geodetic_to_ITRS(-90.0, 0.0, 0.0, radians=False)
        nt.assert_almost_equal(x[2], -constants.WGS84.b, decimal = dec)
        
        x = frames.geodetic_to_ITRS(0.0, 0.0, 0.0, radians=False)
        nt.assert_almost_equal(x[0], constants.WGS84.a, decimal = dec)
        
        x = frames.geodetic_to_ITRS(0.0, 90.0, 0.0, radians=False)
        nt.assert_almost_equal(x[1], constants.WGS84.a, decimal = dec)
        
        x = frames.geodetic_to_ITRS(90.0, 0.0, 100., radians=False)
        nt.assert_almost_equal(x[2], constants.WGS84.b+100., decimal = dec)

    def test_ned_to_ecef(self):
        dec = 3
        
        lat, lon, alt = 0.0, 0.0, 0.0
        x = np.array([0.0, 0.0, 0.0])
        g = frames.ned_to_ecef(lat, lon, alt, x, radians=False)
        nt.assert_array_almost_equal(g, x, decimal=dec)
        
        
        lat, lon = 0.0, 0.0
        x = np.array([0.0, 0.0, -100.0])
        g = frames.ned_to_ecef(lat, lon, alt, x, radians=False)
        g_ref  = np.array([-x[2], 0.0, 0.0])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)
    
        x = np.array([0.0, 0.0, 100.0])
        g = frames.ned_to_ecef(lat, lon, alt, x, radians=False)
        g_ref  = np.array([-x[2], 0.0, 0.0])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)

        
        lat, lon = 90.0, 0.0
        x = np.array([0.0, 0.0, 100.0])
        g = frames.ned_to_ecef(lat, lon, alt, x, radians=False)
        g_ref  = np.array([0.0, 0.0, -x[2]])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)

        lat, lon = 45.0, 0.0
        x = np.array([0.0, 0.0, -np.sqrt(2.0)])
        g = frames.ned_to_ecef(lat, lon, alt, x, radians=False)
        g_ref  = np.array([1.0, 0.0, 1.0])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)

    def test_ecef2local(self):
        dec = 3
        
        lat, lon, alt = 0.0, 0.0, 0.0
        x_ref = np.array([0.0, 0.0, 100.0]) #enu
        ecef = np.array([100.0, 0.0, 0.0])
        x = frames.ecef_to_enu(lat, lon, alt, ecef, radians=False)
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        x_ref = np.array([0.0, 100.0, 0.0])
        ecef = np.array([0.0, 0.0, 100.0])
        x = frames.ecef_to_enu(lat, lon, alt, ecef, radians=False)
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        x_ref = np.array([100.0, 0.0, 0.0])
        ecef = np.array([0.0, 100.0, 0.0])
        x = frames.ecef_to_enu(lat, lon, alt, ecef, radians=False)
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        lat, lon = 0.0, 180.0
        x_ref = np.array([0.0, 0.0, 100.0])
        ecef = np.array([-100.0, 0.0, 0.0])
        x = frames.ecef_to_enu(lat, lon, alt, ecef, radians=False)
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        lat, lon = 90.0, 0.0
        x_ref = np.array([0.0, 0.0, 100.0])
        ecef = np.array([0.0, 0.0, 100.0])
        x = frames.ecef_to_enu(lat, lon, alt, ecef, radians=False)
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        lat, lon = 45.0, 0.0
        x_ref = np.array([0.0, 0.0, np.sqrt(200.0)])
        ecef = np.array([10.0, 0.0, 10.0])
        x = frames.ecef_to_enu(lat, lon, alt, ecef, radians=False)
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)


    def test_ecef2local_inverse(self):
        dec = 3

        lat, lon, alt = 0.0, 0.0, 0.0
        x = np.array([0.0, 0.0, 0.0])
        g = frames.enu_to_ecef(lat, lon, alt, x, radians=False)
        x_ref = frames.ecef_to_enu(lat, lon, alt, g, radians=False)
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        
        lat, lon = 0.0, 0.0
        x = np.array([0.0, 0.0, -100.0])
        g = frames.enu_to_ecef(lat, lon, alt, x, radians=False)
        x_ref = frames.ecef_to_enu(lat, lon, alt, g, radians=False)
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
    
        x = np.array([0.0, 0.0, 100.0])
        g = frames.enu_to_ecef(lat, lon, alt, x, radians=False)
        x_ref = frames.ecef_to_enu(lat, lon, alt, g, radians=False)
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        
        lat, lon = 90.0, 0.0
        x = np.array([0.0, 0.0, 100.0])
        g = frames.enu_to_ecef(lat, lon, alt, x, radians=False)
        x_ref = frames.ecef_to_enu(lat, lon, alt, g, radians=False)
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        lat, lon = 45.0, 0.0
        x = np.array([0.0, 0.0, -np.sqrt(2.0)])
        g = frames.enu_to_ecef(lat, lon, alt, x, radians=False)
        x_ref = frames.ecef_to_enu(lat, lon, alt, g, radians=False)
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
    
    
    def test_cart_to_sph(self):
        y = np.array([10.0, 0.0, 0.0])
        az, el, r = frames.cart_to_sph(y)
        self.assertAlmostEqual(az, 90.0)
        self.assertAlmostEqual(el, 0.0)
        self.assertAlmostEqual(r, 10.0)

        y = np.array([0.0, 10.0, 0.0])
        az, el, r = frames.cart_to_sph(y)
        self.assertAlmostEqual(az, 0.0)
        self.assertAlmostEqual(el, 0.0)
        self.assertAlmostEqual(r, 10.0)

        y = np.array([0.0, 0.0, 10.0])
        az, el, r = frames.cart_to_sph(y)
        self.assertAlmostEqual(az, 0.0)
        self.assertAlmostEqual(el, 90.0)
        self.assertAlmostEqual(r, 10.0)

        y = np.array([10.0, 0.0, 10.0])
        az, el, r = frames.cart_to_sph(y)
        self.assertAlmostEqual(az, 90.0)
        self.assertAlmostEqual(el, 45.0)
        self.assertAlmostEqual(r, np.sqrt(200.0))
        
    def test_sph_to_cart(self):
        dec = 7
        y = np.array([10.0, 0.0, 0.0])
        y_ref = frames.sph_to_cart(np.array([90.0, 0.0, 10.0]), radians=False)
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)

        y = np.array([0.0, 10.0, 0.0])
        y_ref = frames.sph_to_cart(np.array([0.0, 0.0, 10.0]), radians=False)
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)

        y = np.array([0.0, 0.0, 10.0])
        y_ref = frames.sph_to_cart(np.array([0.0, 90.0, 10.0]), radians=False)
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)

        y = np.array([10.0, 0.0, 10.0])
        y_ref = frames.sph_to_cart(np.array([90.0, 45.0, np.sqrt(200.0)]), radians=False)
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)
        
    
    def test_ecef_geo_inverse(self):
        dec = 3
        y = np.array((90.0, 0.0, 0.0))
        x = frames.geodetic_to_ITRS(y[0], y[1], y[2])
        y_ref = frames.ITRS_to_geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = np.array((-90.0, 0.0, 0.0))
        x = frames.geodetic_to_ITRS(y[0], y[1], y[2])
        y_ref = frames.ITRS_to_geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = np.array((0.0, 0.0, 0.0))
        x = frames.geodetic_to_ITRS(y[0], y[1], y[2])
        y_ref = frames.ITRS_to_geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = np.array((0.0, 90.0, 0.0))
        x = frames.geodetic_to_ITRS(y[0], y[1], y[2])
        y_ref = frames.ITRS_to_geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = np.array((90.0, 0.0, 100.0))
        x = frames.geodetic_to_ITRS(y[0], y[1], y[2])
        y_ref = frames.ITRS_to_geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)
        
    def test_ITRS_to_geodetic(self):
        dec = 3
        x = frames.ITRS_to_geodetic(0.0, 0.0, constants.WGS84.b)
        y = np.array((90.0, 0.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = frames.ITRS_to_geodetic(0.0, 0.0, -constants.WGS84.b)
        y = np.array((-90.0, 0.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = frames.ITRS_to_geodetic(constants.WGS84.a, 0.0, 0.0)
        y = np.array((0.0, 0.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = frames.ITRS_to_geodetic(0.0, constants.WGS84.a, 0.0)
        y = np.array((0.0, 90.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = frames.ITRS_to_geodetic(0.0, 0.0, constants.WGS84.b+100.)
        y = np.array((90.0, 0.0, 100.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)

    
    def test_vector_angle(self):
        x = np.array([1,0,0])
        y = np.array([1,1,0])
        theta = frames.vector_angle(x, y)
        self.assertAlmostEqual(theta, 45.0)
        
        y = np.array([0,1,0])
        theta = frames.vector_angle(x, y)
        self.assertAlmostEqual(theta, 90.0)
        
        theta = frames.vector_angle(x, x)
        self.assertAlmostEqual(theta, 0.0)
        
        theta = frames.vector_angle(x, -x)
        self.assertAlmostEqual(theta, 180.0)


        X = np.array([0.11300039,-0.85537661,0.50553118])
        theta = frames.vector_angle(X, X)
        self.assertAlmostEqual(theta, 0.0)
