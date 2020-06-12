import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as n
import numpy.testing as nt

import coord


class TestCoord(unittest.TestCase):

    def test_geodetic2ecef(self):
        dec = 3
        x = coord.geodetic2ecef(90.0, 0.0, 0.0)
        nt.assert_almost_equal(x[2], coord.b, decimal = dec)
        
        x = coord.geodetic2ecef(-90.0, 0.0, 0.0)
        nt.assert_almost_equal(x[2], -coord.b, decimal = dec)
        
        x = coord.geodetic2ecef(0.0, 0.0, 0.0)
        nt.assert_almost_equal(x[0], coord.a, decimal = dec)
        
        x = coord.geodetic2ecef(0.0, 90.0, 0.0)
        nt.assert_almost_equal(x[1], coord.a, decimal = dec)
        
        x = coord.geodetic2ecef(90.0, 0.0, 100.)
        nt.assert_almost_equal(x[2], coord.b+100., decimal = dec)

    def test_ned2ecef(self):
        dec = 3
        
        lat, lon, alt = 0.0, 0.0, 0.0
        x = n.array([0.0, 0.0, 0.0])
        g = coord.ned2ecef(lat, lon, alt, x[0], x[1], x[2])
        nt.assert_array_almost_equal(g, x, decimal=dec)
        
        
        lat, lon = 0.0, 0.0
        x = n.array([0.0, 0.0, -100.0])
        g = coord.ned2ecef(lat, lon, alt, x[0], x[1], x[2])
        g_ref  = n.array([-x[2], 0.0, 0.0])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)
    
        x = n.array([0.0, 0.0, 100.0])
        g = coord.ned2ecef(lat, lon, alt, x[0], x[1], x[2])
        g_ref  = n.array([-x[2], 0.0, 0.0])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)

        
        lat, lon = 90.0, 0.0
        x = n.array([0.0, 0.0, 100.0])
        g = coord.ned2ecef(lat, lon, alt, x[0], x[1], x[2])
        g_ref  = n.array([0.0, 0.0, -x[2]])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)

        lat, lon = 45.0, 0.0
        x = n.array([0.0, 0.0, -n.sqrt(2.0)])
        g = coord.ned2ecef(lat, lon, alt, x[0], x[1], x[2])
        g_ref  = n.array([1.0, 0.0, 1.0])
        nt.assert_array_almost_equal(g, g_ref, decimal=dec)

    def test_ecef2local(self):
        dec = 3
        
        lat, lon, alt = 0.0, 0.0, 0.0
        x_ref = n.array([0.0, 0.0, 100.0]) #enu
        ecef = n.array([100.0, 0.0, 0.0])
        x = coord.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        x_ref = n.array([0.0, 100.0, 0.0])
        ecef = n.array([0.0, 0.0, 100.0])
        x = coord.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        x_ref = n.array([100.0, 0.0, 0.0])
        ecef = n.array([0.0, 100.0, 0.0])
        x = coord.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        lat, lon = 0.0, 180.0
        x_ref = n.array([0.0, 0.0, 100.0])
        ecef = n.array([-100.0, 0.0, 0.0])
        x = coord.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        lat, lon = 90.0, 0.0
        x_ref = n.array([0.0, 0.0, 100.0])
        ecef = n.array([0.0, 0.0, 100.0])
        x = coord.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        lat, lon = 45.0, 0.0
        x_ref = n.array([0.0, 0.0, n.sqrt(200.0)])
        ecef = n.array([10.0, 0.0, 10.0])
        x = coord.ecef2local(lat, lon, alt, ecef[0], ecef[1], ecef[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)


    def test_ecef2local_inverse(self):
        dec = 3

        lat, lon, alt = 0.0, 0.0, 0.0
        x = n.array([0.0, 0.0, 0.0])
        g = coord.enu2ecef(lat, lon, alt, x[0], x[1], x[2])
        x_ref = coord.ecef2local(lat, lon, alt, g[0], g[1], g[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
        
        
        lat, lon = 0.0, 0.0
        x = n.array([0.0, 0.0, -100.0])
        g = coord.enu2ecef(lat, lon, alt, x[0], x[1], x[2])
        x_ref = coord.ecef2local(lat, lon, alt, g[0], g[1], g[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
    
        x = n.array([0.0, 0.0, 100.0])
        g = coord.enu2ecef(lat, lon, alt, x[0], x[1], x[2])
        x_ref = coord.ecef2local(lat, lon, alt, g[0], g[1], g[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        
        lat, lon = 90.0, 0.0
        x = n.array([0.0, 0.0, 100.0])
        g = coord.enu2ecef(lat, lon, alt, x[0], x[1], x[2])
        x_ref = coord.ecef2local(lat, lon, alt, g[0], g[1], g[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)

        lat, lon = 45.0, 0.0
        x = n.array([0.0, 0.0, -n.sqrt(2.0)])
        g = coord.enu2ecef(lat, lon, alt, x[0], x[1], x[2])
        x_ref = coord.ecef2local(lat, lon, alt, g[0], g[1], g[2])
        nt.assert_array_almost_equal(x_ref, x, decimal=dec)
    
    
    def test_cart_to_azel(self):
        y = n.array([10.0, 0.0, 0.0])
        az, el, r = coord.cart_to_azel(y)
        self.assertAlmostEqual(az, 90.0)
        self.assertAlmostEqual(el, 0.0)
        self.assertAlmostEqual(r, 10.0)

        y = n.array([0.0, 10.0, 0.0])
        az, el, r = coord.cart_to_azel(y)
        self.assertAlmostEqual(az, 0.0)
        self.assertAlmostEqual(el, 0.0)
        self.assertAlmostEqual(r, 10.0)

        y = n.array([0.0, 0.0, 10.0])
        az, el, r = coord.cart_to_azel(y)
        self.assertAlmostEqual(az, 0.0)
        self.assertAlmostEqual(el, 90.0)
        self.assertAlmostEqual(r, 10.0)

        y = n.array([10.0, 0.0, 10.0])
        az, el, r = coord.cart_to_azel(y)
        self.assertAlmostEqual(az, 90.0)
        self.assertAlmostEqual(el, 45.0)
        self.assertAlmostEqual(r, n.sqrt(200.0))
        
    def test_azel_to_cart(self):
        dec = 7
        y = n.array([10.0, 0.0, 0.0])
        y_ref = coord.azel_to_cart(90.0, 0.0, 10.0)
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)

        y = n.array([0.0, 10.0, 0.0])
        y_ref = coord.azel_to_cart(0.0, 0.0, 10.0)
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)

        y = n.array([0.0, 0.0, 10.0])
        y_ref = coord.azel_to_cart(0.0, 90.0, 10.0)
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)

        y = n.array([10.0, 0.0, 10.0])
        y_ref = coord.azel_to_cart(90.0, 45.0, n.sqrt(200.0))
        nt.assert_array_almost_equal(y, y_ref, decimal=dec)
        
    
    def test_ecef_geo_inverse(self):
        dec = 3
        y = n.array((90.0, 0.0, 0.0))
        x = coord.geodetic2ecef(y[0], y[1], y[2])
        y_ref = coord.ecef2geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = n.array((-90.0, 0.0, 0.0))
        x = coord.geodetic2ecef(y[0], y[1], y[2])
        y_ref = coord.ecef2geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = n.array((0.0, 0.0, 0.0))
        x = coord.geodetic2ecef(y[0], y[1], y[2])
        y_ref = coord.ecef2geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = n.array((0.0, 90.0, 0.0))
        x = coord.geodetic2ecef(y[0], y[1], y[2])
        y_ref = coord.ecef2geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)

        y = n.array((90.0, 0.0, 100.0))
        x = coord.geodetic2ecef(y[0], y[1], y[2])
        y_ref = coord.ecef2geodetic(x[0], x[1], x[2])
        nt.assert_array_almost_equal(y, y_ref, decimal = dec)
        
    def test_ecef2geodetic(self):
        dec = 3
        x = coord.ecef2geodetic(0.0, 0.0, coord.b)
        y = n.array((90.0, 0.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = coord.ecef2geodetic(0.0, 0.0, -coord.b)
        y = n.array((-90.0, 0.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = coord.ecef2geodetic(coord.a, 0.0, 0.0)
        y = n.array((0.0, 0.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = coord.ecef2geodetic(0.0, coord.a, 0.0)
        y = n.array((0.0, 90.0, 0.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)
        
        x = coord.ecef2geodetic(0.0, 0.0, coord.b+100.)
        y = n.array((90.0, 0.0, 100.0))
        nt.assert_array_almost_equal(x, y, decimal = dec)

    
    def test_geodetic_to_az_el_r(self):
        lat, lon, alt = 0.0, 0.0, 0.0
        x = coord.geodetic_to_az_el_r(lat, lon, alt, lat, lon, alt + 100.0)
        #self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(x[1], 90.0)
        self.assertAlmostEqual(x[2], 100.0)

        lat, lon, alt = 90.0, 0.0, 0.0
        x = coord.geodetic_to_az_el_r(lat, lon, alt, lat, lon, alt + 100.0)
        #self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(x[1], 90.0)
        self.assertAlmostEqual(x[2], 100.0)

        lat, lon, alt = 90.0, 45.0, 100.0
        x = coord.geodetic_to_az_el_r(lat, lon, alt, lat, lon, alt + 100.0)
        #self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(x[1], 90.0)
        self.assertAlmostEqual(x[2], 100.0)

        lat, lon, alt = 42.61950, 288.50827, 146.0
        delta = 1e-3
        x = coord.geodetic_to_az_el_r(lat, lon, alt, lat+delta, lon, alt)
        self.assertAlmostEqual(x[0], 0.0)
        nt.assert_almost_equal(x[1], 0.0, decimal = 3)
        x = coord.geodetic_to_az_el_r(lat, lon, alt, lat-delta, lon, alt)
        self.assertAlmostEqual((x[0] + 360.0) % 360.0, 180.0)
        nt.assert_almost_equal(x[1], 0.0, decimal = 3)
        x = coord.geodetic_to_az_el_r(lat, lon, alt, lat, lon+delta, alt)
        nt.assert_almost_equal(x[0], 90.0, decimal = 3)
        nt.assert_almost_equal(x[1], 0.0, decimal = 3)
        x = coord.geodetic_to_az_el_r(lat, lon, alt, lat, lon-delta, alt)
        nt.assert_almost_equal((x[0]+360.0) % 360.0, 270.0, decimal = 3)
        nt.assert_almost_equal(x[1], 0.0, decimal = 3)

    def test_az_el_r2geodetic(self):
        raise Exception('Test not written')
    
    def test_angle_deg(self):
        x = n.array([1,0,0])
        y = n.array([1,1,0])
        theta = coord.angle_deg(x, y)
        self.assertAlmostEqual(theta, 45.0)
        
        y = n.array([0,1,0])
        theta = coord.angle_deg(x, y)
        self.assertAlmostEqual(theta, 90.0)
        
        theta = coord.angle_deg(x, x)
        self.assertAlmostEqual(theta, 0.0)
        
        theta = coord.angle_deg(x, -x)
        self.assertAlmostEqual(theta, 180.0)


        X = n.array([0.11300039,-0.85537661,0.50553118])
        theta = coord.angle_deg(X, X)
        self.assertAlmostEqual(theta, 0.0)

'''
def test_coord():
    result = geodetic2ecef(69.0,19.0,10.0)
    print(resigrf)
    result3 = ned2ecef(69.0, 19.0, 10.0, 10679.6, 1288.2, 49873.3)

    print("North")
    print(geodetic_to_az_el_r(42.61950, 288.50827, 146.0, 43.61950, 288.50827, 100e3))
    print("az_el_r2geodetic")
    print(az_el_r2geodetic(42.61950, 288.50827, 146.0, 0.00000000e+00, 4.12258606e+01, 1.50022653e+05))
    print("East")
    print(geodetic_to_az_el_r(42.61950, 288.50827, 146.0, 42.61950, 289.50827, 100e3))
    print("West")
    print(geodetic_to_az_el_r(42.61950, 288.50827, 146.0, 42.61950, 287.50827, 100e3))
    print("South")
    print(geodetic_to_az_el_r(42.61950, 288.50827, 146.0, 41.61950, 288.50827, 100e3))
    print("Southwest")
    print(geodetic_to_az_el_r(42.61950, 288.50827, 146.0, 41.61950, 287.50827, 100e3))
'''