import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as n
import numpy.testing as nt

import propagator_base

class new_propagator(propagator_base.PropagatorBase):
    def get_orbit(self, t, a, e, inc, raan, aop, mu0, mjd0, **kwargs):
        pass
    def get_orbit_cart(self, t, x, y, z, vx, vy, vz, mjd0, **kwargs):
        pass

class TestBaseProp(unittest.TestCase):

    def test_base_prop_methods(self):

        prop = new_propagator()

        assert prop.get_orbit(0,
            0,0,0,
            0,0,0,
            0,
        ) is None

        assert prop.get_orbit_cart(0,
            0,0,0,
            0,0,0,
            0,
        ) is None

    def test_meta_raise_no_method(self):

        class new_wrong_propagator(propagator_base.PropagatorBase):
            pass

        self.assertRaises(TypeError, new_wrong_propagator)

    def test_meta_raise_wrong_method(self):

        class new_wrong_propagator(propagator_base.PropagatorBase):
            def get_orbit(self):
                pass
            def get_orbit_cart(self, t, x, y, z, vx, vy, vz, mjd0, **kwargs):
                pass

        self.assertRaises(AssertionError, new_wrong_propagator)


        class new_wrong_propagator(propagator_base.PropagatorBase):
            def get_orbit(self, x, a, e, inc, raan, aop, mu0, mjd0, **kwargs):
                pass
            def get_orbit_cart(self, t, x, y, z, vx, vy, vz, mjd0, **kwargs):
                pass

        self.assertRaises(AssertionError, new_wrong_propagator)


    def test_numpy_conv_float(self):

        prop = new_propagator()

        x = 5.3
        x_conv = prop._make_numpy(x)

        assert isinstance(x_conv, n.ndarray)

        nt.assert_almost_equal(x_conv[0], x, decimal=9)

    def test_numpy_conv_list(self):

        prop = new_propagator()

        x = [5.3]
        x_conv = prop._make_numpy(x)

        assert isinstance(x_conv, n.ndarray)

        nt.assert_almost_equal(x_conv[0], x[0], decimal=9)

    def test_numpy_conv_numpy(self):

        prop = new_propagator()

        x = n.array([5.3], dtype=n.float)
        x_conv = prop._make_numpy(x)

        assert isinstance(x_conv, n.ndarray)

        assert x is x_conv

        nt.assert_array_almost_equal(x_conv, x, decimal=9)

    def test_numpy_conv_raise(self):

        prop = new_propagator()

        x = '4'
        self.assertRaises(Exception, prop._make_numpy, x)



if __name__ == '__main__':
    unittest.main(verbosity=2)