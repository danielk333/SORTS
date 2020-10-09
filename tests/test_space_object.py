import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as np
import numpy.testing as nt


import sorts
import pyorb
from sorts import SpaceObject

from astropy.time import TimeDelta


class TestSpaceObject(unittest.TestCase):

    def setUp(self):
        self.init_obj = { 
            'parameters': {
                'C_D': 2.3,
                'C_R': 1.0,
                'A': 1.0,
                'm': 1.0,
            },
            'propagator': sorts.propagator.SGP4,
        }
        self.orb_init = {
            'a': 7000e3,
            'e': 0.0,
            'i': 69,
            'raan': 0.0,
            'aop': 0.0,
            'mu0': 0.0,
        }
        self.orb_init.update(self.init_obj)

        self.orb_alt_init = {
            'a': 7000e3,
            'e': 0.0,
            'i': 69,
            'omega': 0.0,
            'Omega': 0.0,
            'anom': 0.0,
        }
        self.orb_alt_init.update(self.init_obj)

        self.cart_init = {
            'x': 7000e3,
            'y': 0.0,
            'z': 0,
            'vx': 0.0,
            'vy': 0.0,
            'vz': 7e3,
        }
        self.cart_init.update(self.init_obj)

        self.state_init = {
            'state': {
                'sgfd': 'this is a state',
                'sdf': 'even though it dsent look like it',
                'sss': 69,
                'aaa': 0.0,
            },
        }
        self.state_init.update(self.init_obj)

        self.R_e = 6371.0e3


    def test_orb_init(self):
        obj = SpaceObject(**self.orb_init)

    def test_orb_alt_init(self):
        obj = SpaceObject(**self.orb_alt_init)

    def test_cart_init(self):
        obj = SpaceObject(**self.cart_init)

    def test_state_init(self):
        obj = SpaceObject(**self.state_init)


    def test_Orbit_init(self):

        obj = SpaceObject(**self.orb_init)
        assert type(obj.state) == pyorb.Orbit

        obj = SpaceObject(**self.orb_alt_init)
        assert type(obj.state) == pyorb.Orbit

        obj = SpaceObject(**self.cart_init)
        assert type(obj.state) == pyorb.Orbit


    def test_orbit_property(self):

        obj = SpaceObject(**self.orb_init)
        orb = obj.orbit
        assert orb == obj.state

        obj = SpaceObject(**self.orb_alt_init)
        orb = obj.orbit
        assert orb == obj.state

        obj = SpaceObject(**self.cart_init)
        orb = obj.orbit
        assert orb == obj.state

        obj = SpaceObject(**self.state_init)
        with self.assertRaises(AttributeError):
            orb = obj.orbit


    def test_d_property(self):

        obj = SpaceObject(**self.orb_init)
        assert obj.d == np.sqrt(self.init_obj['parameters']['A']/np.pi)*2

        del self.orb_init['parameters']['A']
        self.orb_init['parameters']['d'] = 2.0
        obj = SpaceObject(**self.orb_init)
        assert obj.d == 2.0

        self.orb_init['parameters']['A'] = 1.0
        obj = SpaceObject(**self.orb_init)
        assert obj.d == 2.0

        self.orb_init['parameters']['diam'] = 3.0
        self.orb_init['parameters']['r'] = 5.0

        obj = SpaceObject(**self.orb_init)
        assert obj.d == 2.0

        del self.orb_init['parameters']['d']
        obj = SpaceObject(**self.orb_init)
        assert obj.d == 3.0

        del self.orb_init['parameters']['diam']
        obj = SpaceObject(**self.orb_init)
        assert obj.d == 5.0*2

        del self.orb_init['parameters']['A']
        del self.orb_init['parameters']['r']
        
        obj = SpaceObject(**self.orb_init)
        #test default value
        x = obj.d

        del obj.parameters['A']
        with self.assertRaises(AttributeError):
            x = obj.d



    def test_return_sizes(self):

        obj = SpaceObject(**self.orb_init)

        t = np.array([0.0], dtype=np.float)

        x = obj.get_position(t)
        v = obj.get_velocity(t)
        xv = obj.get_state(t)

        self.assertEqual(x.shape, (3,1))
        self.assertEqual(v.shape, (3,1))
        self.assertEqual(xv.shape, (6,1))

        nt.assert_array_equal(x[:,0], xv[:3,0])
        nt.assert_array_equal(v[:,0], xv[3:,0])

        t = np.array([0.0, 1.0], dtype=np.float)

        x = obj.get_position(t)
        v = obj.get_velocity(t)
        xv = obj.get_state(t)

        self.assertEqual(x.shape, (3,2))
        self.assertEqual(v.shape, (3,2))
        self.assertEqual(xv.shape, (6,2))

        nt.assert_array_equal(x[:,:], xv[:3,:])
        nt.assert_array_equal(v[:,:], xv[3:,:])

        t = 1.0

        x = obj.get_position(t)
        v = obj.get_velocity(t)
        xv = obj.get_state(t)

        self.assertEqual(x.shape, (3,1))
        self.assertEqual(v.shape, (3,1))
        self.assertEqual(xv.shape, (6,1))

        nt.assert_array_equal(x[:], xv[:3])
        nt.assert_array_equal(v[:], xv[3:])


    def test_time_input_types(self):

        obj = SpaceObject(**self.orb_init)

        states = []
        states += [obj.get_state(np.array([1.0], dtype=np.float))]
        states += [obj.get_state(1.0)]
        states += [obj.get_state(1)]
        states += [obj.get_state(TimeDelta(1.0, format='sec'))]
        states += [obj.get_state(obj.epoch + TimeDelta(1.0, format='sec'))]

        for state in states:
            self.assertEqual(state.shape, (6,1))

        for state in states:
            nt.assert_array_equal(state, states[0])


    def test_update_elements_cart(self):
        t = np.array([0.0], dtype=np.float)
        obj = SpaceObject(**self.orb_init)

        xv0 = obj.get_state(t)

        ox0 = obj.orbit.x[0]

        #move 100 km in x direction
        obj.orbit.update(x = obj.orbit.x[0] + 100.0)
        #print(o)
        xv1 = obj.get_state(t)

        ox1 = obj.orbit.x[0]

        #move 100 km back
        obj.orbit.update(x = obj.orbit.x[0] - 100.0)
        #print(o)
        xv2 = obj.get_state(t)

        ox2 = obj.orbit.x[0]

        self.assertAlmostEqual(ox0 + 100.0, ox1)
        self.assertAlmostEqual(ox1 - 100.0, ox2)
        self.assertAlmostEqual(ox0, ox2)

        pos_diff01 = np.linalg.norm(xv0[:3,0] - xv1[:3,0])
        pos_diff12 = np.linalg.norm(xv1[:3,0] - xv2[:3,0])
        pos_diff02 = np.linalg.norm(xv0[:3,0] - xv2[:3,0])
        self.assertLess(np.abs(pos_diff01 - 100.0), 1.0)
        self.assertLess(np.abs(pos_diff12 - 100.0), 1.0)
        self.assertLess(pos_diff02, 1.0)

        nt.assert_almost_equal(xv0, xv2, decimal=2)

    def test_update_elements_kep(self):
        t = np.array([0.0], dtype=np.float)
        obj = SpaceObject(**self.orb_init)

        xv0 = obj.get_state(t)

        ox0 = obj.orbit.a[0]

        #move 100 km in x direction
        obj.orbit.update(a = obj.orbit.a[0] + 100.0)
        #print(o)
        xv1 = obj.get_state(t)

        ox1 = obj.orbit.a[0]

        #move 100 km back
        obj.orbit.update(a = obj.orbit.a[0] - 100.0)
        #print(o)
        xv2 = obj.get_state(t)

        ox2 = obj.orbit.a[0]

        self.assertAlmostEqual(ox0 + 100.0, ox1)
        self.assertAlmostEqual(ox1 - 100.0, ox2)
        self.assertAlmostEqual(ox0, ox2)

        pos_diff01 = np.linalg.norm(xv0[:3,0] - xv1[:3,0])
        pos_diff12 = np.linalg.norm(xv1[:3,0] - xv2[:3,0])
        pos_diff02 = np.linalg.norm(xv0[:3,0] - xv2[:3,0])
        self.assertLess(np.abs(pos_diff01 - 100.0), 1.0)
        self.assertLess(np.abs(pos_diff12 - 100.0), 1.0)
        self.assertLess(pos_diff02, 1.0)

        nt.assert_almost_equal(xv0, xv2, decimal=2)



    def test_update_error(self):
        obj = SpaceObject(**self.orb_init)
        with self.assertRaises(ValueError):
            obj.orbit.update(a = obj.orbit.a + 100.0, x = obj.orbit.x + 100.0)


    def test_update_wrap(self):
        obj = SpaceObject(**self.orb_init)

        a0 = obj.orbit.a[0]
        
        obj.orbit.update(a = obj.orbit.a + 100.0)
        obj.update(a = obj.orbit.a - 100.0)

        a1 = obj.orbit.a[0]

        nt.assert_almost_equal(a0, a1, decimal=5)

    def test_change_frame(self):
        obj = SpaceObject(**self.orb_init)

        obj.out_frame = 'TEME'
        xv1 = obj.get_state(0)

        obj.out_frame = 'ITRS'
        xv0 = obj.get_state(0)

        self.assertEqual(xv1.shape, xv0.shape)
        for ind in range(6):
            self.assertNotEqual(xv1[ind,0], xv0[ind,0])

    def test_propagate_epoch(self):
        self.orb_init['propagator'] = sorts.propagator.Kepler
        obj = SpaceObject(**self.orb_init)
        obj.out_frame = obj.in_frame

        state0 = obj.get_state(10.0)
        obj.propagate(10.0)
        state1 = obj.get_state(0.0)
        state2 = obj.orbit.cartesian

        nt.assert_almost_equal(state0, state1, decimal=5)
        nt.assert_almost_equal(state0, state2, decimal=5)
        nt.assert_almost_equal(state2, state1, decimal=5)



if __name__ == '__main__':
    unittest.main(verbosity=2)