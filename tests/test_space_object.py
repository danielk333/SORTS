import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as n
import numpy.testing as nt


import space_object as so


class TestSpaceObject(unittest.TestCase):

    def setUp(self):
        self.init_obj = {
            'a': 7000,
            'e': 0.0,
            'i': 69,
            'raan': 0.0,
            'aop': 0.0,
            'mu0': 0.0,
            'C_D': 2.3,
            'C_R': 1.0,
            'A': 1.0,
            'm': 1.0,
        }
        self.o = so.SpaceObject(**self.init_obj)
        self.R_e = 6371.0

    def test_return_sizes(self):
        t = n.array([0.0], dtype=n.float)

        x = self.o.get_orbit(t)
        xv = self.o.get_state(t)

        self.assertEqual(x.shape, (3,1))
        self.assertEqual(xv.shape, (6,1))

        nt.assert_array_equal(x[:,0], xv[:3,0])

        t2 = n.array([0.0, 1.0], dtype=n.float)

        x = self.o.get_orbit(t2)
        xv = self.o.get_state(t2)

        self.assertEqual(x.shape, (3,2))
        self.assertEqual(xv.shape, (6,2))

        nt.assert_array_equal(x[:,:], xv[:3,:])

    def test_kep_cart_init(self):
        t = n.array([0.0], dtype=n.float)
        xv = self.o.get_state(t)

        o2 = so.SpaceObject.cartesian(
            self.o.x, self.o.y, self.o.z, 
            self.o.vx, self.o.vy, self.o.vz, 
            d=self.o.d, C_D=self.o.C_D, A=self.o.A, m=self.o.m,
            mjd0=self.o.mjd0, oid=self.o.oid, C_R=self.o.C_R,
            propagator = self.o._propagator, M_cent = self.o.M_cent,
            propagator_options = self.o.propagator_options,
        )

        xv2 = o2.get_state(t)

        pos_diff = n.linalg.norm(xv[:3,0] - xv2[:3,0])
        vel_diff = n.linalg.norm(xv[3:,0] - xv2[3:,0])
        self.assertLess(pos_diff*1e-3, 10.0)
        self.assertLess(vel_diff*1e-3, 5.0)
        nt.assert_almost_equal(xv*1e-3/self.R_e, xv2*1e-3/self.R_e, decimal=2)

    def test_update_elements_cart(self):
        t = n.array([0.0], dtype=n.float)
        xv0 = self.o.get_state(t)

        ox0 = self.o.x

        #print(o)

        #move 100 km in x direction
        self.o.update(x = self.o.x + 100.0)
        #print(o)
        xv1 = self.o.get_state(t)

        ox1 = self.o.x

        #move 100 km back
        self.o.update(x = self.o.x - 100.0)
        #print(o)
        xv2 = self.o.get_state(t)

        ox2 = self.o.x

        self.assertAlmostEqual(ox0 + 100.0, ox1)
        self.assertAlmostEqual(ox1 - 100.0, ox2)
        self.assertAlmostEqual(ox0, ox2)

        pos_diff01 = n.linalg.norm(xv0[:3,0] - xv1[:3,0])
        pos_diff12 = n.linalg.norm(xv1[:3,0] - xv2[:3,0])
        pos_diff02 = n.linalg.norm(xv0[:3,0] - xv2[:3,0])
        self.assertLess(n.abs(pos_diff01*1e-3 - 100.0), 1.0)
        self.assertLess(n.abs(pos_diff12*1e-3 - 100.0), 1.0)
        self.assertLess(pos_diff02*1e-3, 1.0)

        nt.assert_almost_equal(xv0, xv2, decimal=2)

    def test_update_elements_kep(self):
        t = n.array([0.0], dtype=n.float)
        xv0 = self.o.get_state(t)

        oa0 = self.o.a

        #move 100 km in x direction
        self.o.update(a = self.o.a + 100.0)
        xv1 = self.o.get_state(t)

        oa1 = self.o.a

        #move 100 km back
        self.o.update(a = self.o.a - 100.0)
        xv2 = self.o.get_state(t)

        oa2 = self.o.a

        self.assertAlmostEqual(oa0 + 100.0, oa1)
        self.assertAlmostEqual(oa1 - 100.0, oa2)
        self.assertAlmostEqual(oa0, oa2)

        pos_diff02 = n.linalg.norm(xv0[:3,0] - xv2[:3,0])
        self.assertLess(pos_diff02, 1.0)

        nt.assert_almost_equal(xv0, xv2, decimal=2)


    def test_update_error(self):
        with self.assertRaises(TypeError):
            self.o.update(a = self.o.a + 100.0, x = self.o.x + 100.0)


    def test_propagator_options_sgp4(self):
        from propagator_sgp4 import PropagatorSGP4

        obj0 = so.SpaceObject(
            propagator = PropagatorSGP4,
            propagator_options = {'polar_motion': False},
            **self.init_obj
        )

        obj1 = so.SpaceObject(
            propagator = PropagatorSGP4,
            propagator_options = {'polar_motion': True},
            **self.init_obj
        )
        t = n.array([0.0], dtype=n.float)
        xv1 = obj0.get_state(t)
        xv0 = obj1.get_state(t)


    def test_propagator_change_sgp4(self):
        from propagator_sgp4 import PropagatorSGP4

        obj = so.SpaceObject(
            propagator = PropagatorSGP4,
            propagator_options = {},
            **self.init_obj
        )
        t = n.array([0.0], dtype=n.float)
        xv1 = obj.get_state(t)
        xv0 = self.o.get_state(t)

        self.assertEqual(xv1.shape, xv0.shape)



    def test_propagator_change_orekit(self):
        from propagator_orekit import PropagatorOrekit

        obj = so.SpaceObject(
            propagator = PropagatorOrekit,
            propagator_options = {},
            **self.init_obj
        )
        t = n.array([0.0], dtype=n.float)
        xv1 = obj.get_state(t)
        xv0 = self.o.get_state(t)

        self.assertEqual(xv1.shape, xv0.shape)


    def test_propagator_options_orekit(self):
        from propagator_orekit import PropagatorOrekit

        obj0 = so.SpaceObject(
            propagator = PropagatorOrekit,
            propagator_options = {'drag_force': False},
            **self.init_obj
        )

        obj1 = so.SpaceObject(
            propagator = PropagatorOrekit,
            propagator_options = {'drag_force': True},
            **self.init_obj
        )
        t = n.array([0.0], dtype=n.float)
        xv1 = obj0.get_state(t)
        xv0 = obj1.get_state(t)

        nt.assert_array_almost_equal(xv0, xv1, decimal=2)



if __name__ == '__main__':
    unittest.main(verbosity=2)