#!/usr/bin/env python

'''Test Orekit python implementation

'''

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import unittest
import numpy as np
import numpy.testing as nt

from sorts.targets.propagator import Orekit

@pytest.mark.usefixtures("orekit_data")
class TestOrekit(unittest.TestCase):

    def setUp(self):
        if self.orekit_data is None:
            self.skipTest('No path to orekit-data given: skipping orekit unit tests.')

        self.init_data = dict(
            state0 = np.array([-7100297.113,-3897715.442,18568433.707,86.771,-3407.231,2961.571]),
            epoch = 57125.7729,
            C_D = 2.3,
            C_R = 1.0,
            m = 8000,
            A = 1.0,
        )
        self.t = np.linspace(0,2*3600.0, num=1000, dtype=np.float)
        self.t0 = np.array([0.0], dtype=np.float)


    def test_get_orbit_kep(self):
        p = Orekit(orekit_data = self.orekit_data)

        ecefs = p.propagate(self.t, **self.init_data)
        self.assertEqual(ecefs.shape, (6,len(self.t)))


    def test_options_tidal(self):
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                frame_tidal_effects=True,
            ),
        )
        ecefs = p.propagate(self.t0, **self.init_data)


    def test_options_frames(self):
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                in_frame='ITRS',
                out_frame='GCRS',
            ),
        )
        ecefs = p.propagate(self.t0, **self.init_data)


    def test_options_integrator(self):
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                integrator='GraggBulirschStoer',
            ),
        )
        ecefs = p.propagate(self.t0, **self.init_data)


    def test_options_tolerance(self):
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                position_tolerance=1.0,
            ),
        )
        ecefs1 = p.propagate(self.t0, **self.init_data)
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                position_tolerance=100.0,
            ),
        )
        ecefs2 = p.propagate(self.t0, **self.init_data)
        nt.assert_array_almost_equal(ecefs2, ecefs1, decimal=1)


    def test_options_gravity_order(self):
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                earth_gravity='HolmesFeatherstone',
                gravity_order=(3,3),
            ),
        )
        ecefs = p.propagate(self.t0, **self.init_data)


    def test_options_gravity_kep(self):
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                earth_gravity='Newtonian',
            ),
        )
        ecefs = p.propagate(self.t0, **self.init_data)


    def test_options_more_solarsystem(self):
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                solarsystem_perturbers=['Moon', 'Sun', 'Jupiter', 'Saturn'],
            ),
        )
        ecefs = p.propagate(self.t0, **self.init_data)


    def test_options_drag_off(self):
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                drag_force=False,
            ),
        )
        ecefs = p.propagate(self.t0, **self.init_data)


    def test_options_rad_off(self):
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                radiation_pressure=False,
            ),
        )
        ecefs = p.propagate(self.t0, **self.init_data)


    def test_options_jpliau(self):
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                constants_source='JPL-IAU',
            ),
        )
        ecefs = p.propagate(self.t0, **self.init_data)
        

    def test_raise_frame(self):
        with self.assertRaises(Exception):
            p = Orekit(
                orekit_data = self.orekit_data,
                settings=dict(
                    in_frame='THIS DOES NOT EXIST',
                ),
            )
            ecefs = p.propagate(self.t0, **self.init_data)
        with self.assertRaises(Exception):
            p = Orekit(
                orekit_data = self.orekit_data,
                settings=dict(
                    out_frame='THIS DOES NOT EXIST',
                ),
            )
            ecefs = p.propagate(self.t0, **self.init_data)


    def test_raise_models(self):
        with self.assertRaises(Exception):
            p = Orekit(
                orekit_data = self.orekit_data,
                settings=dict(
                    earth_gravity='THIS DOES NOT EXIST',
                ),
            )
            ecefs = p.propagate(self.t0, **self.init_data)
        with self.assertRaises(Exception):
            p = Orekit(
                orekit_data = self.orekit_data,
                settings=dict(
                    atmosphere='THIS DOES NOT EXIST',
                ),
            )
            ecefs = p.propagate(self.t0, **self.init_data)
        with self.assertRaises(Exception):
            p = Orekit(
                orekit_data = self.orekit_data,
                settings=dict(
                    solar_activity='THIS DOES NOT EXIST',
                ),
            )
            ecefs = p.propagate(self.t0, **self.init_data)


    def test_raise_bodies(self):
        with self.assertRaises(Exception):
            p = Orekit(
                orekit_data = self.orekit_data,
                settings=dict(
                    solarsystem_perturbers=['THIS DOES NOT EXIST'],
                ),
            )
            ecefs = p.propagate(self.t0, **self.init_data)


    def test_raise_sc_params_missing(self):
        p = Orekit(
            orekit_data = self.orekit_data,
            settings=dict(
                radiation_pressure=True,
                drag_force=True,
            ),
        )
        with self.assertRaises(Exception):
            init__ = self.init_data.copy()
            del init__['C_R']

            ecefs = p.propagate(self.t0, **init__)

        with self.assertRaises(Exception):
            init__ = self.init_data.copy()
            del init__['C_D']

            ecefs = p.propagate(self.t0, **init__)


if __name__ == '__main__':
    unittest.main(verbosity=2)
