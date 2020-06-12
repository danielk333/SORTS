import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as n
import numpy as np
import numpy.testing as nt

import time
import dpt_tools as dpt
from propagator_sgp4 import M_earth
from propagator_kepler import PropagatorKepler


class TestPropagatorKepler(unittest.TestCase):

    def setUp(self):
        self.init_data = {
            'a': 7500e3,
            'e': 0,
            'inc': 90.0,
            'raan': 10,
            'aop': 10,
            'mu0': 40.0,
            'mjd0': 57125.7729,
            'm': 8000,
        }

        self.t = n.linspace(0,2*3600.0, num=1000, dtype=n.float)
        self.t0 = n.array([0.0], dtype=n.float)

    def _gen_cart(self, p):
        orb = n.array([
            self.init_data['a'],
            self.init_data['e'],
            self.init_data['inc'],
            self.init_data['aop'],
            self.init_data['raan'],
            dpt.mean2true(self.init_data['mu0'], self.init_data['e'], radians=False),
        ])
        cart = dpt.kep2cart(orb, m=self.init_data['m'], M_cent=M_earth, radians=False)

        self.init_data_cart = {
            'x': cart[0],
            'y': cart[1],
            'z': cart[2],
            'vx': cart[3],
            'vy': cart[4],
            'vz': cart[5],
            'mjd0': 57125.7729,
            'm': 8000,
        }

    def test_get_orbit_kep(self):
        p = PropagatorKepler()

        ecefs = p.get_orbit(self.t, **self.init_data)
        self.assertEqual(ecefs.shape, (6,len(self.t)))

    def test_circ_orbit(self):
        p = PropagatorKepler(in_frame='TEME', out_frame='TEME')
        self.init_data['a'] = 36000e3

        ecefs = p.get_orbit(self.t, **self.init_data)
        rn = n.sum(ecefs[:3,:]**2, axis=0)/36000.0e3**2

        nt.assert_array_almost_equal(rn, n.ones(rn.shape, dtype=ecefs.dtype), decimal=4)

    def test_options_frames(self):
        p = PropagatorKepler(
                in_frame='ITRF',
                out_frame='EME',
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
    
    def test_get_orbit_cart(self):
        p = PropagatorKepler()

        self._gen_cart(p)

        ecefs = p.get_orbit_cart(self.t, **self.init_data_cart)
        self.assertEqual(ecefs.shape, (6,len(self.t)))

    def test_kep_cart(self):
        p = PropagatorKepler(
                in_frame='EME',
                out_frame='EME',
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)

        ecefs2 = p.get_orbit_cart(self.t0,
            ecefs[0], ecefs[1], ecefs[2],
            ecefs[3], ecefs[4], ecefs[5],
            mjd0=self.init_data['mjd0'],
            m=self.init_data['m'],
        )

        nt.assert_array_almost_equal(ecefs/ecefs2, n.ones(ecefs.shape, dtype=ecefs.dtype), decimal=7)

    def test_frame_conversion(self):
        p0 = PropagatorKepler(
                in_frame='EME',
                out_frame='EME',
        )
        p = PropagatorKepler(
                in_frame='EME',
                out_frame='ITRF',
        )
        p2 = PropagatorKepler(
                in_frame='ITRF',
                out_frame='EME',
        )
        ecefs0 = p0.get_orbit(self.t0, **self.init_data)
        ecefs = p.get_orbit(self.t0, **self.init_data)

        ecefs2 = p2.get_orbit_cart(self.t0,
            ecefs[0], ecefs[1], ecefs[2],
            ecefs[3], ecefs[4], ecefs[5],
            mjd0=self.init_data['mjd0'],
            m=self.init_data['m'],
        )

        nt.assert_array_almost_equal(ecefs0/ecefs2, n.ones(ecefs.shape, dtype=ecefs.dtype), decimal=7)

    def _gen_orbits(self,num):
        R_E = 6353.0e3
        
        a = R_E*2.0
        orb_init_list = []

        orb_init_list.append(n.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 270], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 1e-9, 0.0, 0.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 75.0, 0.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 0.0, 120.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 0.0, 0.0, 35.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 75.0, 120.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 75.0, 0.0, 35.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 75.0, 120.0, 35.0, 0.0], dtype=n.float))

        n.random.seed(12398774)

        orb_range = n.array([a, 0.9, 180, 360, 360, 360], dtype=n.float)
        orb_offset = n.array([R_E*1.1, 0, 0.0, 0.0, 0.0, 0.0], dtype=n.float)
        while len(orb_init_list) < num:
            orb = n.random.rand(6)
            orb = orb_offset + orb*orb_range
            if orb[0]*(1.0 - orb[1]) > R_E+200e3:
                orb_init_list.append(orb)

        n.random.seed(None)

        return orb_init_list

    def test_orbit_inverse_error_cart(self):
        mjd0 = dpt.jd_to_mjd(2457126.2729)

        orb_init_list = self._gen_orbits(1000)

        prop = PropagatorKepler(
                in_frame='TEME',
                out_frame='TEME',
        )

        t = n.array([0.0])
        orbs_done = 0
        for kep in orb_init_list:
            state_ref = dpt.kep2cart(kep, m=self.init_data['m'], M_cent=M_earth, radians=False)

            state = prop.get_orbit_cart(
                t=t, mjd0=mjd0,
                x=state_ref[0], y=state_ref[1], z=state_ref[2],
                vx=state_ref[3], vy=state_ref[4], vz=state_ref[5],
                m=self.init_data['m'],
            )

            state_diff1 = n.abs(state_ref - state[:,0])
            if n.linalg.norm(state_diff1[:3]) > 1e-5:
                print(kep)
                print(orbs_done)
            nt.assert_array_less(state_diff1[:3], n.full((3,), 1e-5, dtype=state_diff1.dtype))
            nt.assert_array_less(state_diff1[3:], n.full((3,), 1e-7, dtype=state_diff1.dtype))
            orbs_done += 1


    def test_orbit_inverse_error_kep(self):
        mjd0 = dpt.jd_to_mjd(2457126.2729)

        orb_init_list = self._gen_orbits(1000)

        prop = PropagatorKepler(
                in_frame='EME',
                out_frame='EME',
        )

        t = n.array([0.0])
        
        for kep in orb_init_list:
            state_ref = dpt.kep2cart(kep, m=self.init_data['m'], M_cent=M_earth, radians=False)

            state = prop.get_orbit(
                t=t, mjd0=mjd0,
                a=kep[0], e=kep[1], inc=kep[2],
                raan=kep[4], aop=kep[3], mu0=dpt.true2mean(kep[5], kep[1], radians=False),
                m=self.init_data['m'],
                radians=False,
            )

            state_diff1 = n.abs(state_ref - state[:,0])

            nt.assert_array_less(state_diff1[:3], n.full((3,), 1e-5, dtype=state_diff1.dtype))
            nt.assert_array_less(state_diff1[3:], n.full((3,), 1e-7, dtype=state_diff1.dtype))

    def test_orbit_kep_cart_correspondance(self):
        mjd0 = dpt.jd_to_mjd(2457126.2729)

        orb_init_list = self._gen_orbits(100)

        prop = PropagatorKepler(
                in_frame='EME',
                out_frame='EME',
        )

        t = n.linspace(0, 12*3600, num=100, dtype=n.float)
        
        for kep in orb_init_list:
            state_ref = dpt.kep2cart(kep, m=self.init_data['m'], M_cent=M_earth, radians=False)

            state_kep = prop.get_orbit(
                t=t, mjd0=mjd0,
                a=kep[0], e=kep[1], inc=kep[2],
                raan=kep[4], aop=kep[3], mu0=dpt.true2mean(kep[5], kep[1], radians=False),
                m=self.init_data['m'], 
                radians=False,
            )
            state_cart = prop.get_orbit_cart(
                t=t, mjd0=mjd0,
                x=state_ref[0], y=state_ref[1], z=state_ref[2],
                vx=state_ref[3], vy=state_ref[4], vz=state_ref[5],
                m=self.init_data['m'],
            )

            state_diff1 = n.abs(state_kep - state_cart)

            nt.assert_array_less(state_diff1[:3,:], n.full((3,t.size), 1e-5, dtype=state_diff1.dtype))
            nt.assert_array_less(state_diff1[3:,:], n.full((3,t.size), 1e-7, dtype=state_diff1.dtype))


if __name__ == '__main__':
    unittest.main(verbosity=2)
