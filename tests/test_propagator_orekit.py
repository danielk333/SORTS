import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as n
import numpy as np
import numpy.testing as nt

import time
import dpt_tools as dpt
import orbit_verification as over

from propagator_orekit import PropagatorOrekit


class TestPropagatorOrekit(unittest.TestCase):

    def setUp(self):
        self.init_data = {
            'a': 7500e3,
            'e': 0,
            'inc': 90.0,
            'raan': 10,
            'aop': 10,
            'mu0': 40.0,
            'mjd0': 57125.7729,
            'C_D': 2.3,
            'C_R': 1.0,
            'm': 8000,
            'A': 1.0,
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
        cart = dpt.kep2cart(orb, m=self.init_data['m'], M_cent=p.M_earth, radians=False)

        self.init_data_cart = {
            'x': cart[0],
            'y': cart[1],
            'z': cart[2],
            'vx': cart[3],
            'vy': cart[4],
            'vz': cart[5],
            'mjd0': 57125.7729,
            'C_D': 2.3,
            'C_R': 1.0,
            'm': 8000,
            'A': 1.0,
        }

    def test_get_orbit_kep(self):
        p = PropagatorOrekit()

        ecefs = p.get_orbit(self.t, **self.init_data)
        self.assertEqual(ecefs.shape, (6,len(self.t)))

    def test_circ_orbit(self):
        p = PropagatorOrekit(solarsystem_perturbers=[], radiation_pressure=False)
        self.init_data['a'] = 36000e3

        ecefs = p.get_orbit(self.t, **self.init_data)
        rn = n.sum(ecefs[:3,:]**2, axis=0)/36000.0e3**2

        nt.assert_array_almost_equal(rn, n.ones(rn.shape, dtype=ecefs.dtype), decimal=4)

    def test_options_tidal(self):
        p = PropagatorOrekit(
                frame_tidal_effects=True,
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
    def test_options_frames(self):
        p = PropagatorOrekit(
                in_frame='ITRF',
                out_frame='EME',
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
    def test_options_integrator(self):
        p = PropagatorOrekit(
                integrator='GraggBulirschStoer',
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
    def test_options_tolerance(self):
        p = PropagatorOrekit(
                position_tolerance=1.0,
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
        p = PropagatorOrekit(
                position_tolerance=100.0,
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
    def test_options_gravity_order(self):
        p = PropagatorOrekit(
                earth_gravity='HolmesFeatherstone',
                gravity_order=(3,3),
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
    def test_options_gravity_kep(self):
        p = PropagatorOrekit(
                earth_gravity='Newtonian',
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
    def test_options_more_solarsystem(self):
        p = PropagatorOrekit(
                solarsystem_perturbers=['Moon', 'Sun', 'Jupiter', 'Saturn'],
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
    def test_options_drag_off(self):
        p = PropagatorOrekit(
                drag_force=False,
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
    def test_options_rad_off(self):
        p = PropagatorOrekit(
                radiation_pressure=False,
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
    def test_options_jpliau(self):
        p = PropagatorOrekit(
                constants_source='JPL-IAU',
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)
        
    def test_get_orbit_cart(self):
        p = PropagatorOrekit()

        self._gen_cart(p)

        ecefs = p.get_orbit_cart(self.t, **self.init_data_cart)
        self.assertEqual(ecefs.shape, (6,len(self.t)))

    def test_kep_cart(self):
        p = PropagatorOrekit(
                in_frame='EME',
                out_frame='EME',
        )
        ecefs = p.get_orbit(self.t0, **self.init_data)

        ecefs2 = p.get_orbit_cart(self.t0,
            ecefs[0], ecefs[1], ecefs[2],
            ecefs[3], ecefs[4], ecefs[5],
            mjd0=self.init_data['mjd0'],
            m=self.init_data['m'],
            A=self.init_data['A'],
            C_R=self.init_data['C_R'],
            C_D=self.init_data['C_D'],
        )

        nt.assert_array_almost_equal(ecefs/ecefs2, n.ones(ecefs.shape, dtype=ecefs.dtype), decimal=7)

    def test_frame_conversion(self):
        p0 = PropagatorOrekit(
                in_frame='EME',
                out_frame='EME',
        )
        p = PropagatorOrekit(
                in_frame='EME',
                out_frame='ITRF',
        )
        p2 = PropagatorOrekit(
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
            A=self.init_data['A'],
            C_R=self.init_data['C_R'],
            C_D=self.init_data['C_D'],
        )

        nt.assert_array_almost_equal(ecefs0/ecefs2, n.ones(ecefs.shape, dtype=ecefs.dtype), decimal=7)

    def test_raise_frame(self):
        with self.assertRaises(Exception):
            p = PropagatorOrekit(
                in_frame='THIS DOES NOT EXIST',
            )
        with self.assertRaises(Exception):
            p = PropagatorOrekit(
                out_frame='THIS DOES NOT EXIST',
            )
    def test_raise_models(self):
        with self.assertRaises(Exception):
            p = PropagatorOrekit(
                earth_gravity='THIS DOES NOT EXIST',
            )
        with self.assertRaises(Exception):
            p = PropagatorOrekit(
                atmosphere='THIS DOES NOT EXIST',
            )
            ecefs = p.get_orbit(self.t0, **self.init_data)
        with self.assertRaises(Exception):
            p = PropagatorOrekit(
                solar_activity='THIS DOES NOT EXIST',
            )
            ecefs = p.get_orbit(self.t0, **self.init_data)
    def test_raise_bodies(self):
        with self.assertRaises(Exception):
            p = PropagatorOrekit(
                solarsystem_perturbers=['THIS DOES NOT EXIST'],
            )

    def test_raise_sc_params_missing(self):
        with self.assertRaises(Exception):
            p = PropagatorOrekit()
            del self.init_data['C_R']
            ecefs = p.get_orbit(self.t0, **self.init_data)
        with self.assertRaises(Exception):
            p = PropagatorOrekit()
            del self.init_data['C_D']
            ecefs = p.get_orbit(self.t0, **self.init_data)


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

        prop = PropagatorOrekit(
                in_frame='EME',
                out_frame='EME',
        )

        t = n.array([0.0])
        orbs_done = 0
        for kep in orb_init_list:
            state_ref = dpt.kep2cart(kep, m=self.init_data['m'], M_cent=prop.M_earth, radians=False)

            state = prop.get_orbit_cart(
                t=t, mjd0=mjd0,
                x=state_ref[0], y=state_ref[1], z=state_ref[2],
                vx=state_ref[3], vy=state_ref[4], vz=state_ref[5],
                C_D=self.init_data['C_D'], m=self.init_data['m'], A=self.init_data['A'],
                C_R=self.init_data['C_R'],
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

        prop = PropagatorOrekit(
                in_frame='EME',
                out_frame='EME',
        )

        t = n.array([0.0])
        
        for kep in orb_init_list:
            state_ref = dpt.kep2cart(kep, m=self.init_data['m'], M_cent=prop.M_earth, radians=False)

            state = prop.get_orbit(
                t=t, mjd0=mjd0,
                a=kep[0], e=kep[1], inc=kep[2],
                raan=kep[4], aop=kep[3], mu0=dpt.true2mean(kep[5], kep[1], radians=False),
                C_D=self.init_data['C_D'], m=self.init_data['m'], A=self.init_data['A'],
                C_R=self.init_data['C_R'],
                radians=False,
            )

            state_diff1 = n.abs(state_ref - state[:,0])

            nt.assert_array_less(state_diff1[:3], n.full((3,), 1e-5, dtype=state_diff1.dtype))
            nt.assert_array_less(state_diff1[3:], n.full((3,), 1e-7, dtype=state_diff1.dtype))

    def test_orbit_kep_cart_correspondance(self):
        mjd0 = dpt.jd_to_mjd(2457126.2729)

        orb_init_list = self._gen_orbits(100)

        prop = PropagatorOrekit(
                in_frame='EME',
                out_frame='EME',
        )

        t = n.linspace(0, 12*3600, num=100, dtype=n.float)
        
        for kep in orb_init_list:
            state_ref = dpt.kep2cart(kep, m=self.init_data['m'], M_cent=prop.M_earth, radians=False)

            state_kep = prop.get_orbit(
                t=t, mjd0=mjd0,
                a=kep[0], e=kep[1], inc=kep[2],
                raan=kep[4], aop=kep[3], mu0=dpt.true2mean(kep[5], kep[1], radians=False),
                C_D=self.init_data['C_D'], m=self.init_data['m'], A=self.init_data['A'],
                C_R=self.init_data['C_R'],
                radians=False,
            )
            state_cart = prop.get_orbit_cart(
                t=t, mjd0=mjd0,
                x=state_ref[0], y=state_ref[1], z=state_ref[2],
                vx=state_ref[3], vy=state_ref[4], vz=state_ref[5],
                C_D=self.init_data['C_D'], m=self.init_data['m'], A=self.init_data['A'],
                C_R=self.init_data['C_R'],
            )

            state_diff1 = n.abs(state_kep - state_cart)

            nt.assert_array_less(state_diff1[:3,:], n.full((3,t.size), 1e-5, dtype=state_diff1.dtype))
            nt.assert_array_less(state_diff1[3:,:], n.full((3,t.size), 1e-7, dtype=state_diff1.dtype))



class TestSentinel(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.p = PropagatorOrekit(in_frame='ITRF', out_frame='ITRF', frame_tidal_effects=True)
        super(TestSentinel, self).__init__(*args, **kwargs)

    def setUp(self):
        self.sv, _, _ = over.read_poe('data/S1A_OPER_AUX_POEORB_OPOD_20150527T122640_V20150505T225944_20150507T005944.EOF')



    def test_tg_cart0(self):
        '''
        See if cartesian orbit interface recovers starting state
        '''
        

        # Statevector from Sentinel-1 precise orbit (in ECEF frame)
        #sv = n.array([('2015-04-30T05:45:44.000000000',
        #    [2721793.785377, 1103261.736653, 6427506.515945],
        #    [ 6996.001258,  -171.659563, -2926.43233 ])],
        #      dtype=[('utc', '<M8[ns]'), ('pos', '<f8', (3,)), ('vel', '<f8', (3,))])
        sv = self.sv

        x,  y,  z  = sv[0].pos
        vx, vy, vz = sv[0].vel
        mjd0 = (sv[0]['utc'] - n.datetime64('1858-11-17')) / n.timedelta64(1, 'D')

        t = [0]
        pv = self.p.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0,
                              m=2300., C_R=1., C_D=2.3, A=4*2.3)

        print('pos error:')
        dp = sv['pos'] - pv[:3].T
        print('{:.5e}, {:.5e}, {:.5e}'.format(*dp[0].tolist()))

        print('vel error:')
        dv = sv['vel'] - pv[3:].T
        print('{:.5e}, {:.5e}, {:.5e}'.format(*dv[0].tolist()))

        nt.assert_array_almost_equal(sv['pos'] / pv[:3].T, n.ones((1,3)), decimal=7)
        nt.assert_array_almost_equal(sv['vel'] / pv[3:].T, n.ones((1,3)), decimal=7)


    def test_tg_cartN(self):
        '''
        See if cartesian orbit propagation interface matches actual orbit
        '''

        # Statevector from Sentinel-1 precise orbit (in ECEF frame)
        #sv = n.array([
        #    ('2015-04-30T05:45:44.000000000',
        #        [2721793.785377, 1103261.736653, 6427506.515945],
        #        [ 6996.001258,  -171.659563, -2926.43233 ]),
        #    ('2015-04-30T05:45:54.000000000',
        #        [2791598.832403, 1101432.471307, 6397880.289842],
        #        [ 6964.872299,  -194.182612, -2998.757484]),
        #    ('2015-04-30T05:46:04.000000000',
        #        [2861088.520266, 1099378.309568, 6367532.487662],
        #        [ 6932.930021,  -216.638226, -3070.746198]),
        #    ('2015-04-30T05:46:14.000000000',
        #        [2930254.733863, 1097099.944255, 6336466.514344], 
        #        [ 6900.178053,  -239.022713, -3142.39037 ]), 
        #    ('2015-04-30T05:46:24.000000000',
        #        [2999089.394834, 1094598.105058, 6304685.855646],
        #        [ 6866.620117,  -261.332391, -3213.681933]),
        #    ('2015-04-30T05:46:34.000000000', 
        #        [3067584.462515, 1091873.55841 , 6272194.077798], 
        #        [ 6832.260032,  -283.563593, -3284.612861])],
        #    dtype=[('utc', '<M8[ns]'), ('pos', '<f8', (3,)), ('vel', '<f8', (3,))])
        sv = self.sv

        x,  y,  z  = sv[0].pos
        vx, vy, vz = sv[0].vel
        mjd0 = (sv[0]['utc'] - n.datetime64('1858-11-17')) / n.timedelta64(1, 'D')

        N = 7

        t = 10*n.arange(N)
        pv = self.p.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0,
                              m=2300., C_R=1., C_D=2.3, A=4*2.3)

        nt.assert_array_less(n.abs(sv[:N]['pos'] - pv[:3].T), n.full((N,3), 1.0, dtype=pv.dtype)) #m
        nt.assert_array_less(n.abs(sv[:N]['vel'] - pv[3:].T), n.full((N,3), 1.0e-3, dtype=pv.dtype)) #mm/s

    def test_longer(self):
        sv = self.sv

        x,  y,  z  = sv[0].pos
        vx, vy, vz = sv[0].vel
        mjd0 = (sv[0]['utc'] - np.datetime64('1858-11-17')) / np.timedelta64(1, 'D')

        N = len(sv)
        t = 10*n.arange(N)

        pv = self.p.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0,
                              m=2300., C_R=0., C_D=.0, A=4*2.3)

        perr = np.linalg.norm(pv[:3].T - sv.pos, axis=1)
        verr = np.linalg.norm(pv[3:].T - sv.vel, axis=1)

        # f, ax = plt.subplots(2,1)
        # ax[0].plot(t/3600., perr)
        # ax[0].set_title('Errors from propagation')
        # ax[0].set_ylabel('Position error [m]')
        # ax[1].plot(t/3600., verr)
        # ax[1].set_ylabel('Velocity error [m/s]')
        # ax[1].set_xlabel('Time [h]')


    def test_s3_pair(self):

        '''
        Statevectors from S3 product
        S3A_OL_1_EFR____20180926T093816_20180926T094049_20180927T140153_0153_036_136_1620_LN1_O_NT_002
        (https://code-de.org/Sentinel3/OLCI/2018/09/26/)
        '''
        def _decomp(v):
            return n.array([float(v['x']), float(v['y']), float(v['z'])])

        sv = [{'epoch': {'TAI': '2018-09-26T09:11:26.318893',
                         'UTC': '2018-09-26T09:10:49.318893',
                         'UT1': '2018-09-26T09:10:49.371198'},
               'position': {'x': -7018544.618, 'y': -1531717.645, 'z': 0.001},
               'velocity': {'x': -341.688439, 'y': 1605.354223, 'z': 7366.313136}},
              {'epoch': {'TAI': '2018-09-26T10:52:25.494280',
                         'UTC': '2018-09-26T10:51:48.494280',
                         'UT1': '2018-09-26T10:51:48.546508'},
               'position': {'x': -7001440.416, 'y': 1608173.405, 'z': 0.004},
               'velocity': {'x': 375.658124, 'y': 1597.811148, 'z': 7366.302506}}]


        t0 = n.datetime64(sv[0]['epoch']['TAI'])
        t1 = n.datetime64(sv[1]['epoch']['TAI'])

        t0pos = _decomp(sv[0]['position'])
        t0vel = _decomp(sv[0]['velocity'])
        t1pos = _decomp(sv[1]['position'])
        t1vel = _decomp(sv[1]['velocity'])

        x0,  y0,  z0  = t0pos
        vx0, vy0, vz0 = t0vel
        x1,  y1,  z1  = t1pos
        vx1, vy1, vz1 = t1vel

        t = (t1-t0)/dpt.sec

        mjd0 = dpt.npdt2mjd(dpt.tai2utc(t0))
        mjd1 = dpt.npdt2mjd(dpt.tai2utc(t1))
        # mjd0 = (sv[0]['utc'] - n.datetime64('1858-11-17')) / n.timedelta64(1, 'D')

        # Propagate forwards from t0 to t1
        # If I disable drag and radiation pressure here, errors increase
        # by a factor of almost 300
        pv1 = self.p.get_orbit_cart(t, x0, y0, z0, vx0, vy0, vz0, mjd0,
                              m=1250., C_R=1., C_D=2.3, A=2.2*2.2)

        nt.assert_(np.linalg.norm(t1pos - pv1[:3].T) < 150, 'S3 propagate position')     # m
        nt.assert_(np.linalg.norm(t1vel - pv1[3:].T) < .15, 'S3 propagate velocity')     # m/s

        # Propagate backwards from t1 to t0
        # Need to disable drag and solar radiation pressure,
        # and increase tolerances by factor of 2.33
        # So does that mean Orekit bungles drag when propagating backwards in time?
        pv0 = self.p.get_orbit_cart(-t, x1, y1, z1, vx1, vy1, vz1, mjd1,
                              m=1250., C_R=0., C_D=0., A=2.2*2.2)

        nt.assert_(np.linalg.norm(t0pos - pv0[:3].T) < 350, 'S3 propagate position')     # m
        nt.assert_(np.linalg.norm(t0vel - pv0[3:].T) < .35, 'S3 propagate velocity')     # m/s

if __name__ == '__main__':
    unittest.main(verbosity=2)
