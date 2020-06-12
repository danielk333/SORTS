import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as n
import numpy.testing as nt

import antenna
import coord

class TestAntennaRX(unittest.TestCase):

    def setUp(self):
        self.beam_u = antenna.BeamPattern(lambda k, sf: sf.I_0, az0=0.0, el0=90.0, I_0=1.0, f=1.0, beam_name='Test')

    def test_init(self):
        ant = antenna.AntennaRX(name='a rx', lat=2.43, lon=11.0, el_thresh=30.0, freq=1e4, rx_noise=10e3, beam=self.beam_u, alt = 0.0)

    def test_ecef(self):
        dec = 4
        ant = antenna.AntennaRX(name='a rx', lat=0, lon=0, el_thresh=30.0, freq=1e4, rx_noise=10e3, beam=self.beam_u, alt = 0.0)
        nt.assert_almost_equal(ant.ecef[0], coord.a, decimal=dec)

        ant = antenna.AntennaRX(name='a rx', lat=0, lon=0, el_thresh=30.0, freq=1e4, rx_noise=10e3, beam=self.beam_u, alt = 100.0)
        nt.assert_almost_equal(ant.ecef[0], coord.a+100, decimal=dec)

        ant = antenna.AntennaRX(name='a rx', lat=90, lon=0, el_thresh=30.0, freq=1e4, rx_noise=10e3, beam=self.beam_u, alt = 0.0)
        nt.assert_almost_equal(ant.ecef[2], coord.b, decimal=dec)
        
        ant = antenna.AntennaRX(name='a rx', lat=0, lon=90, el_thresh=30.0, freq=1e4, rx_noise=10e3, beam=self.beam_u, alt = 0.0)
        nt.assert_almost_equal(ant.ecef[1], coord.a, decimal=dec)
    
    def test_str(self):
        ant = antenna.AntennaRX(name='a rx', lat=2.43, lon=11.0, el_thresh=30.0, freq=1e4, rx_noise=10e3, beam=self.beam_u, alt = 0.0)
        print(ant)



class TestAntennaTX(unittest.TestCase):

    def setUp(self):
        self.beam_u = antenna.BeamPattern(lambda k, sf: sf.I_0, az0=0.0, el0=90.0, I_0=1.0, f=1.0, beam_name='Test')

    def test_init(self):
        ant = antenna.AntennaTX(
            name='a tx',
            lat=2.43,
            lon=11.0,
            el_thresh=30.0,
            freq=1e4,
            rx_noise=10e3,
            beam=self.beam_u,
            alt = 0.0,
            scan=None,
            tx_power=1.0,
            tx_bandwidth=1.0,
            duty_cycle=1.0,
            pulse_length=1e-3,
            ipp=10e-3,
            n_ipp=20,
        )

    def test_get_scan(self):
        def scan_cont(ant, t):
            if t > 10:
                return ant.extra_scans[0]
            else:
                return ant.scan

        ant = antenna.AntennaTX(
            name='a tx',
            lat=2.43,
            lon=11.0,
            el_thresh=30.0,
            freq=1e4,
            rx_noise=10e3,
            beam=self.beam_u,
            alt = 0.0,
            scan='SCAN1',
            tx_power=1.0,
            tx_bandwidth=1.0,
            duty_cycle=1.0,
            pulse_length=1e-3,
            ipp=10e-3,
            n_ipp=20,
        )

        ant.extra_scans = ['SCAN2', 'SCAN3']
        ant.scan_controler = scan_cont

        scn = ant.get_scan(2.3)
        self.assertEqual(scn, ant.scan)

        scn = ant.get_scan(20)
        self.assertEqual(scn, ant.extra_scans[0])
        
    def test_set_scan(self):
        from radar_scans import RadarScan
        
        def pfun(t, **kw):
            if t < 2.5:
                return 0, 0, 1
            else:
                return 180, 0, 1
        
        def pfun2(t, **kw):
            return 0, 90, 1
        scan1 = RadarScan(23, 123, 101, pfun, 0.1, pointing_coord='azel', name='a scan')
        scan2 = RadarScan(23, 123, 101, pfun2, 0.1, pointing_coord='azel', name='12 scan')
        scan3 = RadarScan(23, 123, 101, pfun, 0.1, pointing_coord='azel', name='32 scan')
        
        def scan_cont(ant, t):
            if t > 10:
                return ant.extra_scans[0]
            else:
                return ant.scan

        ant = antenna.AntennaTX(
            name='a tx',
            lat=0.0,
            lon=0.0,
            alt = 0.0,
            el_thresh=30.0,
            freq=1e4,
            rx_noise=10e3,
            beam=self.beam_u,
            scan=scan1,
            tx_power=1.0,
            tx_bandwidth=1.0,
            duty_cycle=1.0,
            pulse_length=1e-3,
            ipp=10e-3,
            n_ipp=10,
        )

        ant.set_scan(extra_scans = [scan2, scan3], scan_controler = scan_cont)

        scn = ant.get_scan(2.3)
        assert scn is scan1

        scn = ant.get_scan(20)
        assert scn is scan2
        
        dec = 3
        p0, k0 = ant.get_pointing(2.3)
        nt.assert_almost_equal(k0[2], 1.0, decimal=dec)
    
        p0, k0 = ant.get_pointing(7.8)
        nt.assert_almost_equal(k0[2], -1.0, decimal=dec)

        p0, k0 = ant.get_pointing(15.)
        nt.assert_almost_equal(k0[0], 1.0, decimal=dec)



class TestGainConv(unittest.TestCase):

    def test_inst_gain2full_gain(self):
        IG = 1.0
        FG = antenna.inst_gain2full_gain(gain = IG, groups = 1, N_IPP = 1, IPP_scale=1.0, units = '')
        self.assertAlmostEqual(IG, FG)

        FG = antenna.inst_gain2full_gain(gain = IG, groups = 1, N_IPP = 1, IPP_scale=1.0, units = 'dB')
        self.assertAlmostEqual(IG, FG)

        FG = 1.0
        IG = antenna.full_gain2inst_gain(gain = FG, groups = 1, N_IPP = 1, IPP_scale=1.0, units = '')
        self.assertAlmostEqual(IG, FG)

        IG = antenna.full_gain2inst_gain(gain = FG, groups = 1, N_IPP = 1, IPP_scale=1.0, units = 'dB')
        self.assertAlmostEqual(IG, FG)

        IG = antenna.inst_gain2full_gain(gain = 10.0, groups = 1, N_IPP = 1)
        FG = antenna.full_gain2inst_gain(gain = IG, groups = 1, N_IPP = 1)
        self.assertAlmostEqual(10.0, FG)

        IG = antenna.inst_gain2full_gain(gain = 20.0, groups = 1, N_IPP = 1)
        FG = antenna.full_gain2inst_gain(gain = IG, groups = 1, N_IPP = 1)
        self.assertAlmostEqual(20.0, FG)

    def test_inst_gain2full_gain_types(self):
        gains = n.linspace(1.0,100, dtype=n.float)
        FGgains = IG = antenna.inst_gain2full_gain(gain = gains, groups = 1, N_IPP = 1)

        self.assertEqual(type(gains), type(FGgains))
        self.assertEqual(gains.shape, FGgains.shape)
        self.assertEqual(gains.dtype, FGgains.dtype)


class TestBeamPattern(unittest.TestCase):

    def setUp(self):
        fun_quad = lambda k, sf: sf.I_0*(sf.angle_k(k)/180.0)**2
        self.beam_u = antenna.BeamPattern(lambda k, sf: sf.I_0, az0=0.0, el0=90.0, I_0=1.0, f=1.0, beam_name='Test')
        self.beam_quad = antenna.BeamPattern(fun_quad, az0=0.0, el0=90.0, I_0=1.0, f=1.0, beam_name='Test')

    def test_on_axis(self):
        nt.assert_array_almost_equal(self.beam_quad.on_axis, n.array([0,0,1], dtype=n.float) )

        self.beam_u.point(0,0)
        self.assertAlmostEqual(self.beam_u.az0,0)
        self.assertAlmostEqual(self.beam_u.el0,0)
        
        nt.assert_array_almost_equal(self.beam_u.on_axis, n.array([0,1,0], dtype=n.float) )

        self.beam_u.point_k0(n.array([1,0,0], dtype=n.float))

        self.assertAlmostEqual(self.beam_u.az0,90)
        self.assertAlmostEqual(self.beam_u.el0,0)
        
        nt.assert_array_almost_equal(self.beam_u.on_axis, n.array([1,0,0], dtype=n.float) )


    def test_gain(self):
        x_vecs = n.random.rand(3,1000)*2.0 - 1
        for ind in range(1000):
            self.assertAlmostEqual( self.beam_u.gain(x_vecs[:,ind]), 1.0 )

            self.assertAlmostEqual( self.beam_quad.gain(x_vecs[:,ind]), (coord.angle_deg(x_vecs[:,ind], n.array([0,0,1], dtype=n.float))/180.0)**2 )

            self.assertLess( self.beam_quad.gain(x_vecs[:,ind]), self.beam_quad.I_0 )




if __name__ == '__main__':
    unittest.main(verbosity=2)