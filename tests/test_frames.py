import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import time
import re

import unittest
import numpy as np
import numpy.testing as nt
import scipy
import scipy.constants as consts

import sgp4

from sgp4.api import Satrec

twoline2rv = Satrec.twoline2rv
def _propagate(rv, t):
    e, p, v = rv.sgp4(t.jd1, t.jd2)
    if e != 0: raise
    return p, v

from sorts import frames, dates
from sorts.propagator import SGP4

from astropy.coordinates import CartesianRepresentation, CartesianDifferential
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates.builtin_frames import ITRS, TEME


def vdot(va, vb):
    """dot products between equal or shape-compatible arrays of N-vectors"""
    return np.sum(np.asarray(va) * np.asarray(vb), axis=-1)

def vnorm2(vector):
    """L2-norm of array of N-vectors"""
    return np.sqrt(vdot(vector, vector))


def Time_from_rv(rv):
    # epoch is in UTC or UT1 (not clear which). The difference is < 1 sec always.
    return Time(rv.jdsatepoch, rv.jdsatepochF, format='jd')



def load_tle_table(fname):
    tab = {}
    src = open(fname, 'rt')
    try:
        while True:
            name = next(src).strip()
            line1 = next(src)
            line2 = next(src)
            tab[name] = twoline2rv(line1, line2)
    except StopIteration:
        src.close()
        return tab
    raise

# Descriptor for numpy dtype representing a state vector
svec_d = [
   ('UTC',        'datetime64[ns]'),          # absolute time in UTC
   ('TAI',        'datetime64[ns]'),          # absolute time in TAI
   ('POS',         np.double, (3,)),          # ECEF position [m]
   ('VEL',         np.double, (3,))]          # ECEF velocity [m/s]

# Scarfed from gdar.readers.sentinel1_orbit.py
def read_statevectors(fname):
    utc_r = re.compile(r'<(UTC|TAI)>\1=([0-9T:.-]+)</\1>')
    tag_r = re.compile(r'<([A-Za-z_]+)>(.+)</\1>')
    xyz_r = re.compile(r'<(V?[XYZ]) unit="m(/s)?">([+-]?\d+\.\d+)</\1>')

    sv_data = []
    current = None

    if isinstance(fname, bytes):
        fname = fname.decode('UTF-8')

    with open(fname) as src:
      for line in src:
        line = str(line)
        if '<OSV>' in line:
            current = {}
        elif '</OSV>' in line:
            newitem = {}
            newitem['pos'] = [current.pop('X', None),
                              current.pop('Y', None),
                              current.pop('Z', None)]
            newitem['vel'] = [current.pop('VX', None),
                              current.pop('VY', None),
                              current.pop('VZ', None)]
            newitem['tai'] = current.pop('TAI', None)
            newitem['utc'] = current.pop('UTC', None)
            newitem['abs_orbit'] = current.pop('Absolute_Orbit', None)
            newitem['quality'] = current.pop('Quality', None)
            sv_data.append(newitem)
            current = None
        else:
            if current is None:
                continue
            try:
                m = utc_r.search(line)
                if m:
                    g = m.groups()
                    current[g[0]] = g[1]
                    continue
                m = xyz_r.search(line)
                if m:
                    g = m.groups()
                    current[g[0]] = float(g[2])
                    continue
                m = tag_r.search(line)
                if m:
                    g = m.groups()
                    if g[0] == 'Absolute_Orbit':
                        current[g[0]] = int(g[1])
                    else:
                        current[g[0]] = g[1]  # qual[m.groups()[1]]
            except TypeError:
                print("Trying add fields to abandoned statevector")
            except ValueError:
                print("Line <{0}> not understood".format(line))

    sv = np.recarray(len(sv_data), dtype=svec_d)
    sv['TAI'] = [s['tai'] for s in sv_data]
    sv['UTC'] = [s['utc'] for s in sv_data]
    sv['POS'] = [s['pos'] for s in sv_data]
    sv['VEL'] = [s['vel'] for s in sv_data]

    return sv


class TestFrames(unittest.TestCase):

    def setUp(self):
        # Test vector using ISS (https://en.wikipedia.org/wiki/Two-line_element_set)
        # self.line1 = '1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927'
        # self.line2 = '2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537'

        # "Earth Resources" TLE file downloaded as 'tests/eos_tle.txt'
        # from https://celestrak.com/NORAD/elements/ on 2020-07-21
        self.tab = load_tle_table('tests/eos_tle.txt')
        self.resorb = read_statevectors('tests/S1A_OPER_AUX_RESORB_OPOD_20200721T073339_V20200721T023852_20200721T055622.EOF')


    def test_sgp4_version_workaround(self):
        rv = self.tab['SENTINEL-1A']

        rv_ept = Time_from_rv(rv)

        # Extract timetuple, using DK's methods
        # tuple: year, month, day, hr, min, sec, microseconds
        year, month, day, hour, minute, sec = rv_ept.ymdhms

        # Extract TEME Cartesian state vector at epoch
        err, po0, ve0 = rv.sgp4(rv_ept.jd1, rv_ept.jd2)
        assert err == 0
        x0 = np.array([po0 + ve0])

        # unified
        pos, vel = _propagate(rv, rv_ept)
        x = np.array([pos + vel])


        nt.assert_array_equal(x0,x)


    def test_tle_init(self):
        """
        See whether SGP4 class in sorts/propagator/pysgp4.py
        initialised from TLE values is consistent with the above.
        The propagator uses Astropy to rotate the TEME results into
        ITRS.  Hence, if this test succeeds, then the transformation is reasonably accurate
        when initialized from an ECI (TEME) cartesian statevector.
        """

        # Extract one statevector for initialization
        rv = self.tab['SENTINEL-1A']
        s1_res = self.resorb

        rv_ept = Time_from_rv(rv)

        # Find first statevector past TLE epoch
        ix = np.where(s1_res.UTC > rv_ept.datetime64)[0][0]
        svec = s1_res[ix]

        # Extract TEME statevector for initialization of SORTS propagation object
        pos, vel = _propagate(rv, rv_ept)

        state_eci = np.array(pos + vel) * 1e3

        # SORTS-style SGP4 propagator
        prp = SGP4(settings=dict(in_frame='TEME', out_frame='ITRS'))

        # Should be Cartesian ITRF coordinates in SI units
        posvel = prp.propagate(Time(svec.UTC)-rv_ept, state_eci, rv_ept)
        ppos, pvel = posvel[:3], posvel[3:]

        assert vnorm2(svec.POS - ppos) < 800.0, 'Inaccurate position'
        assert vnorm2(svec.VEL - pvel) < 1.0,   'Inaccurate velocity'

        for ii in range(200):
            # dt between statevectors is 10.0 seconds
            svec = s1_res[ix+ii]

            posvel = prp.propagate(Time(svec.UTC)-rv_ept, state_eci, rv_ept)
            ppos, pvel = posvel[:3], posvel[3:]

            assert vnorm2(svec.POS - ppos) < 800.0, f'Inaccurate position: {ii}'
            assert vnorm2(svec.VEL - pvel) < 1.0,   f'Inaccurate velocity: {ii}'



    def test_resorb_init(self):
        """
        See whether SGP4 class in sorts/propagator/pysgp4.py
        initialised from RES orbit ITRF statevector values
        is consistent with the above.
        The propagator uses frames.TEME_to_ECEF to rotate the TEME results into
        ITRF.  Hence, if this test succeeds, then TEME_to_ECEF is reasonably accurate
        when initialized from a ECEC (ITRS) cartesian statevector rotated to
        ECI (TEME) using astropy-derived rotation matrix (see above)
        """

        # Extract one statevector for initialization
        rv = self.tab['SENTINEL-1A']
        s1_res = self.resorb


        rv_ept = Time_from_rv(rv)

        # Find first statevector past TLE epoch
        ix = np.where(s1_res.UTC > rv_ept.datetime64)[0][0]
        svec = s1_res[ix]

        epoch0 = Time(svec.UTC, scale='utc')

        state_ecef = np.r_[svec.POS, svec.VEL]

        # SORTS-style SGP4 propagator
        prp = SGP4(settings=dict(in_frame='ITRS', out_frame='ITRS'))

        # Should be Cartesian ITRF coordinates in SI units
        posvel = prp.propagate(0., state_ecef, epoch0)
        ppos, pvel = posvel[:3], posvel[3:]

        assert vnorm2(svec.POS - ppos) < 800.0, 'Inaccurate position'
        assert vnorm2(svec.VEL - pvel) < 1.0,   'Inaccurate velocity'

        for ii in range(200):
            # dt between statevectors is 10.0 seconds
            svec = s1_res[ix+ii]
            posvel = prp.propagate(10.0*ii, state_ecef, epoch0)
            ppos, pvel = posvel[:3], posvel[3:]

            assert vnorm2(svec.POS - ppos) < 800.0, 'Inaccurate position'
            assert vnorm2(svec.VEL - pvel) < 1.0,   'Inaccurate velocity'



    def propagate_from_RES(self, nprop=400):
        """
        Initialise SPG4 propagation from a RES statevector, and compare to
        actually observed trajectory.
        """
        # Extract one statevector for initialization
        rv = self.tab['SENTINEL-1A']
        s1_res = self.resorb

        ix = 50
        svec = s1_res[ix]

        state_ecef = np.r_[svec.POS, svec.VEL]

        # SORTS-style SGP4 propagator
        prp = SGP4(settings=dict(in_frame='ITRS', out_frame='ITRS'))

        ii = np.arange(nprop)

        pv_itrf = prp.propagate(10*ii, state_ecef, Time(svec.UTC))
        pos = pv_itrf[:3].T
        vel = pv_itrf[3:].T

        rpos = s1_res[ix+ii].POS
        rvel = s1_res[ix+ii].VEL

        return pos, vel, rpos, rvel


    def test_propagate_from_RES(self, nprop=100):

        pos, vel, rpos, rvel = self.propagate_from_RES(nprop)

        np.testing.assert_array_less(vnorm2(pos[:nprop]-rpos[:nprop]), 100), \
            'Position errors too large'

        np.testing.assert_array_less(vnorm2(vel[:nprop]-rvel[:nprop]), 0.24), \
            'Velocity errors too large'

    def plot_propagation_errors(self, nprop=400):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(2, 2, sharex='col')

        pos, vel, rpos, rvel = self.propagate_from_RES(nprop)

        tt = 10 * np.arange(len(pos))

        def do_plot(ax, t, comp, rcomp, lab):
            ax.plot(t, comp[:,0]-rcomp[:,0], 'r-')
            ax.plot(t, comp[:,1]-rcomp[:,1], 'g-')
            ax.plot(t, comp[:,2]-rcomp[:,2], 'b-')
            ax.plot(t, vnorm2(comp-rcomp), 'k--')
            ax.legend([ lab + ' x', lab + ' y', lab + ' z', '|' + lab + '|'])

        do_plot(ax[0,0], tt[:150], pos[:150], rpos[:150], 'pos error')
        do_plot(ax[0,1], tt, pos, rpos, 'pos error')

        do_plot(ax[1,0], tt[:150], vel[:150], rvel[:150], 'vel error')
        do_plot(ax[1,1], tt, vel, rvel, 'vel error')







