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
# import TLE_tools as tle
# import dpt_tools as dpt

import sgp4
if hasattr(sgp4, 'api'):
    # version 2.0 of sgp4 or newer
    from sgp4.Satrec import twoline2rv
    def _propagate(rv, t):
        e, p, v = rv.sgp4(t.jd1, t.jd2)
        if e != 0: raise
        return p, v
    sgp4_v2 = True
else:
    from sgp4.io import twoline2rv as _twoline
    from sgp4.earth_gravity import wgs72
    def twoline2rv(lin1, lin2):
        return _twoline(lin1, lin2, wgs72)
    def _propagate(rv, t):
        dt = Time(t, format='datetime').value
        return rv.propagate(dt.year, dt.month, dt.day,
                    dt.hour, dt.minute, dt.second + 1e-6*dt.microsecond)

    sgp4_v2 = False

from sorts import frames, dates
from sorts import using_astropy as uap
from sorts.propagator.pysgp4 import SGP4

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.coordinates.builtin_frames import ITRS


def vdot(va, vb):
    """dot products between equal or shape-compatible arrays of N-vectors"""
    return np.sum(np.asarray(va) * np.asarray(vb), axis=-1)

def vnorm2(vector):
    """L2-norm of array of N-vectors"""
    return np.sqrt(vdot(vector, vector))

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

        # epoch is in UTC or UT1 (not clear which). The difference is < 1 sec always.
        rv_ept = Time(rv.epoch, scale='utc')

        # Extract timetuple, using DK's methods
        # tuple: year, month, day, hr, min, sec, microseconds
        epochdate = dates.npdt_to_date(dates.mjd_to_npdt(dates.jd_to_mjd(rv.jdsatepoch)))
        year, month, day, hour, minute, sec, microseconds = epochdate

        # microseconds are off. Recreate (float) number of seconds from rv.epoch
        seconds = rv.epoch.second + 1e-6 * rv.epoch.microsecond

        # Find first given statevector _after_ TLE epoch:
        rvep_utc = np.datetime64(rv.epoch, 'us')


        # Extract TEME Cartesian state vector at epoch
        if sgp4_v2:
            err, po0, ve0 = rv.sgp4(rv_ept.jd1, rv_ept.jd2)
            assert err == 0
        else:
            po0, ve0 = rv.propagate(year, month, day, hour, minute, seconds)

        # unified
        pos, vel = _propagate(rv, rv_ept)

        assert po0 == pos
        assert ve0 == vel


    def test_one_astropy_ITRF(self):
        """
        Using TLE and restituted orbit files for Sentinel-1A,
        ensure that we can get a reasonably accurate ITRF statevector
        from the underlying SGP4 propagator and the astropy-provided
        matrix for rotating TEME into ITRF
        https://astropy.slack.com/archives/C0LAGR5TP/p1583221032005400?thread_ts=1582732263.004700&cid=C0LAGR5TP
        """

        rv = self.tab['SENTINEL-1A']
        s1_res = self.resorb

        rv_epd = np.datetime64(rv.epoch, 'us')              # Numpy datetime64 value

        # Find first statevector past TLE epoch
        ix = np.where(s1_res.UTC > rv_epd)[0][0]
        svec = s1_res[ix]

        svt = Time(svec.UTC, scale='utc')                # Astropy Time object
        pos, vel = _propagate(rv, svt)

        pmat, pmatp = uap.teme_to_itrs_mat(svt, derivative=True)
        rpos = 1e3 *  pmat @ pos
        rvel = 1e3 * (pmat @ vel + pmatp @ pos)

        # print(f"diff |pRES - pTEME->ITRF| {vnorm2(svec.POS - rpos)} (m)")
        # print(f"diff |vRES - vTEME->ITRF| {vnorm2(svec.VEL - rvel)} (m/s)")

        assert vnorm2(svec.POS - rpos) < 800.0, 'Inaccurate position'
        assert vnorm2(svec.VEL - rvel) < 0.5,   'Inaccurate velocity'

        return True

    def test_array_astropy_ITRF(self):
        """Same as test_one_astropy_ITRF but with an array of states"""

        rv = self.tab['SENTINEL-1A']
        s1_res = self.resorb

        rv_ept = Time(rv.epoch, scale='utc')                # Astropy Time object
        rv_epd = np.datetime64(rv.epoch, 'us')              # Numpy datetime64 value

        # Find first statevector past TLE epoch
        i0 = np.where(s1_res.UTC > rv_epd)[0][0]
        ix = range(i0, i0+200, 10)
        svec = s1_res[ix]
        for ii in ix:
            svec = s1_res[ii]
            svt = Time(svec.UTC, scale='utc')                # Astropy Time object
            pos, vel = _propagate(rv, svt)

            pmat, pmatp = uap.teme_to_itrs_mat(svt, derivative=True)
            rpos = 1e3 *  pmat @ pos
            rvel = 1e3 * (pmat @ vel + pmatp @ pos)

            # print(f"diff |pRES - pTEME->ITRF| {vnorm2(svec.POS - rpos)} (m)")
            # print(f"diff |vRES - vTEME->ITRF| {vnorm2(svec.VEL - rvel)} (m/s)")

            assert vnorm2(svec.POS - rpos) < 800.0, 'Inaccurate position'
            assert vnorm2(svec.VEL - rvel) < 1.0,   'Inaccurate velocity'

        return True

    def test_astropythonesque(self):
        """
        As above, but using the astropy frame transform framework

        Compare hand-unrolled transformations with those from astropy
        """
        rv = self.tab['SENTINEL-1A']
        s1_res = self.resorb

        # rv_ept = Time(rv.epoch, scale='utc')                # Astropy Time object
        rv_epd = np.datetime64(rv.epoch, 'us')              # Numpy datetime64 value

        # Find first statevector past TLE epoch
        ix = np.where(s1_res.UTC > rv_epd)[0][0]
        svec = s1_res[ix]

        svt = Time(svec.UTC, scale='utc')
        pos, vel = _propagate(rv, svt)

        teme_p = CartesianRepresentation(pos * u.km)
        teme_v = CartesianDifferential(vel * u.km/u.s)
        teme = uap.TEME(teme_p.with_differentials(teme_v), obstime=svt)

        itrs = teme.transform_to(ITRS(obstime=teme.obstime))

        pmat, pmatp = uap.teme_to_itrs_mat(svt, derivative=True)
        rpos = 1e3 *  pmat @ pos
        rvel = 1e3 * (pmat @ vel + pmatp @ pos)

        np.testing.assert_allclose(rpos*u.m, itrs.data.xyz)
        np.testing.assert_allclose(rvel*u.m/u.s, itrs.velocity.d_xyz)

        # How about inverse transform?
        teme2 = itrs.transform_to(uap.TEME(obstime=itrs.obstime))

        np.testing.assert_allclose(teme.data.xyz, teme2.data.xyz)
        np.testing.assert_allclose(teme.velocity.d_xyz, teme2.velocity.d_xyz)



    def test_tle_init(self):
        """
        See whether SGP4 class in sorts/propagator/pysgp4.py
        initialised from TLE values is consistent with the above.

        The propagator uses frames.TEME_to_ECEF to rotate the TEME results into
        ITRF.  Hence, if this test succeeds, then TEME_to_ECEF is reasonably accurate
        when initialized from an ECI (TEME) cartesian statevector.
        """

        # Extract one statevector for initialization
        rv = self.tab['SENTINEL-1A']
        s1_res = self.resorb

        # rv_ept = Time(rv.epoch, scale='utc')                # Astropy Time object
        rv_epd = np.datetime64(rv.epoch, 'us')              # Numpy datetime64 value

        # Find first statevector past TLE epoch
        ix = np.where(s1_res.UTC > rv_epd)[0][0]
        svec = s1_res[ix]

        # def propagate(self, t, state0, mjd0, **kwargs):

        # Extract TEME statevector for initialization of SORTS propagation object
        yr, mth, day, hr, mn, sec, usec = dates.npdt_to_date(svec.UTC)
        pos, vel = rv.propagate(yr, mth, day, hr, mn, second=sec+1e-6*usec)
        state_eci = np.array(pos + vel) * 1e3
        epoch_mjd = dates.npdt_to_mjd(svec.UTC)

        # SORTS-style SGP4 propagator
        prp = SGP4()

        # Should be Cartesian ITRF coordinates in SI units
        posvel = prp.propagate(0., state_eci, epoch_mjd)[:,0]
        ppos, pvel = posvel[:3], posvel[3:]

        assert vnorm2(svec.POS - ppos) < 800.0, 'Inaccurate position'
        assert vnorm2(svec.VEL - pvel) < 1.0,   'Inaccurate velocity'

        for ii in range(200):
            # dt between statevectors is 10.0 seconds
            svec = s1_res[ix+ii]
            posvel = prp.propagate(10.0*ii, state_eci, epoch_mjd)[:,0]
            ppos, pvel = posvel[:3], posvel[3:]

            assert vnorm2(svec.POS - ppos) < 800.0, 'Inaccurate position'
            assert vnorm2(svec.VEL - pvel) < 1.0,   'Inaccurate velocity'


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

        # rv_ept = Time(rv.epoch, scale='utc')                # Astropy Time object
        rv_epd = np.datetime64(rv.epoch, 'us')              # Numpy datetime64 value

        # Find first statevector past TLE epoch
        ix = np.where(s1_res.UTC > rv_epd)[0][0]
        svec = s1_res[ix]

        # def propagate(self, t, state0, mjd0, **kwargs):
        pmat, pmatp = uap.teme_to_itrs_mat(Time(svec.UTC, scale='utc'), derivative=True)

        # rotation matrices ): transpose is inverse
        # Get TEME statevectors from POE (ITRS) using inverse relations
        rpos = pmat.T @ svec.POS
        rvel = pmat.T @ (svec.VEL - pmatp @ rpos)

        state_eci = np.r_[rpos, rvel]
        epoch_mjd = dates.npdt_to_mjd(svec.UTC)

        # SORTS-style SGP4 propagator
        prp = SGP4()

        # Should be Cartesian ITRF coordinates in SI units
        posvel = prp.propagate(0., state_eci, epoch_mjd)[:,0]
        ppos, pvel = posvel[:3], posvel[3:]

        assert vnorm2(svec.POS - ppos) < 800.0, 'Inaccurate position'
        assert vnorm2(svec.VEL - pvel) < 1.0,   'Inaccurate velocity'

        for ii in range(200):
            # dt between statevectors is 10.0 seconds
            svec = s1_res[ix+ii]
            posvel = prp.propagate(10.0*ii, state_eci, epoch_mjd)[:,0]
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

        p = CartesianRepresentation(svec.POS * u.m)
        v = CartesianDifferential(svec.VEL * u.m/u.s)
        svt = Time(svec.UTC, scale='utc')
        itrs = ITRS(p.with_differentials(v), obstime=svt)

        teme = itrs.transform_to(uap.TEME(obstime=itrs.obstime))
        state_eci = np.r_[teme.data.xyz.to(u.m).value,
                          teme.velocity.d_xyz.to(u.m/u.s).value]

        # Find mean elements from osculating elements
        state = frames.TEME_to_TLE(state_eci, svt.mjd)
        # SORTS-style SGP4 propagator
        prp = SGP4()
        prp.settings['out_frame'] = 'ITRF'

        ii = np.arange(nprop)

        pv_itrf = prp.propagate(10*ii, state_eci, svt.mjd)
        pos = pv_itrf[:3].T
        vel = pv_itrf[3:].T

        rpos = s1_res[ix+ii].POS
        rvel = s1_res[ix+ii].VEL

        return pos, vel, rpos, rvel

    def test_propagate_from_RES(self, nprop=400):

        pos, vel, rpos, rvel = propagate_from_RES(self, nprop)

        np.testing.assert_array_less(vnorm2(pos[:150]-rpos[:150]), 100), \
            'Position errors too large'

        np.testing.assert_array_less(vnorm2(vel[:150]-rvel[:150]), 0.24), \
            'Position errors too large'

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






    def skip_test_TEME_to_ITRF(self):

        # First, try without polar motion

        # Extract one statevector for initialization
        rv = self.tab['SENTINEL-1A']
        s1_res = self.resorb

        # rv_ept = Time(rv.epoch, scale='utc')                # Astropy Time object
        rv_epd = np.datetime64(rv.epoch, 'us')              # Numpy datetime64 value

        # Find first statevector past TLE epoch
        ix = np.where(s1_res.UTC > rv_epd)[0][0]
        svec = s1_res[ix]

        # def propagate(self, t, state0, mjd0, **kwargs):

        # Extract TEME statevector for initialization of SORTS propagation object
        yr, mth, day, hr, mn, sec, usec = dates.npdt_to_date(svec.UTC)
        pos, vel = rv.propagate(yr, mth, day, hr, mn, second=sec+1e-6*usec)
        state_eci = np.array(pos + vel) * 1e3
        epoch_mjd = dates.npdt_to_mjd(svec.UTC)

        # SORTS-style SGP4 propagator
        prp = SGP4()

        # Should be Cartesian TEME coordinates in SI units
        prp.settings['out_frame'] = 'TEME'
        pv_teme = prp.propagate(0., state_eci, epoch_mjd)[:,0]
        prp.settings['out_frame'] = 'ITRF'
        pv_itrf = prp.propagate(0., state_eci, epoch_mjd)[:,0]

        # rotate          # def TEME_to_ITRF(TEME, jd_ut1, xp, yp)
        # UT1 is close enough to UTC so as to make no difference here
        pv_rot = frames.TEME_to_ITRF(pv_teme, dates.mjd_to_jd(epoch_mjd), 0, 0)

        p_itrf, v_itrf = pv_itrf[:3], pv_itrf[3:]
        p_rot, v_rot = pv_rot[:3], pv_rot[3:]

        assert vnorm2(p_rot - p_itrf) < 800.0, 'Inaccurate position after rotation'
        assert vnorm2(v_rot - v_itrf) < 1.0,   'Inaccurate velocity after rotation'





if __name__ == '__main__':

    import sys

    print(sys.argv)

    if len(sys.argv) > 1:
        unittest.main()

    else:

        rv = load_tle_table('tests/eos_tle.txt')['SENTINEL-1A']
        s1_res = read_statevectors('tests/S1A_OPER_AUX_RESORB_OPOD_20200721T073339_V20200721T023852_20200721T055622.EOF')

        rv_epd = np.datetime64(rv.epoch, 'us')              # Numpy datetime64 value

        # Find first statevector past TLE epoch
        ix = np.where(s1_res.UTC > rv_epd)[0][0]
        svec = s1_res[ix]

        svt = Time(svec.UTC, scale='utc')

