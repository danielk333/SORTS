#!/usr/bin/env python

'''

'''

import numpy as np
import pyorb
import pyant

#Local import
from . import frames

class Pass:
    '''Saves the local coordinate data for a single pass. Optionally also indicates the location of that pass in a bigger dataset.
    '''

    def __init__(self, t, enu, inds=None, cache=True, stations=1):
        self.inds = inds
        self.t = t
        self.enu = enu
        self.cache = cache
        self.stations = stations

        self.snr = None

        self._start = None
        self._end = None
        self._r = None
        self._zang = None

        self.__zenith = np.array([0,0,1], dtype=np.float64)

    def start(self):
        if self.cache:
            if self._start is None:
                self._start = self.t.min()
            return self._start
        else:
            return self.t.min()

    def end(self):
        if self.cache:
            if self._end is None:
                self._end = self.t.max()
            return self._end
        else:
            return self.t.max()


    def get_range(self):
        if self.stations > 1:
            return [np.linalg.norm(enu, axis=0) for enu in self.enu]
        else:
            return np.linalg.norm(self.enu, axis=0)

    def range(self):
        if self.cache:
            if self._r is None:
                self._r = self.get_range()
            return self._r
        else:
            return self.get_range()


    def get_zenith_angle(self, radians=False):
        if self.stations > 1:
            return [
                pyant.coordinates.vector_angle(self.__zenith, enu, radians=radians)
                for enu in self.enu
            ]
        else:
            return pyant.coordinates.vector_angle(self.__zenith, self.enu, radians=radians)

    def zenith_angle(self, radians=False):
        if self.cache:
            if self._zang is None:
                self._zang = self.get_zenith_angle(radians=radians)
            return self._zang
        else:
            return self.get_zenith_angle(radians=radians)




def equidistant_sampling(orbit, start_t, end_t, max_dpos=1e3, eccentricity_tol=0.3):
    '''Find the temporal sampling of an orbit which is sufficient to achieve a maximum spatial separation.
    Assume elliptic orbit and uses Keplerian propagation to find sampling, does not take perturbation patterns into account. 
    If eccentricity is small, uses periapsis speed and uniform sampling in time.
    
    :param pyorb.Orbit orbit: Orbit to find temporal sampling of.
    :param float start_t: Start time in seconds
    :param float end_t: End time in seconds
    :param float max_dpos: Maximum separation between evaluation points in meters.
    :param float eccentricity_tol: Minimum eccentricity below which the orbit is approximated as a circle and temporal samples are uniform in time.
    :return: Vector of sample times in seconds.
    :rtype: numpy.ndarray
    '''
    if len(orbit) > 1:
        raise ValueError(f'Cannot use vectorized orbits: len(orbit) = {len(orbit)}')

    if orbit.e <= eccentricity_tol:
        r = pyorb.elliptic_radius(0.0, orbit.a, orbit.e, degrees=False)
        v = pyorb.orbital_speed(r, orbit.a, orbit.G*(orbit.M0 + orbit.m))[0]
        return np.arange(start_t, end_t, max_dpos/v)

    tmp_orb = orbit.copy()
    tmp_orb.auto_update = False

    tmp_orb.propagate(start_t)
    period = tmp_orb.period

    t_curr = start_t
    t = [t_curr]
    t_repeat = None
    while t_curr < end_t:
        if t_curr - start_t > period:
            if t_repeat is None:
                t_repeat = len(t)
            dt = t[-t_repeat+1] - t[-t_repeat]
            t_curr += dt
        else:
            v = tmp_orb.speed[0]
            dt = max_dpos/v
            t_curr += dt
            tmp_orb.propagate(dt)

        t.append(t_curr)
    return np.array(t, dtype=np.float64)


def find_passes(t, states, station):
    '''Find passes inside the FOV of a radar station given a series of times for a space object.
    
    :param numpy.ndarray t: Vector of times in seconds to use as a base to find passes.
    :param numpy.ndarray states: ECEF states of the object to find passes for.
    :param sorts.Station station: Radar station that defines the FOV.
    :return: list of passes
    :rtype: sorts.Pass

    '''
    passes = []
    zenith = np.array([0,0,1], dtype=np.float64)

    enu = station.enu(states[:3,:])
    zenith_ang = pyant.coordinates.vector_angle(zenith, enu, radians=False)
    check = zenith_ang < 90.0-station.min_elevation
    inds = np.where(check)[0]

    dind = np.diff(inds)
    splits = np.where(dind > 1)[0]

    splits = np.insert(splits, 0, inds[0])
    splits = np.insert(splits, len(splits), inds[-1])
    for si in range(len(splits)-1):
        ps_inds = inds[(splits[si]+1):splits[si+1]]
        ps = Pass(
            t=t[ps_inds], 
            enu=enu[:,ps_inds], 
            inds=ps_inds, 
            cache=True,
            stations=1,
        )
        ps._zang = zenith_ang[ps_inds]
        passes.append(ps)

    return passes


def find_simultaneous_passes(t, states, stations):
    '''Finds all passes that are simultaneously inside a multiple stations FOV's.
    
    :param numpy.ndarray t: Vector of times in seconds to use as a base to find passes.
    :param numpy.ndarray states: ECEF states of the object to find passes for.
    :param list of sorts.Station stations: Radar stations that defines the FOV's.
    :return: list of passes
    :rtype: list of sorts.Pass

    '''
    passes = []
    zenith = np.array([0,0,1], dtype=np.float64)

    enu = []
    zenith_ang = []
    check = np.full((len(t),), True, dtype=np.bool)
    for station in stations:

        enu_st = station.enu(states[:3,:])
        enu.append(enu_st)

        zenith_st_ang = pyant.coordinates.vector_angle(zenith, enu_st, radians=False)
        zenith_ang.append(zenith_st_ang)

        check_st = zenith_st_ang < 90.0-station.min_elevation
        check = np.logical_and(check, check_st)

    inds = np.where(check)[0]

    dind = np.diff(inds)
    splits = np.where(dind > 1)[0]

    splits = np.insert(splits, 0, inds[0])
    splits = np.insert(splits, len(splits), inds[-1])
    for si in range(len(splits)-1):
        ps_inds = inds[(splits[si]+1):splits[si+1]]
        ps = Pass(
            t=t[ps_inds], 
            enu=[r[:3,ps_inds] for r in enu], 
            inds=ps_inds, 
            cache=True,
            stations=2,
        )
        ps._zang = [z[ps_inds] for z in zenith_ang]
        passes.append(ps)

    return passes