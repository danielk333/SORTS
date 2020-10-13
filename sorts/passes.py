#!/usr/bin/env python

'''

'''

import numpy as np
import pyorb
import pyant

#Local import
from . import frames
from .signals import hard_target_snr

class Pass:
    '''Saves the local coordinate data for a single pass. Optionally also indicates the location of that pass in a bigger dataset.
    '''

    def __init__(self, t, enu, inds=None, cache=True, station_id=None):
        self.inds = inds
        self.t = t
        self.enu = enu
        self.cache = cache

        self.station_id = station_id

        self.snr = None

        self._start = None
        self._end = None
        self._r = None
        self._v = None
        self._zang = None


    def calculate_max_snr(self, tx, rx, diameter):
        '''
        '''
        assert self.stations == 2, 'Can only calculate SNR for TX-RX pairs'

        ranges = self.range()
        enus = self.enu

        self.snr = np.empty((len(self.t),), dtype=np.float64)
        for ti in range(len(self.t)):

            tx.beam.point(enus[0][:3,ti])
            rx.beam.point(enus[1][:3,ti])

            self.snr[ti] = hard_target_snr(
                tx.beam.gain(enus[0][:3,ti]),
                rx.beam.gain(enus[1][:3,ti]),
                rx.wavelength,
                tx.power,
                ranges[0][ti],
                ranges[1][ti],
                diameter=diameter,
                bandwidth=tx.coh_int_bandwidth,
                rx_noise_temp=rx.noise,
            )



    @property
    def stations(self):
        if self.station_id is not None:
            if isinstance(self.station_id, list):
                return len(self.station_id)
            else:
                return 1
        else:
            return 1


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

    @staticmethod
    def calculate_range(enu):
        return np.linalg.norm(enu[:3,:], axis=0)

    @staticmethod
    def calculate_range_rate(enu):
        return np.sum(enu[3:,:]*(enu[:3,:]/np.linalg.norm(enu[:3,:], axis=0)), axis=0)

    @staticmethod
    def calculate_zenith_angle(enu, radians=False):
        return pyant.coordinates.vector_angle(np.array([0,0,1], dtype=np.float64), enu[:3,:], radians=radians)

    def get_range(self):
        if self.stations > 1:
            return [Pass.calculate_range(enu) for enu in self.enu]
        else:
            return Pass.calculate_range(self.enu)

    def range(self):
        if self.cache:
            if self._r is None:
                self._r = self.get_range()
            return self._r
        else:
            return self.get_range()

    def get_range_rate(self):
        if self.stations > 1:
            return [Pass.calculate_range_rate(enu) for enu in self.enu]
        else:
            return Pass.calculate_range_rate(self.enu)

    def range_rate(self):
        if self.cache:
            if self._v is None:
                self._v = self.get_range_rate()
            return self._v
        else:
            return self.get_range_rate()


    def get_zenith_angle(self, radians=False):
        if self.stations > 1:
            return [
                Pass.calculate_zenith_angle(enu, radians=radians)
                for enu in self.enu
            ]
        else:
            return Pass.calculate_zenith_angle(self.enu, radians=radians)

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


def find_passes(t, states, station, cache_data=True):
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

        if cache_data:
            ps = Pass(
                t=t[ps_inds], 
                enu=enu[:,ps_inds], 
                inds=ps_inds, 
                cache=True,
            )
            ps._zang = zenith_ang[ps_inds]
        else:
            ps = Pass(
                t=None, 
                enu=None, 
                inds=ps_inds, 
                cache=True,
            )
            ps._start = t[ps_inds].min()
            ps._end = t[ps_inds].max()

        passes.append(ps)

    return passes


def find_simultaneous_passes(t, states, stations, cache_data=True):
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

        enu_st = station.enu(states)
        enu.append(enu_st)

        zenith_st_ang = pyant.coordinates.vector_angle(zenith, enu_st[:3,:], radians=False)
        zenith_ang.append(zenith_st_ang)

        check_st = zenith_st_ang < 90.0-station.min_elevation
        check = np.logical_and(check, check_st)

    inds = np.where(check)[0]

    if len(inds) == 0:
        return passes

    dind = np.diff(inds)
    splits = np.where(dind > 1)[0]

    splits = np.insert(splits, 0, inds[0])
    splits = np.insert(splits, len(splits), inds[-1])
    for si in range(len(splits)-1):
        ps_inds = inds[(splits[si]+1):splits[si+1]]
        if len(ps_inds) == 0:
            continue
        if cache_data:
            ps = Pass(
                t=t[ps_inds], 
                enu=[xv[:,ps_inds] for xv in enu], 
                inds=ps_inds, 
                cache=True,
                station_id=[None, None],
            )
            ps._zang = [z[ps_inds] for z in zenith_ang]
        else:
            ps = Pass(
                t=None, 
                enu=None, 
                inds=ps_inds, 
                cache=True,
                station_id=[None, None],
            )
            ps._start = t[ps_inds].min()
            ps._end = t[ps_inds].max()

        passes.append(ps)

    return passes


def group_passes(passes):
    '''Takes a list of passes structured as [tx][rx][pass] and find all simultaneous passes and groups them according to [tx], resulting in a [tx][pass][rx] structure.
    '''

    def overlap(ps1, ps2):
        return ps1.start() <= ps2.end() and ps2.start() <= ps1.end()

    grouped_passes = []
    for tx_passes in passes:
        grouped_passes.append([])


        #first flatten
        flat_passes = [x for rx_passes in tx_passes for x in rx_passes]

        if len(flat_passes) > 0:
            grouped_passes[-1].append([flat_passes[0]])
        else:
            continue

        for x in range(1,len(flat_passes)):
            for y in range(len(grouped_passes[-1])):
                member = False
                for gps in grouped_passes[-1][y]:
                    if overlap(gps, flat_passes[x]):
                        member = True
                        break

                if member:
                    member_id = y
                    break

            if member:
                grouped_passes[-1][member_id].append(flat_passes[x])
            else:
                grouped_passes[-1].append([flat_passes[x]])

    return grouped_passes

