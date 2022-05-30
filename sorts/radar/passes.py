#!/usr/bin/env python

'''Encapsulates a fundamental component of tracking space objects: a pass over a geographic location. 
Also provides convenience functions for finding passes given states and stations and sorting structures of passes in particular ways.

'''
import datetime

import numpy as np
import pyorb
import pyant

#Local import
from ..transformations import frames
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


    def __str__(self):
        str_ = 'Pass '
        if self.station_id is not None:
            str_ += f'Station {self.station_id} | '
        str_ += f'Rise {str(datetime.timedelta(seconds=self.start()))} ({(self.end() - self.start())/60.0:.1f} min) {str(datetime.timedelta(seconds=self.end()))} Fall'
        
        return str_


    def __repr__(self):
        return str(self)


    def calculate_snr(self, tx, rx, diameter):
        '''Uses the :code:`signals.hard_target_snr` function to calculate the optimal SNR curve of a target during the pass **if the TX and RX stations are pointing at the object**.
        The SNR's are returned from the function but also stored in the property :code:`self.snr`. 

        :param TX tx: The TX station that observed the pass.
        :param RX rx: The RX station that observed the pass.
        :param float diameter: The diameter of the object.
        :return: (n,) Vector with the SNR's during the pass.
        :rtype: numpy.ndarray

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
        return self.snr



    @property
    def stations(self):
        '''The number of stations that can observe the pass.
        '''
        if self.station_id is not None:
            if isinstance(self.station_id, list):
                return len(self.station_id)
            else:
                return 1
        else:
            return 1


    def start(self):
        '''The Start time of the pass (uses cached value after first call if :code:`self.cache=True`).
        '''
        if self.cache:
            if self._start is None:
                self._start = self.t.min()
            return self._start
        else:
            return self.t.min()

    def end(self):
        '''The ending time of the pass (uses cached value after first call if :code:`self.cache=True`).
        '''
        if self.cache:
            if self._end is None:
                self._end = self.t.max()
            return self._end
        else:
            return self.t.max()

    @staticmethod
    def calculate_range(enu):
        '''Norm of the ENU coordinates.
        '''
        return np.linalg.norm(enu[:3,:], axis=0)

    @staticmethod
    def calculate_range_rate(enu):
        '''Projected ENU velocity along the ENU range.
        '''
        return np.sum(enu[3:,:]*(enu[:3,:]/np.linalg.norm(enu[:3,:], axis=0)), axis=0)

    @staticmethod
    def calculate_zenith_angle(enu, radians=False):
        '''Zenith angle of the ENU coordinates.
        '''
        return pyant.coordinates.vector_angle(np.array([0,0,1], dtype=np.float64), enu[:3,:], radians=radians)

    def get_range(self):
        '''Get ranges from all stations involved in the pass.

        :rtype: list of numpy.ndarray or numpy.ndarray
        :return: If there is one station observing the pass, return a vector of ranges, otherwise return a list of vectors with ranges.
        '''
        if self.stations > 1:
            return [Pass.calculate_range(enu) for enu in self.enu]
        else:
            return Pass.calculate_range(self.enu)

    def range(self):
        '''Get ranges from all stations involved in the pass (uses cached value after first call if :code:`self.cache=True`).
        The cache is stored in :code:`self._r`.

        :rtype: list of numpy.ndarray or numpy.ndarray
        :return: If there is one station observing the pass, return a vector of ranges, otherwise return a list of vectors with ranges.
        '''
        if self.cache:
            if self._r is None:
                self._r = self.get_range()
            return self._r
        else:
            return self.get_range()

    def get_range_rate(self):
        '''Get range rates from all stations involved in the pass.

        :rtype: list of numpy.ndarray or numpy.ndarray
        :return: If there is one station observing the pass, return a vector of range rates, otherwise return a list of vectors with range rates.
        '''
        if self.stations > 1:
            return [Pass.calculate_range_rate(enu) for enu in self.enu]
        else:
            return Pass.calculate_range_rate(self.enu)

    def range_rate(self):
        '''Get range rates from all stations involved in the pass (uses cached value after first call if :code:`self.cache=True`).
        The cache is stored in :code:`self._v`.

        :rtype: list of numpy.ndarray or numpy.ndarray
        :return: If there is one station observing the pass, return a vector of range rates, otherwise return a list of vectors with range rates.
        '''
        if self.cache:
            if self._v is None:
                self._v = self.get_range_rate()
            return self._v
        else:
            return self.get_range_rate()


    def get_zenith_angle(self, radians=False):
        '''Get zenith angles for all stations involved in the pass.

        :rtype: list of numpy.ndarray or numpy.ndarray
        :return: If there is one station observing the pass, return a vector of zenith angles, otherwise return a list of vectors with zenith angles.
        '''
        if self.stations > 1:
            return [
                Pass.calculate_zenith_angle(enu, radians=radians)
                for enu in self.enu
            ]
        else:
            return Pass.calculate_zenith_angle(self.enu, radians=radians)

    def zenith_angle(self, radians=False):
        '''Get zenith angles from all stations involved in the pass (uses cached value after first call if :code:`self.cache=True`).
        The cache is stored in :code:`self._v`.

        :rtype: list of numpy.ndarray or numpy.ndarray
        :return: If there is one station observing the pass, return a vector of zenith angles, otherwise return a list of vectors with zenith angles.
        '''
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

    enu = station.enu(states[:3,:])

    check = station.field_of_view(states)
    inds = np.where(check)[0]

    if len(inds) == 0:
        return passes

    dind = np.diff(inds)
    splits = np.where(dind > 1)[0]

    splits = np.insert(splits, 0, -1)
    splits = np.insert(splits, len(splits), len(inds)-1)
    splits += 1
    for si in range(len(splits)-1):
        ps_inds = inds[splits[si]:splits[si+1]]

        if cache_data:
            ps = Pass(
                t=t[ps_inds], 
                enu=enu[:, ps_inds], 
                inds=ps_inds, 
                cache=True,
            )
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


def find_simultaneous_passes(t, states, stations, cache_data=True, fov_kw=None):
    '''Finds all passes that are simultaneously inside a multiple stations FOV's.
    
    :param numpy.ndarray t: Vector of times in seconds to use as a base to find passes.
    :param numpy.ndarray states: ECEF states of the object to find passes for.
    :param list of sorts.Station stations: Radar stations that defines the FOV's.
    :return: list of passes
    :rtype: list of sorts.Pass

    '''
    passes = []
    if fov_kw is None:
        fov_kw = {}

    enu = []
    check = np.full((len(t),), True, dtype=np.bool)
    
    for station in stations:
        enu_st = station.enu(states)
        enu.append(enu_st)

        check_st = station.field_of_view(states, **fov_kw)
        check = np.logical_and(check, check_st)

    inds = np.where(check)[0]

    if len(inds) == 0:
        return passes

    dind = np.diff(inds)
    splits = np.where(dind > 1)[0]

    splits = np.insert(splits, 0, -1)
    splits = np.insert(splits, len(splits), len(inds)-1)
    splits += 1
    
    for si in range(len(splits)-1):
        ps_inds = inds[splits[si]:splits[si+1]]
        if len(ps_inds) == 0:
            continue
        if cache_data:
            ps = Pass(
                t=t[ps_inds], 
                enu=[xv[:, ps_inds] for xv in enu], 
                inds=ps_inds, 
                cache=True,
                station_id=[None, None],
            )
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

