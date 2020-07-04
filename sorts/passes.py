#!/usr/bin/env python

'''

'''

import numpy as np
import pyorb
import pyant

#Local import
from . import frames

class Pass:

    def __init__(self, start, end, data = dict()):
        self.start = start
        self.end = end
        self.data = data


def equidistant_sampling(orbit, start_t, end_t, max_dpos=1e3):
    '''Find the temporal sampling of an orbit which is sufficient to achieve a maximum spatial separation.
    Assume elliptic orbit and uses Keplerian propagation to find sampling, does not take perturbation patterns into account.
    
    :param pyorb.Orbit orbit: Orbit to find temporal sampling of.
    :param float start_t: Start time in seconds
    :param float end_t: End time in seconds
    :param float max_dpos: Maximum separation between evaluation points in meters.
    :return: Vector of sample times in seconds.
    :rtype: numpy.ndarray
    '''
    if len(orbit) > 1:
        raise ValueError(f'Cannot use vectorized orbits: len(orbit) = {len(orbit)}')

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




def find_passes(t, states, radar, logger = None, profiler = None):
    '''Find a pass inside the FOV of a radar given a series of times for a space object.
    
    :param numpy.ndarray states: ECEF states of the object to find passes for.
    :param numpy.ndarray t: Vector of times in seconds to use as a base to find passes.
    :param sorts.Radar radar: Radar system that defines the FOV.
    :return: Dictionary of lists of passes, structured as the radar system.
    :rtype: dict

    '''
    
    zenith = np.array([0,0,1], dtype=np.float64)

    output = dict(tx=[list() for x in radar.tx], rx=[list() for x in radar.rx])

    for ti,station in enumerate(radar.tx + radar.rx):
        enu = station.enu(states[:3,:])
        zenith_ang = pyant.coordinates.vector_angle(zenith, enu, radians=False)
        check = zenith_ang < 90.0-station.min_elevation
        #save theese??
        inds = np.where(check)[0]

        dind = np.diff(inds)
        splits = np.where(dind > 1)[0]

        splits = np.insert(splits, 0, inds[0])
        splits = np.insert(splits, len(splits), inds[-1])
        for si in range(len(splits)-1):
            ps_inds = inds[splits[si]:splits[si+1]]
            data = dict(
                zenith = zenith_ang[ps_inds],
                t = t[ps_inds],
                range = np.linalg.norm(enu[:,ps_inds], axis=0),
                inds = ps_inds,
            )

            ps = Pass(start=data['t'].min(), end=data['t'].max(), data=data)
            if ti < len(radar.tx):            
                output['tx'][ti].append(ps)
            else:
                output['rx'][ti-len(radar.tx)].append(ps)

    return output

