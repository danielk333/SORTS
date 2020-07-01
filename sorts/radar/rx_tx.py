#!/usr/bin/env python

'''Defines an antenna's or entire radar system's radiation pattern, also defines physical antennas for RX and TX.

(c) 2016-2019 Juha Vierinen, Daniel Kastinen
'''

#Python standard import
import copy


#Third party import
import numpy as np


#Local import
from .. import frames


class Station(object):
    '''A radar station.

        :param float lat: Geographical latitude of radar station in decimal degrees  (North+).
        :param float lon: Geographical longitude of radar station in decimal degrees (East+).
        :param float alt: Geographical altitude above geoid surface of radar station in meter.
        :param float min_elevation: Elevation threshold for the radar station, i.e. it cannot detect or point below this elevation.
        :param pyant.Beam beam: Radiation pattern for radar station.
        :param sorts.radar.Scan scan: Scanning pattern for radar station.

        :ivar float lat: Geographical latitude of radar station in decimal degrees  (North+).
        :ivar float lon: Geographical longitude of radar station in decimal degrees (East+).
        :ivar float alt: Geographical altitude above geoid surface of radar station in meter.
        :ivar float min_elevation: Elevation threshold for the radar station, i.e. it cannot detect or point below this elevation.
        :ivar numpy.array ecef: The ECEF coordinates of the radar station calculated using :func:`frames.geodetic_to_ecef`.
        :ivar pyant.Beam beam: Radiation pattern for radar station.
        :ivar sorts.radar.Scan scan: Scanning pattern for radar station.

    '''
    def __init__(self, lat, lon, alt, min_elevation, beam, scan = None):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.min_elevation = min_elevation
        self.ecef = frames.geodetic_to_ecef(lat, lon, alt, radians = False)
        self.beam = beam
        self.scan = scan


    @property
    def frequency(self):
        return self.beam.frequency

    @property
    def wavelength(self):
        return self.beam.wavelength


    def point(self, k):
        '''Point Station beam in local NEU coordinates.
        '''
        self.beam.point(k)


    def point_ecef(self, point):
        '''Point Station beam in location of ECEF coordinate. Returns local pointing direction.
        '''
        k = frames.ecef_to_enu(
            self.lat,
            self.lon,
            self.alt,
            point,
            radians=False,
        )
        self.beam.point(k)
        return k/n.linalg.norm(k, axis=0)


    def get_pointing_ecef(self, t):
        '''Return the instantaneous pointing of the Station based on the currently running scan.
        
           :param float/numpy.ndarray t: Time past reference epoch in seconds.
           :return: Current pointing direction in ECEF. Is a :code:`(3,)` numpy ndarray if :code:`len(t) == 1`, else return a :code:`(3, len(t))`.
        '''
        return self.scan.ecef_pointing(t, self)


    def get_pointing(self, t):
        '''Return the instantaneous pointing of the Station based on the currently running scan.
        
           :param float/numpy.ndarray t: Time past reference epoch in seconds.
           :return: Current pointing direction in local coordinates. Is a :code:`(3,)` numpy ndarray if :code:`len(t) == 1`, else return a :code:`(3, len(t))`.
        '''
        return self.scan.pointing(t)

    def scan_pointing(self, t):
        '''Set the current pointing to the scan pointing. If `t` is iterable, sets pointing and yields self.'''
        iter_ = False
        if isinstance(t, np.ndarray) or isinstance(t, list) or isinstance(t, set):
            if len(t) > 1:
                iter_ = True

        if iter_:
            k = self.get_pointing(t)
            for ind in range(len(t)):
                self.point(k[:,ind])
                yield self
        else:
            k = self.get_pointing(t)
            self.point(k)


    def __str__(self):
        pass


class RX(Station):
    '''A radar receiving system.

        :param float rx_noise: Receiver noise in Kelvin, i.e. system temperature.

        :ivar float rx_noise: Receiver noise in Kelvin, i.e. system temperature.
    '''
    def __init__(self, lat, lon, alt, min_elevation, beam, rx_noise, scan = None):
        super().__init__(lat, lon, alt, min_elevation, beam, scan = scan)
        self.rx_noise = rx_noise



class TX(Station):
    '''A transmitting radar station
        
        :param float tx_bandwidth: Transmissions bandwidth.
        :param float duty_cycle: Maximum duty cycle, i.e. fraction of time transmission can occur at maximum power.
        :param float tx_power: Transmissions power in watts.
        :param float pulse_length: Length of transmission pulse.
        :param float ipp: Time between consecutive pulses.
        :param int n_ipp: Number of pulses to coherently integrate.

        :ivar float tx_bandwidth: Transmissions bandwidth.
        :ivar float duty_cycle: Maximum duty cycle, i.e. fraction of time transmission can occur at maximum power.
        :ivar float tx_power: Transmissions power in watts.
        :ivar float pulse_length: Length of transmission pulse.
        :ivar float ipp: Time between consecutive pulses.
        :ivar int n_ipp: Number of pulses to coherently integrate.
        :ivar float coh_int_bandwidth: Effective bandwidth of receiver noise after coherent integration.
    '''
    def __init__(self, lat, lon, alt, min_elevation, beam, tx_power, tx_bandwidth, duty_cycle, scan = None, pulse_length=1e-3, ipp=10e-3, n_ipp=20):
        super().__init__(lat, lon, alt, min_elevation, beam, scan = scan)

        self.tx_bandwidth = tx_bandwidth
        self.duty_cycle = duty_cycle
        self.tx_power = tx_power
        self.pulse_length = pulse_length
        self.ipp = ipp
        self.n_ipp = n_ipp
        self.coh_int_bandwidth = 1.0/(pulse_length*n_ipp)

        self.scan.check_dwell_tx(self)