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
        :param float min_elevation: Elevation threshold for the radar station in degrees, i.e. it cannot detect or point below this elevation.
        :param pyant.Beam beam: Radiation pattern for radar station.


        :ivar float lat: Geographical latitude of radar station in decimal degrees  (North+).
        :ivar float lon: Geographical longitude of radar station in decimal degrees (East+).
        :ivar float alt: Geographical altitude above geoid surface of radar station in meter.
        :ivar float min_elevation: Elevation threshold for the radar station in degrees, i.e. it cannot detect or point below this elevation.
        :ivar numpy.array ecef: The ECEF coordinates of the radar station calculated using :func:`frames.geodetic_to_ITRS`.
        :ivar pyant.Beam beam: Radiation pattern for radar station.
        :ivar bool enabled: Indicates if this station is turned on or off.

    '''
    def __init__(self, lat, lon, alt, min_elevation, beam):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.min_elevation = min_elevation
        self.ecef = frames.geodetic_to_ITRS(lat, lon, alt, radians = False)
        self.beam = beam
        self.enabled = True
        self.pointing_range = None

    def copy(self):
        st = Station(
            lat = self.lat,
            lon = self.lon,
            alt = self.alt,
            min_elevation = self.min_elevation,
            beam = self.beam.copy(),
        )
        st.enabled = self.enabled
        return st


    @property
    def frequency(self):
        return self.beam.frequency

    @property
    def wavelength(self):
        return self.beam.wavelength


    def enu(self, ecefs):
        '''Converts a set of ECEF states to local ENU coordinates.

        '''
        rel_ = ecefs.copy()
        rel_[:3,:] = rel_[:3,:] - self.ecef[:,None]
        rel_[:3,:] = frames.ecef_to_enu(
            self.lat,
            self.lon,
            self.alt,
            rel_[:3,:],
            radians=False,
        )
        if ecefs.shape[0] > 3:
            rel_[3:,:] = frames.ecef_to_enu(
                self.lat,
                self.lon,
                self.alt,
                rel_[3:,:],
                radians=False,
            )
        return rel_


    def point(self, k):
        '''Point Station beam in local ENU coordinates.
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
        k_norm = np.linalg.norm(k, axis=0)
        
        self.pointing_range = k_norm

        k = k/k_norm
        self.beam.point(k)
        return k

    @property
    def pointing(self):
        '''Station beam pointing in local ENU coordinates.
        '''
        return self.beam.pointing.copy()


    @property
    def pointing_ecef(self):
        '''Station beam pointing in local ENU coordinates.
        '''
        return frames.enu_to_ecef(
            self.lat, 
            self.lon, 
            self.alt, 
            self.beam.pointing, 
            radians=False,
        )


    def __str__(self):
        pass


class RX(Station):
    '''A radar receiving system.

        :param float noise: Receiver noise in Kelvin, i.e. system temperature.

        :ivar float noise: Receiver noise in Kelvin, i.e. system temperature.
    '''
    def __init__(self, lat, lon, alt, min_elevation, beam, noise):
        super().__init__(lat, lon, alt, min_elevation, beam)
        self.noise = noise

    def copy(self):
        st = RX(
            lat = self.lat,
            lon = self.lon,
            alt = self.alt,
            min_elevation = self.min_elevation,
            beam = self.beam.copy(),
            noise = self.noise,
        )
        st.enabled = self.enabled
        return st



class TX(Station):
    '''A transmitting radar station
        
        :param float bandwidth: Transmissions bandwidth.
        :param float duty_cycle: Maximum duty cycle, i.e. fraction of time transmission can occur at maximum power.
        :param float power: Transmissions power in watts.
        :param float pulse_length: Length of transmission pulse.
        :param float ipp: Time between consecutive pulses.
        :param int n_ipp: Number of pulses to coherently integrate.

        :ivar float bandwidth: Transmissions bandwidth.
        :ivar float duty_cycle: Maximum duty cycle, i.e. fraction of time transmission can occur at maximum power.
        :ivar float power: Transmissions power in watts.
        :ivar float pulse_length: Length of transmission pulse.
        :ivar float ipp: Time between consecutive pulses.
        :ivar int n_ipp: Number of pulses to coherently integrate.
        :ivar float coh_int_bandwidth: Effective bandwidth of receiver noise after coherent integration.
    '''
    def __init__(self, lat, lon, alt, min_elevation, beam, power, bandwidth, duty_cycle, pulse_length=1e-3, ipp=10e-3, n_ipp=20):
        super().__init__(lat, lon, alt, min_elevation, beam)

        self.bandwidth = bandwidth
        self.duty_cycle = duty_cycle
        self.power = power
        self.pulse_length = pulse_length
        self.ipp = ipp
        self.n_ipp = n_ipp
        self.coh_int_bandwidth = 1.0/(pulse_length*n_ipp)


    def copy(self):
        st = TX(
            lat = self.lat,
            lon = self.lon,
            alt = self.alt,
            min_elevation = self.min_elevation,
            beam = self.beam.copy(),
            power = self.power,
            bandwidth = self.bandwidth,
            duty_cycle = self.duty_cycle,
            pulse_length = self.pulse_length,
            ipp = self.ipp,
            n_ipp = self.n_ipp,
        )
        st.enabled = self.enabled
        return st