#!/usr/bin/env python

'''Defines an antenna's or entire radar system's radiation pattern, also defines physical antennas for RX and TX.

(c) 2016-2019 Juha Vierinen, Daniel Kastinen
'''
import copy

import numpy as n
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.constants as c
import scipy.special as s

# SORTS imports


def inst_gain2full_gain(gain,groups,N_IPP,IPP_scale=1.0,units = 'dB'):
    '''Using pulse encoding schema, subgroup setup and coherrent integration setup; convert from instantanius gain to coherrently integrated gain.
    
    :param float gain: Instantanius gain, linear units or in dB.
    :param int groups: Number of subgroups from witch signals are coherrently combined, assumes subgroups are identical.
    :param int N_IPP: Number of pulses to coherrently integrate.
    :param float IPP_scale: Scale the IPP effective length in case e.g. the IPP is the same but the actual TX length is lowered.
    :param str units: If string equals 'dB', assume input and output units should be dB, else use linear scale.
    
    :return float: Gain after coherrent integration, linear units or in dB.
    '''
    if units == 'dB':
        return gain + 10.0*n.log10( groups*N_IPP*IPP_scale )
    else:
        return gain*(groups*N_IPP*IPP_scale)


def full_gain2inst_gain(gain,groups,N_IPP,IPP_scale=1.0,units = 'dB'):
    '''Using pulse encoding schema, subgroup setup and coherrent integration setup; convert from coherrently integrated gain to instantanius gain.
    
    :param float gain: Coherrently integrated gain, linear units or in dB.
    :param int groups: Number of subgroups from witch signals are coherrently combined, assumes subgroups are identical.
    :param int N_IPP: Number of pulses to coherrently integrate.
    :param float IPP_scale: Scale the IPP effective length in case e.g. the IPP is the same but the actual TX length is lowered.
    :param str units: If string equals 'dB', assume input and output units should be dB, else use linear scale.
    
    :return float: Instantanius gain, linear units or in dB.
    '''
    if units == 'dB':
        return gain - 10.0*n.log10( groups*N_IPP*IPP_scale )
    else:
        return gain/(groups*N_IPP*IPP_scale)


class AntennaRX(object):
    '''A receiving radar system (antenna or array of antennas).

        :param str name: Name of transmitting radar.
        :param float lat: Geographical latitude of radar system in decimal degrees  (North+).
        :param float lon: Geographical longitude of radar system in decimal degrees (East+).
        :param float alt: Geographical altitude above geoid surface of radar system in meter.
        :param float el_thresh: Elevation threshold for radar station, i.e. it cannot detect or point below this elevation.
        :param float freq: Operating frequency of radar station in Hz, i.e. carrier wave frequncy.
        :param float rx_noise: Receiver noise in Kelvin, i.e. system temperature.
        :param BeamPattern ant: Radiation pattern for radar station.
        :param bool phased: Is this a phased array that can perform post-analysis beam-forming?

        :ivar str name: Name of transmitting radar.
        :ivar float lat: Geographical latitude of radar system in decimal degrees  (North+).
        :ivar float lon: Geographical longitude of radar system in decimal degrees (East+).
        :ivar float alt: Geographical altitude above geoid surface of radar system in meter.
        :ivar float el_thresh: Elevation threshold for radar station, i.e. it cannot detect or point below this elevation.
        :ivar float freq: Operating frequency of radar station in Hz, i.e. carrier wave frequncy.
        :ivar float wavelength: Operating wavelength of radar station in meter.
        :ivar float rx_noise: Reviver noise in Kelvin, i.e. system temperature.
        :ivar BeamPattern beam: Radiation pattern for radar station.
        :ivar bool phased: Is this a phased array that can perform post-analysis beam-forming?
        :ivar numpy.array ecef: The ECEF coordinates of the radar system calculated using :func:`coord.geodetic2ecef`.
        
    '''
    def __init__(self, name, lat, lon, alt, el_thresh, freq, rx_noise, beam, scan = None, phased=True):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.el_thresh = el_thresh
        self.rx_noise = rx_noise
        self.beam = beam
        self.freq = freq
        self.phased = phased
        self.wavelength = c.c/freq
        self.ecef = coord.geodetic2ecef(lat, lon, alt)

        self.scan = scan
        self.extra_scans = None
        self.scan_controler = None

    def point_ecef(self, point):
        '''Point antenna beam in location of ECEF coordinate. Returns local pointing direction.
        '''
        k_obj = coord.ecef2local(
            lat = self.lat,
            lon = self.lon,
            alt = self.alt,
            x = point[0],
            y = point[1],
            z = point[2],
        )
        self.beam.point_k0(k_obj)
        return k_obj/n.linalg.norm(k_obj)

    def set_scan(self, scan = None, extra_scans = None, scan_controler = None):
        '''Set the scan this TX-antenna will use.
        
        :param RadarScan scan: The main observation mode of the transmitter. If not given or :code:`None` the scan set at initialization will be used.
        :param list extra_scans: List of additional observation schemes the transmitter will switch between, i.e. instances of :class:`radar_scans.radar_scan`.
        :param function scan_controler: The scan_controler function takes the :class:`antenna.AntennaTX` instance and the time as arguments. The function should, based on the time, return either the :attr:`antenna.AntennaTX.scan` attribute, or one of the scans in the list :attr:`antenna.AntennaTX.extra_scans` attribute. If the function pointer is set to ``None``, it is assumed only one scan exists and by default :attr:`antenna.AntennaTX.scan` is returned.
        '''
        self.extra_scans = extra_scans
        self.scan_controler = scan_controler
        if scan is not None:
            self.scan = scan
        self.scan.set_tx_location(self)
        self.scan.check_tx_compatibility(self)
        for sc in self.extra_scans:
            sc.set_tx_location(self)
            sc.check_tx_compatibility(self)

    def get_scan(self, t):
        '''Return the current scan at a particular time.
           
           Depending on the scan_controler function return the current observation schema that the system is running. If no scan_controler function is set, return the default scan.
           
           The :attr:`antenna.AntennaTX.scan_controler` function takes the :class:`antenna.AntennaTX` instance and a time as arguments.
           
           :param float t: Current time.
           
           :return: The currently running radar scan at time :code:`t`.
           :rtype: RadarScan
        '''
        if self.scan_controler is None:
            return self.scan
        else:
            return self.scan_controler(self, t)
    
    def get_pointing(self, t):
        '''Return the instantanius pointing of the TX antenna based on the currently running scan. Uses :func:`antenna.AntennaTX.get_scan`.
        
           :param float t: Current time.
           :return: Current TX-location in WGS84 ECEF and current pointing direction in ECEF. Both are 1-D arrays of 3 elements (lists, tuples or numpy.ndarray).
        '''
        return self.get_scan(t).antenna_pointing(t)
        

    def __str__(self):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("_")]
        string = "Antenna %s\n\n"%(self.name)
        for m in members:
            string += ("%s = %s\n"%(m, str(getattr(self, m))))
        return(string)
    

class AntennaTX(AntennaRX):
    '''A transmitting radar system (antenna or array of antennas)
        
        :param str name: Name of transmitting radar.
        :param float lat: Geographical latitude of radar system in decimal degrees (North+).
        :param float lon: Geographical longitude of radar system in decimal degrees (East+).
        :param float alt: Geographical altitude above geoid surface of radar system in meter.
        :param float el_tresh: Elevation threshold for radar station, i.e. it cannot detect or point below this elevation.
        :param float freq: Operating frequency of radar station in Hz, i.e. carrier wave frequency.
        :param float rx_noise: Receiver noise in Kelvin, i.e. system temperature.
        :param BeamPattern beam: Radiation pattern for radar station.
        :param float tx_bandwidth: Transmissions bandwidth.
        :param float duty_cycle: Maximum duty cycle, i.e. fraction of time transmission can occur at maximum power.
        :param float tx_power: Transmissions power in watts.
        :param float pulse_length: Length of transmission pulse.
        :param float ipp: Time between consecutive pulses.
        :param int n_ipp: Number of pulses to coherently integrate.

        :ivar str name: Name of transmitting radar.
        :ivar float lat: Geographical latitude of radar system in decimal degrees  (North+).
        :ivar float lon: Geographical longitude of radar system in decimal degrees (East+).
        :ivar float alt: Geographical altitude above geoid surface of radar system in meter.
        :ivar float el_thresh: Elevation threshold for radar station, i.e. it cannot detect or point below this elevation.
        :ivar float freq: Operating frequency of radar station in Hz, i.e. carrier wave frequency.
        :ivar float wavelength: Operating wavelength of radar station in meter.
        :ivar float rx_noise: Reviver noise in Kelvin, i.e. system temperature.
        :ivar BeamPattern beam: Radiation pattern for radar station.
        :ivar numpy.array ecef: The ECEF coordinates of the radar system calculated using :func:`coord.geodetic2ecef`.
        :ivar float tx_bandwidth: Transmissions bandwidth.
        :ivar float duty_cycle: Maximum duty cycle, i.e. fraction of time transmission can occur at maximum power.
        :ivar float tx_power: Transmissions power in watts.
        :ivar float enr_thresh: Minimum detectable target SNR (after coherent integration)
        :ivar float pulse_length: Length of transmission pulse.
        :ivar float ipp: Time between consecutive pulses.
        :ivar int n_ipp: Number of pulses to coherently integrate.
        :ivar float coh_int_bandwidth: Effective bandwidth of receiver noise after coherent integration.
        :ivar list extra_scans: List of additional observation schemes the transmitter will switch between, i.e. instances of :class:`radar_scans.radar_scan`.
        :ivar radar_scan scan: The main observation mode of the transmitter.
        :ivar function scan_controler: The scan_controler function takes the :class:`antenna.AntennaTX` instance and the time as arguments. The function should, based on the time, return either the :attr:`antenna.AntennaTX.scan` attribute, or one of the scans in the list :attr:`antenna.AntennaTX.extra_scans` attribute. If the function pointer is set to ``None``, it is assumed only one scan exists and by default :attr:`antenna.AntennaTX.scan` is returned.

    '''
    def __init__(self, name, lat, lon, alt, el_thresh, freq, rx_noise, beam, scan, tx_power, tx_bandwidth, duty_cycle, pulse_length=1e-3, ipp=10e-3, n_ipp=20, **kwargs):
        super(AntennaTX, self).__init__(name, lat, lon, alt, el_thresh, freq, rx_noise, beam, scan = scan, **kwargs)
        self.tx_bandwidth = tx_bandwidth
        
        self.duty_cycle = duty_cycle
        self.tx_power = tx_power
        self.enr_thresh = 10.0
        self.pulse_length = pulse_length
        self.ipp = ipp
        self.n_ipp = n_ipp
        self.coh_int_bandwidth = 1.0/(pulse_length*n_ipp)


