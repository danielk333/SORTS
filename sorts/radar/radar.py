#!/usr/bin/env python

'''This module is used to define the radar network configuration.

# TODO: Change all attribute names according to convention: 'var' public, '_var' internal, '__var' private.
# TODO: It would make sens to change it so that a rx antenna is always a reciver but a TX antenna inherrents RX antenna as it is now but that every TX antenna is also automatically counted as a RX antenna so that you do not e.g. have to specify 'skibotten RX' and 'skibotten TX', instead it would only be 'skibotten' but a instance of TX instead of RX and then you intead only loop over stations in radar system and know that all have RX capabilities but at least one have to have TX capabilities.
# TODO: Change name of this module to radar.py
'''
import numpy as np
import scipy.constants as c

import matplotlib.pyplot as plt

# SORTS imports
import coord
import radar_scans as rs
import antenna

try:
    import geopandas
    import pandas as pd
    from shapely.geometry import Point
except ImportError:
    geopandas = None
    pd = None
    Point = None

class RadarSystem(object):
    '''A network of transmitting and receiving radar systems.
        
        :ivar list _tx: List of transmitting sites, i.e. instances of :class:`antenna.AntennaTX`
        :ivar list _rx: List of receiving sites, i.e. instances of :class:`antenna.AntennaRX`
        :ivar float max_on_axis: Maximum angle between pointing direction and a received signal.
        :ivar string name: Verbose name of the radar system
        :ivar float _horizon_elevation: Elevation in degrees of the horizon, i.e. minimum elevation the radar system can measure and point.
        :ivar float min_SNRdb: Minimum SNR detectable by radar system in dB.

        :param list tx_lst: List of transmitting sites, i.e. instances of :class:`antenna.AntennaTX`
        :param list rx_lst: List of receiving sites, i.e. instances of :class:`antenna.AntennaRX`
        :param string name: Verbose name of the radar system
        :param float max_on_axis: Maximum angle between pointing direction and a received signal.
        :param float min_SNRdb: Minimum SNR detectable by radar system in dB.
    '''
    def __init__(self, tx_lst, rx_lst, name, max_on_axis=90.0, min_SNRdb=1.0):
        self._tx = tx_lst
        self._rx = rx_lst
        self.name = name
        self.max_on_axis = max_on_axis
        self.min_SNRdb = min_SNRdb
        self._horizon_elevation = None


    def set_FOV(self, max_on_axis, horizon_elevation):
        '''Set the Field of View (FOV) for this radar system. The FOV is imposed on every receiving station and transmitting station in the network. The FOV is assumed to be azimutally symmetric.
        
        :param float max_on_axis: Maximum angle in degrees from the pointing direction at witch a detection can be made.
        :param float horizon_elevation: The elevation angle in degrees of the FOV.
        '''
        self.max_on_axis = max_on_axis
        self._horizon_elevation = horizon_elevation
        for rx in self._rx:
            rx.el_thresh = horizon_elevation
        for tx in self._tx:
            tx.el_thresh = horizon_elevation


    def set_SNR_limits(self, min_total_SNRdb, min_pair_SNRdb):
        '''Set the Signal to Noise Ratio (SNR) limits for the system.
        
        :param float min_total_SNRdb: The minimum SNR in dB that is required on at least one transmitter-receiver pair for a detection to be made.
        :param float min_pair_SNRdb: The minimum SNR in dB that is required for a transmitter-receiver pair to have a detection.
        '''
        self.min_SNRdb = min_total_SNRdb
        for tx in self._tx:
            tx.enr_thresh = 10.0**(min_pair_SNRdb/10.0)


    def set_scan(self, SST, secondary_list = None):
        '''Set the observation schema that the radar system will use.
        
        :param radar_scan SST: Sets the main SST observation schema.
        :param list secondary_list: Sets a list of other observation schema's, i.e. instances of :class:`radar_scans.radar_scan`, that are interleaved with the main SST scan.
        '''
        for tx in self._tx:
            tx.scan = SST.copy()
            tx.scan.set_tx_location(tx)

        if secondary_list is not None:
            for tx in self._tx:
                tx.extra_scans = [None]*len(secondary_list)
                for ind, scan in enumerate(secondary_list):
                    tx.extra_scans[ind] = scan.copy()
                    tx.extra_scans[ind].set_tx_location(tx)


    def set_TX_bandwith(self, bw):
        '''Set the transmission bandwidth in Hz of all transmitters in the radar system.
        
        :param float bw: Transmission bandwidth in Hz. This is basically what range of frequencies available for wave forming the transmission, e.g. how fast bit-key-shifting code can switch from 0 to :math:`\pi` and can then be calculated as the inverse of the baud length.
        
        '''
        for tx in self._tx:
            tx.tx_bandwidth = bw
    
    
    def set_beam(self, beam, mode='all'):
        '''Sets the radiation pattern for transmitters, receivers or entire radar system.
        
        To manually set custom beams for each transmitter and receiver in the radar system, set the attributes directly using instances of :class:`antenna.BeamPattern`.
        
        :param BeamPattern beam: The radiation pattern to set for radar system.
        :param str mode: String describing what part of radar system to set beam for: Options are ``'TX'`` for transmission, ``'RX'`` for reception, or both when left unset.
        
        **Example:**

        .. code-block:: python

            import antenna_library as alib
            from my_radar import radar
           
            #radar is a instance of RadarSystem
            radar.set_beam(
                alib.planar_beam(az0=0, el0=90, lat=68, lon=0, I_0=10**4.5, a0=40.0, az1=0.0, el1=90.0, f=233e6),
                'TX'
            )

        '''
        if mode.lower() == 'tx':
            for tx in self._tx:
                tx.beam = beam.copy()
        elif mode.lower() == 'rx':
            for rx in self._rx:
                rx.beam = beam.copy()
        elif mode.lower() == 'all':
            for rx in self._rx:
                rx.beam = beam.copy()
            for tx in self._tx:
                tx.beam = beam.copy()
        else:
            raise Exception('Mode "{}" not found, cannot set beam to radar system.'.format(mode))
