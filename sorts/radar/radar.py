#!/usr/bin/env python

'''This module is used to define the radar network configuration.

'''
import numpy as np


class Radar(object):
    '''A network of transmitting and receiving radar stations.
        
        :ivar list tx: List of transmitting sites, i.e. instances of :class:`sorts.radar.TX`
        :ivar list rx: List of receiving sites, i.e. instances of :class:`sorts.radar.RX`
        :ivar float max_off_axis: Maximum angle between pointing direction and a received signal.
        :ivar float min_SNRdb: Minimum SNR detectable by radar system in dB (after coherent integration).

        :param list tx: List of transmitting sites, i.e. instances of :class:`antenna.AntennaTX`
        :param list rx: List of receiving sites, i.e. instances of :class:`antenna.AntennaRX`
        :param float max_off_axis: Maximum angle between pointing direction and a received signal.
        :param float min_SNRdb: Minimum SNR detectable by radar system in dB.
    '''
    def __init__(self, tx, rx, max_off_axis=90.0, min_SNRdb=10.0):
        self.tx = tx_lst
        self.rx = rx_lst
        self.max_off_axis = max_off_axis
        self.min_SNRdb = min_SNRdb


    def set_tx_bandwith(self, bw):
        '''Set the transmission bandwidth in Hz of all transmitters in the radar system.
        
        :param float bw: Transmission bandwidth in Hz. This is basically what range of frequencies available for wave forming the transmission, e.g. how fast bit-key-shifting code can switch from 0 to :math:`\pi` and can then be calculated as the inverse of the baud length.
        
        '''
        for tx in self.tx:
            tx.tx_bandwidth = bw
    
    
    def set_beam(self, beam):
        '''Sets the radiation pattern for transmitters and receivers.
        
        :param pyant.Beam beam: The radiation pattern to set for radar system.
        '''
        self.set_tx_beam(beam)
        self.set_rx_beam(beam)


    def set_tx_beam(self, beam):
        '''Sets the radiation pattern for transmitters.
        
        :param pyant.Beam beam: The radiation pattern to set for radar system.
        '''
        for tx in self.tx:
            tx.beam = beam.copy()

    def set_rx_beam(self, beam):
        '''Sets the radiation pattern for receivers.
        
        :param pyant.Beam beam: The radiation pattern to set for radar system.
        '''
        for rx in self.rx:
            rx.beam = beam.copy()
