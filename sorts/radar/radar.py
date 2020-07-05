#!/usr/bin/env python

'''This module is used to define the radar network configuration.

'''
import numpy as np

from .. import passes

class Radar(object):
    '''A network of transmitting and receiving radar stations.
        
        :ivar list tx: List of transmitting sites, i.e. instances of :class:`sorts.radar.TX`
        :ivar list rx: List of receiving sites, i.e. instances of :class:`sorts.radar.RX`
        :ivar float max_off_axis: Maximum angle between pointing direction and a received signal.
        :ivar float min_SNRdb: Minimum SNR detectable by radar system in dB (after coherent integration).

        :param list tx: List of transmitting sites, i.e. instances of :class:`sorts.radar.TX`
        :param list rx: List of receiving sites, i.e. instances of :class:`sorts.radar.RX`
        :param float max_off_axis: Maximum angle between pointing direction and a received signal.
        :param float min_SNRdb: Minimum SNR detectable by radar system in dB (after coherent integration).

    '''
    def __init__(self, tx, rx, max_off_axis=90.0, min_SNRdb=10.0):
        self.tx = tx
        self.rx = rx
        self.max_off_axis = max_off_axis
        self.min_SNRdb = min_SNRdb


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


    def find_passes(self, t, states):
        '''Finds all passes that are simultaneously inside a transmitter station FOV and a receiver station FOV. 
        '''
        rd_ps = []
        for tx in self.tx:
            rd_ps.append([])
            for rx in self.rx:
                txrx = passes.find_simultaneous_passes(t, states, [tx, rx])
                rd_ps[-1].append(txrx)
        return rd_ps