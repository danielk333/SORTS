#!/usr/bin/env python

"""This module is used to define the radar system"""
import copy
from datetime import datetime
import numpy as np
import numpy.typing as npt
from .. import passes, passes_v2
from .radars.composite_key import RadarCompositeKey


class Radar(object):
    """A network of transmitting and receiving radar stations.

    :ivar list tx: List of transmitting sites,
        i.e. instances of :class:`sorts.radar.TX`
    :ivar list rx: List of receiving sites,
        i.e. instances of :class:`sorts.radar.RX`
    :ivar float min_SNRdb: Minimum SNR detectable by radar system in dB
        (after coherent integration).
    :ivar list joint_stations: A list of (tx,rx) indecies of stations that
        share hardware. This can be used to e.g. turn of receivers when
        the same hardware is transmitting.

    :param list tx: List of transmitting sites,
        i.e. instances of :class:`sorts.radar.TX`
    :param list rx: List of receiving sites,
        i.e. instances of :class:`sorts.radar.RX`
    :param float min_SNRdb: Minimum SNR detectable by radar system in dB
        (after coherent integration).

    """

    def __init__(self, tx, rx, min_SNRdb=10.0, joint_stations=None):
        self.tx = tx
        self.rx = rx
        self.min_SNRdb = min_SNRdb
        if joint_stations is None:
            self.joint_stations = []
        else:
            self.joint_stations = joint_stations

    def copy(self):
        """Create a deep copy of the radar system."""
        ret = Radar(
            tx=[],
            rx=[],
            min_SNRdb=copy.deepcopy(self.min_SNRdb),
            joint_stations=copy.deepcopy(self.joint_stations),
        )
        for tx in self.tx:
            ret.tx.append(tx.copy())
        for rx in self.rx:
            ret.rx.append(rx.copy())
        return ret

    def set_beam(self, beam):
        """Sets the radiation pattern for transmitters and receivers.

        :param pyant.Beam beam: The radiation pattern to set for radar system.
        """
        self.set_tx_beam(beam)
        self.set_rx_beam(beam)

    def set_tx_beam(self, beam):
        """Sets the radiation pattern for transmitters.

        :param pyant.Beam beam: The radiation pattern to set for radar system.
        """
        for tx in self.tx:
            tx.beam = beam.copy()

    def set_rx_beam(self, beam):
        """Sets the radiation pattern for receivers.

        :param pyant.Beam beam: The radiation pattern to set for radar system.
        """
        for rx in self.rx:
            rx.beam = beam.copy()

    def find_passes(self, t, states, cache_data=True, fov_kw=None):
        """Finds all passes that are simultaneously inside a transmitter
        station FOV and a receiver station FOV.

            :param numpy.ndarray t: Vector of times in seconds to use as a
                base to find passes.
            :param numpy.ndarray states: ECEF states of the object to find
                passes for.
            :return: list of passes indexed by first tx-station and then
                rx-station.
            :rtype: list of list of sorts.Pass
        """
        rd_ps = []
        for txi, tx in enumerate(self.tx):
            rd_ps.append([])
            for rxi, rx in enumerate(self.rx):
                txrx = passes.find_simultaneous_passes(
                    t,
                    states,
                    [tx, rx],
                    cache_data=cache_data,
                    fov_kw=fov_kw,
                )
                for ps in txrx:
                    ps.station_id = [txi, rxi]
                rd_ps[-1].append(txrx)
        return rd_ps

    def find_passes_v2(
        self,
        dt_arr: npt.NDArray[np.float64],
        states: npt.NDArray[np.float64],
        radar_composite_key: RadarCompositeKey,
        epoch: datetime,
        fov_kw=None,
    ):
        """
        ver 2 of the `find_passes()` func,
        same logic as v1 but return v2 `Pass` class instead

        Finds all passes that are simultaneously inside a transmitter
        station FOV and a receiver station FOV.

            :param numpy.ndarray t: Vector of times in seconds to use as a
                base to find passes.
            :param numpy.ndarray states: ECEF states of the object to find
                passes for.
            :return: list of passes indexed by first tx-station and then
                rx-station.
            :rtype: list of list of sorts.Pass

        TODO: clean up the v1 func and remove the v2 suffix
        TODO: radar should know RadarCompositeKey, RadarStationCompositeKey, and not need to get them from param
        TODO: this `stations=[tx, rx],` feels a bit hard-coded?
        """

        rd_ps: list[list[list[passes_v2.Pass]]] = []
        for txi, tx in enumerate(self.tx):
            rd_ps.append([])
            for rxi, rx in enumerate(self.rx):
                txrx = passes_v2.find_simultaneous_passes(
                    dt_arr=dt_arr,
                    states=states,
                    stations=[tx, rx],
                    radar_station_composite_keys=[
                        (*radar_composite_key, "tx", f"{txi}"),
                        (*radar_composite_key, "rx", f"{rxi}"),
                    ],  # TODO: should not hard-code it here, each radar can have their own radar key to station key logic
                    epoch=epoch,
                    fov_kw=fov_kw,
                )

                rd_ps[-1].append(txrx)

        return rd_ps
