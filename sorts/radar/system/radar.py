#!/usr/bin/env python

'''This module is used to define the radar system

'''
import copy
import ctypes
import numpy as np

from .. import passes
from .. import measurements
from ...common import interpolation
from . import station
from sorts import clibsorts

class Radar(object):
    '''A network of transmitting and receiving radar stations.
        
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
    '''

    TIME_VARIABLES = [
        "t",
        "t_slice",
    ]

    CONTROL_VARIABLES = ["pointing_direction"]


    def __init__(self, tx, rx, min_SNRdb=10.0, logger=None, profiler=None, joint_stations=None, measurement=measurements.Measurement):
        self.tx = tx
        self.rx = rx

        self.logger = logger
        self.logger = profiler

        self.min_SNRdb = min_SNRdb
        self.measurement_class = measurement(logger=logger, profiler=profiler)

        if joint_stations is None:
            self.joint_stations = []
        else:
            self.joint_stations = joint_stations

        # add all properties from station.PROPERTIES
        for txi in range(len(self.tx)):

            # intializes all default properties of tx station
            for property_name in self.tx[txi].PROPERTIES:
                self.tx[txi].add_property(property_name)

        # add all properties from station.PROPERTIES
        for rxi in range(len(self.rx)):

            # intializes all default properties of tx station
            for property_name in self.rx[rxi].PROPERTIES:
                self.rx[rxi].add_property(property_name)


    def copy(self):
        '''Create a deep copy of the radar system.
        '''
        ret = Radar(
            tx = [],
            rx = [],
            min_SNRdb = copy.deepcopy(self.min_SNRdb),
            joint_stations = copy.deepcopy(self.joint_stations),
        )
        for tx in self.tx:
            ret.tx.append(tx.copy())
        for rx in self.rx:
            ret.rx.append(rx.copy())
        return ret


    def get_station_id_and_type(self, station):
        station_id = None
        station_type = None
        end = False

        # get type and id of station
        for station_type_ in ("tx", "rx"):
            if end is True:
                break

            stations = getattr(self, station_type_)

            for sid, station_ in enumerate(stations):
                if station_ == station:
                    station_id = sid
                    station_type = station_type_

                    end = True
                    break
        
        if station_id is None: raise Exception(f"could not find station {station} in radar {self}")

        return (station_id, station_type)


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


    def find_passes(self, t, states, cache_data=True, fov_kw=None):
        '''Finds all passes that are simultaneously inside a transmitter 
        station FOV and a receiver station FOV. 

            :param numpy.ndarray t: Vector of times in seconds to use as a 
                base to find passes.
            :param numpy.ndarray states: ECEF states of the object to find 
                passes for.
            :return: list of passes indexed by first tx-station and then 
                rx-station.
            :rtype: list of list of sorts.Pass
        '''
        rd_ps = []
        for txi, tx in enumerate(self.tx):
            rd_ps.append([])
            for rxi, rx in enumerate(self.rx):
                txrx = passes.find_simultaneous_passes(
                    t, states, 
                    [tx, rx], 
                    cache_data=cache_data, fov_kw=fov_kw,
                )
                for ps in txrx:
                    ps.station_id = [txi, rxi]
                rd_ps[-1].append(txrx)
        return rd_ps


    def check_control_feasibility(self, control_sequence):
        """
        This function verifies if any of the controls are in conflict with the physical limitations of the RADAR (for example 
        its speed, elevation, power, ...).

        This function can be freely overrided to check the feasibility of the controls wby a custom Radar system.
        
        Parameters :
        ------------
            control_sequence : dict()
                dictionnary containing the controls to be sent to the radar. Such controls give the commanded states of the radar
                (such as power, pointing direction, ...) for each control time slice.

        Returns :
        ---------
            Boolean :
                True if the controls are compatible with the radar system, False otherwise.
        """

        return True


    def control(self, control_sequence):
        n_tx = len(self.tx)
        n_rx = len(self.rx)

        if self.check_control_feasibility(control_sequence) is False:
            self.logger.error("radar:system:control: control sequence is not compatible with the radar system !")
            return None

        # Initialize radar states
        radar_states = dict()
        
        radar_states["meta"] = dict()
        radar_states["meta"]["radar"] = self

        n_periods = len(control_sequence["t"])

        # check if the control sequence is valid
        for ctrl_var_name in self.TIME_VARIABLES:
            if ctrl_var_name not in control_sequence.keys(): 
                raise KeyError(f"The control sequence must possess the time variable {ctrl_var_name}")

        for ctrl_var_name in self.CONTROL_VARIABLES:
            if ctrl_var_name not in control_sequence["controls"].keys(): 
                raise KeyError(f"The control sequence must possess the control variable {ctrl_var_name}")

        # copy t and t_slice to radar states
        for ctrl_var_name in self.TIME_VARIABLES:
            radar_states[ctrl_var_name] = control_sequence[ctrl_var_name]
        
        for ctrl_var_name in self.CONTROL_VARIABLES + self.PROPERTIES:
            if ctrl_var_name not in control_sequence["controls"].keys():
                default_param_value = np.asarray(getattr(self, ctrl_var_name))
                radar_states[ctrl_var_name] = np.ndarray((n_periods,), dtype=object)
                
                for period_id in range(n_periods):
                    default_param_value_arr = np.repeat(default_param_value[:, None], len(control_sequence["t"][period_id]), axis=1)
                    
                    for station_id in range(np.size(default_param_value, axis=0)):
                        radar_states[ctrl_var_name][period_id] = default_param_value_arr
            else:
                radar_states[ctrl_var_name] = np.empty((n_periods,), dtype=object)

                for period_id in range(n_periods):
                   radar_states[ctrl_var_name][period_id] = next(control_sequence["controls"][ctrl_var_name])

        return radar_states


    def compute_measurements(
        self, 
        radar_states, 
        space_object,
        rx_indices=None,
        tx_indices=None,
        epoch=None, 
        calculate_snr=True, 
        doppler_spread_integrated_snr=False,
        interpolator=interpolation.Linear, 
        max_dpos=50e3,
        snr_limit=True, 
        save_states=False, 
        logger=None,
        profiler=None
    ):
        return self.measurement_class.measure(
            radar_states, 
            space_object, 
            self, 
            rx_indices=rx_indices,
            tx_indices=tx_indices,
            epoch=epoch, 
            calculate_snr=calculate_snr, 
            doppler_spread_integrated_snr=doppler_spread_integrated_snr,
            interpolator=interpolator, 
            max_dpos=100e3,
            snr_limit=snr_limit, 
            save_states=save_states, 
            logger=logger,
            profiler=profiler,
        )


    def compute_intersection_points(
        self,
        tx_directions,
        rx_directions,
        rtol=0.05,
        ):
        """ 
        Computes the ECEF point (if it exists) targetted by pointing direction controls.
        Given a set of pointing directions 

        :math:($\hat{k_{tx}}$, \hat{k_{rx}}$) 

        and the positions of the stations (from which we can exctract the vector 

        :math:$\hat{k_{tx, rx}$

        ), one can compute the theoretical point targetted (given that the 3 prevous vectors lie in the same plane) using the formnula :

        TODO

        """
        if np.shape(tx_directions[0]) != np.shape(rx_directions[0]):
            raise Exception(f"tx and rx pdirs directions are not the same shape : {np.shape(tx_directions)} != {np.shape(rx_directions)}")

        # radar pointing directions
        tx_directions = np.ascontiguousarray(np.atleast_1d(tx_directions), dtype=np.float64)
        rx_directions = np.ascontiguousarray(np.atleast_1d(rx_directions), dtype=np.float64)

        # radar station positions in ECEF coordinates
        tx_ecef = np.array([tx.ecef for tx in self.tx], dtype=np.float64)
        rx_ecef = np.array([rx.ecef for rx in self.rx], dtype=np.float64)
        
        # initialization of the computation results
        n_points = len(rx_directions[0, 0, 0]) # number of time points

        intersection_points = np.empty((3, n_points,), dtype=np.float64)
        keep = np.zeros((n_points,), dtype=np.int32)

        # Calling c library
        clibsorts.compute_intersection_points.argtypes = [
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=tx_directions.ndim, shape=tx_directions.shape, flags="C"),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=rx_directions.ndim, shape=rx_directions.shape, flags="C"),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=tx_ecef.ndim, shape=tx_ecef.shape, flags="C"),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=rx_ecef.ndim, shape=rx_ecef.shape, flags="C"),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=intersection_points.ndim, shape=intersection_points.shape, flags="C"),
            ctypes.c_double,
            np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=keep.ndim, shape=keep.shape),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        clibsorts.compute_intersection_points(
            tx_directions,
            rx_directions,
            tx_ecef,
            rx_ecef,
            intersection_points,
            ctypes.c_double(rtol),
            keep,
            ctypes.c_int(len(self.tx)),
            ctypes.c_int(len(self.rx)),
            ctypes.c_int(n_points),
        ) 

        return intersection_points[:, np.where(keep==1)[0]]