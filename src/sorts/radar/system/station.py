#!/usr/bin/env python

'''

'''

#Python standard import
import copy

#Third party import
import numpy as np
import pyant

#Local import
from ...transformations import frames

class PropertyError(Exception):
    pass

class Station(object):
    """ Encapsulates a RADAR station.

    The role of the RADAR station is to transmit/receive Radar pulses (of given wavelength, power, gain pattern, ...) in a given direction to 
    perform Radar measurements.

    Parameters
    ---------- 
    lat : float
        Geographical latitude of radar station in decimal degrees (North+).
    lon : float
        Geographical longitude of radar station in decimal degrees (East+).
    alt : float
        Geographical altitude above geoid surface of radar station in meter.
    min_elevation: float
        Elevation threshold for the radar station in degrees, i.e. it cannot detect or point below this elevation.
    beam : pyant.Beam 
        Radiation pattern for radar station.
    """

    __slots__  = ["lat", "lon", "alt", "min_elevation", "ecef", "beam", "enabled", "pointing_range", "_type"]

    CONTROL_VARIABLES = [
        "pointing_direction",
        ]
    """ list of Radar :class:`Station<Station>` control variables. A control variable is a 
    variable which must be controlled by a Radar Controller for the Radar controls to be 
    considered valid (see :ref:`radar_controller<radar_controller>` for more information).

    .. note:: 
        Please note that in the current implementation of sorts, ``CONTROL_VARIABLES`` is only supposed
        to be modified internally. Therefore do not advise the manual modification of the
        ``CONTROL_VARIABLES`` list during runtime.
    """

    PROPERTIES = []
    """ list of properties of a radar :class:`Station<Station>`. Those properties can be controlled 
    by Radar controllers (see :ref:`radar_controller<radar_controller>` for more information).

    .. note:: 
        Please note that in the current implementation of sorts, ``PROPERTIES`` is only supposed
        to be modified internally. Therefore do not advise the manual modification of the
        ``PROPERTIES`` list during runtime.

        See :func:`~Station.add_property` to obtain more information on how to add new controllable
        Radar properties. 
    """

    BEAM_PROPERTIES = [
        "phase_steering",
        "wavelength",
    ]
    """ list of radar beam properties of a radar :class:`Station<Station>`. Those properties can be controlled 
    by Radar controllers (see :ref:`radar_controller<radar_controller>` for more information).

    .. note:: 
        Please note that in the current implementation of sorts, ``BEAM_PROPERTIES`` is only supposed
        to be modified internally. Therefore do not advise the manual modification of the
        ``BEAM_PROPERTIES`` list during runtime.

        See :func:`~Station.add_property` to obtain more information on how to add new controllable
        Radar properties. 
    """

    def __init__(self, lat, lon, alt, min_elevation, beam, **kwargs):
        self.lat = lat 
        """ Station geographical latitude (in degrees). """
        
        self.lon = lon    
        """ Station geographical longitude (in degrees). """    

        self.alt = alt
        """ Station geographical altitude (in meters). """    

        self.min_elevation = min_elevation
        """ Station minimal beam elevation angle (in degrees). 

        The beam elevation angle is defined as the angle between the local vertical 
        (:math:`\hat{n_z} = [0 \hspace{0.5mm}0 \hspace{0.5mm} 1]^{T}` in the ENU frame) 
        and the pointing direction of the Radar beam :math:`\hat{k}`.
        The minimial elevation angle is used to compute passes (sequence of 
        radar states within a given field of view) used for data reduction purposes.
        """    

        self.ecef = frames.geodetic_to_ITRS(lat, lon, alt, radians = False)
        """ Station position vector in the ECEF frame. """    

        self.beam = beam
        """ Station gain pattern (taking into account the properties of the transmitter/receiver and of the signal itself). """    

        self.enabled = True
        """ State of the station. """    
        self.pointing_range = None
        """ Range (in meters) from the point being targetted by the radar beam. """    

        self._type = None 
        """ Station type ("tx"/"rx"). """

        # add properties to access beam properties from the station
        for prop_name in self.BEAM_PROPERTIES:
            if hasattr(self.beam, prop_name) and not hasattr(self, prop_name) and not prop_name in self.PROPERTIES:
                self.PROPERTIES.append(prop_name)

                property_code = f"""@property   
                \ndef prop(self,):
                \n    val = self.beam.{prop_name}
                \n    return val

                \n@prop.setter
                \ndef prop(self, value):
                \n    self.beam.{prop_name} = value

                \nsetattr(self.__class__, '{prop_name}', prop)
                """

                exec(property_code)



    def add_property(self, name):
        ''' Adds a new property to the station available for the given station.

        Each station property can be individually controlled by Radar controls (see :class:`RadarControls`).
        The addition of new controls during runtime allows for a standard interface for the creation of custom 
        station instances without the need to create new station classes. If a property with the same name already
        exists with the current station instance, the 
    
        .. note:
            In the current implementation, a property is defined by its name and can only take scalar values at
            each control time point.

        Parameters
        ----------
        name : str
            Name of the radar property to be added.     

        Returns
        -------
        None   
        '''
        if not hasattr(self, name):
            setattr(self, name, None)

            if not name in self.PROPERTIES:
                self.PROPERTIES.append(name)

    def get_properties(self):
        """ Returns all properties available for the given station. 

        Each Radar station possesses a set of properties which can be controlled by a Radar
        Controller (see :attr:`Station.PROPERTIES` for more information). The function 
        :func:`Station.get_properties` returns the list of all controllable properties associated
        with the current station.

        Parameters 
        ----------
        None

        Returns
        -------
        list 
            List of Radar properties within the current station. 
        
        Examples
        --------
        By default, all stations possess a single property named ``wavelength`` which corresponds to the wavelength
        of the received or transmitted signal. 

        Therefore, suppose that we have previously created a station named ``radar_station``,
        calling :func:`Station.has_property` over a station this property yields :

        >>> radar_station.get_properties()
        ["wavelength"]

        """
        property_list = []
        
        for prop in self.PROPERTIES:
            if hasattr(self, prop):
                property_list.append(prop)

        return property_list


    def has_property(self, name):
        """ Checks if the station possesses a given property. 
        
        Each Radar station possesses a set of properties which can be controlled by a Radar
        Controller (see :attr:`Station.PROPERTIES` for more information). The function 
        :func:`Station.has_property` checks if the station has a given property

        Parameters 
        ----------
        name : str
            Station property name.

        Returns
        -------
        bool 
            ``True`` if the station possesses the given property, and ``False`` if not.

        Examples
        --------
        All stations possess a property named ``wavelength`` which corresponds to the wavelength
        of the received or transmitted signal. 

        Therefore, suppose that we have previously created a station named ``radar_station``,
        calling :func:`Station.has_property` over a station this property yields :

        >>> radar_station.has_property("wavelength")
        True
        """
        return name in self.PROPERTIES and hasattr(self, name)


    def field_of_view(self, ecef):
        """ Checks if a set of ECEF point in FOV.

        This function checks if a given set of ECEF points are within the field of view 
        of the station. Used to determine when a "pass" is occurring based on the input 
        ECEF states and times. The default implementation relies on a minimum elevation 
        check.

        Parameters 
        ----------
            ecef : numpy.ndarray of float (6, N)
                set of ecef points which are to be checked.

        Returns 
        -------
            numpy.ndarray of bools (N,)
                Mask indicating the points which are within the field of view of the station 

        Examples
        --------
        This function is used for the generation of Tracking  Radar Controls for reducing 
        the number of states by removing all the states which are outside of the FOV of the
        stations.

        First we get the propagated space object states we want to track ``states`` (see
        :class:SpaceObject<sorts.targets.space_object.SpaceObject>). 

        >>> states = space_object.get_state(t)
        >>> t # radar controller time points
        array([0., 100., 200., 300., 400., 500.])

        For the purpose of simplifying this example, we will assume that only the states at time
        t = 200, 300 and 400 seconds are within the FOV of the radar station. Therefore, we get :

        >>> pass_mask = radar_station.field_of_view(states)
        >>> pass_mask
        >>> array([False, False, True, True, True, False])

        we can then use the ``pass_mask`` array to only keep the time points which are within 
        the FOV :

        >>> t = t[pass_mask] # remove time points outside of the FOV
        >>> t
        >>> array([200., 300., 400.])

        """
        ecef = np.asarray(ecef).reshape(6, -1)[0:3]

        # compute station local normal direction
        local_normal_ecef = self.get_local_vertical()

        # compute direction from station to each ecef point to check
        station_to_points_ecef = ecef - self.ecef[:, None]
        station_to_points_ecef = station_to_points_ecef/np.linalg.norm(station_to_points_ecef, axis=0)[None, :]

        # compute dot product to get angle between vectors
        dotp = np.einsum("ij, i->j", station_to_points_ecef, local_normal_ecef) # since all vector are normalized, dotp = cos(theta)

        return dotp >= np.cos(np.pi/2.0 - self.min_elevation*np.pi/180.0)


    def rebase(self, lat, lon, alt):
        """ Change geographical location of the station. 

        The position of a station can be defined either by a vector in the ``ECEF`` frame 
        (:math:`r_{ecef}=[x, y, z]^T`), or by a vector in the ``Geodetic`` frame 
        (:math:`r_{geo}=[\phi, \lambda, h]^T`, which elements correspond 
        respectively to latitude [deg], longitude [deg] and altitude [m]). 

        The function :func:`Station.rebase` allows the user to change the position of the
        radar station to a new set of ``Geodetic`` coordinates :math:`r_{geo}=[\phi
        , \lambda, h]^T`. The function will also update the value of the position vector in
        the ``ECEF`` frame.

        Parameters
        ----------
        lat : float
            Station geographical latitude (in degrees).
        lon : float
            Station geographical longitude (in degrees).
        alt : float
            Station geographical altitude (in meters).

        Returns
        -------
        None

        Examples
        --------
        Assume that we have previously created a station named ``radar_station`` : 
        
        >>> import sorts
        >>> radar = sorts.radars.eiscat3d
        >>> radar_station = radar.tx[0]

        The station is by default located at 69.34023844°N, 20.313166°E, 0.0m :

        >>> radar_station.lon
        20.313166
        >>> radar_station.lat
        69.34023844
        >>> radar_station.alt
        0.0
        >>> radar_station.ecef
        array([2116582.84832936,  783500.73846828, 5945469.15155332])

        We can relocate this station to 0.0°N, 0.0°E, 0.0m by calling :

        >>> radar_station.rebase(0.0, 0.0, 0.0)
        >>> radar_station.lon
        0.0
        >>> radar_station.lat
        0.0
        >>> radar_station.alt
        0.0
        >>> radar_station.ecef
        array([6378137., 0., 0.])

        """
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.ecef = frames.geodetic_to_ITRS(lat, lon, alt, radians = False)


    def copy(self):
        """ Performs a deepcopy of the Radar station. """
        st = Station(
            lat = self.lat,
            lon = self.lon,
            alt = self.alt,
            min_elevation = self.min_elevation,
            beam = self.beam.copy(),
        )
        st.enabled = self.enabled
        return st

    def enu(self, ecefs):
        """ Converts a set of ECEF points to local Radar station coordinates (ENU). """
        rel_ = ecefs.copy()

        rel_[:3,:] = rel_[:3,:] - self.ecef[:, None]
        rel_[:3,:] = frames.ecef_to_enu(
            self.lat,
            self.lon,
            self.alt,
            rel_[:3,:],
            radians=False,
        ).reshape(3, -1)
        if ecefs.shape[0] > 3:
            rel_[3:,:] = frames.ecef_to_enu(
                self.lat,
                self.lon,
                self.alt,
                rel_[3:,:],
                radians=False,
            ).reshape(3, -1)
        return rel_

    def to_ecef(self, enus):
        """ Converts a set of ECEF points to local Radar station coordinates (ENU). """
        ecefs = enus.copy()
        ecefs[:3,:] = frames.enu_to_ecef(
            self.lat,
            self.lon,
            self.alt,
            enus[:3,:],
            radians=False,
        ).reshape(3, -1) + self.ecef[:, None]

        # transform velocity to ecef
        if enus.shape[0] > 3:
            ecefs[3:,:] = frames.enu_to_ecef(
                self.lat,
                self.lon,
                self.alt,
                enus[3:,:],
                radians=False,
            ).reshape(3, -1)            
        return ecefs


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
        ).reshape(3, -1)

        if np.size(k) == 3:
            k = k.reshape(3,)

        self.beam.point(k)

    @property
    def pointing(self):
        '''Station beam pointing direction in local ENU coordinates.
        '''
        return self.beam.pointing.copy()


    @property
    def pointing_ecef(self):
        '''Station beam pointing direction in ECEF coordinates.
        '''
        return frames.enu_to_ecef(
            self.lat, 
            self.lon, 
            self.alt, 
            self.beam.pointing, 
            radians=False,
        )

    @property
    def type(self):
        '''Station type ("tx"/"rx").'''
        return self._type


    def get_local_vertical(self):
        """
        Computes the local normal direction of the station.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray of float (3,)
            local vertical pointing direction in ECEF coordinates.
        """
        return frames.enu_to_ecef(self.lat, self.lon, self.alt, np.array([0.0, 0.0, 1.0]))

class RX(Station):
    """ 
    Defines a Radar Receiving station.

    Receiving stations (called :class:`RX<RX>` in SORTS) are responsible for 
    aquireing and processing the incomming Radar signal scattered by the `target` which 
    properties (such as size, position or velocity) we want to determine. 

    The :class:`RX<RX>` class defines all the methods and arguments common to all 
    `receiving stations`. 

    .. note:: 
       One has to keep in mind that the :class:`RX<RX>` only represents the reciving 
       part of station, and not the entirety of the physical Radar station. 

       In reality, some stations are able to perform both the transmission and reception of 
       the signal. Such stations are modelled by two different instances :class:`TX<TX>` 
       and :class:`RX<RX>` which are then joint together within the 
       :class:`Radar<sorts.radar.system.radar.Radar>` class.

    Examples
    --------
    There are two main ways to create a :class:`RX<RX>` Radar station.

    1. Predifined Radar stations:
        One can used the predifined radar stations which are defined within the :ref:`sorts.radars<radar_instances>`
        module.

        .. admonition:: Example

            To import a predifined radar instance, one first needs to import the ``sorts`` toolbox, 
            and then to get an instance from the :ref:`sorts.radars<radar_instances>` module.
            
            >>> import sorts
            >>> radar = sorts.radars.eiscat3d

            When the radar is imported, one can then access the stations as follow :
            
            >>> station_rx = radar.rx[0]
            >>> station_rx.type
            "rx"

    2. Create a new Radar stations:
        The other solution to create a Radar station is to create a brand new station using the 
        default :class:`RX<RX>` constructor.

        .. admonition:: Example

            First import the station module from sorts
            
            >>> from sorts import station

            Then, import the :class:`Beam<pyant.Beam>` from pyant to create the radiation
            pattern of the station's antenna.
            
            >>> from pyant import Beam

            Define the position of the station is defined as follow :
            
            >>> lon = 20.22 # in degrees
            >>> lat = 67.86 # in degrees
            >>> alt = 530.0 # in meters

            Set the station minimal elevation (which defines the FOV of the station)
            
            >>> min_el = 30.0 # in degrees

            Define the station radiation pattern :
            
            >>> az = 0.0 # beam azimuth (degrees)
            >>> el = 0.0 # beam elevation (degrees)
            >>> f = 500e6 # bram frequency at 500MHz
            >>> beam_rx = pyant.Beam(az, el, f)

            Define the properties of the receiver processing unit :
            
            >>> noise_temperature = 150 # in Kelvins

            Finally, one can create the station by calling :
            
            >>> station_rx = station.RX(
                    lat, 
                    lon, 
                    alt, 
                    min_el, 
                    beam_rx, 
                    noise_temperature)
            >>> station_rx.type
            "rx"

    Parameters
    ----------
    lat : float
        Geographical latitude of radar station in decimal degrees (North+).
    lon : float
        Geographical longitude of radar station in decimal degrees (East+).
    alt : float
        Geographical altitude above geoid surface of radar station in meter.
    min_elevation: 
        Elevation threshold for the radar station in degrees, i.e. it cannot detect or point below this elevation.
    beam : pyant.Beam 
        Radiation pattern for radar station.
    noise_temperature : float
        Temperature of the receiver noise. The noise power can then be computed as follows :

        .. math::       P_n=k_B T_{noise} b_n

        With :math:`P_n` the ``noise power``, :math:`k_B` the ``Boltzman constant`` (equal to 
        :math:`1.380649 . 10^{-23} m^{2} kg s^{-2} K^{-1}`), :math:`T_{noise}` the ``noise temperature``
        and :math:`b_n` the ``effective receiver noise bandwidth``.
    """

    PROPERTIES = Station.PROPERTIES + []
    """ list of properties of a :class:`RX<RX>` Radar station. Those properties can be controlled 
    by Radar controllers (see :ref:`radar_controller<radar_controller>` for more information).

    .. note:: 
        Please note that in the current implementation of sorts, ``PROPERTIES`` is only supposed
        to be modified internally. Therefore do not advise the manual modification of the
        ``PROPERTIES`` list.

        See :func:`~TX.add_property` to obtain more information on how to add new controllable
        Radar properties. 
    """

    def __init__(self, lat, lon, alt, min_elevation, beam, noise_temperature):
        """ __init__(self, lat, lon, alt, min_elevation, beam, noise_temperature)

        Default :class:`RX<RX>` class constructor.
        """
        super().__init__(lat, lon, alt, min_elevation, beam)
        self.noise_temperature = noise_temperature
        ''' Receiver noise temperature.

        It is possible to express the noise power density :math:`P_N` (in W/Hz) as an equivalent temperature  
        :math:`T_n` (called *noise temperature*) that would produce the same **Johnson–Nyquist noise** as the 
        receiver :

        .. math::           P_N = k_B T_n b

        with :math:`b` the bandwidth of the receiver and :math:`k_B = 1.381 \\times 10^{-23} J/K` the Boltzmann 
        constant.
        '''

        self._type = "rx"


    def copy(self):
        """ Performs a deep copy of the Radar station. 

        This method is used to perform a copy of a radar station.

        Examples
        --------
        Assuming that we have created a :class:`TX<TX>` station named ``station_old``, the  
        station can be copied by simply calling :

        >>> station_new = station_old.copy()
        """
        st = RX(
            lat = self.lat,
            lon = self.lon,
            alt = self.alt,
            min_elevation = self.min_elevation,
            beam = self.beam.copy(),
            noise_temperature = self.noise_temperature,
        )
        st.enabled = self.enabled
        return st


class TX(Station):
    """ 
    Defines a Radar Transmitting station.

    Transmittig stations (called :class:`TX<TX>` in SORTS) are responsible for 
    amplifying and transmitting the Radar signal in the direction of the `target` which 
    properties (such as size, position or velocity) we want to determine. As such, the 
    definition of the radar pulse properties is entirely done within the :class:`TX<TX>` 
    station. 

    The :class:`TX<TX>` class defines all the methods and arguments common to all 
    `transmitting stations`. 

    .. note:: 
       One has to keep in mind that the :class:`TX<TX>` only represents the transmitting 
       part of station, and not the entirety of the physical Radar station. 

       In reality, some stations are able to perform both the transmission and reception of 
       the signal. Such stations are modelled by two different instances :class:`TX<TX>` 
       and :class:`RX<RX>` which are then joint together within the 
       :class:`Radar<sorts.radar.system.radar.Radar>` class.

    Examples
    --------
    There are two main ways to create a :class:`TX<TX>` Radar station.

    1. Predifined Radar stations:
        One can used the predifined radar stations which are defined within the :ref:`sorts.radars<radar_instances>`
        module.

        .. admonition:: Example

            To import a predifined radar instance, one first needs to import the ``sorts`` toolbox, 
            and then to get an instance from the :ref:`sorts.radars<radar_instances>` module.
            
            >>> import sorts
            >>> radar = sorts.radars.eiscat3d

            When the radar is imported, one can then access the stations as follow :
            
            >>> station_tx = radar.tx[0]
            >>> station_tx.type
            "tx"

    2. Create a new Radar stations:
        The other solution to create a Radar station is to create a brand new station using the 
        default :class:`TX<TX>` constructor.

        .. admonition:: Example

            First import the station module from sorts
            
            >>> from sorts import station

            Then, import the :class:`Beam<pyant.Beam>` from pyant to create the radiation
            pattern of the station's antenna.
            
            >>> from pyant import Beam

            Define the position of the station is defined as follow :
            
            >>> lon = 20.22 # in degrees
            >>> lat = 67.86 # in degrees
            >>> alt = 530.0 # in meters

            Set the station minimal elevation (which defines the FOV of the station)
            
            >>> min_el = 30.0 # in degrees

            Define the station radiation pattern :
            
            >>> az = 0.0 # beam azimuth (degrees)
            >>> el = 0.0 # beam elevation (degrees)
            >>> f = 500e6 # bram frequency at 500MHz
            >>> beam_tx = pyant.Beam(az, el, f)

            Define the properties of the radar pulse :
            
            >>> power = 500e3 # 500kW power
            >>> bandwidth = 10.0 # 10Hz emission bandwidth
            >>> duty_cycle = 0.1 # 10% duty cycle
            >>> pulse_length = 1e-3 # 1ms pulse length
            >>> ipp = 10e-3 # 10ms inter-pulse period
            >>> n_ipp = 20 # 20 IPPs per time slice

            Finally, one can create the station by calling :
            
            >>> station_tx = station.TX(
                    lat, 
                    lon, 
                    alt, 
                    min_el, 
                    beam_tx, 
                    power, 
                    bandwidth, 
                    duty_cycle, 
                    pulse_length=pulse_length, 
                    ipp=ipp, 
                    n_ipp=n_ipp)
            >>> station_tx.type
            "tx"

    Parameters
    ----------
    lat : float
        Geographical latitude of radar station in decimal degrees (North+).
    lon : float
        Geographical longitude of radar station in decimal degrees (East+).
    alt : float
        Geographical altitude above geoid surface of radar station in meter.
    min_elevation: 
        Elevation threshold for the radar station in degrees, i.e. it cannot detect or point below this elevation.
    beam : pyant.Beam 
        Radiation pattern for radar station.
    bandwidth : float
        Emission bandwidth of the transmitted Radar signal.
    duty_cycle : float
        Duty cycle of the :class:`TX<TX>` station.

        The duty cycle is defined as : 
    
    .. math:: D=\\frac{t_{pulse}}{t_{IPP}}


    pulse_length : float
        Length of the Radar pulse.
    ipp : float
        Inter-pulse perdiod (in seconds).
    n_ipp : int
        Number of Inter-pulse perdiod per control time slice.
    """

    PROPERTIES = Station.PROPERTIES + [
        "power",
        "ipp",
        "n_ipp",
        "pulse_length",
        "coh_int_bandwidth",
        "duty_cycle",
        "bandwidth",
        ]
    """ list of properties of a :class:`TX<TX>` Radar station. Those properties can be controlled 
    by Radar controllers (see :ref:`radar_controller<radar_controller>` for more information).

    .. note:: 
        Please note that in the current implementation of sorts, ``PROPERTIES`` is only supposed
        to be modified internally. Therefore do not advise the manual modification of the
        ``PROPERTIES`` list.

        See :func:`~TX.add_property` to obtain more information on how to add new controllable
        Radar properties. 
    """

    def __init__(self, 
                 lat, 
                 lon, 
                 alt, 
                 min_elevation, 
                 beam, 
                 power, 
                 bandwidth, 
                 duty_cycle, 
                 pulse_length=1e-3, 
                 ipp=10e-3, 
                 n_ipp=20, 
                 **kwargs):
        super().__init__(lat, lon, alt, min_elevation, beam)
        """ __init__(self, lat, lon, alt, min_elevation, beam, power, bandwidth, duty_cycle, pulse_length=1e-3, ipp=10e-3, n_ipp=20, **kwargs):

        Default :class:`TX<TX>` class constructor. 

        See :class:`TX<TX>` for more information about the input parameters.
        """

        self.bandwidth = bandwidth
        """ Bandwidth (in Hertz) of the transmitted signal. """
        self.duty_cycle = duty_cycle
        """ Station duty cycle.

        The duty cycle is defined as the ratio : :math:`D=\\frac{t_{pulse}}{t_{IPP}}`, where 
        :math:`t_{pulse}` and :math:`t_{IPP}` represent respectively the duration of the
        pulse and the duration of the inter-pulse period. 
        
        """
        self.power = power
        """ Transmitted Radar pulse power (in seconds). """
        self.pulse_length = pulse_length
        """ Transmitted Radar pulse length (in seconds). """
        self.ipp = ipp
        """ Inter-pulse period duration (in seconds). """

        self.n_ipp = n_ipp
        """ Number of pulses (or inter-pulse periods) per time slice. """
        self.coh_int_bandwidth = 1.0/(pulse_length*n_ipp)
        """ Bandwidth (in Hertz) of the coherently integrated transmitted signal. """

        self._type = "tx"


    def copy(self):
        """ Performs a deep copy of the Radar station. 

        This method is used to perform a copy of a radar station.

        Examples
        --------
        Assuming that we have created a :class:`TX<TX>` station named ``station_old``, the  
        station can be copied by simply calling :

        >>> station_new = station_old.copy()
        """
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
