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
import coord


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

def plot_gains(beams, res=1000, min_el = 0.0, alpha = 0.5):
    '''Plot the gain of a list of beam patterns as a function of elevation at :math:`0^\circ` degrees azimuth.
    
    :param list beams: List of instances of :class:`antenna.BeamPattern`.
    :param int res: Number of points to devide the set elevation range into.
    :param float min_el: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`.
    '''

    #turn on TeX interperter
    plt.rc('text', usetex=True)

    
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111)
    
    
    theta=n.linspace(min_el,90.0,num=res)
    
    S=n.zeros((res,len(beams)))
    for b,beam in enumerate(beams):
        for i,th in enumerate(theta):
            k=coord.azel_to_cart(0.0, th, 1.0)
            S[i,b]=beam.gain(k)
    for b in range(len(beams)):
        ax.plot(90-theta,n.log10(S[:,b])*10.0,label="Gain " + beams[b].beam_name, alpha=alpha)
    ax.legend()
    bottom, top = plt.ylim()
    plt.ylim((0,top))
    ax.set_xlabel('Zenith angle [deg]',fontsize=24)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    ax.set_ylabel('Gain $G$ [dB]',fontsize=24)
    ax.set_title('Gain patterns',fontsize=28)

    return fig, ax

def plot_gain_heatmap(beam, res=201, min_el = 0.0, title = None, title_size = 28, ax = None):
    '''Creates a heatmap of the beam-patters as a function of azimuth and elevation in terms of wave vector ground projection coordinates.
    
    :param BeamPattern beam: Beam pattern to plot.
    :param int res: Number of points to devide the wave vector x and y component range into, total number of caluclation points is the square of this number.
    :param float min_el: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`. This number defines the half the length of the square that the gain is calculated over, i.e. :math:`\cos(el_{min})`.
    '''
    #turn on TeX interperter
    plt.rc('text', usetex=True)

    if ax is None:
        fig = plt.figure(figsize=(15,7))
        ax = fig.add_subplot(111)
    else:
        fig = None


    kx=n.linspace(
        beam.on_axis[0] - n.cos(min_el*n.pi/180.0),
        beam.on_axis[0] + n.cos(min_el*n.pi/180.0),
        num=res,
    )
    ky=n.linspace(
        beam.on_axis[1] - n.cos(min_el*n.pi/180.0),
        beam.on_axis[1] + n.cos(min_el*n.pi/180.0),
        num=res,
    )
    
    S=n.zeros((res,res))
    K=n.zeros((res,res,2))
    for i,x in enumerate(kx):
        for j,y in enumerate(ky):
            z2_c = (beam.on_axis[0]-x)**2 + (beam.on_axis[1]-y)**2
            z2 = x**2 + y**2
            if z2_c < n.cos(min_el*n.pi/180.0)**2 and z2 <= 1.0:
                k=n.array([x, y, n.sqrt(1.0 - z2)])
                S[i,j]=beam.gain(k)
            else:
                S[i,j] = 0;
            K[i,j,0]=x
            K[i,j,1]=y
    SdB = n.log10(S)*10.0
    SdB[SdB < 0] = 0
    conf = ax.contourf(K[:,:,0], K[:,:,1], SdB, cmap=cm.plasma, vmin=0, vmax=n.max(SdB))
    ax.set_xlabel('$k_x$ [1]',fontsize=24)
    ax.set_ylabel('$k_y$ [1]',fontsize=24)
    
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    cbar = plt.colorbar(conf, ax=ax)
    cbar.ax.set_ylabel('Gain $G$ [dB]',fontsize=24)
    if title is not None:
        ax.set_title(title + ': ' + beam.beam_name + ' gain pattern', fontsize=title_size)
    else:
        ax.set_title('Gain pattern ' + beam.beam_name, fontsize=title_size)

    return fig, ax

def plot_gain(beam,res=1000,min_el = 0.0):
    '''Plot the gain of a beam patterns as a function of elevation at :math:`0^\circ` degrees azimuth.
    
    :param BeamPattern beam: Beam pattern to plot.
    :param int res: Number of points to devide the set elevation range into.
    :param float min_el: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`.
    '''
    #turn on TeX interperter
    plt.rc('text', usetex=True)

    
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111)
    
    
    theta=n.linspace(min_el,90.0,num=res)
    
    S=n.zeros((res,))
    for i,th in enumerate(theta):
        k=coord.azel_ecef(beam.lat, beam.lon, 0.0, 0, th)
        S[i]=beam.gain(k)

    ax.plot(theta,n.log10(S)*10.0)
    bottom, top = plt.ylim()
    plt.ylim((0,top))
    ax.set_xlabel('Elevation [deg]',fontsize=24)
    ax.set_ylabel('Gain $G$ [dB]',fontsize=24)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    ax.set_title('Gain pattern ' + beam.beam_name,\
        fontsize=28)
    
    plt.show()

def plot_gain3d(beam, res=200, min_el = 0.0):
    '''Creates a 3d plot of the beam-patters as a function of azimuth and elevation in terms of wave vector ground projection coordinates.
    
    :param BeamPattern beam: Beam pattern to plot.
    :param int res: Number of points to devide the wave vector x and y component range into, total number of caluclation points is the square of this number.
    :param float min_el: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`. This number defines the half the length of the square that the gain is calculated over, i.e. :math:`\cos(el_{min})`.
    '''
    #turn on TeX interperter
    plt.rc('text', usetex=True)

    
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111, projection='3d')
    
    
    kx=n.linspace(-n.cos(min_el*n.pi/180.0),n.cos(min_el*n.pi/180.0),num=res)
    ky=n.linspace(-n.cos(min_el*n.pi/180.0),n.cos(min_el*n.pi/180.0),num=res)
    
    S=n.zeros((res,res))
    K=n.zeros((res,res,2))
    for i,x in enumerate(kx):
        for j,y in enumerate(ky):
            z2 = x**2 + y**2
            if z2 < n.cos(min_el*n.pi/180.0)**2:
                k=n.array([x, y, n.sqrt(1.0 - z2)])
                S[i,j]=beam.gain(k)
            else:
                S[i,j] = 0;
            K[i,j,0]=x
            K[i,j,1]=y
    SdB = n.log10(S)*10.0
    SdB[SdB < 0] = 0
    surf = ax.plot_surface(K[:,:,0],K[:,:,1],SdB,cmap=cm.plasma, linewidth=0, antialiased=False, vmin=0, vmax=n.max(SdB))
    #surf = ax.plot_surface(K[:,:,0],K[:,:,1],S.T,cmap=cm.plasma,linewidth=0)
    ax.set_xlabel('$k_x$ [1]',fontsize=24)
    ax.set_ylabel('$k_y$ [1]',fontsize=24)
    ax.set_zlabel('Gain $G$ [dB]',fontsize=24)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    ax.set_title('Gain pattern ' + beam.beam_name,\
        fontsize=28)
    plt.show()



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




class BeamPattern(object):
    '''Defines the radiation pattern of a radar station.
    
   
    :param float I_0: Peak intensity of radiation pattern in linear scale, i.e. the peak gain.
    :param float f: Frequency of radiation pattern.
    :param float az0: Azimuth of pointing direction in dgreees.
    :param float el0: Elevation of pointing direction in degrees.
    :attr numpy.array on_axis: Cartesian vector in ECEF describing pointing direction.
    :param function gain_func: Function describing gain as a function of incoming wave vector direction.
    :param str beam_name: Name of the radiation pattern model.
    

    :ivar float I_0: Peak intensity of radiation pattern in linear scale, i.e. the peak gain.
    :ivar float f: Frequency of radiation pattern.
    :ivar float az0: Azimuth of pointing direction in dgreees.
    :ivar float el0: Elevation of pointing direction in degrees.
    :ivar numpy.array on_axis: Cartesian vector in local coordinates describing pointing direction.
    :ivar function gain_func: Function describing gain as a function of incoming wave vector direction.
    :ivar str beam_name: Name of the radiation pattern model.
    
    '''


    def __init__(self, gain_func, az0, el0, I_0, f, beam_name=''):

        self.I_0 = I_0
        self.f = f
        self.az0 = az0
        self.el0 = el0
        self.on_axis = coord.azel_to_cart(az0, el0, 1.0)
        self.gain_func = gain_func
        self.beam_name = beam_name


    def copy(self):
        '''Return a copy of the current instance.
        '''

        beam = BeamPattern(
            gain_func=self.gain_func,
            az0=self.az0,
            el0=self.el0,
            I_0=self.I_0,
            f=self.f,
            beam_name=self.beam_name)

        #add any additional attributes created by the model.
        attrs_self = dir(self)
        attrs_copy = dir(beam)
        for attr in attrs_self:
            if attr not in attrs_copy:
                setattr(beam, attr, copy.deepcopy(getattr(self, attr)))

        return beam


    def point(self, az0, el0):
        '''Point beam towards azimuth and elevation coordinate.
        
            :param float az0: Azimuth of pointing direction in dgreees east of north.
            :param float el0: Elevation of pointing direction in degrees from horizon.
        '''
        self.az0 = az0
        self.el0 = el0
        self.on_axis = coord.azel_to_cart(az0, el0, 1.0)


    def point_k0(self, k0):
        '''Point beam in local direction.
        
            :param numpy.ndarray k0: Pointing direction in local coordinates.
        '''
        self.on_axis = k0/n.linalg.norm(k0)
        az0, el0, r0 = coord.cart_to_azel(self.on_axis)
        self.az0 = az0
        self.el0 = el0
        

    def angle(self, az, el):
        '''Get angle between azimuth and elevation and pointing direction.
        
            :param float az: Azimuth in dgreees east of north to measure from.
            :param float el: Elevation in degrees from horizon to measure from.
            
            :return: Angle in degrees.
            :rtype: float
        '''
        direction = coord.azel_to_cart(az, el, 1.0)
        return coord.angle_deg(self.on_axis, direction)

    def angle_k(self, k):
        '''Get angle between azimuth and elevation and pointing direction.
        
            :param numpy.array k: Direction to evaluate angle to.

            :return: Angle in degrees.
            :rtype: float
        '''
        return coord.angle_deg(self.on_axis, k)


    def gain(self,k):
        '''Return the gain using the gain-function. The gain function may change gain result for a specific direction based on the instance state, i.e. pointing direction.
        
        :param numpy.array k: Direction in local coordinates to evaluate gain in.
        
        :return float: Gain evaluated using current configuration.
        '''
        return self.gain_func(k, self)
