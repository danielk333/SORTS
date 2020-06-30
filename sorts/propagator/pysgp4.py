#!/usr/bin/env python

'''rapper for the SGP4 propagator

'''

#Python standard import


#Third party import
import numpy as np

import sgp4
import sgp4.earth_gravity
import sgp4.io
import sgp4.propagation
import sgp4.model

#Local import
from .base import Propagator
from .. import dates
from .. import frames


class SGP4_module_wrapper:
    """
    The SGP4 class acts as a wrapper around the sgp4 module
    uploaded by Brandon Rhodes (http://pypi.python.org/pypi/sgp4/).

        
    It converts orbital elements into the TLE-like 'satellite'-structure which
    is used by the module for the propagation.

    Notes:
        The class can be directly used for propagation. Alternatively,
        a simple propagator function is provided below.
    """


    # Geophysical constants (WGS 72 values) for notational convinience
    WGS     = sgp4.earth_gravity.wgs72     # Model used within SGP4
    R_EARTH = WGS.radiusearthkm            # Earth's radius [km]
    GM      = WGS.mu                       # Grav.coeff.[km^3/s^2]
    RHO0    = 2.461e-8                     # Density at q0[kg/m^3]
    Q0      = 120.0                        # Reference height [km]
    S0      = 78.0                         # Reference height [km]

    # Time constants

    MJD_0 = 2400000.5

    def __init__(self, mjd_epoch, mean_elements, B):
        """
        Initialize SGP4 object from mean orbital elements and
        ballistic coefficient

        Creates a sgp4.model.Satellite object for mean element propagation
        First all units are converted to the ones used for TLEs, then
        they are modified to the sgp4 module standard.
        
        Input
        -----
        mjd_epoch     : epoch as Modified Julian Date (MJD)
        mean_elements : [a0,e0,i0,raan0,aop0,M0]
        B             : Ballistic coefficient ( 0.5*C_D*A/m )
        
        Remarks
        -------
        
        This object is not usable for TLE generation, but only for propagation
        as internal variables are modified by sgp4init.
        """

        a0     = mean_elements[0]         # Semi-major (a') at epoch [km]
        e0     = mean_elements[1]         # Eccentricity at epoch
        i0     = mean_elements[2]         # Inclination at epoch
        raan0  = mean_elements[3]         # RA of the ascending node at epoch
        aop0   = mean_elements[4]         # Argument of perigee at epoch
        M0     = mean_elements[5]         # Mean anomaly at epoch
    
        # Compute ballistic coefficient
        bstar  = 0.5*B*SGP4_module_wrapper.RHO0 # B* in [1/m]
        n0 = np.sqrt(SGP4_module_wrapper.GM) / (a0**1.5)
        
        # Scaling
        n0    = n0*(86400.0/(2*np.pi))          # Convert to [rev/d]
        bstar = bstar*(SGP4_module_wrapper.R_EARTH*1000.0)     # Convert from [1/m] to [1/R_EARTH]
    
        # Compute year and day of year
        d   = mjd_epoch - 16480.0               # Days since 1904 Jan 1.0
        y   = int(int(d) / 365.25)                # Number of years since 1904
        doy = d - int(365.25*y)                 # Day of year
        if (y%4==0):
            doy+=1.0
                    
        # Create Satellite object and fill member variables
        sat = sgp4.model.Satellite()
        #Unique satellite number given in the TLE file.
        sat.satnum = 12345
        #Full four-digit year of this element set's epoch moment.
        sat.epochyr = 1904+y
        #Fractional days into the year of the epoch moment.
        sat.epochdays = doy
        #Julian date of the epoch (computed from epochyr and epochdays).
        sat.jdsatepoch = mjd_epoch + SGP4_module_wrapper.MJD_0
        
        #First time derivative of the mean motion (ignored by SGP4).
        #sat.ndot
        #Second time derivative of the mean motion (ignored by SGP4).
        #sat.nddot
        #Ballistic drag coefficient B* in inverse earth radii.
        sat.bstar = bstar
        #Inclination in radians.
        sat.inclo = i0
        #Right ascension of ascending node in radians.
        sat.nodeo = raan0
        #Eccentricity.
        sat.ecco = e0
        #Argument of perigee in radians.
        sat.argpo = aop0
        #Mean anomaly in radians.
        sat.mo = M0
        #Mean motion in radians per minute.
        sat.no = n0 / ( 1440.0 / (2.0 *np.pi) )
        #
        sat.whichconst = SGP4_module_wrapper.WGS
        
        sat.a = pow( sat.no*SGP4_module_wrapper.WGS.tumin , (-2.0/3.0) )
        sat.alta = sat.a*(1.0 + sat.ecco) - 1.0
        sat.altp = sat.a*(1.0 - sat.ecco) - 1.0
    
        sgp4.propagation.sgp4init(SGP4_module_wrapper.WGS, 'i', \
            sat.satnum, sat.jdsatepoch-2433281.5, sat.bstar,\
            sat.ecco, sat.argpo, sat.inclo, sat.mo, sat.no,\
            sat.nodeo, sat)

        # Store satellite object and epoch
        self.sat       = sat
        self.mjd_epoch = mjd_epoch
        
    def state(self, mjd):
        """
        Inertial position and velocity ([m], [m/s]) at epoch mjd
        

        :param float mjd: epoch where satellite should be propagated to
        
        """
        # minutes since reference epoch
        m = (mjd - self.mjd_epoch) * 1440.0
        r,v = sgp4.propagation.sgp4(self.sat, m)
        # print(r,v)
        return np.hstack((np.array(r),np.array(v)))*1e3



def sgp4_propagation( mjd_epoch, mean_elements, B=0.0, dt=0.0, method=None):
    """
    Lazy SGP4 propagation using SGP4 class
    
    Create a satellite object from mean elements and propagate it

    :param list/numpy.ndarray mean_elements : [a0,e0,i0,raan0,aop0,M0]
    :param float B: Ballistic coefficient ( 0.5*C_D*A/m )
    :param float dt: Time difference w.r.t. element epoch in seconds
    :param float mjd_epoch: Epoch of elements as Modified Julian Date (MJD) Can be ignored if the exact epoch is unimportant.
    :param str method: Forces use of SGP4 or SDP4 depending on string 'n' or 'd'
    
    """
    mjd_ = mjd_epoch + dt / 86400.0
    obj = SGP4_module_wrapper(mjd_epoch, mean_elements, B)
    if method is not None and method in ['n', 'd']:
        obj.sat.method = method
    return obj.state(mjd_)
    


class SGP4(Propagator):
    '''Propagator class implementing the SGP4 propagator.

    :ivar bool polar_motion: Determines if polar motion should be used in calculating ITRF frame.
    :ivar str out_frame: String identifying the output frame. Options are 'ITRF' or 'TEME'.

    :param bool polar_motion: Determines if polar motion should be used in calculating ITRF frame.
    :param str out_frame: String identifying the output frame. Options are 'ITRF' or 'TEME'.
    '''

    DEFAULT_SETTINGS = dict(
        out_frame = 'ITRF',
        polar_motion = False,
        polar_motion_model = '80',
    )


    def _check_settings(self):
        for key_s, val_s in self.settings.items():
            if key_s not in SGP4.DEFAULT_SETTINGS:
                raise KeyError('Setting "{}" does not exist'.format(key_s))
            if type(SGP4.DEFAULT_SETTINGS[key_s]) != type(val_s):
                raise ValueError('Setting "{}" does not support "{}"'.format(key_s, type(val_s)))


    def __init__(self, settings=None, **kwargs):
        super(SGP4, self).__init__(**kwargs)

        if self.logger is not None:
            self.logger.info(f'sorts.propagator.SGP4:init')

        self.settings.update(SGP4.DEFAULT_SETTINGS)
        if settings is not None:
            self.settings.update(settings)
            self._check_settings()

        if self.logger is not None:
            for key in self.settings:
                self.logger.debug(f'SGP4:settings:{key} = {self.settings[key]}')


    def propagate(self, t, state0, mjd0, **kwargs):
        '''Propagate a state

        All state-vector are given in SI units.

        Keyword arguments contain only information needed for ballistic coefficient :code:`B` used by SGP4. Either :code:`B` or :code:`C_D`, :code:`A` and :code:`m` must be supplied.
        They also contain a option to give angles in radians or degrees. By default input is assumed to be degrees.

        **Frame:**

        The input frame is ECI (TEME) for orbital elements and Cartesian. The output frame is as standard ECEF (ITRF). But can be set to TEME.

        :param float/list/numpy.ndarray t: Time in seconds to propagate relative the initial state epoch.
        :param float mjd0: The epoch of the initial state in fractional Julian Days.
        :param numpy.ndarray state0: 6-D Cartesian state vector in SI-units.
        :param float B: Ballistic coefficient
        :param float C_D: Drag coefficient
        :param float A: Cross-sectional Area
        :param float m: Mass
        :param bool degrees: If false, all angles are assumed to be in radians.
        :return: 6-D Cartesian state vectors in SI-units.

        '''
        if self.profiler is not None:
            self.profiler.start('SGP4-propagate')

        t = self._make_numpy(t)

        if self.logger is not None:
            self.logger.info(f'SGP4:propagate:len(t) = {len(t)}')

        if 'B' in kwargs:
            B = kwargs['B']
        else:
            B = 0.5*kwargs.get('C_D',2.3)*kwargs.get('A',1.0)/kwargs.get('m',1.0)

        if self.logger is not None:
            self.logger.debug(f'SGP4:propagate:B = {B}')

        mean_elements = frames.TEME_to_TLE(state0, mjd0=mjd0, kepler=False)

        if np.any(np.isnan(mean_elements)):
            raise Exception('Could not compute SGP4 initial state: {}'.format(mean_elements))

        # Create own SGP4 object
        obj = SGP4_module_wrapper(mjd0, mean_elements, B)

        mjdates = mjd0 + t/86400.0
        pos=np.zeros([3,t.size])
        vel=np.zeros([3,t.size])

        for mi,mjd in enumerate(mjdates):
            if self.profiler is not None:
                self.profiler.start('SGP4-propagate-step')

            y = obj.state(mjd)
            pos[:,mi] = y[:3]
            vel[:,mi] = y[3:]

            if self.profiler is not None:
                self.profiler.stop('SGP4-propagate-step')

        if self.settings['out_frame'] == 'TEME':
            states=np.empty((6,t.size), dtype=np.float)
            states[:3,:] = pos
            states[3:,:] = vel
            
        elif self.settings['out_frame'] == 'ITRF':
            if self.settings['polar_motion']:
                PM_data = frames.get_polar_motion(dates.mjd_to_jd(mjdates))
                xp = PM_data[:,0]
                xp.shape = (1,xp.size)
                yp = PM_data[:,1]
                yp.shape = (1,yp.size)

            else:
                xp = 0.0
                yp = 0.0

            states = frames.TEME_to_ECEF(t, pos, vel, mjd0=mjd0, xp=xp, yp=yp , model=self.settings['polar_motion_model'])
        else:
            raise Exception('Output frame {} not found'.format(self.out_frame))

        if self.profiler is not None:
            self.profiler.stop('SGP4-propagate')
        if self.logger is not None:
            self.logger.info(f'SGP4:propagate:completed')

        return states
