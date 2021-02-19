#!/usr/bin/env python

'''Wrapper for the Orekit propagator

**Links:**
    * `orekit <https://www.orekit.org/>`_
    * `orekit python <https://gitlab.orekit.org/orekit-labs/python-wrapper>`_
    * `orekit python guide <https://gitlab.orekit.org/orekit-labs/python-wrapper/wikis/Manual-Installation-of-Python-Wrapper>`_
    * `Hipparchus <https://www.hipparchus.org/>`_
    * `orekit 9.3 api <https://www.orekit.org/static/apidocs/index.html>`_
    * `JCC <https://pypi.org/project/JCC/>`_


**Example usage:**

Simple propagation showing time difference due to loading of model data.

.. code-block:: python

    from sorts.propagator import Orekit
 
'''

#Python standard import
import os
import time
import copy
import urllib.request
import pathlib

#Third party import
import numpy as np
import scipy
import scipy.constants
import orekit
import pyorb

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

#Local import
from .base import Propagator
from .. import dates as dates
from .. import frames


orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.frames import FramesFactory
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.orbits import KeplerianOrbit, EquinoctialOrbit, CartesianOrbit, PositionAngle
from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.python import PythonOrekitStepHandler
import org

from orekit import JArray_double


def npdt2absdate(dt, utc):
    '''
    Converts a numpy datetime64 value to an orekit AbsoluteDate
    '''

    year, month, day, hour, minutes, seconds, microsecond = dates.npdt_to_date(dt)
    return AbsoluteDate(
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minutes),
        float(seconds + microsecond*1e-6),
        utc,
    )


def mjd2absdate(mjd, utc):
    '''
    Converts a Modified Julian Date value to an orekit AbsoluteDate
    '''

    return npdt2absdate(dates.mjd_to_npdt(mjd), utc)



class Orekit(Propagator):
    '''Propagator class implementing the Orekit propagator.

    #TODO: Update docs according to new "settings dict" method.
    #TODO: Valdiate Orekit v10 compatibility

    :ivar list solarsystem_perturbers: List of strings of names of objects in the solarsystem that should be used for third body perturbation calculations. All objects listed at `CelestialBodyFactory <https://www.orekit.org/static/apidocs/org/orekit/bodies/CelestialBodyFactory.html>`_ are available.
    :ivar str in_frame: String identifying the input frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :ivar str out_frame: String identifying the output frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :ivar bool drag_force: Should drag force be included in propagation.
    :ivar bool radiation_pressure: Should radiation pressure force be included in propagation.
    :ivar bool frame_tidal_effects: Should coordinate frames include Tidal effects.
    :ivar str integrator: String representing the numerical integrator from the Hipparchus package to use. Any integrator listed at `Hipparchus nonstiff ode <https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/package-summary.html>`_ is available.
    :ivar float minStep: Minimum time step allowed in the numerical orbit propagation given in seconds.
    :ivar float maxStep: Maximum time step allowed in the numerical orbit propagation given in seconds.
    :ivar float position_tolerance: Position tolerance in numerical orbit propagation errors given in meters.
    :ivar str earth_gravity: Gravitation model to use for calculating central acceleration force. Currently avalible options are `'HolmesFeatherstone'` and `'Newtonian'`. See `gravity <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/package-summary.html>`_.
    :ivar tuple gravity_order: A tuple of two integers for describing the order of spherical harmonics used in the `HolmesFeatherstoneAttractionModel <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/HolmesFeatherstoneAttractionModel.html>`_ model.
    :ivar str atmosphere: Atmosphere model used to calculate atmospheric drag. Currently available options are `'DTM2000'`. See `atmospheres <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/package-summary.html>`_.
    :ivar str solar_activity: The model used for calculating solar activity and thereby the influx of solar radiation. Used in the atmospheric drag force model. Currently available options are `'Marshall'` for the `MarshallSolarActivityFutureEstimation <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/data/MarshallSolarActivityFutureEstimation.html>`_.
    :ivar str constants_source: Controls which source for Earth constants to use. Currently avalible options are `'WGS84'` and `'JPL-IAU'`. See `constants <https://www.orekit.org/static/apidocs/org/orekit/utils/Constants.html>`_.
    :ivar float mu: Standard gravitational constant for the Earth.  Definition depend on the :class:`sorts.propagator.Orekit` constructor parameter :code:`constants_source`
    :ivar float R_earth: Radius of the Earth in m. Definition depend on the :class:`sorts.propagator.Orekit` constructor parameter :code:`constants_source`
    :ivar float f_earth: Flattening of the Earth (i.e. :math:`\\frac{a-b}{a}` ). Definition depend on the :class:`sorts.propagator.Orekit` constructor parameter :code:`constants_source`.
    :ivar float M_earth: Mass of the Earth in kg. Definition depend on the :class:`sorts.propagator.Orekit` constructor parameter :code:`constants_source`
    :ivar org.orekit.frames.Frame inputFrame: The orekit frame instance for the input frame.
    :ivar org.orekit.frames.Frame outputFrame: The orekit frame instance for the output frame.
    :ivar org.orekit.frames.Frame inertialFrame: The orekit frame instance for the inertial frame. If inputFrame is pseudo innertial this is the same as inputFrame.
    :ivar org.orekit.bodies.OneAxisEllipsoid body: The model ellipsoid representing the Earth.
    :ivar dict _forces: Dictionary of forces to include in the numerical integration. Contains instances of children of :class:`org.orekit.forces.AbstractForceModel`.
    :ivar list _tolerances: Contains the absolute and relative tolerances calculated by the `tolerances <https://www.orekit.org/static/apidocs/org/orekit/propagation/numerical/NumericalPropagator.html#tolerances(double,org.orekit.orbits.Orbit,org.orekit.orbits.OrbitType)>`_ function.
    :ivar org.orekit.propagation.numerical.NumericalPropagator propagator: The numerical propagator instance.
    :ivar org.orekit.forces.drag.atmosphere.data.MarshallSolarActivityFutureEstimation.StrengthLevel SolarStrengthLevel: The strength of the solar activity. Options are 'AVRAGE', 'STRONG', 'WEAK'.

    The constructor creates a propagator instance with supplied options.

    :param list solarsystem_perturbers: List of strings of names of objects in the solarsystem that should be used for third body perturbation calculations. All objects listed at `CelestialBodyFactory <https://www.orekit.org/static/apidocs/org/orekit/bodies/CelestialBodyFactory.html>`_ are available.
    :param str in_frame: String identifying the input frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :param str out_frame: String identifying the output frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :param bool drag_force: Should drag force be included in propagation.
    :param bool radiation_pressure: Should radiation pressure force be included in propagation.
    :param bool frame_tidal_effects: Should coordinate frames include Tidal effects.
    :param str integrator: String representing the numerical integrator from the Hipparchus package to use. Any integrator listed at `Hipparchus nonstiff ode <https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/package-summary.html>`_ is available.
    :param float min_step: Minimum time step allowed in the numerical orbit propagation given in seconds.
    :param float max_step: Maximum time step allowed in the numerical orbit propagation given in seconds.
    :param float position_tolerance: Position tolerance in numerical orbit propagation errors given in meters.
    :param str atmosphere: Atmosphere model used to calculate atmospheric drag. Currently available options are `'DTM2000'`. See `atmospheres <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/package-summary.html>`_.
    :param str solar_activity: The model used for calculating solar activity and thereby the influx of solar radiation. Used in the atmospheric drag force model. Currently available options are `'Marshall'` for the `MarshallSolarActivityFutureEstimation <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/data/MarshallSolarActivityFutureEstimation.html>`_.
    :param str constants_source: Controls which source for Earth constants to use. Currently avalible options are `'WGS84'` and `'JPL-IAU'`. See `constants <https://www.orekit.org/static/apidocs/org/orekit/utils/Constants.html>`_.
    :param str earth_gravity: Gravitation model to use for calculating central acceleration force. Currently avalible options are `'HolmesFeatherstone'` and `'Newtonian'`. See `gravity <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/package-summary.html>`_.
    :param tuple gravity_order: A tuple of two integers for describing the order of spherical harmonics used in the `HolmesFeatherstoneAttractionModel <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/HolmesFeatherstoneAttractionModel.html>`_ model.
    :param str solar_activity_strength: The strength of the solar activity. Options are 'AVRAGE', 'STRONG', 'WEAK'.
    '''


    class OrekitVariableStep(PythonOrekitStepHandler):
        '''Class for handling the steps.
        '''
        def set_params(self, t, start_date, states_pointer, outputFrame, heartbeat, profiler=None):
            self.t = t
            self.start_date = start_date
            self.states_pointer = states_pointer
            self.outputFrame = outputFrame
            self.profiler = profiler
            self.__heartbeat = heartbeat

        def init(self, s0, t):
            pass

        def handleStep(self, interpolator, isLast):
            if self.profiler is not None:
                self.profiler.start('Orekit:propagate:steps:step-handler')

            state1 = interpolator.getCurrentState()
            state0 = interpolator.getPreviousState()

            t0 = state0.getDate().durationFrom(self.start_date)
            t1 = state1.getDate().durationFrom(self.start_date)

            t_filt = np.logical_and(np.abs(self.t) >= np.abs(t0), np.abs(self.t) <= np.abs(t1))

            for ti, t in zip(np.where(t_filt)[0], self.t[t_filt]):
                if self.profiler is not None:
                    self.profiler.start('Orekit:propagate:steps:step-handler:getState')

                t_date = self.start_date.shiftedBy(float(t))

                _state = interpolator.getInterpolatedState(t_date)

                PVCoord = _state.getPVCoordinates(self.outputFrame)

                x_tmp = PVCoord.getPosition()
                v_tmp = PVCoord.getVelocity()

                self.states_pointer[0,ti] = x_tmp.getX()
                self.states_pointer[1,ti] = x_tmp.getY()
                self.states_pointer[2,ti] = x_tmp.getZ()
                self.states_pointer[3,ti] = v_tmp.getX()
                self.states_pointer[4,ti] = v_tmp.getY()
                self.states_pointer[5,ti] = v_tmp.getZ()

                if self.__heartbeat is not None:
                    self.__heartbeat(float(t), self.states_pointer[:,ti], interpolator=interpolator)

                if self.profiler is not None:
                    self.profiler.stop('Orekit:propagate:steps:step-handler:getState')

            if self.profiler is not None:
                self.profiler.stop('Orekit:propagate:steps:step-handler')

    DEFAULT_SETTINGS = copy.copy(Propagator.DEFAULT_SETTINGS)
    DEFAULT_SETTINGS.update(
        dict(
            in_frame='Orekit-EME',
            out_frame='Orekit-ITRF',
            frame_tidal_effects=False,
            integrator='DormandPrince853',
            min_step=0.001,
            max_step=120.0,
            position_tolerance=10.0,
            earth_gravity='HolmesFeatherstone',
            gravity_order=(10,10),
            solarsystem_perturbers=['Moon', 'Sun'],
            drag_force=True,
            atmosphere='DTM2000',
            radiation_pressure=True,
            solar_activity='Marshall',
            constants_source='WGS84',
            solar_activity_strength='WEAK',
        )
    )


    @staticmethod
    def download_quickstart_data(path, url=None, headers=None, timeout=10.0, block_size=2048, verbose=True):
        if url is None:
            #Standard URL of the example data distributed on orekit.org
            url = 'https://gitlab.orekit.org/orekit/orekit-data/-/archive/master/orekit-data-master.zip'
        if headers is None:
            headers = {}

        pbar_str = f'Downloading from {url[:15]}...{url.split("/")[-1]} [kb]'

        urlopener = urllib.request.build_opener()
        request = urllib.request.Request(url, headers=headers)
        with urlopener.open(request, timeout=timeout) as remote:
            info = remote.info()
            try:
                size = int(info['Content-Length'])
                size_kb = int(size//1024)
            except (KeyError, ValueError, TypeError):
                size = None
                size_kb = None

            if verbose:
                pbar = tqdm(total=size_kb)

            with open(path, 'wb') as f:
                try:
                    bytes_done = 0
                    block = remote.read(block_size)
                    while block:
                        f.write(block)
                        bytes_done += len(block)

                        if verbose:
                            pbar.set_description(pbar_str)
                            pbar.update(int(len(block)//1024))

                        block = remote.read(block_size)

                    if size is not None and bytes_done != size:
                        raise Exception(f'Content-Length {size} did not match download size {bytes_done}, aborting...')
                except BaseException:
                    if path.is_file():
                        try:
                            print('Attempting to remove failed partial download')
                            path.unlink()
                        except:
                            pass
                    raise

    def __init__(self,
                orekit_data,
                settings=None,
                **kwargs
            ):

        super(Orekit, self).__init__(settings=settings, **kwargs)

        if self.logger is not None:
            self.logger.debug(f'sorts.propagator.Orekit:init')
        if self.profiler is not None:
            self.profiler.start('Orekit:init')

        if isinstance(orekit_data, pathlib.Path):
            orekit_data = str(orekit_data)

        setup_orekit_curdir(filename = orekit_data)

        if self.logger is not None:
            self.logger.debug(f'Orekit:init:orekit-data = {orekit_data}')

        self.utc = TimeScalesFactory.getUTC()

        self.__settings = dict()
        self.__settings['SolarStrengthLevel'] = getattr(org.orekit.forces.drag.atmosphere.data.MarshallSolarActivityFutureEstimation.StrengthLevel, self.settings['solar_activity_strength'])
        self._tolerances = None
        
        if self.settings['constants_source'] == 'JPL-IAU':
            self.mu = Constants.JPL_SSD_EARTH_GM
            self.R_earth = Constants.IAU_2015_NOMINAL_EARTH_EQUATORIAL_RADIUS
            self.f_earth = (Constants.IAU_2015_NOMINAL_EARTH_EQUATORIAL_RADIUS - Constants.IAU_2015_NOMINAL_EARTH_POLAR_RADIUS)/Constants.IAU_2015_NOMINAL_EARTH_POLAR_RADIUS
        else:
            self.mu = Constants.WGS84_EARTH_MU
            self.R_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
            self.f_earth = Constants.WGS84_EARTH_FLATTENING

        self.M_earth = self.mu/scipy.constants.G

        self.__params = None

        self._forces = {}

        if self.settings['radiation_pressure']:
            self._forces['radiation_pressure'] = None
        if self.settings['drag_force']:
            self._forces['drag_force'] = None

        if self.settings['earth_gravity'] == 'HolmesFeatherstone':
            provider = org.orekit.forces.gravity.potential.GravityFieldFactory.getNormalizedProvider(self.settings['gravity_order'][0], self.settings['gravity_order'][1])
            holmesFeatherstone = org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel(
                FramesFactory.getITRF(IERSConventions.IERS_2010, True),
                provider,
            )
            self._forces['earth_gravity'] = holmesFeatherstone

        elif self.settings['earth_gravity'] == 'Newtonian':
            Newtonian = org.orekit.forces.gravity.NewtonianAttraction(self.mu)
            self._forces['earth_gravity'] = Newtonian

        else:
            raise Exception('Supplied Earth gravity model "{}" not recognized'.format(self.settings['earth_gravity']))

        if self.settings['solarsystem_perturbers'] is not None:
            for body in self.settings['solarsystem_perturbers']:
                body_template = getattr(CelestialBodyFactory, 'get{}'.format(body))
                body_instance = body_template()
                perturbation = org.orekit.forces.gravity.ThirdBodyAttraction(body_instance)

                self._forces['perturbation_{}'.format(body)] = perturbation

        if self.logger is not None:
            for key in self._forces:
                if self._forces[key] is not None:                
                    self.logger.debug(f'Orekit:init:_forces:{key} = {type(self._forces[key])}')
                else:
                    self.logger.debug(f'Orekit:init:_forces:{key} = None')

        if self.profiler is not None:
            self.profiler.stop('Orekit:init')

    def __str__(self):
        
        ret = ''
        ret += 'Orekit instance @ {}:'.format(hash(self)) + '\n' + '-'*25 + '\n'
        ret += '{:20s}: '.format('Integrator') + self.settings['integrator'] + '\n'
        ret += '{:20s}: '.format('Minimum step') + str(self.settings['min_step']) + ' s' + '\n'
        ret += '{:20s}: '.format('Maximum step') + str(self.settings['max_step']) + ' s' + '\n'
        ret += '{:20s}: '.format('Position Tolerance') + str(self.settings['position_tolerance']) + ' m' + '\n'
        if self._tolerances is not None:
            ret += '{:20s}: '.format('Absolute Tolerance') + str(JArray_double.cast_(self._tolerances[0])) + ' m' + '\n'
            ret += '{:20s}: '.format('Relative Tolerance') + str(JArray_double.cast_(self._tolerances[1])) + ' m' + '\n'
        ret += '\n'
        ret += '{:20s}: '.format('Input frame') + self.settings['in_frame'] + '\n'
        ret += '{:20s}: '.format('Output frame') + self.settings['out_frame'] + '\n'
        ret += '{:20s}: '.format('Gravity model') + self.settings['earth_gravity'] + '\n'
        if self.settings['earth_gravity'] == 'HolmesFeatherstone':
            ret += '{:20s} - Harmonic expansion order {}'.format('', self.settings['gravity_order']) + '\n'
        ret += '{:20s}: '.format('Atmosphere model') + self.settings['atmosphere'] + '\n'
        ret += '{:20s}: '.format('Solar model') + self.settings['solar_activity'] + '\n'
        ret += '{:20s}: '.format('Constants') + self.settings['constants_source'] + '\n'
        ret += 'Included forces:' + '\n'
        for key in self._forces:
            ret += ' - {}'.format(' '.join(key.split('_'))) + '\n'
        ret += 'Third body perturbations:' + '\n'
        for body in self.settings['solarsystem_perturbers']:
            ret += ' - {:}'.format(body) + '\n'
        
        return ret



    def _get_frame(self, name):
        '''Uses a string to identify which coordinate frame to initialize from Orekit package.

        See `Orekit FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_
        '''

        if self.profiler is not None:
            self.profiler.start('Orekit:get_frame')

        if name == 'EME':
            frame = FramesFactory.getEME2000()
        elif name == 'EME2000':
            frame = FramesFactory.getEME2000()
        elif name == 'GCRF':
            frame = FramesFactory.getGCRF()
        elif name == 'CIRF':
            frame = FramesFactory.getCIRF(IERSConventions.IERS_2010, not self.settings['frame_tidal_effects'])
        elif name == 'ITRF':
            frame = FramesFactory.getITRF(IERSConventions.IERS_2010, not self.settings['frame_tidal_effects'])
        elif name == 'TIRF':
            frame = FramesFactory.getTIRF(IERSConventions.IERS_2010, not self.settings['frame_tidal_effects'])
        elif name == 'ITRFEquinox':
            frame = FramesFactory.getITRFEquinox(IERSConventions.IERS_2010, not self.settings['frame_tidal_effects'])
        elif name == 'TEME':
            frame = FramesFactory.getTEME()
        else:
            raise Exception('Frame "{}" not recognized'.format(name))

        if self.profiler is not None:
            self.profiler.stop('Orekit:get_frame')

        return frame

    def _construct_propagator(self, initialOrbit):
        '''
        Get the specified integrator from hipparchus package. List available at: `nonstiff ode <https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/package-summary.html>`_

        Configure the integrator tolerances using the orbit.
        '''

        if self.profiler is not None:
            self.profiler.start('Orekit:propagate:construct_propagator')

        self._tolerances = NumericalPropagator.tolerances(
                self.settings['position_tolerance'],
                initialOrbit,
                initialOrbit.getType()
            )

        integrator_constructor = getattr(
            org.hipparchus.ode.nonstiff,
            '{}Integrator'.format(self.settings['integrator']),
        )

        integrator = integrator_constructor(
            self.settings['min_step'],
            self.settings['max_step'], 
            JArray_double.cast_(self._tolerances[0]),
            JArray_double.cast_(self._tolerances[1]),
        )

        propagator = NumericalPropagator(integrator)
        propagator.setOrbitType(initialOrbit.getType())

        self.propagator = propagator

        if self.profiler is not None:
            self.profiler.stop('Orekit:propagate:construct_propagator')


    def _set_forces(self, A, cd, cr):
        '''Using the spacecraft specific parameters, set the drag force and radiation pressure models.
        
        **See:**
            * `drag <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/package-summary.html>`_
            * `radiation <https://www.orekit.org/static/apidocs/org/orekit/forces/radiation/package-summary.html>`_
        '''

        if self.profiler is not None:
            self.profiler.start('Orekit:propagate:set_forces')

        if self.logger is not None:
            self.logger.debug(f'Orekit:set_forces:A = {A}')
            self.logger.debug(f'Orekit:set_forces:cd = {cd}')
            self.logger.debug(f'Orekit:set_forces:cr = {cr}')

        __params = [A, cd, cr]

        re_calc = True
        if self.__params is None:
            re_calc = True
        else:
            if not np.allclose(np.array(__params,dtype=np.float), self.__params, rtol=1e-3):
                re_calc = True

        if self.logger is not None:
            self.logger.debug(f'Orekit:set_forces:re_calc = {re_calc}')

        self.atmosphere_instance = None
        self.spacecraft_drag_model = None

        if re_calc:        
            self.__params = __params
            self.force_params = {'A': A, 'C_D': cd, 'C_R': cr}
            if self.settings['drag_force']:
                if self.settings['solar_activity'] == 'Marshall':
                    msafe = org.orekit.forces.drag.atmosphere.data.MarshallSolarActivityFutureEstimation(
                        "(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\p{Digit}\\p{Digit}\\p{Digit}\\p{Digit}F10\\.(?:txt|TXT)",
                        self.__settings['SolarStrengthLevel'],
                    )
                    manager = org.orekit.data.DataProvidersManager.getInstance()
                    manager.feed(msafe.getSupportedNames(), msafe)

                    if self.settings['atmosphere'] == 'DTM2000':
                        self.atmosphere_instance = org.orekit.forces.drag.atmosphere.DTM2000(msafe, CelestialBodyFactory.getSun(), self.body)
                    else:
                        raise Exception('Atmosphere model not recognized')

                    self._forces['drag_force'] = self.GetIsotropicDragForce(A, cd)
                else:
                    raise Exception('Solar activity model not recognized')

            if self.settings['radiation_pressure']:
                radiation_pressure_model = org.orekit.forces.radiation.SolarRadiationPressure(
                    CelestialBodyFactory.getSun(),
                    self.body.getEquatorialRadius(),
                    org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient(float(A), float(cr)),
                )

                self._forces['radiation_pressure'] = radiation_pressure_model

            #self.propagator.removeForceModels()

            self.UpdateForces()

        if self.profiler is not None:
            self.profiler.stop('Orekit:propagate:set_forces')
        

    def UpdateForces(self):
        for force_name, force in self._forces.items():
            self.propagator.addForceModel(force)


    def GetIsotropicDragForce(self, A, cd):
        self.spacecraft_drag_model = org.orekit.forces.drag.IsotropicDrag(float(A), float(cd))
        force_ = org.orekit.forces.drag.DragForce(
            self.atmosphere_instance,
            self.spacecraft_drag_model,
        )
        return force_

    def propagate(self, t, state0, epoch, **kwargs):
        '''
        **Implementation:**
    
        Keyword arguments are:

            * float A: Area in m^2
            * float C_D: Drag coefficient
            * float C_R: Radiation pressure coefficient
            * float m: Mass of object in kg
        '''
        if self.profiler is not None:
            self.profiler.start('Orekit:propagate')

        if self.settings['in_frame'].startswith('Orekit-'):
            self.inputFrame = self._get_frame(self.settings['in_frame'].replace('Orekit-', ''))
            orekit_in_frame = True
        else:
            self.inputFrame = self._get_frame('GCRF')
            orekit_in_frame = False

        if self.settings['out_frame'].startswith('Orekit-'):
            self.outputFrame = self._get_frame(self.settings['out_frame'].replace('Orekit-', ''))
            orekit_out_frame = True
        else:
            self.outputFrame = self._get_frame('GCRF')
            orekit_out_frame = False


        if self.inputFrame.isPseudoInertial():
            self.inertialFrame = self.inputFrame
        else:
            self.inertialFrame = FramesFactory.getEME2000()

        self.body = OneAxisEllipsoid(self.R_earth, self.f_earth, self.outputFrame)


        if isinstance(state0, pyorb.Orbit):
            state0_cart = np.squeeze(state0.cartesian)
        else:
            state0_cart = state0

        if self.settings['radiation_pressure']:
            if 'C_R' not in kwargs:
                raise Exception('Radiation pressure force enabled but no coefficient "C_R" given')
        else:
            kwargs['C_R'] = 1.0

        if self.settings['drag_force']:
            if 'C_D' not in kwargs:
                raise Exception('Drag force enabled but no drag coefficient "C_D" given')
            if 'A' not in kwargs:
                raise Exception('Drag force enabled but no area "A" given')
        else:
            if 'C_D' not in kwargs:
                kwargs['C_D'] = 2.3
            if 'A' not in kwargs:
                kwargs['A'] = 1.0

        if 'm' not in kwargs:
            kwargs['m'] = 0.0

        t, epoch = self.convert_time(t, epoch)
        times = epoch + t

        mjd0 = epoch.mjd
        t = t.sec
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        if self.logger is not None:
            self.logger.debug(f'Orekit:propagate:len(t) = {len(t)}')

        if not orekit_in_frame:
            state0_cart = frames.convert(
                epoch, 
                state0_cart, 
                in_frame=self.settings['in_frame'], 
                out_frame='GCRS',
                profiler = self.profiler,
                logger = self.logger,
            )


        initialDate = mjd2absdate(mjd0, self.utc)

        pos = org.hipparchus.geometry.euclidean.threed.Vector3D(float(state0_cart[0]), float(state0_cart[1]), float(state0_cart[2]))
        vel = org.hipparchus.geometry.euclidean.threed.Vector3D(float(state0_cart[3]), float(state0_cart[4]), float(state0_cart[5]))
        PV_state = PVCoordinates(pos, vel)

        if not self.inputFrame.isPseudoInertial():
            transform = self.inputFrame.getTransformTo(self.inertialFrame, initialDate)
            PV_state = transform.transformPVCoordinates(PV_state)

        initialOrbit = CartesianOrbit(
            PV_state,
            self.inertialFrame,
            initialDate,
            self.mu + float(scipy.constants.G*kwargs['m']),
        )

        self._construct_propagator(initialOrbit)
        self._set_forces(kwargs['A'], kwargs['C_D'], kwargs['C_R'])

        initialState = SpacecraftState(initialOrbit)

        self.propagator.setInitialState(initialState)

        tb_inds = t < 0.0
        t_back = t[tb_inds]

        tf_indst = t >= 0.0
        t_forward = t[tf_indst]

        if len(t_forward) == 1:
            if np.any(t_forward == 0.0):
                t_back = t
                t_forward = []
                tb_inds = t <= 0

        state = np.empty((6, len(t)), dtype=np.float)
        step_handler = Orekit.OrekitVariableStep()

        if self.profiler is not None:
            self.profiler.start('Orekit:propagate:steps')

        if self.settings['heartbeat']:
            __heartbeat = self.heartbeat
        else:
            __heartbeat = None

        if len(t_back) > 0:
            _t = t_back
            _t_order = np.argsort(np.abs(_t))
            _t_res = np.argsort(_t_order)
            _t = _t[_t_order]
            _state = np.empty((6, len(_t)), dtype=np.float) 

            step_handler.set_params(_t, initialDate, _state, self.outputFrame, __heartbeat, profiler = self.profiler)

            self.propagator.setMasterMode(step_handler)

            self.propagator.propagate(initialDate.shiftedBy(float(_t[-1])))
            
            #now _state is full and in the order of _t
            state[:, tb_inds] = _state[:, _t_res]

        if len(t_forward) > 0:
            _t = t_forward
            _t_order = np.argsort(np.abs(_t))
            _t_res = np.argsort(_t_order)
            _t = _t[_t_order]
            _state = np.empty((6, len(_t)), dtype=np.float) 
            step_handler.set_params(_t, initialDate, _state, self.outputFrame, __heartbeat, profiler = self.profiler)

            self.propagator.setMasterMode(step_handler)

            self.propagator.propagate(initialDate.shiftedBy(float(_t[-1])))
            
            #now _state is full and in the order of _t
            state[:, tf_indst] = _state[:, _t_res]

        if not orekit_out_frame:
            state = frames.convert(
                times, 
                state, 
                in_frame='GCRS', 
                out_frame=self.settings['out_frame'],
                profiler = self.profiler,
                logger = self.logger,
            )

        if self.profiler is not None:
            self.profiler.stop('Orekit:propagate:steps')
            self.profiler.stop('Orekit:propagate')

        if self.logger is not None:
            self.logger.debug(f'Orekit:propagate:completed')

        return state
