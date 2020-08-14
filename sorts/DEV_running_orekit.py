
"""
Petri's notebooks don't all work unchanged for Orekit 9.x,
trying to see what it must look like to work with 10.x
"""


################################################################################
# https://gitlab.orekit.org/orekit-labs/python-wrapper/-/blob/master/examples/The_Basics.ipynb

from numpy import radians, degrees, pi
from matplotlib.pyplot import plot, legend, ylabel, xlabel, ylim

from math import radians, pi
import matplotlib.pyplot as plt

import orekit
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime

vm = orekit.initVM()
setup_orekit_curdir('/home/tom/Downloads/orekit-data-master.zip')

from org.orekit.utils import Constants

print(Constants.WGS84_EARTH_EQUATORIAL_RADIUS)


from org.orekit.bodies import CelestialBodyFactory
from org.orekit.utils import PVCoordinatesProvider
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory

sun = CelestialBodyFactory.getSun()     # Here we get it as an CelestialBody
sun = PVCoordinatesProvider.cast_(sun)  # But we want the PVCoord interface

# Get the position of the sun in the Earth centered coordinate system EME2000
date = AbsoluteDate(2020, 2, 28, 23, 30, 0.0, TimeScalesFactory.getUTC())
sun.getPVCoordinates(date, FramesFactory.getEME2000()).getPosition()

from org.hipparchus.geometry.euclidean.threed import Vector3D
a = Vector3D(2.0, 0.0, 0.0)
print(a.getX(), a.getY(), a.getZ())
print(a.x, a.y, a.z)


################################################################################
# https://gitlab.orekit.org/orekit-labs/python-wrapper/-/blob/master/examples/Time.ipynb
from org.orekit.time import TimeScalesFactory, AbsoluteDate

utc = TimeScalesFactory.getUTC()
start = AbsoluteDate(2005, 12, 31, 23, 59, 59.0, utc)
stop  = AbsoluteDate(2006,  1,  1,  0,  0,  0.0, utc)

print (stop.offsetFrom(start, utc))
print (stop.durationFrom(start))


utc = TimeScalesFactory.getUTC()
referenceDate = AbsoluteDate(2005, 12, 31, 23, 59, 59.0, utc)
date1         =  AbsoluteDate(referenceDate, 1.0, utc)
date2         =  AbsoluteDate(referenceDate, 2.0)

print (date1, date2)

tt = TimeScalesFactory.getTT()
mjd = AbsoluteDate.MODIFIED_JULIAN_EPOCH

print ('TTScale', mjd.toString(tt))
print ('UTC    ', mjd.toString(utc))


utc = TimeScalesFactory.getUTC()
tai = TimeScalesFactory.getTAI()

timerange = range(1957,2019)

utc_date = [AbsoluteDate(t, 1,  1, 0, 0, 0.0, utc) for t in timerange]
tai_date = [AbsoluteDate(t, 1,  1, 0, 0, 0.0, tai) for t in timerange]

diff_utc = [t1.durationFrom(t2) for t1, t2 in zip(tai_date, utc_date)]
# diff_tt = [t1.durationFrom(t2) for t1, t2 in zip(tai_date, utc_date)]

plot(timerange, diff_utc, label='utc offset to TAI Scale', drawstyle='steps-mid')

legend()
ylabel('TAI - UTC in seconds at 1st Jan of year'); ylim(-40,5)
xlabel('year');

################################################################################
# https://gitlab.orekit.org/orekit-labs/python-wrapper/-/blob/master/examples/Orbit_Definition.ipynb

from org.orekit.orbits import KeplerianOrbit, PositionAngle, OrbitType
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants
from org.orekit.time import AbsoluteDate

k_orbit = KeplerianOrbit(24464560.0, # Semimajor Axis (m)
                0.7311,    # Eccentricity
                0.122138,  # Inclination (radians)
                3.10686,   # Perigee argument (radians)
                1.00681,   # Right ascension of ascending node (radians)
                0.048363,  # Anomaly (rad/s)
                PositionAngle.MEAN,  # Sets which type of anomaly we use
                FramesFactory.getEME2000(), # The frame in which the parameters are defined (must be a pseudo-inertial frame)
                AbsoluteDate.J2000_EPOCH,   # Sets the date of the orbital parameters
                Constants.WGS84_EARTH_MU)   # Sets the central attraction coefficient (m³/s²)

k_orbit.getType()


# Convert between different orbit types

OrbitType.CARTESIAN.convertType(k_orbit)

OrbitType.EQUINOCTIAL.convertType(k_orbit)

OrbitType.CIRCULAR.convertType(k_orbit)


from org.orekit.frames import TopocentricFrame, FramesFactory
from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint
from org.orekit.utils import IERSConventions

earthFrame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        earthFrame)

longitude = float(radians(21.063))              # need 'float' here in my installation, don't know why
latitude  = float(radians(67.878))
altitude  = 341.0
station_point = GeodeticPoint(latitude, longitude, altitude)
station_frame = TopocentricFrame(earth, station_point, "Esrange")


# The following then obtains the state for the satellite in the station frame:
k_orbit.getPVCoordinates(station_frame)



################################################################################
# https://gitlab.orekit.org/orekit-labs/python-wrapper/-/blob/master/examples/Frames_and_Coordinate_Systems.ipynb

from org.hipparchus.geometry.euclidean.threed import Vector3D, SphericalCoordinates

from org.orekit.data import DataProvidersManager, ZipJarCrawler
from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint, CelestialBodyFactory
from org.orekit.time import TimeScalesFactory, AbsoluteDate, DateComponents, TimeComponents
from org.orekit.utils import IERSConventions, Constants, PVCoordinates, PVCoordinatesProvider # , AbsolutePVCoordinates

from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from java.io import File

from math import radians, pi, degrees
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sun_frame = CelestialBodyFactory.getSun().getBodyOrientedFrame()

print(sun_frame, '\n',
      sun_frame.getParent(), '\n',
      sun_frame.getParent().getParent(), '\n',
      sun_frame.getParent().getParent().getParent(), '\n',
      sun_frame.getParent().getParent().getParent().getParent(),'\n',
)

icrf_frame = FramesFactory.getICRF()

icrf_frame.getParent().getParent()

eme_frame = FramesFactory.getEME2000()


position = Vector3D(3220103., 69623., 6449822.)
velocity = Vector3D(6414.7, -2006., -3180.)
pv_eme = PVCoordinates(position, velocity)
initDate = AbsoluteDate.J2000_EPOCH.shiftedBy(584.)


# ITRF

# getITRF(IERSConventions conventions, boolean simpleEOP)

ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

ITRF

p2 = eme_frame.getTransformTo(ITRF, initDate).transformPVCoordinates(pv_eme)

p3 = ITRF.getTransformTo(eme_frame, initDate).transformPVCoordinates(p2)


# Topocentric frame
# (local horizon on surface)

# As above, earthFrame .. station_frame

pv_topo = eme_frame.getTransformTo(station_frame, initDate).transformPVCoordinates(pv_eme)


pv_topo.getPosition()


# Calculate the elevation, azimuth and range to the Moon, the Sun and Mars, as
# seen from a local observer on the Earth, expressed in topocentric spherical
# coordinates.

sun = CelestialBodyFactory.getSun()

sun_pv = PVCoordinatesProvider.cast_(sun).getPVCoordinates(initDate, eme_frame)

sun_pv

# Orbit example

import math

from org.orekit.orbits import KeplerianOrbit, PositionAngle
from org.orekit.propagation.analytical import KeplerianPropagator



rp = 400 * 1000                 #  Perigee      (had to switch around rp and ra from notebook)
ra = 2000 * 1000                #  Apogee
i = math.radians(90.0)          # inclinationa
omega = math.radians(90.0)      # perigee argument
raan = math.radians(0.0)        # right ascension of ascending node
lv = math.radians(0.0)          # True anomaly

epochDate = AbsoluteDate(2016, 1, 1, 0, 0, 00.000, utc)
initialDate = epochDate

a = (rp + ra + 2 * Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / 2.0
e = 1.0 - (rp + Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / a

## Inertial frame where the satellite is defined
inertialFrame = FramesFactory.getEME2000()

## Orbit construction as Keplerian
initialOrbit = KeplerianOrbit(a, e, i, omega, raan, lv, PositionAngle.TRUE,
        inertialFrame, epochDate, Constants.WGS84_EARTH_MU)

propagator = KeplerianPropagator(initialOrbit)

el=[]
pv=[]
t = []
s = []

extrapDate = initialDate;
finalDate = extrapDate.shiftedBy(60.0*60*24*1) #seconds

while (extrapDate.compareTo(finalDate) <= 0.0):
    s.append(propagator.propagate(extrapDate))
    t.append(extrapDate)
    extrapDate = extrapDate.shiftedBy(10.0)

x_inert  = [tmp.getPVCoordinates().getPosition().getX()/1000 for tmp in s]
y_inert  = [tmp.getPVCoordinates().getPosition().getY()/1000 for tmp in s]

plt.plot(x_inert,y_inert)
plt.axes().set_aspect('equal', 'datalim')


# In Earth frame:

target_frame = earthFrame
x_earth = []
y_earth = []

for time, tmp_s in zip(t,s):
    trans = inertialFrame.getTransformTo(target_frame, time)
    pos = trans.transformPosition(tmp_s.getPVCoordinates().getPosition())
    x_earth.append(pos.getX()/1000)
    y_earth.append(pos.getY()/1000)

plt.plot(x_earth,y_earth);
plt.axes().set_aspect('equal', 'datalim')

# Sun oriented inertial frame:

sun = CelestialBodyFactory.getSun()
target_frame = sun.getInertiallyOrientedFrame()

x_earth = []
y_earth = []

for tmp_t, tmp_s in zip(t,s):
    trans = inertialFrame.getTransformTo(target_frame, tmp_t)
    pos = trans.transformPosition(tmp_s.getPVCoordinates().getPosition())
    x_earth.append(pos.getX()/1000)
    y_earth.append(pos.getY()/1000)

plt.plot(x_earth,y_earth)

# Moon centered moon body oriented frame

moon = CelestialBodyFactory.getMoon()
target_frame = moon.getBodyOrientedFrame()

x_earth = []
y_earth = []

for tmp_t, tmp_s in zip(t,s):
    trans = inertialFrame.getTransformTo(target_frame, tmp_t)
    pos = trans.transformPosition(tmp_s.getPVCoordinates().getPosition())
    x_earth.append(pos.getX()/1000)
    y_earth.append(pos.getY()/1000)

plt.plot(x_earth,y_earth)
plt.axes().set_aspect('equal', 'datalim')

# AbsolutePVCoordinates not in  orekit 9.3.1
# pv = AbsolutePVCoordinates(eme_frame, initDate, position, velocity)
# apv
#
# apv.getPVCoordinates(station_frame)
#
# station_frame.getElevation(apv.getPosition(), apv.getFrame(), apv.getDate())


################################################################################
# https://gitlab.orekit.org/orekit-labs/python-wrapper/-/blob/master/examples/TLE_Propagation.ipynb


from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint
from org.orekit.time import TimeScalesFactory, AbsoluteDate, DateComponents, TimeComponents
from org.orekit.utils import IERSConventions, Constants

from org.orekit.propagation.analytical.tle import TLE, TLEPropagator



tle_line1 = "1 27421U 02021A   02124.48976499 -.00021470  00000-0 -89879-2 0    20"
tle_line2 = "2 27421  98.7490 199.5121 0001333 133.9522 226.1918 14.26113993    62"

mytle = TLE(tle_line1,tle_line2)

# ITRF, earth, station_frame, inertialFrame as above
propagator = TLEPropagator.selectExtrapolator(mytle)


extrapDate = AbsoluteDate(2002, 5, 7, 12, 0, 0.0, TimeScalesFactory.getUTC())
finalDate = extrapDate.shiftedBy(60.0*60*24) #seconds

el=[]
pos=[]

while (extrapDate.compareTo(finalDate) <= 0.0):
    pv = propagator.getPVCoordinates(extrapDate, inertialFrame)
    pos_tmp = pv.getPosition()
    pos.append((pos_tmp.getX(),pos_tmp.getY(),pos_tmp.getZ()))

    el_tmp = station_frame.getElevation(pv.getPosition(),
                    inertialFrame,
                    extrapDate)*180.0/pi
    el.append(el_tmp)
    #print extrapDate, pos_tmp, vel_tmp
    extrapDate = extrapDate.shiftedBy(10.0)

plt.plot(el)
plt.ylim(0,90)
plt.title('Elevation')
plt.grid(True)


# Evaluate the maximum elevation of the International Space Station from your
# current location and for the coming 5 days, and at what time and date it
# occurs. The Two-Line Elements needs to be fresh, and fetched from internet. One
# site that is publishing recent elements is Celestrack. Verify that the used TLE
# is fresh using the epoch.



# Propagation
# https://gitlab.orekit.org/orekit-labs/python-wrapper/-/blob/master/examples/Propagation.ipynb


from org.orekit.orbits import KeplerianOrbit, PositionAngle
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants
from org.orekit.frames import FramesFactory

ra = 400 * 1000         #  Apogee
rp = 500 * 1000         #  Perigee
i = radians(87.0)      # inclination
omega = radians(20.0)   # perigee argument
raan = radians(10.0)  # right ascension of ascending node
lv = radians(0.0)    # True anomaly

epochDate = AbsoluteDate(2020, 1, 1, 0, 0, 00.000, utc)
initialDate = epochDate

a = (rp + ra + 2 * Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / 2.0
e = 1.0 - (rp + Constants.WGS84_EARTH_EQUATORIAL_RADIUS) / a

## Inertial frame where the satellite is defined
inertialFrame = FramesFactory.getEME2000()

## Orbit construction as Keplerian
initialOrbit = KeplerianOrbit(a, e, i, omega, raan, lv,
                              PositionAngle.TRUE,
                              inertialFrame, epochDate, Constants.WGS84_EARTH_MU)

propagator = KeplerianPropagator(initialOrbit)

# Initial state -- no pretty-printing of state in Orekit 9.3.  Must use st.getOrbit() accessor
st0 = propagator.getInitialState()

st0.getOrbit()

# 48 hours later ...
st48 = propagator.propagate(initialDate, initialDate.shiftedBy(3600.0 * 48))

st48.getOrbit()


# Eckstein-Hechler Propagator

# The EH propagator is only applicable for near circular orbits, typically used for LEO satellites.


from org.orekit.propagation.analytical import EcksteinHechlerPropagator
from org.orekit.orbits import OrbitType

propagator_eh = EcksteinHechlerPropagator(initialOrbit,
                                        Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS,
                                        Constants.EIGEN5C_EARTH_MU, Constants.EIGEN5C_EARTH_C20,
                                        Constants.EIGEN5C_EARTH_C30, Constants.EIGEN5C_EARTH_C40,
                                        Constants.EIGEN5C_EARTH_C50, Constants.EIGEN5C_EARTH_C60)
st_eh0 = propagator_eh.getInitialState()

OrbitType.CARTESIAN.convertType(st_eh0.getOrbit())

# 48 hours later ...
end_state = propagator_eh.propagate(initialDate, initialDate.shiftedBy(3600.0 * 48))

OrbitType.KEPLERIAN.convertType(end_state.getOrbit())




# Numerical Propagators

# LEO propagation parameters

from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.utils import IERSConventions
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel

from orekit import JArray_double

minStep = 0.001
maxstep = 1000.0
initStep = 60.0

positionTolerance = 1.0

tolerances = NumericalPropagator.tolerances(positionTolerance,
                                            initialOrbit,
                                            initialOrbit.getType())

integrator = DormandPrince853Integrator(minStep, maxstep,
    JArray_double.cast_(tolerances[0]),  # Double array of doubles needs to be casted in Python
    JArray_double.cast_(tolerances[1]))
integrator.setInitialStepSize(initStep)

satellite_mass = 100.0  # The models need a spacecraft mass, unit kg.
initialState = SpacecraftState(initialOrbit, satellite_mass)

initialState.getOrbit()

OrbitType.CARTESIAN.convertType(initialState.getOrbit())


propagator_num = NumericalPropagator(integrator)
propagator_num.setOrbitType(OrbitType.CARTESIAN)
propagator_num.setInitialState(initialState)


gravityProvider = GravityFieldFactory.getNormalizedProvider(10, 10)
propagator_num.addForceModel(HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider))


end_state.getOrbit()

OrbitType.KEPLERIAN.convertType(end_state.getOrbit())  # Note that this is the Osculating orbit!



##################### ##################### ##################### #####################
##  Using Orekit propagators from given ITRF precision orbit statevectors
##################### ##################### ##################### #####################

from org.orekit.orbits import CartesianOrbit
from sorts import dates

def absolutedate_from_npdt(dt, scale=TimeScalesFactory.getUTC()):
    if isinstance(scale, str):
        # 'utc' => TimeScalesFactory.getUTC(), etc
        try:
            scale = getattr(TimeScalesFactory, 'get' + scale.upper())()
        except AttributeError:
            raise NameError(f'Time scale {scale} unknown')
    yy, mm, dd, HH, MM, SS, usec = dates.npdt_to_date(dt)
    # float() again necessary
    return AbsoluteDate(yy, mm, dd, HH, MM, float(SS + usec*1e-6), scale)


def plot4(t0, tt, res, prop, lab=''):
    # t0 - AbsoluteDate for propagator's initial state
    # tt - time offsets for propagated states (usu. 10 sec increments)
    # res  - array of ITRF statevectors (from e.g. resituted orbit files)
    # prop - propagator
    fh, ax = plt.subplots(2, 2, sharex='col')


def propagate_t(t0, tt, prop, pframe=None):
    """
    Propagate, and return PVCoordinates in terrestric (ITRF) frame
    """
    if pframe is None:
        pframe = FramesFactory.getEME2000()
    ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    return 





# Initialize CARTESIAN orbit from a given ITRF statevector

# read_statevectors() from tests/test_frames.py
# resorb = read_statevectors('tests/S1A_OPER_AUX_RESORB_OPOD_20200721T073339_V20200721T023852_20200721T055622.EOF')


# Constructor from any kind of orbital parameters.
# CartesianOrbit(PVCoordinates pvaCoordinates, Frame frame, AbsoluteDate date, double mu)

pos = Vector3D(tuple(float(x) for x in resorb[0].POS))
vel = Vector3D(tuple(float(x) for x in resorb[0].VEL))

# ITRF statevector
pvc = PVCoordinates(pos, vel)   # Not tied to specific frame

# svd = AbsoluteDate(2004, 1, 1, 23, 30, 00.000, TimeScalesFactory.getUTC())
svd = absolutedate_from_npdt(resorb[0].UTC)

# Transform ITRF statevector coordinates to (inertial) EME frame
pvi = ITRF.getTransformTo(eme_frame, svd).transformPVCoordinates(pvc)

# Create initial Cartesian orbit:
orb_c = CartesianOrbit(pvi, eme_frame, svd, Constants.WGS84_EARTH_MU)


# The EH propagator is only applicable for near circular orbits, typically used for LEO satellites.
propagator_eh = EcksteinHechlerPropagator(orb_c,
                                        Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS,
                                        Constants.EIGEN5C_EARTH_MU, Constants.EIGEN5C_EARTH_C20,
                                        Constants.EIGEN5C_EARTH_C30, Constants.EIGEN5C_EARTH_C40,
                                        Constants.EIGEN5C_EARTH_C50, Constants.EIGEN5C_EARTH_C60)

# Propagate for 180 minutes (two orbits) every 10 seconds
duration = 180 * 60
tstep = 10

tt = [svd.shiftedBy(float(dt)) for dt in np.arange(0, duration, tstep)]

# Propagated states (inertial frame)
estate_i = [propagator_eh.propagate(t) for t in tt]

# Propagated states (terrestrial frame)
estate_t = [eme_frame.getTransformTo(ITRF, t).transformPVCoordinates(st.getOrbit().getPVCoordinates())
            for ii, (st, t) in enumerate(zip(estate_i, tt))]


# Plot discrepancies
ppos_t = np.array([s.getPosition().toArray() for s in estate_t])
pvel_t = np.array([s.getVelocity().toArray() for s in estate_t])

rpos = resorb[:len(state_t)].POS
rvel = resorb[:len(state_t)].VEL

plt.plot(ppos_t - rpos)
plt.plot(vnorm2(ppos_t - rpos), 'k--')
plt.legend(['pos x', 'pos y', 'pos z', '|pos|'])

plt.plot(pvel_t - rvel)
plt.plot(vnorm2(pvel_t - rvel), 'k--')
plt.legend(['vel x', 'vel y', 'vel z', '|vel|'])


# Same exercise as above, using numerical propagator (DormandPrince853)
# https://gitlab.orekit.org/orekit-labs/python-wrapper/-/blob/master/examples/Example_numerical_prop.ipynb

# Initial orbit is orb_c from above
s1a_mass = 2300.0               # kg, per Wikipedia
ini_state = SpacecraftState(orb_c, s1a_mass)

minStep = 0.001;
maxstep = 10.0;
initStep = 1.0

positionTolerance = 0.04 
orbitType = OrbitType.CARTESIAN
tol = NumericalPropagator.tolerances(positionTolerance, initialOrbit, orbitType)

integrator = DormandPrince853Integrator(minStep, maxstep, 
    JArray_double.cast_(tol[0]),  # Double array of doubles needs to be casted in Python
    JArray_double.cast_(tol[1]))
integrator.setInitialStepSize(initStep)

propagator_num = NumericalPropagator(integrator)
propagator_num.setOrbitType(orbitType)
propagator_num.setInitialState(ini_state)

itrf    = FramesFactory.getITRF(IERSConventions.IERS_2010, True) # International Terrestrial Reference Frame, earth fixed
earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                         Constants.WGS84_EARTH_FLATTENING,
                         itrf)
gravityProvider = GravityFieldFactory.getNormalizedProvider(8, 8)
propagator_num.addForceModel(HolmesFeatherstoneAttractionModel(earth.getBodyFrame(), gravityProvider))
                                                  
# Propagated states (inertial frame)
nstate_i = [propagator_num.propagate(t) for t in tt]

# Propagated states (terrestrial frame)
nstate_t = [eme_frame.getTransformTo(ITRF, t).transformPVCoordinates(st.getOrbit().getPVCoordinates())
            for ii, (st, t) in enumerate(zip(nstate_i, tt))]


# Plot discrepancies
npos_t = np.array([s.getPosition().toArray() for s in nstate_t])
nvel_t = np.array([s.getVelocity().toArray() for s in nstate_t])

# rpos, rvel as above
plt.plot(npos_t - rpos)
plt.plot(vnorm2(ppos_t - rpos), 'k--')
plt.legend(['pos x', 'pos y', 'pos z', '|pos|'])

plt.plot(pvel_t - rvel)
plt.plot(vnorm2(pvel_t - rvel), 'k--')
plt.legend(['vel x', 'vel y', 'vel z', '|vel|'])

