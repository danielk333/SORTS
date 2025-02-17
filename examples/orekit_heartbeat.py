#!/usr/bin/env python

'''
Using the heartbeat for dynamics
========================================
'''
import pathlib

import numpy as np
import pyorb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time

from sorts.propagator import Orekit

try:
    pth = pathlib.Path(__file__).parent.resolve()
except NameError:
    pth = pathlib.Path('.').parent.resolve()
pth = pth / 'data' / 'orekit-data-master.zip'


if not pth.is_file():
    sorts.propagator.Orekit.download_quickstart_data(pth, verbose=True)


class MyOrekit(Orekit):
    def heartbeat(self, t, state, interpolator):

        A = float(1.0 + 0.5*np.random.randn(1)[0])*self.force_params['A']

        self.propagator.removeForceModels()
        self._forces['drag_force'] = self.GetIsotropicDragForce(A, self.force_params['C_D'])
        self.UpdateForces()


prop = MyOrekit(
    orekit_data = pth, 
    settings = dict(
        heartbeat=True,
        drag_force=True,
    ),
)
orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees = True, a=6600e3, e=0, i=69, omega=0, Omega=0, anom=0)

#area in m^2
A = 2.0

#change the area every 1 minutes 
dt = 60.0

#propagate for 6h
t = np.arange(0, 6*3600.0, dt)

#for reproducibility
np.random.seed(23984)

states = []

for mci in range(10):
    print(f'MC-iteration {mci+1}/10')
    state = prop.propagate(
        t, 
        orb.cartesian[:,0], 
        epoch=Time(53005, format='mjd', scale='utc'), 
        A=A, 
        C_R = 1.0,
        C_D = 2.3,
    )
    states += [state]

print('With heartbeat')

#Reference simulation without force modifying
prop.set(heartbeat=False)
states0 = prop.propagate(
    t, 
    orb.cartesian[:,0], 
    epoch=Time(53005, format='mjd', scale='utc'), 
    A=A, 
    C_R = 1.0,
    C_D = 2.3,
)

print('No heartbeat')

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

ax.plot([states0[0,-1]], [states0[1,-1]], [states0[2,-1]], ".r", alpha=1)
for mci in range(len(states)):
    ax.plot([states[mci][0,-1]], [states[mci][1,-1]], [states[mci][2,-1]], ".b", alpha=1)

ax.set_title('Anomalous diffusion by variation in area')

plt.show()
