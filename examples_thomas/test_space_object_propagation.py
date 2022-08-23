import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time

from sorts.propagator import Kepler
from sorts import SpaceObject

# define propagator options
options = dict(
    settings=dict(
        in_frame='GCRS',
        out_frame='GCRS',
    ),
)

# define propagation time interval (2 days)
t = np.linspace(0, 3600*24.0*2, num=5000)

# instantiate space object
obj = SpaceObject(
    Kepler,
    propagator_options = options,
    a = 7000e3, 
    e = 0.0, 
    i = 69, 
    raan = 0, 
    aop = 0, 
    mu0 = 0, 
    epoch = Time(57125.7729, format='mjd'),
    parameters = dict(
        d = 0.2,
    )
)

print(obj)

# propagate
states = obj.get_state(t)

# plot states
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0,:], states[1,:], states[2,:],"-b")

max_range = np.linalg.norm(states[0:3,0])*2

ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

plt.show()