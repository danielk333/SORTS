import numpy as np
import matplotlib.pyplot as plt

# n_tx = 1
# n_rx = 3

# ktx = np.array([[1, 0.1, 0.05]])
# krx = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

# rtx = np.array([[0, 0, 1]])
# rrx = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])

# A = np.zeros((9, 4), dtype=float)
# b = np.zeros((9,), dtype=float)

# for txi in range(n_tx):
# 	start_id = n_rx*3*txi

# 	for rxi in range(n_rx):
# 		A[start_id + 3*rxi:start_id + 3*(rxi+1), 0] = ktx[txi]
# 		A[start_id + 3*rxi:start_id + 3*(rxi+1), 1 + rxi] = -krx[rxi]

# 		b[start_id + 3*rxi:start_id + 3*(rxi+1)] =  rrx[rxi] - rtx[txi]

# M = A.T.dot(A)
# print(A)
# print(b)

# print(M)
# if np.linalg.det(M) == 0:
# 	l = b.dot(ktx)/np.sum(ktx**2)
# else:
# 	l = np.linalg.inv(M).dot(A.T).dot(b)

# r = l[0]*ktx[0] + rtx[0]

# print(r)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# def plot_vector(r, vec, ax, fmt="-b"):
# 	A = np.asfarray([r, r+vec]).T
# 	ax.plot(A[0], A[1], A[2], fmt)

# for txi in range(n_tx):
# 	plot_vector(rtx[txi], ktx[txi], ax, fmt="-r")

# 	for rxi in range(n_rx):
# 		plot_vector(rrx[rxi], krx[rxi], ax, fmt="-b")

# ax.scatter(r[0], r[1], r[2], c="purple")

# r = l[1]*krx[1] + rrx[1]
# ax.scatter(r[0], r[1], r[2], c="green")
# plt.show()

'''
Implements the unit testing of the Tracker controller
'''

import pyorb

import sorts
from sorts import radars
from sorts import controllers
from sorts import space_object
from sorts import find_simultaneous_passes, equidistant_sampling
from sorts import plotting

from sorts.common import profiling
from sorts.common import interpolation
from sorts.targets.propagator import Kepler

# Profiler
eiscat3d = sorts.radars.eiscat3d

p = profiling.Profiler()
logger = profiling.get_logger('scanning')
epoch = 53005.0
t_slice = 10
tracking_period=40
end_t = 3600*24
max_dpos=100e3

max_points = 100

# Propagator
Prop_cls = sorts.propagator.Kepler
Prop_opts = dict(
    settings = dict(
        out_frame='ITRS',
        in_frame='TEME',
    ),
)

space_object = sorts.SpaceObject(
	Prop_cls,
	propagator_options = Prop_opts,
	a = 1.25*6378e3, 
	e = 0.0,
	i = 72.2,
	raan = 0,
	aop = 66.6,
	mu0 = 0,

	epoch = epoch,
	parameters = dict(
		d = 0.1,
	),
)
# create state time array
t_states = equidistant_sampling(orbit = space_object.state, start_t = 0, end_t = end_t, max_dpos=50e3,)

print("running propagator")

# get object states in ECEF frame
object_states = space_object.get_state(t_states)

print("propagator done")

print(t_states)
print(object_states)

eiscat_passes = find_simultaneous_passes(t_states, object_states, eiscat3d.tx + eiscat3d.rx)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

# Plotting station ECEF positions
plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

# Plotting station ECEF positions
for tx in eiscat3d.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in eiscat3d.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

# plotting object states
ax.plot(object_states[0], object_states[1], object_states[2], "--b", alpha=0.2)


print("passes found")
print(eiscat_passes)
# compute and plot controls for each pass
for pass_id in range(np.shape(eiscat_passes)[0]):
    tracking_states = object_states[:, eiscat_passes[pass_id].inds]
    t_states_i = t_states[eiscat_passes[pass_id].inds]
    t_controller = np.arange(t_states_i[0], t_states_i[-1]+tracking_period, tracking_period)
    
    tracker_controller = controllers.Tracker(logger=logger, profiler=p)
    controls = tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, max_points=max_points, states_per_slice=5, interpolator=interpolation.Legendre8)
    
    for period_id in range(controls.n_periods):
        pdirs = controls.get_pdirs(period_id)

        txdirs = pdirs["tx"].copy()
        rxdirs = pdirs["rx"].copy()
        p.start("old")
        ipoints_old = eiscat3d.compute_intersection_points(txdirs, rxdirs)
        p.stop("old")
        p.start("new")
        ipoints_new = eiscat3d.compute_intersection_points_new(txdirs.copy(), rxdirs.copy())
        p.stop("new")

        distance = np.linalg.norm(ipoints_old-ipoints_new, axis=0)
        print(distance)
        print(max(distance))

        plotting.plot_beam_directions(pdirs, eiscat3d, ax=ax, logger=logger, profiler=p, tx_beam=True, rx_beam=True, zoom_level=0.9, azimuth=10, elevation=10)

        ax.scatter(ipoints_new[0], ipoints_new[1], ipoints_new[2])
    ax.plot(tracking_states[0], tracking_states[1], tracking_states[2], "-", color="blue")

print(p)
plt.show()
