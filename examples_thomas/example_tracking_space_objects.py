import numpy as np
import matplotlib.pyplot as plt

import sorts

# RADAR definition
eiscat3d = sorts.radars.eiscat3d

# Object definition
# Propagator
Prop_cls = sorts.propagator.Kepler
Prop_opts = dict(
    settings = dict(
        out_frame='ITRS',
        in_frame='TEME',
    ),
)

# Object properties
orbits_a = np.array([7200, 7200, 8500, 12000, 10000])*1e3 # m
orbits_i = np.array([80, 80, 105, 105, 80]) # deg
orbits_raan = np.array([86, 86, 160, 180, 90]) # deg
orbits_aop = np.array([0, 0, 50, 40, 55]) # deg
orbits_mu0 = np.array([60, 50, 5, 30, 8]) # deg
priorities = np.array([4, 3, 1, 2, 5])
epoch = 53005.0

# Creating space objects
space_objects = []
for so_id in range(len(orbits_a)):
    space_objects.append(sorts.SpaceObject(
            Prop_cls,
            propagator_options = Prop_opts,
            a=orbits_a[so_id], 
            e=0.1,
            i=orbits_i[so_id],
            raan=orbits_raan[so_id],
            aop=orbits_aop[so_id],
            mu0=orbits_mu0[so_id],
            epoch=epoch,
            parameters = dict(
                d=0.1,
            ),
        ))



# Radar controller parameters
tracking_period = 50.0
t_slice = 2.0
t_start = 0.0
t_end = 3600*5

# intialization of the space object tracker controller
so_tracking_controller = sorts.SpaceObjectTracker()

# generate controls
t_tracking = np.arange(t_start, t_end, tracking_period)
controls = so_tracking_controller.generate_controls(t_tracking, eiscat3d, space_objects, epoch, t_slice, space_object_priorities=priorities, save_states=True)



# plotting results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting station ECEF positions and earth grid
sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res=100, color='black', hide_ax=True)
for tx in eiscat3d.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')

for rx in eiscat3d.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')       

# plot all space object states
for space_object_index in range(len(space_objects)):
    states = controls.meta["objects_states"][space_object_index]
    ax.plot(states[0], states[1], states[2], '--', label=f"so-{space_object_index} (p={priorities[space_object_index]})", alpha=0.35)


# plot states being tracked and 
ecef_tracking = controls.meta["tracking_states"]
object_ids = controls.meta["state_priorities"]

for period_id in range(controls.n_periods):
    # compute transitions between the tracking of multiple objects (used to plot segments)
    mask = np.logical_or(np.abs(controls.t[period_id][1:] - controls.t[period_id][:-1]) > tracking_period, object_ids[period_id][1:] - object_ids[period_id][:-1] != 0)
    transition_ids = np.where(mask)[0]+1

    # plot control sequence beam directions
    ax = sorts.plotting.plot_beam_directions(controls.get_pdirs(period_id), eiscat3d, ax=ax, zoom_level=0.6, azimuth=10, elevation=20)

    # plot states being tracked as segments
    for i in range(len(transition_ids)+1):
        if i == 0:
            i_start = 0
        else:
            i_start = transition_ids[i-1]

        if i == len(transition_ids):
            i_end = len(t_tracking)+1
        else:
            i_end = transition_ids[i]
             
        ax.plot(ecef_tracking[period_id][0, i_start:i_end], ecef_tracking[period_id][1, i_start:i_end], ecef_tracking[period_id][2, i_start:i_end], '-b')
             
ax.legend()
plt.show()