'''
Implements the unit testing of the Tracker controller
'''

import numpy as np
import matplotlib.pyplot as plt
import pyorb

from sorts import radars
from sorts import controllers
from sorts import scans
from sorts import space_object
from sorts import find_simultaneous_passes, equidistant_sampling
from sorts import plotting
from sorts.radar import scheduler

from sorts.common import interpolation
from sorts.common import profiling
from sorts.targets.propagator import Kepler

# Profiler
p = profiling.Profiler()
logger = profiling.get_logger('scanning')

end_t = 24*3600*10
t_slice = 7.5
tracking_period = 10

# RADAR definition
eiscat3d = radars.eiscat3d
logger.info(f"test_tracker_controller -> initializing radar insance eiscat3d={eiscat3d}")

# Object definition
# Propagator
Prop_cls = Kepler
Prop_opts = dict(
    settings = dict(
        out_frame='ITRS',
        in_frame='TEME',
    ),
)
logger.info(f"test_tracker_controller -> initializing propagator ({Kepler}) options ({Prop_opts})")

# scheduler properties
t0 = 0
scheduler_period = 120 # [s] -> 2 minutes - can go up to 10mins or more depending on the available RAM

# create scheduler
scheduler = scheduler.StaticPriorityScheduler(eiscat3d, t0, scheduler_period=scheduler_period)

logger.info("test_scan_controller_w_scheduler -> scheduler initialized:")
logger.info(f"test_scan_controller_w_scheduler:scheduler_variables -> t0 = {t0}")
logger.info(f"test_scan_controller_w_scheduler:scheduler_variables -> scheduler_period = {scheduler_period}\n")
logger.info(f"test_scan_controller_w_scheduler:scheduler_variables -> scheduler = {scheduler}\n")

# Creating space object
# Object properties
orbits_a = np.array([7200, 8500, 12000, 10000])*1e3 # km
orbits_i = np.array([80, 105, 105, 80]) # deg
orbits_raan = np.array([86, 160, 180, 90]) # deg
orbits_aop = np.array([0, 50, 40, 55]) # deg
orbits_mu0 = np.array([50, 5, 30, 8]) # deg
obj_id = 0

p.start('Total')
p.start('object_initialization')

# Object instanciation
logger.info("test_tracker_controller -> creating new object\n")

space_object = space_object.SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = orbits_a[obj_id], 
        e = 0.0,
        i = orbits_i[obj_id],
        raan = orbits_raan[obj_id],
        aop = orbits_aop[obj_id],
        mu0 = orbits_mu0[obj_id],
        
        epoch = 53005.0,
        parameters = dict(
            d = 0.1,
        ),
    )

p.stop('object_initialization')
logger.info("test_tracker_controller -> object created :")
logger.info(f"test_tracker_controller -> {space_object}")

logger.info("test_tracker_controller -> sampling equidistant states on the orbit")

# create state time array
p.start('equidistant_sampling')
t_states = equidistant_sampling(
    orbit = space_object.state, 
    start_t = 0, 
    end_t = end_t, 
    max_dpos=50e3,
)
p.stop('equidistant_sampling')

logger.info(f"test_tracker_controller -> sampling done : t_states -> {t_states.shape}")

# get object states in ECEF frame
p.start('get_state')
object_states = space_object.get_state(t_states)
p.stop('get_state')

logger.info(f"test_tracker_controller -> object states computation done ! ")
logger.info(f"test_tracker_controller -> t_states -> {t_states.shape}")


# reduce state array
p.start('find_simultaneous_passes')
eiscat_passes = find_simultaneous_passes(t_states, object_states, [*eiscat3d.tx, *eiscat3d.rx])
p.stop('find_simultaneous_passes')
logger.info(f"test_tracker_controller -> Passes : eiscat_passes={eiscat_passes}")

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

# compute and plot controls for each pass
tracker_controller = controllers.tracker.Tracker(logger=logger, profiler=p)

for pass_id in range(np.shape(eiscat_passes)[0]):
    logger.info(f"test_tracker_controller -> Computing tracking controls for pass {pass_id}:")

    tracking_states = object_states[:, eiscat_passes[pass_id].inds]
    t_states_i = t_states[eiscat_passes[pass_id].inds]
    
    p.start('intitialize_controller')
    t_controller = np.arange(t_states_i[0], t_states_i[-1]+tracking_period, tracking_period)
    p.stop('intitialize_controller')
    
    p.start('generate_tracking_controls')
    controls = tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, scheduler=scheduler, states_per_slice=4, interpolator=interpolation.Linear)
    p.stop('generate_tracking_controls')

    logger.info("test_tracker_controller -> Controls generated")

    for period_id in range(controls.n_periods):
        ctrl = controls.get_pdirs(period_id)
        plotting.plot_beam_directions(ctrl, eiscat3d, ax=ax, logger=logger, profiler=p, tx_beam=True, rx_beam=True, zoom_level=0.9, azimuth=10, elevation=10)
        
        logger.info(f"test_tracker_controller -> ploting data for sub control {period_id}")

        ax.plot(tracking_states[0], tracking_states[1], tracking_states[2], "-", color="blue")

plt.show()

del tracking_states, object_states, eiscat_passes, t_states
del ax, fig

logger.info("test_tracker_controller -> test script execution done !")
logger.info("showing results :")

p.stop('Total')

print(p)
del p

