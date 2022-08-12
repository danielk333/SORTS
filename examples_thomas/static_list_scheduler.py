import numpy as np
import matplotlib.pyplot as plt
import sorts

class StaticListScheduler(sorts.RadarSchedulerBase):
	def __init__(self, radar, t0, scheduler_period, logger=None, profiler=None):
		super().__init__(radar, t0, scheduler_period, logger=logger, profiler=profiler)

	def run(self, controls):
		t_max = max([ctrl.t[-1][-1] for ctrl in controls])
		t_min = min([ctrl.t[0][0] for ctrl in controls])
		t_start = t_min//self.scheduler_period*self.scheduler_period
		self.t0 = t_start
		print(t_min)
		print(t_start)


		n_scheduler_periods = int((t_max-t_start)/self.scheduler_period)+1

		time_slice_start_time   = np.ndarray((n_scheduler_periods,), dtype=object)
		time_slice_duration     = np.ndarray((n_scheduler_periods,), dtype=object)
		controls_id             = np.ndarray((n_scheduler_periods,), dtype=object)

		for ctrl_id, ctrl in enumerate(controls):
			i_start = int((ctrl.t[0][0]-t_start)/self.scheduler_period)
			# print(ctrl.t)
			print(i_start)
			# print(i_start + ctrl.n_periods)
			# print(n_scheduler_periods)

			for scheduler_period_id in range(i_start, min(i_start + ctrl.n_periods, n_scheduler_periods)):
				control_period_id 	= ctrl.get_control_period_id(scheduler_period_id)
				print(control_period_id)

				if control_period_id > -1:
					if time_slice_start_time[scheduler_period_id] is not None:
						controls_id[scheduler_period_id]         	= np.append(controls_id[scheduler_period_id], np.full(len(ctrl.t[control_period_id]), ctrl_id, int))
						time_slice_start_time[scheduler_period_id]  = np.append(time_slice_start_time[scheduler_period_id], ctrl.t[control_period_id])
						time_slice_duration[scheduler_period_id]    = np.append(time_slice_duration[scheduler_period_id], ctrl.t_slice[control_period_id])
					else:
						controls_id[scheduler_period_id]         	= np.full(len(ctrl.t[control_period_id]), ctrl_id, int)
						time_slice_start_time[scheduler_period_id]  = ctrl.t[control_period_id]
						time_slice_duration[scheduler_period_id]    = ctrl.t_slice[control_period_id]
						print(controls_id)

					print(time_slice_start_time)

		
		final_control_sequence = sorts.RadarControls(self.radar, None, scheduler=self, priority=None)
		final_control_sequence.set_time_slices(time_slice_start_time, time_slice_duration)
		
		print(final_control_sequence.t)
		print(final_control_sequence.t_slice)

		final_control_sequence.active_control = controls_id
		final_control_sequence.meta["scheduled_controls"] = controls

		final_control_sequence = self.extract_control_sequence(controls, final_control_sequence)
		print(final_control_sequence)
		return final_control_sequence

# simulation parameters
end_t = 3600*10
t_slice = 7.5
tracking_period = 10

# RADAR definition
eiscat3d = sorts.radars.eiscat3d

# Propagator
Prop_cls = sorts.propagator.Kepler
Prop_opts = dict(
    settings = dict(
        out_frame='ITRS',
        in_frame='TEME',
    ),
)

# scheduler properties
t0 = 0
scheduler_period = 120 # [s] -> 2 minutes - can go up to 10mins or more depending on the available RAM

# create scheduler
scheduler = StaticListScheduler(eiscat3d, t0, scheduler_period)

# Creating space object
# Object properties
orbits_a = np.array([7200, 8500, 12000, 10000])*1e3 # km
orbits_i = np.array([80, 105, 105, 80]) # deg
orbits_raan = np.array([86, 160, 180, 90]) # deg
orbits_aop = np.array([0, 50, 40, 55]) # deg
orbits_mu0 = np.array([50, 5, 30, 8]) # deg
obj_id = 0

# Object instanciation
space_object = sorts.SpaceObject(
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


# create state time array
t_states = sorts.equidistant_sampling(
    orbit = space_object.state, 
    start_t = 0, 
    end_t = end_t, 
    max_dpos=50e3,
)

# get object states in ECEF frame
object_states = space_object.get_state(t_states)

# reduce state array
eiscat_passes = sorts.find_simultaneous_passes(t_states, object_states, [*eiscat3d.tx, *eiscat3d.rx])

# compute and plot controls for each pass
tracker_controller = sorts.Tracker()

controls = []
for pass_id in range(np.shape(eiscat_passes)[0]):
    tracking_states = object_states[:, eiscat_passes[pass_id].inds]
    t_states_i = t_states[eiscat_passes[pass_id].inds]
    t_controller = np.arange(t_states_i[0], t_states_i[-1]+tracking_period, tracking_period)
    controls.append(tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, scheduler=scheduler, states_per_slice=4, interpolator=sorts.interpolation.Linear))

final_control_sequence = scheduler.run(controls)

# plot results
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

# Plotting station ECEF positions
sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

# Plotting station ECEF positions
for tx in eiscat3d.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in eiscat3d.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

# plotting object states
ax.plot(object_states[0], object_states[1], object_states[2], "--b", alpha=0.2)

# for ctrl_i in range(len(controls)):
# 	for period_id in range(controls[ctrl_i].n_periods):
# 		ax = sorts.plotting.plot_beam_directions(controls[ctrl_i].get_pdirs(period_id), eiscat3d, ax=ax, linewidth_rx=0.08, linewidth_tx=0.08, alpha=0.001)

# plot scheduler pointing directions
for period_id in range(final_control_sequence.n_periods):
	pdirs = final_control_sequence.get_pdirs(period_id)
	if pdirs is not None:
		ax = sorts.plotting.plot_beam_directions(pdirs, eiscat3d, ax=ax)

plt.show()
