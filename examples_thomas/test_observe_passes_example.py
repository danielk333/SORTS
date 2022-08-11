import sorts
import numpy as np
radar = sorts.radars.eiscat3d

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
        a = 7000e3, 
        e = 0.0,
        i = 78,
        raan = 86,
        aop = 0, 
        mu0 = 50,
        epoch = 53005.0,
        parameters = dict(
                d = 0.1,
        ),
)
t_states = sorts.equidistant_sampling(
        orbit=space_object.state, 
        start_t=0, 
        end_t=3600.0, 
        max_dpos=10e3
)
object_states = space_object.get_state(t_states)

passes = sorts.passes.find_simultaneous_passes(t_states, object_states, radar.tx+radar.rx, cache_data=True)
controller = sorts.Tracker()

tracking_states = object_states[:, passes[0].inds]
t_states_i = t_states[passes[0].inds]
t_controller = np.arange(t_states_i[0], t_states_i[-1], 10)
controls = controller.generate_controls(
        t_controller, 
        radar, 
        t_states_i, 
        tracking_states, 
        t_slice=0.1, 
        scheduler=None, 
        states_per_slice=1, 
        interpolator=sorts.interpolation.Legendre8)

radar_states = radar.control(controls)
pass_list = radar.find_passes(t_states_i, tracking_states, cache_data=True)
data = radar.observe_passes(pass_list, radar_states, space_object)

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, sharex=ax1)
ax3 = fig.add_subplot(313, sharex=ax1)
fmt = ["-r", "-g", "-b"]
for station_id in range(len(radar.rx)):
        measurements = data[0][station_id][0]["measurements"] # extract measurements for each rx station for the first pass
        ax1.plot(measurements['t_measurements'], measurements['range']*1e-3, fmt[station_id], label=f"rx{station_id}")
        ax2.plot(measurements['t_measurements'], measurements['range_rate']*1e-3, fmt[station_id], label=f"rx{station_id}")
        ax3.plot(measurements['t_measurements'], 10*np.log10(measurements['snr']), fmt[station_id], label=f"rx{station_id}")

ax1.set_ylabel("$R$ [$km$]")
ax2.set_ylabel("$v_r$ [$km/s$]")
ax3.set_ylabel("$\\rho$ [$dB$]")
ax3.set_xlabel("$t$ [$s$]")
ax1.tick_params(labeltop=True, labelbottom=False)
ax2.tick_params(labeltop=False, labelbottom=False)
ax3.tick_params(labeltop=False)
ax1.grid()
ax2.grid()
ax3.grid()
fig.subplots_adjust(hspace=0)
plt.legend()
plt.show()