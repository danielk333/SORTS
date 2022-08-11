import numpy as np
import matplotlib.pyplot as plt
import sorts

radar = sorts.radars.eiscat3d

# Object definition
# Propagator
Prop_cls = sorts.propagator.Kepler
Prop_opts = dict(
    settings = dict(
        out_frame='ITRS',
        in_frame='TEME',
    ),
)

# Object
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
print(space_object)

# get states and passes
t_states = sorts.equidistant_sampling(
    orbit=space_object.state, 
    start_t=0, 
    end_t=3600.0, 
    max_dpos=10e3)

object_states = space_object.get_state(t_states)
radar_passes = radar.find_passes(t_states, object_states, cache_data=True) 

snr = np.ndarray((len(radar.rx),), dtype=object)
for rxi in range(len(radar.rx)):
    snr[rxi] = radar_passes[0][rxi][0].calculate_snr(radar.tx[0], radar.rx[rxi], 0.1, parallelization=True, n_processes=16)


fig = plt.figure()
ax = fig.add_subplot(111)

fmt = ["-r", "-g", "-b"]
for rxi in range(len(radar.rx)):
    ax.plot(t_states[radar_passes[0][rxi][0].inds], 10*np.log10(snr[rxi]), fmt[rxi], label=f"Rx {rxi}")

ax.set_xlabel("$t$ [$s$]")
ax.set_ylabel("$SNR$ [$-$]")
ax.grid()
ax.legend()




fig = plt.figure()
ax = fig.add_subplot(111)
fmt = ["-r", "-g", "-b"]
for rxi in range(len(radar.rx)):
    enu = radar.rx[rxi].enu(object_states[:, radar_passes[0][rxi][0].inds])
    range_ = radar_passes[0][rxi][0].calculate_zenith_angle(enu)
    ax.plot(t_states[radar_passes[0][rxi][0].inds], range_, fmt[rxi], label=f"Rx {rxi}")

ax.set_xlabel("$t$ [$s$]")
ax.set_ylabel("$R_{rx}$ [$m$]")
ax.grid()
ax.legend()
plt.show()






fig = plt.figure()
ax = fig.add_subplot(111)

fmt = ["-r", "-g", "-b"]
for rxi in range(len(radar.rx)):
    ax.plot(t_states[radar_passes[0][rxi][0].inds], radar_passes[0][rxi][0].range()[1], fmt[rxi], label=f"Rx {rxi}")

ax.set_xlabel("$t$ [$s$]")
ax.set_ylabel("$R_{rx}$ [$m$]")
ax.grid()
ax.legend()


fig = plt.figure()
ax = fig.add_subplot(111)

fmt = ["-r", "-g", "-b"]
for rxi in range(len(radar.rx)):
    ax.plot(t_states[radar_passes[0][rxi][0].inds], radar_passes[0][rxi][0].range_rate()[1], fmt[rxi], label=f"Rx {rxi}")

ax.set_xlabel("$t$ [$s$]")
ax.set_ylabel("$v_{r, rx}$ [$m/s$]")
ax.grid()
ax.legend()

fig = plt.figure()
ax = fig.add_subplot(111)

fmt = ["-r", "-g", "-b"]
for rxi in range(len(radar.rx)):
    ax.plot(t_states[radar_passes[0][rxi][0].inds], radar_passes[0][rxi][0].zenith_angle()[1], fmt[rxi], label=f"Rx {rxi}")

ax.set_xlabel("$t$ [$s$]")
ax.set_ylabel("$\\theta_{rx}$ [$deg$]")
ax.grid()
ax.legend()


# # plot trajectory
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

# for tx in radar.tx:
#     ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
# for rx in radar.rx:
#     ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

# ax.plot(object_states[0], object_states[1], object_states[2], '--b', alpha=0.15)

# for pi in radar_passes[0][0]:
#     ax.plot(object_states[0, pi.inds], object_states[1, pi.inds], object_states[2, pi.inds], '-r')

plt.show()
