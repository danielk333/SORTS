import numpy as np
import numpy.testing as nt
import matplotlib.pyplot as plt

import pyorb
import sorts

def plot_vector(start_point, vec, ax, fmt="-r"):
	A = np.array([start_point, start_point + vec]).T

	ax.plot(A[0], A[1], A[2], fmt)


# radar = sorts.radars.eiscat3d
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")

# # Plotting station ECEF positions
# sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

# # Plotting station ECEF positions
# for tx in radar.tx:
#     ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
# for rx in radar.rx:
#     ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

# epoch = 53005.0
# t_slice = 1
# tf = 2000
# max_dpos=10e3

# max_points = 2
# n_tracking_points = 10
# states_per_slice = 2

# # Propagator
# Prop_cls = sorts.propagator.Kepler
# Prop_opts = dict(
#     settings = dict(
#         out_frame='ITRS',
#         in_frame='TEME',
#     ),
# )

# space_object_1 = sorts.SpaceObject(
# 	Prop_cls,
# 	propagator_options = Prop_opts,
# 	a = 1.5*6378e3, 
# 	e = 0.0,
# 	i = 72.2,
# 	raan = 0,
# 	aop = 66.6,
# 	mu0 = 0,

# 	epoch = epoch,
# 	parameters = dict(
# 		d = 0.1,
# 	),
# )

# space_object_2 = sorts.SpaceObject(
# 	Prop_cls,
# 	propagator_options = Prop_opts,
# 	a = 2.5*6378e3, 
# 	e = 0.0,
# 	i = 72,
# 	raan = 20,
# 	aop = 66.6,
# 	mu0 = 12,

# 	epoch = epoch,
# 	parameters = dict(
# 		d = 0.1,
# 	),
# )

# t = np.linspace(0, tf, n_tracking_points)

# space_objects = np.array([space_object_1, space_object_2])
# obj_priorities = np.array([0, 1], dtype=int)
# object_indices_ref 	= np.array([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.int64)

# for so in space_objects:
# 	states = so.get_state(t)
# 	ax.plot(states[0], states[1], states[2])

# so_tracking_controller = sorts.SpaceObjectTracker()
# controls = so_tracking_controller.generate_controls(t, radar, space_objects, epoch, t_slice, states_per_slice=states_per_slice, space_object_priorities=obj_priorities, save_states=True, max_points=max_points, max_dpos=max_dpos)

# # verify correct objects are being observed
# nt.assert_array_equal(controls.meta["object_indices"], object_indices_ref)
# object_indices_ref = np.repeat(object_indices_ref, 2)

# # generate tracking time array (adding itermediate time points to reach states_per_slice)
# t_ref = t[0:-1] # last point is discarded by the controller (no space object in FOV)
# dt_tracking = t_slice/float(states_per_slice)
# t_dirs = np.repeat(t_ref, states_per_slice)
# for ti in range(states_per_slice):
# 	t_dirs[ti::states_per_slice] = t_dirs[ti::states_per_slice] + dt_tracking*ti

# # split arrays according to control periods
# splitting_inds = np.arange(max_points, len(t_ref), max_points)
# t_ref = np.hsplit(t_ref, splitting_inds)
# t_dirs = np.hsplit(t_dirs, splitting_inds*states_per_slice)
# object_indices_ref = np.hsplit(object_indices_ref, splitting_inds*states_per_slice)

# # compute reference ecef states
# ecef_ref = np.ndarray((len(t_dirs),), dtype=object)
# for period_id in range(controls.n_periods):
# 	ecef_ref[period_id] = np.ndarray((3, len(t_dirs[period_id])), dtype=float)
# 	for soid, so in enumerate(space_objects):
# 		obj_msk = object_indices_ref[period_id] == soid
# 		ecef_ref[period_id][:, obj_msk] = so.get_state(t_dirs[period_id])[0:3][:, obj_msk]

# # station positions
# tx_ecef = np.array([tx.ecef for tx in controls.radar.tx], dtype=np.float64) # get the position of each Tx station (ECEF frame)
# rx_ecef = np.array([rx.ecef for rx in controls.radar.rx], dtype=np.float64) # get the position of each Rx station (ECEF frame)

# i_start = 0
# for period_id in range(controls.n_periods):
# 	nt.assert_array_equal(controls.t[period_id], t_ref[period_id])

# 	tx_dirs = ecef_ref[period_id][None, None, :, :] - tx_ecef[:, None, :, None]
# 	rx_dirs = ecef_ref[period_id][None, None, :, :] - rx_ecef[:, None, :, None]

# 	pdirs_ref = dict()
# 	pdirs_ref['tx'] = tx_dirs/np.linalg.norm(tx_dirs, axis=2)[:, :, None, :]
# 	pdirs_ref['rx'] = rx_dirs/np.linalg.norm(rx_dirs, axis=2)[:, :, None, :]

# 	pdirs = controls.get_pdirs(period_id)

# 	nt.assert_array_almost_equal(pdirs_ref['tx'], pdirs['tx'], decimal=5)
# 	nt.assert_array_almost_equal(pdirs_ref['rx'], pdirs['rx'], decimal=5)
# 	nt.assert_array_almost_equal(t_dirs[period_id], pdirs['t'], decimal=5)


# for period_id in range(controls.n_periods):
#     ax = sorts.plotting.plot_beam_directions(controls.get_pdirs(period_id), radar, ax=ax, zoom_level=0.6, azimuth=10, elevation=20)

# plt.show()

import numpy as np
import sorts

p = sorts.profiling.Profiler()
logger = sorts.profiling.get_logger('scanning')

radar = sorts.radars.eiscat3d

p.start('total')

end_t = 200
Prop_cls = sorts.propagator.SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)
objs = [
    sorts.SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = 7200e3, 
        e = 0.02, 
        i = 75, 
        raan = 86,
        aop = 0,
        mu0 = 60,
        epoch = 53005.0,
        parameters = dict(
            d = 0.1,
        ),
    ),
]

scan = sorts.scans.Fence(azimuth=90, min_elevation=30, dwell=0.1, num=100)
radar_ctrl = sorts.Scanner(profiler=p)
t = np.arange(0, end_t, scan.dwell())

p.start('generate_controls')
controls = radar_ctrl.generate_controls(t, radar, scan, max_points=1000)
p.stop('generate_controls')

p.start('get_radar_states')
radar_states = radar.control(controls)
p.stop('get_radar_states')

datas = []
for ind in range(len(objs)):
    print(f'Temporal points obj {ind}: {len(t)}')
    data = radar.compute_measurements(
    	radar_states, 
    	objs[ind], 
    	logger=logger, 
    	profiler=p, 
    	max_dpos=50e3, 
    	snr_limit=False,
    	parallelization=True, 
    	save_states=True, 
    	n_processes=16,
    	tx_indices=[0],
    	rx_indices=[0])
    datas.append(data)

p.stop('total')

#print(p.fmt(normalize='total'))
print(p)

# print results
fig = plt.figure(figsize=(15,15))
axes = [
    [
        fig.add_subplot(221, projection='3d'),
        fig.add_subplot(222),
    ],
    [
        fig.add_subplot(223),
        fig.add_subplot(224),
    ],
]

sorts.plotting.grid_earth(axes[0][0])
for tx in radar.tx:
    axes[0][0].plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in radar.rx:
    axes[0][0].plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')


# for period_id in range(controls.n_periods):
#     for txi, tx in enumerate(radar.tx):
#         point_tx = controls.get_pdirs(period_id)["tx"][txi, 0]
#         point_tx = point_tx/np.linalg.norm(point_tx, axis=0)*1000e3 + tx.ecef[:,None]

#         for ti in range(point_tx.shape[1]):
#             axes[0][0].plot([tx.ecef[0], point_tx[0,ti]], [tx.ecef[1], point_tx[1,ti]], [tx.ecef[2], point_tx[2,ti]], 'r-', alpha=0.15)

for ind in range(len(objs)):
	data = datas[ind]

	axes[0][0].plot(data["states"][0], data["states"][1], data["states"][2], "+b")

	for pass_data in data["pass_data"]:
		txi = 0
		rxi = 0

		if pass_data is not None:
			states = pass_data["states"]
			measurements = pass_data["measurements"]

			detection = measurements['detection']
			snr = measurements['snr']
			ranges = measurements['range']
			range_rates = measurements['range_rate']
			t = measurements['t']

			SNRdB = 10*np.log10(snr)
			det_inds = SNRdB > 10.0

			axes[0][1].plot(t/3600.0, ranges*1e-3, '-', label=f'obj{ind}')
			axes[1][0].plot(t/3600.0, range_rates*1e-3, '-')
			axes[1][1].plot(t/3600.0, SNRdB, '-')

			# detections
			axes[0][1].plot(t[det_inds]/3600.0, ranges[det_inds]*1e-3, '.r')
			axes[1][0].plot(t[det_inds]/3600.0, range_rates[det_inds]*1e-3, '.r')
			axes[1][1].plot(t[det_inds]/3600.0, SNRdB[det_inds], '.r')
			# axes[1][1].set_ylim([0, None])

font_ = 18
axes[0][1].set_xlabel('Time [h]', fontsize=font_)
axes[1][0].set_xlabel('Time [h]', fontsize=font_)
axes[1][1].set_xlabel('Time [h]', fontsize=font_)

axes[0][1].set_ylabel('Two way range [km]', fontsize=font_)
axes[1][0].set_ylabel('Two way range rate [km/s]', fontsize=font_)
axes[1][1].set_ylabel('SNR [dB]', fontsize=font_)

#axes[0][1].legend()

dr = 3000e3
axes[0][0].set_xlim([radar.tx[0].ecef[0]-dr, radar.tx[0].ecef[0]+dr])
axes[0][0].set_ylim([radar.tx[0].ecef[1]-dr, radar.tx[0].ecef[1]+dr])
axes[0][0].set_zlim([radar.tx[0].ecef[2]-dr, radar.tx[0].ecef[2]+dr])

plt.show()