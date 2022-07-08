import numpy as np
import numpy.testing as nt

import sorts

print("testing time slices overlapping detection")
eiscat3d = sorts.radars.eiscat3d

t0 = 0
scheduler_period = 100
scheduler = sorts.StaticPriorityScheduler(eiscat3d, t0, scheduler_period)

controls=sorts.radar_controls.RadarControls(eiscat3d, None, scheduler=scheduler)

t = np.linspace(0.0, 9.0, 10)
time_slices = np.ones(10)*1.0
time_slices[4] = 1.1

assert controls.check_time_slice_overlap(t, time_slices) == True


print("testing time array slicing")
radar = sorts.radars.eiscat3d
scheduler = sorts.StaticPriorityScheduler(radar, 0, 2.0)
controls = sorts.RadarControls(radar, None, scheduler=scheduler)

# time slice parameters
t = np.linspace(0.0, 4.0, 5).astype(np.float64)
duration = np.repeat([0.5], 5)

print(t)

controls.set_time_slices(t, duration)
t_th 		= np.array([[0.0, 1.0], [2.0, 3.0], [4.0]], dtype=object)
t_slice_th 	= np.array([[0.5, 0.5], [0.5, 0.5], [0.5]], dtype=object)

print(controls.splitting_indices)
print(controls.t)

for i in range(len(t_th)):
	nt.assert_array_equal(controls.t[i], np.atleast_1d(t_th[i]).astype(np.float64))
	nt.assert_array_equal(controls.t_slice[i], np.atleast_1d(t_slice_th[i]).astype(np.float64))

print("testing array controls setting/getting")
radar = sorts.radars.eiscat3d

scheduler = sorts.StaticPriorityScheduler(radar, 0, 2.0)
controls = sorts.RadarControls(radar, None, scheduler=scheduler, priority=1)

data            = np.array([0, 200, 0.0, 5.184, -14.565]).astype(np.float64)
data_sliced     = np.array([[0.0, 200], [0.0, 5.184], [-14.565]])

t               = np.linspace(0, 4.0, 5)
t_slice         = 0.5

controls.set_time_slices(t, duration)
n_periods = len(controls.t)

for ctrl_var in radar.OPTIONAL_CONTROL_VARIABLES:
    controls.set_control(ctrl_var, data)

    data_retrieved = controls.get_control(ctrl_var)

    assert np.size(data_retrieved) == np.size(data_sliced)

    for period_id in range(n_periods):
        nt.assert_array_almost_equal(np.atleast_1d(data_retrieved[period_id]).astype(np.float64), data_sliced[period_id])
