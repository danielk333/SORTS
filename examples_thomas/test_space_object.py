import sorts
import numpy as np
import numpy.testing as nt
# def grad(N0, N1, n, m):
# 	dC_dN0 = (N0-N1) + (n-1)*(N0*(n-1) + N1 - m)
# 	dC_dn = N0*(n-1) + N1 - m 

# 	return dC_dN0, dC_dn

# m = 71

# n_max = 16

# N0 = int(m/(n_max-1))
# n  = int(n_max/2)
# N1 = m - N0*(n-1)

# for i in range(10):


# 	for i in range(10):
# 		N0 = int(m/(n_max-1))
# 		N1 = m - N0*(n-1)

# 		if N1 < 0:
# 			N1 = int(N0/2)

# 		dn = N0**2 * (n-1) + N0*N1
# 		dn = np.sign(dn)
# 		n += dn

# 		dN0 = (n-1) * (N0*(n-1) + N1) + N0 - N1
# 		dN0 = np.sign(dN0)
# 		N0 += dN0

# 	dn = N0**2 * (n-1) + N0*N1
# 	dn = np.sign(dn)
# 	n += dn

# 	if n >= n_max:
# 		n = n_max


# print("N0, ", N0)
# print("N1, ", N1)
# print("n, ", n)
# print("mth, ", N0*(n-1) + N1)
# print("m, ", m)

radar = sorts.radars.eiscat3d

scheduler = sorts.StaticPriorityScheduler(radar, 0, 2.0)
controls_ref = sorts.RadarControls(radar, None, scheduler=scheduler)

controller_t0 = 15.0

t               = np.linspace(controller_t0, controller_t0 + 9.0, 10) # 15 to 24 step=1
duration        = np.ones(10)*0.5

controls_ref.set_time_slices(t, duration)
t_ref = np.array([[15.0], [16.0, 17.0], [18.0, 19.0], [20.0, 21.0], [22.0, 23.0], [24.0]], dtype=object)
# control period id 0 		1   			2   		3  				4       	5
# scheduler period  7 		8   			9   		10  			11       	12

for period_id in range(len(controls_ref.t)):
    nt.assert_array_equal(t[period_id], [])

# test outside of range
assert controls_ref.get_control_period_id(5)    == -1 # 10s : t < t0
assert controls_ref.get_control_period_id(14)   == -1 # 28s : t > t0 + control_duration

# test inside of range
assert controls_ref.get_control_period_id(7)    == 0 # 16s
assert controls_ref.get_control_period_id(10)   == 3 # 20s
assert controls_ref.get_control_period_id(12)   == 5 # 26s

