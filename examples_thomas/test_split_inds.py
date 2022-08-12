import numpy as np

scheduler_period = 10
t0 = 20
t = np.array([50.0, 60, 80, 120, 155, 160, 200])

n_periods = int((t[-1] - t[0])//scheduler_period) + 1
print(n_periods)
end_t = np.arange(t[0], t[-1] + scheduler_period, scheduler_period)	

print(end_t)

split_inds = np.ndarray((n_periods,), dtype=int)
start_time = t[0]//scheduler_period*scheduler_period
print(start_time)

i_start = 0
for period_id in range(n_periods):
	if period_id < n_periods-1:
		i_end = int(np.argmax(t[i_start:] >= start_time + (period_id+1)*scheduler_period)) + i_start - 1
	else:
		i_end = len(t) - 1
	
	add_index = False
	if i_end > -1:
		print("t_end period=", start_time + period_id*scheduler_period)
		print("i_end ", i_end)
		print("t_end ", t[i_end])
		print("i_start, ", i_start)
		print("t_start, ", t[i_start])
		
		if i_start == i_end:
			if t[i_end] < start_time + (period_id+1)*scheduler_period:
				add_index = True
		else:
			if t[i_end] - t[i_start] > scheduler_period or i_start > i_end:
				add_index = False
			else:
				add_index = True

	if add_index is True:
		split_inds[period_id] = i_end+1
		i_start = i_end + 1
	else:
		split_inds[period_id] = -1
print(split_inds)

splitted = np.ndarray((n_periods,), dtype=object)

id_start = 0
for period_id in range(len(split_inds)):
	if split_inds[period_id] == -1: # if the periods does not contain any controls
		splitted[period_id] = None
	else:
		# get end index of the control period in the linear array
		id_end = split_inds[period_id]

		print(id_start)
		print(id_end)

		# copy values from the array in the corresponding control period
		splitted[period_id] = t[id_start:id_end]
		id_start = id_end
print(splitted)

# tmp_split_inds = np.array(np.where((period_idx[1:] - period_idx[:-1]) > 1)[0]) + 1

# for i in range(len(tmp_split_inds))

# self.n_periods = len(self.splitting_indices) + 1
# del period_idx