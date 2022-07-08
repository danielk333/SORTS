import numpy as np
import numpy.testing as nt

# import sorts

# radar = sorts.radars.eiscat3d

ctrl_id = 1
active_control = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1]).astype(int)

t_final = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
t_ctrl = np.linspace(0, 9, 100).astype(float)

mask = active_control == ctrl_id
t_final_extract = t_final[mask]

inds = np.intersect1d(t_ctrl, t_final_extract, return_indices=True)
t_ctrl_ids = inds[1]

nt.assert_array_equal(t_ctrl[t_ctrl_ids], t_final_extract)

print(t_final_extract)
print(t_ctrl[t_ctrl_ids])
print(t_ctrl_ids)


	
