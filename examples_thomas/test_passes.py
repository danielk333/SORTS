import time
import numpy as np
import numpy.testing as nt

import sorts

eiscat3d = sorts.radars.eiscat3d

N = 1000
exec_time = np.zeros(N)
exec_time_old = np.zeros(N)

object_states = np.load("object_states.npy")
t_states = np.load("t_states.npy")

def function_a():
    eiscat_passes = sorts.radar.passes.find_simultaneous_passes(t_states, object_states, eiscat3d.tx + eiscat3d.rx)

def function_b():
    eiscat_passes = sorts.radar.passes.find_simultaneous_passes_old(t_states, object_states, eiscat3d.tx + eiscat3d.rx)

eiscat_passes1 = sorts.radar.passes.find_simultaneous_passes(t_states, object_states, eiscat3d.tx + eiscat3d.rx)
eiscat_passes2 = sorts.radar.passes.find_simultaneous_passes_old(t_states, object_states, eiscat3d.tx + eiscat3d.rx)

for pi in range(len(eiscat_passes1)):
    nt.assert_almost_equal(eiscat_passes1[pi], eiscat_passes2[pi])
exit()
# reduce state array
for i in range(N):
    t_start = time.time_ns()
    function_a()
    exec_time[i] = time.time_ns() - t_start

    t_start = time.time_ns()
    function_b()
    exec_time_old[i] = time.time_ns() - t_start

exec_time_old = exec_time_old/1e9
exec_time = exec_time/1e9

print("new")
print(f"mean exec_time : {np.mean(exec_time):.4e}")
print(f"std dev exec_time : {np.var(exec_time)**0.5:.4e}")

print("old")
print(f"mean exec_time_old : {np.mean(exec_time_old):.4e}")
print(f"std dev exec_time_old : {np.var(exec_time_old)**0.5:.4e}")
exit()
