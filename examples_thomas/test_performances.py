import time

import numpy as np
from sorts.radar.measurements.measurement import get_bounds

# init 
n_iter = 10
dt_mean = 0.0

N = 10000
arr = np.linspace(0, 10000, N)
ti0 = 121.5
ti1 = 5135.22

for n in range(1, n_iter):
	t0 = time.time()

	# code section to evaluate :
	a0 = np.argmax(arr >= ti0)
	if arr[a0-1] > ti0:
		a0 -= 1

	a1 = np.argmax(arr[a0:] > ti1)
	if arr[a1+1] <= ti1:
		a1 += 1

	a = arr[a0:a1]

	# compute mean time
	dt = time.time() - t0
	dt_mean = ((n-1)*dt_mean + dt)/n

print(f"Performance estimation (N={n_iter} iterations) : {dt_mean} seconds.")

dt_mean = 0.0
for n in range(1, n_iter):
	t0 = time.time()

	# code section to evaluate :
	mask = np.logical_and(arr >= ti0, arr <= ti1)
	a = arr[mask]

	# compute mean time
	dt = time.time() - t0
	dt_mean = ((n-1)*dt_mean + dt)/n

print(f"Performance estimation (N={n_iter} iterations) : {dt_mean} seconds.")