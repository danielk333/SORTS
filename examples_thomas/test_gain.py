import sorts
import numpy as np


radar = sorts.radars.eiscat3d
station = radar.tx[0]

N = 10
dirs = np.array([0.1, 1, 0])
dirs = (dirs/np.linalg.norm(dirs))[:, None].reshape(3, -1)
dirs = dirs.repeat(N, axis = 1)
k = dirs
k = k/np.linalg.norm(k, axis=0)
r2_ = k[0,...]**2 + k[1,...]**2
print(r2_)
station.point_ecef(dirs)
print(dirs)


station.wavelength = np.repeat(station.wavelength, 10)
enu = dirs.repeat(N, axis = 1)

g = station.beam.gain(station.to_ecef(enu))
print(g)