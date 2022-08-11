import sorts
import numpy as np

radar = sorts.radars.eiscat3d

# uncertainty in pointing direction 
sigma_az = 0.1 # deg
sigma_el = 0.15 # deg

az = 0
el = np.pi/2

J = np.array([	[np.sin(el)*np.cos(az),    -np.cos(el)*np.sin(az)],
				[np.sin(el)*np.sin(az), 	np.cos(el)*np.cos(az)],
				[-np.cos(el), 				np.sin(az)]])

# uncertainty in enu frame
sigma_enu = J.dot(np.diag([sigma_el**2, sigma_az**2])).dot(J.T)
print(sigma_enu)

J = sorts.frames.enu_to_ecef(radar.tx[0].lat, radar.tx[0].lon, radar.tx[0].alt, np.eye(3)).reshape(3, 3)
sigma_ecef = J.dot(sigma_enu).dot(J.T)

print(sigma_ecef)

az = 0 + np.random.normal(0, sigma_az, 1000000)
el = np.pi/2 + np.random.normal(0, sigma_el, 1000000)

k = np.array([	[np.cos(el)*np.cos(az)],
				[np.cos(el)*np.sin(az)],
				[np.sin(el)]]).reshape(3, -1)

print(np.cov(k))

k_ecef = sorts.frames.enu_to_ecef(radar.tx[0].lat, radar.tx[0].lon, radar.tx[0].alt, k).reshape(3, -1)

sigma_ecef = np.cov(k_ecef)
print(sigma_ecef)