import matplotlib.pyplot as plt
import numpy as np
import sorts

grid_size = [10, 10, 10]
R_earth = 6371e3


pop = sorts.population.orbit_grid(
    semi_major_axis_samples=np.linspace(R_earth + 300e3, R_earth + 1200e3, num=grid_size[0]),
    eccentricity_samples=np.array([0]),
    inclination_samples=np.linspace(60, 120, num=grid_size[1]),
    argument_of_periapsis_samples=np.array([0]),
    longitude_of_ascending_node_samples=np.array([0]),
    mean_anomaly_samples=np.array([0]),
    diameter_samples=10 ** np.linspace(-2, 1, num=grid_size[2]),
    propagator=sorts.propagator.SGP4,
    propagator_options={"settings": {"in_frame": "TEME", "out_frame": "ITRF"}},
)

spobj = pop.get_object(0)
print(spobj)
print(spobj.state)

states = spobj.get_state(np.arange(0, 24*3600, 320))

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(221)
ax.plot(pop.data["a"], pop.data["i"])

ax = fig.add_subplot(223)
ax.plot(pop.data["a"], pop.data["d"])

ax = fig.add_subplot(122, projection='3d')
ax.plot(states[0,:], states[1,:], states[2,:])

plt.show()

